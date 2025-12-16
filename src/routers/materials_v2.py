"""
Materials v2 Router with Tree-of-Thoughts.
Code from Section 11.2 of architecture.
"""

import logging
import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from src.agents.input_validation_agent import InputValidationAgent
from src.agents.materials_agent_v2 import ToTOrchestrator
from src.config import get_settings
from src.core.database import MaterialGeneration, get_db
from src.core.llm import TaskType, get_llm_router
from src.core.memory_manager import MemoryManager
from src.exceptions import (
    InvalidInputError,
    LLMUnavailableError,
    PromptInjectionError,
    ToTSearchError,
)
from src.models.schemas import (
    GenerateMaterialV2Request,
    GenerateMaterialV2Response,
    GetGenerationStatusResponse,
    MaterialSource,
    ToTNodeInfo,
    ToTSearchMetrics,
)
from src.prompts.final_generation_prompts import FINAL_MATERIAL_GENERATION_PROMPT

logger = logging.getLogger(__name__)
router = APIRouter()
settings = get_settings()

# In-memory storage for async status tracking
generation_status = {}


@router.post("/generate", response_model=GenerateMaterialV2Response)
async def generate_material_v2(
    request: GenerateMaterialV2Request, db: Annotated[AsyncSession, Depends(get_db)]
):
    """
    Generate educational materials using Tree-of-Thoughts.

    This is the main v2 endpoint that uses ToT orchestrator.
    """

    generation_id = f"gen_{uuid.uuid4().hex[:12]}"

    logger.info(f"🚀 Starting v2 generation {generation_id}")
    logger.info(f"   Topic: {request.topic}")
    logger.info(f"   User: {request.user_id} ({request.user_level})")

    # ═══════════════════════════════════════════════════════════════
    # PHASE 1: INPUT VALIDATION
    # ═══════════════════════════════════════════════════════════════

    validation_agent = InputValidationAgent()

    try:
        await validation_agent.validate(request.topic)
        logger.info("✅ Input validated")
    except PromptInjectionError as e:
        logger.exception(f"❌ Prompt injection detected: {e}")
        raise HTTPException(status_code=400, detail=f"Prompt injection detected: {e!s}")
    except InvalidInputError as e:
        logger.exception(f"❌ Invalid input: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid input: {e!s}")

    # ═══════════════════════════════════════════════════════════════
    # PHASE 2: LOAD MEMORY CONTEXT
    # ═══════════════════════════════════════════════════════════════

    memory_manager = MemoryManager()

    try:
        memory_context = await memory_manager.load_context(
            user_id=request.user_id, query=request.topic
        )
        logger.info(f"🧠 Memory context loaded: session={memory_context.session_id}")
        logger.info(f"   Procedural patterns: {len(memory_context.patterns)}")
    except Exception as e:
        logger.warning(f"⚠️ Memory load failed: {e}, continuing without memory")
        # Create minimal context
        from src.core.memory.memory_schemas import MemoryContext

        memory_context = MemoryContext(
            session_id=f"sess_{uuid.uuid4().hex[:12]}",
            user_id=request.user_id,
            procedural_hints="No memory available",
            patterns=[],
        )

    # ═══════════════════════════════════════════════════════════════
    # PHASE 3: TOT SEARCH
    # ═══════════════════════════════════════════════════════════════

    orchestrator = ToTOrchestrator()

    # Apply request overrides
    if request.max_iterations:
        orchestrator.max_depth = request.max_iterations
    if request.completeness_threshold:
        orchestrator.completeness_threshold = request.completeness_threshold

    try:
        tot_result = await orchestrator.search(
            query=request.topic, user_level=request.user_level, memory_context=memory_context
        )

        logger.info("✅ ToT search complete:")
        logger.info(f"   Iterations: {tot_result.iterations}")
        logger.info(f"   Completeness: {tot_result.final_completeness:.2f}")
        logger.info(f"   Documents: {len(tot_result.collected_documents)}")

    except LLMUnavailableError as e:
        logger.exception(f"❌ LLM unavailable: {e}")
        raise HTTPException(status_code=503, detail=f"LLM service unavailable: {e!s}")
    except ToTSearchError as e:
        logger.exception(f"❌ ToT search failed: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {e!s}")
    except Exception as e:
        logger.exception(f"❌ Unexpected error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal error: {e!s}")

    # ═══════════════════════════════════════════════════════════════
    # PHASE 4: GENERATE FINAL MATERIAL
    # ═══════════════════════════════════════════════════════════════

    llm_router = get_llm_router()
    llm_expensive = llm_router.get_model_for_task(TaskType.FINAL_GENERATION)

    # Format collected documents
    documents_text = "\n\n---\n\n".join([
        f"[{doc.metadata.get('source', 'unknown')}]\n{doc.page_content}"
        for doc in tot_result.collected_documents[:10]  # Top 10
    ])

    # Generate final material

    final_prompt = FINAL_MATERIAL_GENERATION_PROMPT.format(
        topic=request.topic,
        user_level=request.user_level,
        language=request.language,
        collected_documents=documents_text,
    )

    try:
        response = await llm_expensive.ainvoke(final_prompt)
        final_material = response.text

        logger.info(f"✅ Final material generated: {len(final_material)} chars")

    except Exception as e:
        logger.exception(f"❌ Final generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Final generation failed: {e!s}")

    # ═══════════════════════════════════════════════════════════════
    # PHASE 5: SAVE TO MEMORY & DATABASE
    # ═══════════════════════════════════════════════════════════════

    # Save successful pattern to procedural memory
    if tot_result.final_completeness >= settings.memory.memory_procedural_min_success_score:
        try:
            await memory_manager.save_successful_generation(
                session_id=memory_context.session_id,
                tot_result=tot_result,
                query=request.topic,
                user_level=request.user_level,
            )
            logger.info("✅ Saved to procedural memory")
        except Exception as e:
            logger.warning(f"⚠️ Failed to save to procedural memory: {e}")

    # Save to database
    tool_counts = {}
    for tool in tot_result.tools_used:
        tool_counts[tool] = tool_counts.get(tool, 0) + 1

    generation_record = MaterialGeneration(
        generation_id=generation_id,
        user_id=request.user_id,
        topic=request.topic,
        user_level=request.user_level,
        tot_iterations=tot_result.iterations,
        tot_explored_nodes=len(tot_result.explored_nodes),
        tot_dead_end_nodes=len([n for n in tot_result.explored_nodes if n.status == "DEAD_END"]),
        tot_best_path_depth=len(tot_result.best_path),
        tools_used=tot_result.tools_used,
        tool_call_counts=tool_counts,
        gigachat2_max_calls=tot_result.llm_usage.get("gigachat2", 0),
        gigachat3_calls=tot_result.llm_usage.get("gigachat3", 0),
        estimated_cost_usd=_calculate_cost(tot_result.llm_usage),
        success=True,
        final_completeness_score=tot_result.final_completeness,
        documents_collected=len(tot_result.collected_documents),
        material_length=len(final_material),
        material_content=final_material,
        generation_time_seconds=tot_result.total_time,
        memory_hints_used=len(memory_context.patterns) > 0,
        content_guard_filtered=0,  # TODO: Track this
        fallbacks_triggered=[],
    )

    db.add(generation_record)
    await db.commit()

    logger.info(f"✅ Saved to database: {generation_id}")

    # ═══════════════════════════════════════════════════════════════
    # PHASE 6: BUILD RESPONSE
    # ═══════════════════════════════════════════════════════════════

    # Convert sources
    sources = [
        MaterialSource(
            source_type=doc.metadata.get("source", "unknown"),
            url=doc.metadata.get("url"),
            title=doc.metadata.get("title"),
            relevance_score=doc.relevance_score if hasattr(doc, "relevance_score") else 1.0,
        )
        for doc in tot_result.collected_documents
    ]
    if tot_result.best_path:
        best_solution = tot_result.best_path[-1]
        final_relevance = best_solution.relevance_score
        final_quality = best_solution.quality_score
    # Build metrics
    metrics = ToTSearchMetrics(
        total_iterations=tot_result.iterations,
        explored_nodes=len(tot_result.explored_nodes),
        best_path_length=len(tot_result.best_path),
        final_completeness=tot_result.final_completeness,
        tools_used=tot_result.tools_used,
        tool_call_counts=tool_counts,
        gigachat2_max_calls=tot_result.llm_usage.get("gigachat2", 0),
        gigachat3_calls=tot_result.llm_usage.get("gigachat3", 0),
        estimated_cost_usd=_calculate_cost(tot_result.llm_usage),
        total_time_seconds=tot_result.total_time,
        memory_hints_used=len(memory_context.patterns) > 0,
        procedural_patterns_found=len(memory_context.patterns),
        final_relevance=final_relevance,  # pyright: ignore[reportPossiblyUnboundVariable]
        final_quality=final_quality,  # pyright: ignore[reportPossiblyUnboundVariable]
    )

    # Best path (optional)
    best_path_info = None
    if settings.api.include_debug_info:
        best_path_info = [
            ToTNodeInfo(
                node_id=node.node_id,
                depth=node.depth,
                thought=node.thought,
                tool_used=node.planned_action.get("tool_name"),
                completeness_score=node.completeness_score,
                promise_score=node.promise_score,
                status=node.status.value if hasattr(node.status, "value") else str(node.status),
            )
            for node in tot_result.best_path
        ]

    return GenerateMaterialV2Response(
        generation_id=generation_id,
        success=True,
        material=final_material,
        word_count=len(final_material.split()),
        sources=sources,
        documents_collected=len(tot_result.collected_documents),
        tot_metrics=metrics,
        best_path=best_path_info,
        warnings=[],
        fallbacks_used=[],
    )


def _calculate_cost(llm_usage: dict) -> float:
    """
    Calculate estimated cost in USD.

    Pricing (approximate):
    - GigaChat-2-Max: $0.002 per call
    - GigaChat3: $0.0005 per call
    """

    gigachat2_calls = llm_usage.get("gigachat2", 0)
    gigachat3_calls = llm_usage.get("gigachat3", 0)

    cost = (gigachat2_calls * 0.002) + (gigachat3_calls * 0.0005)

    return round(cost, 4)


@router.get("/status/{generation_id}", response_model=GetGenerationStatusResponse)
async def get_generation_status(generation_id: str, db: Annotated[AsyncSession, Depends(get_db)]):
    """
    Get status of a generation request.

    Useful for async/background processing (future feature).
    """

    # For now, just query from database
    from sqlalchemy import select

    result = await db.execute(
        select(MaterialGeneration).where(MaterialGeneration.generation_id == generation_id)
    )
    record = result.scalar_one_or_none()

    if not record:
        raise HTTPException(status_code=404, detail="Generation not found")

    return GetGenerationStatusResponse(
        generation_id=generation_id,
        status="completed" if record.success else "failed",  # pyright: ignore[reportGeneralTypeIssues]
        progress=1.0,
        current_iteration=record.tot_iterations,  # pyright: ignore[reportArgumentType]
        estimated_time_remaining=0.0,
    )
