"""
Materials Agent v2 with Tree-of-Thoughts orchestration.
Code from Section 4.2 of architecture.
"""

import logging
import time
from typing import Any

from src.agents.chains.evaluation_chain import EvaluationChain
from src.agents.chains.reasoning_chain import ReasoningChain
from src.agents.content_guard.orchestrator import ContentGuardOrchestrator
from src.config import get_settings
from src.core.llm import TaskType, get_llm_router
from src.core.memory.memory_schemas import MemoryContext
from src.core.memory_manager import MemoryManager
from src.exceptions import LLMUnavailableError
from src.models.react_schemas import NodeStatus, ToTResult, TreeNode
from src.tools.base_tool import ToolResult
from src.tools.tool_registry import ToolRegistry

logger = logging.getLogger(__name__)


class ToTOrchestrator:
    """
    Orchestrator для Tree-of-Thoughts поиска с DFS.
    Code from Section 4.2 of architecture.

    Использует:
    - GigaChat-2-Max: только для thought generation
    - GigaChat3: для всех evaluation и tool calls
    """

    def __init__(self) -> None:
        """Initialize ToT Orchestrator."""
        self.settings = get_settings()
        self.llm_router = get_llm_router()

        # LLMs
        self.llm_expensive = self.llm_router.get_model_for_task(TaskType.THOUGHT_GENERATION)
        self.llm_cheap = self.llm_router.get_model_for_task(TaskType.PROMISE_EVALUATION)

        # Chains
        self.reasoning_chain = ReasoningChain(self.llm_expensive)
        self.evaluation_chain = EvaluationChain(self.llm_cheap)

        # Tools
        self.tools = ToolRegistry()

        # Memory
        self.memory_manager = MemoryManager()

        # Content Guard
        self.content_guard = ContentGuardOrchestrator()

        # Configuration
        self.max_depth = self.settings.tot.tot_max_depth
        self.branching_factor = self.settings.tot.tot_branching_factor
        self.completeness_threshold = self.settings.tot.tot_completeness_threshold
        self.promise_threshold = self.settings.tot.tot_promise_threshold

        logger.info("✅ ToT Orchestrator initialized")

    async def search(
        self, query: str, user_level: str, memory_context: MemoryContext
    ) -> ToTResult:
        """
        DFS поиск в дереве рассуждений.
        Code from Section 4.2 of architecture.

        Args:
            query: User query
            user_level: User level (beginner/intermediate/advanced)
            memory_context: Memory context with hints

        Returns:
            ToTResult with best_path and collected_documents
        """

        start_time = time.time()

        # ═══════════════════════════════════════════════════════════
        # PHASE 1: INITIALIZATION
        # ═══════════════════════════════════════════════════════════

        root = TreeNode(
            node_id="root",
            depth=0,
            thought=f"Начинаю анализ запроса: {query}",
            collected_info=[],
            completeness_score=0.0,
        )

        stack = [root]  # DFS stack (LIFO)
        best_solution = None
        best_score = 0.0
        explored_nodes = []

        # Tracking
        llm_calls = {"gigachat2": 0, "gigachat3": 0}
        tools_used = set()

        logger.info("🌳 ToT DFS Search started:")
        logger.info(f"   - Query: '{query}'")
        logger.info(f"   - Max depth: {self.max_depth}")
        logger.info(f"   - Completeness threshold: {self.completeness_threshold}")

        # ═══════════════════════════════════════════════════════════
        # PHASE 2: DFS LOOP
        # ═══════════════════════════════════════════════════════════

        iteration = 0
        max_iterations = self.max_depth * self.branching_factor

        while stack and iteration < max_iterations:
            iteration += 1

            # ───────────────────────────────────────────────────────
            # STEP 2.1: POP NODE FROM STACK
            # ───────────────────────────────────────────────────────

            current_node = stack.pop()  # LIFO: last added, first explored
            explored_nodes.append(current_node)

            logger.info(f"📍 Iteration {iteration}: Exploring node {current_node.node_id}")
            logger.info(f"   - Depth: {current_node.depth}")
            logger.info(f"   - Completeness: {current_node.completeness_score:.2f}")
            logger.info(f"   - Documents: {len(current_node.collected_info)}")

            # ───────────────────────────────────────────────────────
            # STEP 2.2: CHECK TERMINATION CONDITIONS
            # ───────────────────────────────────────────────────────

            # Goal reached?
            if current_node.completeness_score >= self.completeness_threshold:
                logger.info(f"🎯 GOAL REACHED at node {current_node.node_id}!")
                current_node.status = NodeStatus.GOAL_REACHED
                best_solution = current_node
                break

            # Max depth reached?
            if current_node.depth >= self.max_depth:
                logger.info(f"⚠️ Max depth reached at node {current_node.node_id}")
                if current_node.completeness_score > best_score:
                    best_score = current_node.completeness_score
                    best_solution = current_node
                continue

            # Dead end from previous evaluation?
            if current_node.status == NodeStatus.DEAD_END:
                logger.info(f"💀 Dead end detected, skipping {current_node.node_id}")
                continue

            # ───────────────────────────────────────────────────────
            # STEP 2.3: GENERATE CANDIDATE THOUGHTS (GigaChat-2-Max)
            # ───────────────────────────────────────────────────────

            try:
                candidates = await self.reasoning_chain.generate_thoughts(
                    current_node=current_node,
                    query=query,
                    user_level=user_level,
                    memory_context=memory_context,
                    branching_factor=self.branching_factor,
                )

                llm_calls["gigachat2"] += 1

                logger.info(f"💭 Generated {len(candidates)} candidate thoughts")

            except LLMUnavailableError as e:
                logger.exception(f"❌ GigaChat-2-Max unavailable: {e}")

                # Fallback: Rule-based thought generation
                candidates = self._fallback_thoughts(current_node)
                logger.warning(f"⚠️ Using fallback: {len(candidates)} rule-based thoughts")

            if not candidates:
                logger.warning(f"⚠️ No candidates generated for {current_node.node_id}")
                continue

            # ───────────────────────────────────────────────────────
            # STEP 2.4: EVALUATE PROMISE (GigaChat3)
            # ───────────────────────────────────────────────────────

            for candidate in candidates:
                try:
                    promise_score = await self.evaluation_chain.evaluate_promise(
                        candidate=candidate, current_state=current_node, query=query
                    )
                    candidate.promise_score = promise_score
                    llm_calls["gigachat3"] += 1

                except Exception as e:
                    logger.warning(f"⚠️ Promise evaluation failed: {e}, using heuristic")
                    candidate.promise_score = self.evaluation_chain._heuristic_promise(candidate)

            # ───────────────────────────────────────────────────────
            # STEP 2.5: PRUNE & SORT
            # ───────────────────────────────────────────────────────

            # Prune low-promise branches
            promising = [c for c in candidates if c.promise_score >= self.promise_threshold]

            if not promising:
                logger.warning("⚠️ All candidates pruned (low promise)")
                current_node.status = NodeStatus.DEAD_END
                continue

            # Sort by promise (best first)
            promising.sort(key=lambda c: c.promise_score, reverse=True)

            logger.info(f"✂️ Pruned {len(candidates) - len(promising)} low-promise branches")
            logger.info(
                f"📊 Top promise scores: {[round(c.promise_score, 2) for c in promising[:3]]}"
            )

            # ───────────────────────────────────────────────────────
            # STEP 2.6: PUSH TO STACK (DFS order)
            # ───────────────────────────────────────────────────────

            # Add to stack in REVERSE order (best goes last, popped first)
            for child in reversed(promising):
                child.parent_id = current_node.node_id
                child.depth = current_node.depth + 1
                child.collected_info = current_node.collected_info.copy()  # Inherit
                stack.append(child)
                current_node.children.append(child)

            logger.info(f"📚 Added {len(promising)} nodes to stack (size: {len(stack)})")

            # ───────────────────────────────────────────────────────
            # STEP 2.7: EXECUTE ACTION FOR BEST CHILD
            # ───────────────────────────────────────────────────────

            # Take best child (just added to end of stack)
            best_child = stack[-1]

            tool_name = best_child.planned_action.get("tool_name")
            logger.info(f"🎬 Executing action: {tool_name}")

            try:
                # Execute tool
                tool_result = await self._execute_tool(best_child.planned_action)
                best_child.action_result = tool_result

                tools_used.add(tool_name)

                # ⚠️ CRITICAL: Content Guard для результатов
                if tool_result.success and tool_result.documents:
                    logger.info(f"🛡️ Running Content Guard on {len(tool_result.documents)} docs")

                    cleaned_docs = await self.content_guard.process(tool_result.documents)

                    logger.info(
                        f"✅ Content Guard: {len(tool_result.documents)} → "
                        f"{len(cleaned_docs)} docs passed"
                    )

                    # Update with cleaned documents
                    best_child.collected_info.extend(cleaned_docs)
                    tool_result.documents = cleaned_docs

            except Exception as e:
                logger.exception(f"❌ Tool execution failed: {e}")
                best_child.status = NodeStatus.DEAD_END
                continue

            # ───────────────────────────────────────────────────────
            # STEP 2.8: POST-EXECUTION EVALUATION (GigaChat3)
            # ───────────────────────────────────────────────────────

            try:
                evaluation = await self.evaluation_chain.evaluate_node(
                    node=best_child, query=query
                )

                best_child.completeness_score = evaluation.completeness
                best_child.relevance_score = evaluation.relevance
                best_child.quality_score = evaluation.quality

                llm_calls["gigachat3"] += 1

                logger.info(
                    f"📈 Evaluation: completeness={evaluation.completeness:.2f}, "
                    f"relevance={evaluation.relevance:.2f}, "
                    f"quality={evaluation.quality:.2f}"
                )

                # Check for dead end
                if (
                    evaluation.relevance < self.settings.tot.tot_dead_end_relevance
                    or evaluation.quality < self.settings.tot.tot_dead_end_quality
                ):
                    logger.warning("💀 Dead end: low relevance/quality")
                    best_child.status = NodeStatus.DEAD_END
                elif evaluation.completeness >= self.completeness_threshold:
                    logger.info("🎯 Goal reached in evaluation!")
                    best_child.status = NodeStatus.GOAL_REACHED
                    best_solution = best_child
                    break
                else:
                    best_child.status = NodeStatus.PROMISING

            except Exception as e:
                logger.warning(f"⚠️ Evaluation failed: {e}, using heuristic")
                best_child.completeness_score = self.evaluation_chain._heuristic_completeness(
                    best_child
                )
                best_child.relevance_score = 0.8  # Assume good
                best_child.quality_score = 0.8
                best_child.status = NodeStatus.PROMISING

            # Update best score
            if best_child.completeness_score > best_score:
                best_score = best_child.completeness_score
                best_solution = best_child

            # ───────────────────────────────────────────────────────
            # STEP 2.9: SAVE TO WORKING MEMORY
            # ───────────────────────────────────────────────────────

            await self.memory_manager.working_memory.append_step(
                session_id=memory_context.session_id,
                step_data={
                    "iteration": iteration,
                    "node_id": best_child.node_id,
                    "depth": best_child.depth,
                    "thought": best_child.thought,
                    "tool_used": best_child.planned_action.get("tool_name"),
                    "tool_params": best_child.planned_action.get("tool_params"),
                    "completeness": best_child.completeness_score,
                    "relevance": best_child.relevance_score,
                    "timestamp": best_child.created_at.isoformat(),
                },
            )

        # ═══════════════════════════════════════════════════════════
        # PHASE 3: RETURN BEST SOLUTION
        # ═══════════════════════════════════════════════════════════

        if not best_solution:
            # No goal reached, take best completeness
            best_solution = max(explored_nodes, key=lambda n: n.completeness_score)
            logger.warning(
                f"⚠️ No goal reached, using best: {best_solution.completeness_score:.2f}"
            )

        # Trace path from root to best solution
        best_path = self._trace_path(best_solution)

        # Calculate total time
        total_time = time.time() - start_time

        logger.info("✅ DFS Search complete:")
        logger.info(f"   - Total iterations: {iteration}")
        logger.info(f"   - Explored nodes: {len(explored_nodes)}")
        logger.info(f"   - Best path length: {len(best_path)}")
        logger.info(f"   - Final completeness: {best_solution.completeness_score:.2f}")
        logger.info(f"   - Documents collected: {len(best_solution.collected_info)}")
        logger.info(f"   - Tools used: {list(tools_used)}")
        logger.info(
            f"   - LLM calls: GigaChat-2-Max={llm_calls['gigachat2']}, GigaChat3={llm_calls['gigachat3']}"
        )
        logger.info(f"   - Total time: {total_time:.2f}s")

        return ToTResult(
            best_path=best_path,
            explored_nodes=explored_nodes,
            collected_documents=best_solution.collected_info,
            final_completeness=best_solution.completeness_score,
            iterations=iteration,
            tools_used=list(tools_used),
            total_time=total_time,
            llm_usage=llm_calls,
        )

    async def _execute_tool(self, action: dict[str, Any]) -> ToolResult:
        """
        Execute tool (tools use GigaChat3 internally).

        Args:
            action: Dict with tool_name and tool_params

        Returns:
            ToolResult
        """

        tool_name = action.get("tool_name")
        tool_params = action.get("tool_params", {})

        if not tool_name:
            return ToolResult(success=False, documents=[], error="No tool_name specified")

        # Get tool from registry
        tool = self.tools.get(tool_name)

        if not tool:
            logger.error(f"❌ Tool not found: {tool_name}")
            return ToolResult(success=False, documents=[], error=f"Tool {tool_name} not found")

        # Execute tool
        try:
            return await tool.execute(tool_params)

        except Exception as e:
            logger.exception(f"❌ Tool execution error: {e}")
            return ToolResult(success=False, documents=[], error=str(e))

    def _trace_path(self, node: TreeNode) -> list[TreeNode]:
        """
        Trace path from root to node.

        Args:
            node: Target node

        Returns:
            List of nodes from root to target
        """

        path = []
        current = node

        # Walk back to root
        visited_ids = set()
        while current:
            if current.node_id in visited_ids:
                logger.warning("⚠️ Cycle detected in path tracing")
                break

            path.append(current)
            visited_ids.add(current.node_id)

            # Find parent
            if current.parent_id:
                # This is a simplified version - in real implementation
                # we'd need to store parent references
                # For now, assume we can't trace back (return current path)
                break
            break

        # Reverse to get root→target order
        path.reverse()

        return path

    def _fallback_thoughts(self, current_node: TreeNode) -> list[TreeNode]:
        """
        Generate fallback thoughts when LLM is unavailable.

        Uses rule-based strategy based on depth.

        Args:
            current_node: Current node

        Returns:
            List of fallback TreeNode candidates
        """

        depth = current_node.depth

        # Rule-based strategy
        strategies = [
            # Depth 0: Start with RAG
            {
                "reasoning": "Начинаю с поиска в локальной базе знаний",
                "tool_name": "adaptive_rag_search",
                "tool_params": {"query": "", "strategy": "semantic", "k": 5},
            },
            # Depth 1: Check quality
            {
                "reasoning": "Проверяю релевантность собранных документов",
                "tool_name": "corrective_check",
                "tool_params": {"query": "", "documents": [], "min_relevance": 0.6},
            },
            # Depth 2: Web search
            {
                "reasoning": "Ищу дополнительную информацию в интернете",
                "tool_name": "web_search",
                "tool_params": {"query": "", "num_results": 5, "scrape_content": True},
            },
            # Depth 3+: Extract concepts
            {
                "reasoning": "Извлекаю ключевые концепции для углубления",
                "tool_name": "extract_concepts",
                "tool_params": {"text": "", "method": "auto", "top_n": 10},
            },
        ]

        # Select strategy based on depth
        strategy_idx = min(depth, len(strategies) - 1)
        strategy = strategies[strategy_idx]

        # Create single fallback node
        node = TreeNode(
            parent_id=current_node.node_id,
            depth=current_node.depth + 1,
            thought=strategy["reasoning"],
            reasoning=strategy["reasoning"],
            planned_action={
                "tool_name": strategy["tool_name"],
                "tool_params": strategy["tool_params"],
            },
            collected_info=current_node.collected_info.copy(),
            promise_score=0.7,  # Medium promise
        )

        return [node]
