"""
Prompts for thought generation in Tree-of-Thoughts.
"""

THOUGHT_GENERATION_PROMPT = """# Задача
Ты — агент для генерации учебных материалов по алгоритмам и структурам данных.

# Текущее состояние
- **Запрос пользователя**: {query}
- **Уровень пользователя**: {user_level}
- **Глубина поиска**: {depth}
- **Текущая полнота материала**: {completeness:.2f} (цель: 0.85)
- **Собрано документов**: {collected_info_summary}

# Доступные инструменты
1. **adaptive_rag_search** — поиск в локальной базе знаний (быстрый, точный)
   - Params: {{"query": str, "strategy": "auto|tfidf|semantic|hybrid", "k": int}}

2. **corrective_check** — проверка релевантности собранных документов
   - Params: {{"query": str, "documents": list[str], "min_relevance": float}}

3. **web_search** — поиск в интернете через 4get (медленнее, но свежая информация)
   - Params: {{"query": str, "num_results": int, "scrape_content": bool}}

4. **web_scraper** — загрузка контента с конкретных URL
   - Params: {{"urls": list[str]}}

5. **extract_concepts** — извлечение ключевых концепций из текста
   - Params: {{"text": str, "method": "auto|keybert|spacy", "top_n": int}}

6. **memory_retrieval** — поиск успешных стратегий в памяти агента
   - Params: {{"query": str, "memory_type": "procedural", "limit": int}}

# Подсказки из памяти
{memory_hints}

# Твоя задача
Сгенерируй {branching_factor} варианта следующего шага (мысли + действия).
Каждая мысль должна:
1. Объяснять, почему этот шаг важен
2. Выбирать наиболее подходящий инструмент
3. Задавать правильные параметры для инструмента

# Формат ответа (JSON)
{{
  "thoughts": [
    {{
      "reasoning": "Почему я делаю этот шаг (2-3 предложения)",
      "tool_name": "adaptive_rag_search",
      "tool_params": {{"query": "...", "k": 5}},
      "explanation": "Почему эта ветка перспективна"
    }},
    ...
  ]
}}

# Важно
- НЕ повторяй уже выполненные действия
- Если completeness >= 0.85, можно завершить поиск
- Приоритет: сначала RAG, потом веб-поиск (RAG быстрее)
- Для сложных тем используй hybrid стратегию в RAG

Ответ (только JSON):"""


THOUGHT_GENERATION_FALLBACK_PROMPT = """# Резервная генерация мыслей

Глубина: {depth}
Полнота: {completeness}

Стандартные стратегии:
- depth=0: adaptive_rag (semantic) → базовая теория
- depth=1: corrective_check → проверка качества
- depth=2: web_search → дополнительные примеры
- depth=3+: extract_concepts → углубление

Выбери подходящий инструмент для depth={depth}.

Ответ (JSON):
{{
  "thoughts": [
    {{
      "reasoning": "...",
      "tool_name": "...",
      "tool_params": {{}}
    }}
  ]
}}
"""
