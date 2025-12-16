# 🎓 АиСД Learning Platform v2: Интеллектуальная платформа для изучения алгоритмов и структур данных

Учебный проект, демонстрирующий применение **агентных систем на базе LLM** с **Tree-of-Thoughts рассуждением**, **Content Guards** и **оптимизацией затрат** для персонализированного обучения алгоритмам и структурам данных.

---

## 🧭 Основная идея

Проект реализует полноценную образовательную экосистему с использованием **FastAPI**, **LangChain**, **ReActивного подхода** с **Tree-of-Thoughts (ToT)**, **Adaptive/Corrective RAG**, **Web Search** и **двух LLM-провайдеров** (GigaChat-2-Max для сложного reasoning, GigaChat3 для быстрых операций).

**Ключевые возможности v2:**

- 🧠 **Tree-of-Thoughts Orchestrator** с DFS для интеллектуального планирования
- 🛡️ **Content Guard Layer** для фильтрации токсичного контента
- 💰 **Cost Optimization** через распределение моделей (экономия ~75%)
- 🔍 **Adaptive RAG** с выбором стратегии поиска (TF-IDF, Semantic, Hybrid)
- ✅ **Corrective RAG** с оценкой релевантности и корректировкой
- 🌐 **Web Search & Scraping** с 4get metasearch и selectolax
- 🧠 **Procedural Memory** для обучения на успешных паттернах
- 💾 **Working Memory** для контекста сессии
- 🔄 **Fallback Chains** для устойчивости к сбоям
- 📊 **Metrics & Monitoring** с Prometheus-совместимым экспортом

---

## 🏗️ Архитектура v2

### High-Level Overview

```
┌────────────────────────────────────────────────────────────────┐
│                     USER REQUEST                               │
│              {topic, user_level, user_id}                      │
└──────────────────────────┬─────────────────────────────────────┘
                           │
                           ▼
    ┌──────────────────────────────────────────────────────┐
    │   INPUT VALIDATION (GigaChat3)                       │
    │   - Prompt injection detection                       │
    │   - Format validation                                │
    └──────────────────────┬───────────────────────────────┘
                           │
                           ▼
    ┌──────────────────────────────────────────────────────┐
    │   MEMORY LOADING (ChromaDB)                          │
    │   - Working Memory (session context)                 │
    │   - Procedural Memory (success patterns)             │
    └──────────────────────┬───────────────────────────────┘
                           │
                           ▼
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃         ToT ORCHESTRATOR (GigaChat-2-Max)                ┃
┃               💰 EXPENSIVE LAYER                         ┃
┃                                                          ┃
┃  THOUGHT → ACTION → OBSERVATION → EVALUATION             ┃
┃  ~5-7 iterations, DFS search                             ┃
┗━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
                         │
                         ▼
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃         TOOL LAYER (GigaChat3 Workers)                    ┃
┃               🆓 FREE/CHEAP LAYER                          ┃
┃                                                           ┃
┃  -  Adaptive RAG Tool (strategy selection)                ┃
┃  -  Corrective RAG Tool (relevance check)                 ┃
┃  -  Web Search Tool (4get metasearch)                     ┃
┃  -  Web Scraper Tool (content extraction)                 ┃
┃  -  Concept Extractor Tool (KeyBERT, spaCy)               ┃
┃  -  Memory Retrieval Tool (procedural patterns)           ┃
┗━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
                         │
                         ▼
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃      CONTENT GUARD LAYER (GigaChat3 Guards)              ┃
┃               🛡️ SAFETY LAYER                            ┃
┃                                                          ┃
┃  1. Toxicity Check (batch processing)                    ┃
┃  2. Policy Compliance Check                              ┃
┃  3. Content Sanitization (rule-based)                    ┃
┃  4. Quality Gate (length, sentences, content type)       ┃
┗━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
                         │
                         ▼
    ┌──────────────────────────────────────────────────────┐
    │   FINAL GENERATION (GigaChat-2-Max)                  │
    │   💰 Second expensive call                           │
    │   - Synthesize cleaned materials                      │
    │   - Apply user_level adaptation                       │
    │   - Generate 5000+ chars Markdown                     │
    └──────────────────────┬───────────────────────────────┘
                           │
                           ▼
    ┌──────────────────────────────────────────────────────┐
    │   PERSISTENCE & MEMORY UPDATE                        │
    │   - SQLite: MaterialGeneration record                 │
    │   - ChromaDB: Procedural Memory update                │
    │   - Working Memory: Clear session                     │
    └──────────────────────┬───────────────────────────────┘
                           │
                           ▼
                    RETURN RESPONSE
```

### Cost Optimization

| Компонент              | Модель         | Частота вызовов       | Назначение                  |
| ---------------------- | -------------- | --------------------- | --------------------------- |
| **ToT Orchestrator**   | GigaChat-2-Max | ~5-7 вызовов/запрос   | Сложное reasoning           |
| **Final Generation**   | GigaChat-2-Max | 1 вызов/запрос        | Высокое качество материала  |
| **Input Validation**   | GigaChat3      | 1 вызов/запрос        | Быстрая проверка            |
| **Promise Evaluation** | GigaChat3      | ~15-20 вызовов/запрос | Оценка перспективности      |
| **Relevance Scoring**  | GigaChat3      | ~5-10 вызовов/запрос  | Классификация релевантности |
| **Content Guards**     | GigaChat3      | ~5-15 вызовов/запрос  | Проверка безопасности       |
| **Completeness Check** | GigaChat3      | ~5-7 вызовов/запрос   | Оценка полноты              |

**Экономия: ~75%** по сравнению с использованием только GigaChat-2-Max для всех операций.

---

## 📁 Структура проекта

```
src/
├── agents/                         # Агентная система (LangChain)
│   ├── orchestrator/               # Tree-of-Thoughts оркестратор
│   │   ├── workers/                # Воркеры для различных задач
│   │   │   ├── base_worker.py      # Базовый класс для воркеров
│   │   │   ├── materials_worker.py # Воркер для работы с материалами
│   │   │   ├── support_worker.py   # Воркер психологической поддержки
│   │   │   ├── test_worker.py      # Воркер генерации тестов
│   │   │   └── verification_worker.py # Воркер проверки ответов
│   │   ├── aggregator.py           # Агрегация результатов от воркеров
│   │   ├── classifier.py           # Классификация запросов студентов
│   │   └── executor.py             # Выполнение задач через воркеров
│   ├── chains/                     # LangChain цепочки
│   │   ├── reasoning_chain.py      # Цепочка генерации мыслей (ToT)
│   │   ├── evaluation_chain.py     # Цепочка оценки узлов (promise, completeness)
│   │   └── output_parsers.py       # Парсеры для структурированного вывода
│   ├── content_guard/              # Content Guard компоненты
│   │   ├── orchestrator.py         # Оркестратор Content Guard pipeline
│   │   ├── toxicity_checker.py     # Проверка токсичности (GigaChat3)
│   │   ├── policy_checker.py       # Проверка соответствия политикам
│   │   ├── content_sanitizer.py    # Санитизация контента (rule-based)
│   │   └── quality_gate.py         # Финальная проверка качества
│   ├── input_validation_agent.py   # Агент валидации ввода
│   ├── materials_agent.py          # Агент для адаптации материалов (legacy)
│   ├── materials_agent_v2.py       # Materials Agent v2 с ToT
│   ├── llm_router_agent.py         # Роутер для выбора подходящей LLM
│   ├── support_agent.py            # Агент психологической поддержки
│   ├── test_generation_agent.py    # Агент генерации тестов и задач
│   ├── verification_agent.py       # Агент проверки ответов с двойной верификацией
│   └── registry.py                 # Реестр всех доступных агентов
├── core/                           # Ядро системы
│   ├── cache/                      # Кеширование
│   │   ├── memory_cache.py         # In-memory LRU cache (fallback)
│   │   └── redis_cache.py          # Redis cache (optional)
│   ├── fallback/                   # Fallback стратегии
│   │   ├── llm_fallback.py         # LLM fallback с retry и переключением
│   │   ├── database_fallback.py    # Database fallback на JSON
│   │   └── chromadb_fallback.py    # ChromaDB fallback на pickle
│   ├── memory/                     # Система памяти
│   │   ├── memory_schemas.py       # Pydantic модели для памяти
│   │   ├── working_memory.py       # Working Memory (ChromaDB + in-memory)
│   │   └── procedural_memory.py    # Procedural Memory (успешные паттерны)
│   ├── database.py                 # SQLAlchemy модели (User, Test, MaterialGeneration и др.)
│   ├── llm.py                      # LLMRouter для выбора модели по типу задачи
│   ├── vector_store.py             # VectorStoreManager для ChromaDB
│   └── memory_manager.py           # MemoryManager (unified interface)
├── data_processing/                # Обработка данных для RAG
│   ├── db_populator.py             # Заполнение ChromaDB документами из PDF
│   ├── pdf_parser.py               # Парсинг LaTeX-PDF с извлечением оглавления
│   └── text_splitter.py            # Умное разбиение текста на чанки
├── metrics/                        # Метрики и мониторинг
│   ├── analytics_service.py        # Аналитика ToT производительности
│   ├── deepeval_metrics.py         # DeepEval метрики качества
│   ├── health_service.py           # Health check сервис
│   ├── metrics_collector.py        # Real-time метрики коллектор
│   └── metrics_exporter.py         # Prometheus exporter
├── models/                         # Pydantic-модели
│   ├── schemas.py                  # Схемы запросов и ответов для API
│   ├── orchestrator_schemas.py     # Схемы для оркестратора (legacy)
│   ├── react_schemas.py            # Схемы для ToT (TreeNode, ToTResult)
│   └── content_guard_schemas.py    # Схемы для Content Guard
├── prompts/                        # Промпты для агентов
│   ├── validation_prompts.py       # Промпты для валидации
│   ├── content_guard_prompts.py    # Промпты для Content Guard
│   ├── reasoning_prompts.py        # Промпты для ToT reasoning
│   └── evaluation_prompts.py       # Промпты для оценки
├── routers/                        # REST API маршруты
│   ├── assessment.py               # API первичной оценки знаний
│   ├── health.py                   # Healthcheck и метрики
│   ├── llm_router.py               # API выбора и использования LLM
│   ├── materials.py                # API получения и генерации материалов
│   ├── orchestrator.py             # API для комплексных запросов
│   ├── support.py                  # API психологической поддержки
│   ├── tests.py                    # API генерации и отправки тестов
│   └── verification.py             # API проверки ответов
├── scripts/                        # Утилиты и скрипты
│   ├── measure_secondary_verification.py # Измерение эффективности верификации
│   ├── populate_db.py              # CLI-скрипт для заполнения ChromaDB
│   └── generate_endpoint_report.py # Генерация отчета по API
├── tools/                          # Tools для Materials Agent v2
│   ├── base_tool.py                # Базовый класс для всех tools
│   ├── adaptive_rag_tool.py        # Adaptive RAG с выбором стратегии
│   ├── corrective_rag_tool.py      # Corrective RAG с проверкой релевантности
│   ├── web_search_tool.py          # Web Search через 4get
│   ├── web_scraper_tool.py         # Web Scraper с selectolax
│   ├── concept_extractor_tool.py   # Извлечение концепций (KeyBERT, spaCy)
│   ├── memory_retrieval_tool.py    # Retrieval из Procedural Memory
│   └── tool_registry.py            # Реестр tools с lazy initialization
├── utils/                          # Утилиты
│   ├── logging_decorators.py       # Декораторы для логирования
│   └── validators.py               # Валидаторы
├── config.py                       # Конфигурация через Pydantic Settings
├── exceptions.py                   # Custom exceptions
└── main.py                         # Точка входа FastAPI приложения
```

---

## ⚙️ Установка и запуск

### 1. Установить uv

**uv** — современный пакетный менеджер для Python, написанный на Rust (в 10x быстрее pip).

**Установка (Linux/macOS):**

```
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Установка (Windows):**

```
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**Или через pipx:**

```
pipx install uv
```

Официальная документация: [https://docs.astral.sh/uv/getting-started/installation/](https://docs.astral.sh/uv/getting-started/installation/)

### 2. Клонировать проект

```
git clone https://github.com/ZUB3C/dsa-learning-agent.git
cd dsa-learning-agent
```

### 3. Установить зависимости

```
uv sync
```

Эта команда автоматически создаст виртуальное окружение и установит все зависимости из `pyproject.toml`.

### 4. Настроить `.env`

Скопировать шаблон конфигурации:

```
cp .env.example .env
```

Заполнить API-ключи и настройки в `.env` (см. раздел [Конфигурация](#-конфигурация) ниже).

### 5. Заполнить базу знаний (ChromaDB)

Вначале надо сохранить файл [algobook.pdf](https://www.babichev.org/books/AlgoBook.pdf) в корень проекта.

**Запуск скрипта для заполнения векторной БД из PDF:**

```
uv run python -m src.scripts.populate_db --pdf algobook.pdf --clear
```

**Параметры:**

- `--pdf` — путь к PDF файлу (по умолчанию: `algobook.pdf`)
- `--clear` — очистить существующую коллекцию перед заполнением
- `--chunk-size` — размер чанка в символах (по умолчанию: 1000)
- `--chunk-overlap` — перекрытие чанков (по умолчанию: 200)

### 6. Запустить приложение (API)

```
uv run uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
```

Откройте документацию Swagger UI:
👉 [http://127.0.0.1:8001/docs](http://127.0.0.1:8001/docs)

### 7. Запустить создание отчёта с примерами работы API

```
uv run python -m src.scripts.generate_endpoint_report
```

Полученный отчёт сохраняется в файл `api-examples-report.md`.

---

## 🔧 Конфигурация

Проект использует **Pydantic Settings** для управления конфигурацией через `.env` файл.

### Основные настройки

```
# ════════════════════════════════════════════════════════════════
# LLM CONFIGURATION
# ════════════════════════════════════════════════════════════════

# GigaChat API Settings
GIGACHAT_API_KEY=your_api_key_here
GIGACHAT_BASE_URL=https://foundation-models.api.cloud.ru/v1

# GigaChat-2-Max (expensive, for ToT Orchestrator)
GIGACHAT_MODEL=GigaChat/GigaChat-2-Max
GIGACHAT_TEMPERATURE=0.2

# GigaChat3 (cheap, for tools and guards)
GIGACHAT3_MODEL=ai-sage/GigaChat3-10B-A1.8B
GIGACHAT3_TEMPERATURE=0.1

# ChromaDB
CHROMA_PERSIST_DIRECTORY=.chromadb
CHROMA_RAG_COLLECTION=aisd_materials

# Database
DATABASE_URL=sqlite+aiosqlite:///./app.db
```

---

## 🧩 API Endpoints

### System

| Метод | Эндпоинт           | Описание                                |
| ----- | ------------------ | --------------------------------------- |
| `GET` | `/health`          | Простая проверка состояния              |
| `GET` | `/health/detailed` | Детальный health check всех компонентов |
| `GET` | `/metrics`         | Системные метрики (последний час)       |

### Materials (v2 с ToT)

| Метод  | Эндпоинт                              | Описание                                       |
| ------ | ------------------------------------- | ---------------------------------------------- |
| `POST` | `/api/v1/materials/generate-material` | **Генерация материала через ToT Orchestrator** |
| `POST` | `/api/v1/materials/get-materials`     | Получить адаптированные материалы              |
| `POST` | `/api/v1/materials/ask-question`      | Задать вопрос по материалу                     |
| `GET`  | `/api/v1/materials/topics`            | Список доступных тем                           |
| `POST` | `/api/v1/materials/search`            | Поиск материалов                               |

### Assessment

| Метод  | Эндпоинт                               | Описание                          |
| ------ | -------------------------------------- | --------------------------------- |
| `POST` | `/api/v1/assessment/start`             | Начать первичное тестирование     |
| `POST` | `/api/v1/assessment/submit`            | Отправить результаты тестирования |
| `GET`  | `/api/v1/assessment/results/{user_id}` | Получить результаты оценки        |

### Tests

| Метод  | Эндпоинт                                | Описание                   |
| ------ | --------------------------------------- | -------------------------- |
| `POST` | `/api/v1/tests/generate`                | Сгенерировать тест по теме |
| `POST` | `/api/v1/tests/generate-task`           | Сгенерировать задачу       |
| `POST` | `/api/v1/tests/submit-for-verification` | Отправить тест на проверку |
| `GET`  | `/api/v1/tests/{test_id}`               | Получить тест по ID        |

### Verification

| Метод  | Эндпоинт                                 | Описание                               |
| ------ | ---------------------------------------- | -------------------------------------- |
| `POST` | `/api/v1/verification/check-test`        | Проверка ответа с двойной верификацией |
| `GET`  | `/api/v1/verification/history/{user_id}` | История проверок пользователя          |

### Support

| Метод  | Эндпоинт                      | Описание                           |
| ------ | ----------------------------- | ---------------------------------- |
| `POST` | `/api/v1/support/get-support` | Получить психологическую поддержку |
| `GET`  | `/api/v1/support/resources`   | Ресурсы поддержки                  |

---

## 🧠 Агентная система v2

### Tree-of-Thoughts Orchestrator

**Файл:** `src/agents/materials_agent_v2.py`

Центральный оркестратор, использующий **DFS (Depth-First Search)** для исследования дерева рассуждений.

**Ключевые особенности:**

- Генерация кандидатных мыслей на каждом шаге (GigaChat-2-Max)
- Оценка перспективности (promise score) через GigaChat3
- Pruning низкоперспективных веток
- Post-execution evaluation (completeness, relevance, quality)
- Backtracking при тупиках
- Цель: completeness ≥ 0.85

### Tools (все используют GigaChat3)

| Tool                  | Файл                        | Назначение                                        |
| --------------------- | --------------------------- | ------------------------------------------------- |
| **Adaptive RAG**      | `adaptive_rag_tool.py`      | Выбор стратегии поиска (TF-IDF, Semantic, Hybrid) |
| **Corrective RAG**    | `corrective_rag_tool.py`    | Проверка релевантности и корректировка запроса    |
| **Web Search**        | `web_search_tool.py`        | Поиск через 4get metasearch engine                |
| **Web Scraper**       | `web_scraper_tool.py`       | Извлечение контента с веб-страниц                 |
| **Concept Extractor** | `concept_extractor_tool.py` | Извлечение концепций (KeyBERT, spaCy)             |
| **Memory Retrieval**  | `memory_retrieval_tool.py`  | Retrieval успешных паттернов                      |

### Content Guard Components (GigaChat3)

| Component             | Файл                   | Назначение                                  |
| --------------------- | ---------------------- | ------------------------------------------- |
| **Toxicity Checker**  | `toxicity_checker.py`  | Batch проверка токсичности                  |
| **Policy Checker**    | `policy_checker.py`    | Проверка соответствия политикам GigaChat    |
| **Content Sanitizer** | `content_sanitizer.py` | Rule-based санитизация (HTML, URLs, emails) |
| **Quality Gate**      | `quality_gate.py`      | Финальная проверка качества                 |

### Chains

| Chain                | Файл                  | Назначение                                      |
| -------------------- | --------------------- | ----------------------------------------------- |
| **Reasoning Chain**  | `reasoning_chain.py`  | Генерация мыслей в ToT (GigaChat-2-Max)         |
| **Evaluation Chain** | `evaluation_chain.py` | Оценка узлов: promise, completeness (GigaChat3) |

### Legacy Agents

| Агент               | Файл                       | Назначение                              |
| ------------------- | -------------------------- | --------------------------------------- |
| **Materials Agent** | `materials_agent.py`       | Адаптация материалов (legacy, без ToT)  |
| **LLM Router**      | `llm_router_agent.py`      | Выбор подходящей LLM по языку           |
| **Support Agent**   | `support_agent.py`         | Психологическая поддержка               |
| **Test Generation** | `test_generation_agent.py` | Генерация тестов и задач                |
| **Verification**    | `verification_agent.py`    | Проверка ответов с двойной верификацией |

---

## 🔬 Технологический стек

- **Backend:** FastAPI + Uvicorn
- **LLM Orchestration:** LangChain
- **LLM Providers:** [GigaChat-2-Max](https://giga.chat) (expensive) + [GigaChat3-10B](https://huggingface.co/ai-sage/GigaChat3-10B-A1.8B) (cheap)
- **Vector Database:** ChromaDB + HuggingFace Embeddings
- **Relational Database:** SQLite (SQLAlchemy ORM, async)
- **PDF Processing:** PyMuPDF (для парсинга LaTeX-PDF)
- **Web Scraping:** aiohttp + selectolax
- **Web Search:** 4get metasearch engine
- **Concept Extraction:** KeyBERT + spaCy
- **Caching:** Redis (optional) + in-memory LRU
- **Metrics:** Prometheus-compatible exporter
- **Package Manager:** uv
- **Task Runner:** just

---

## 📊 Метрики и мониторинг

### Prometheus Metrics

Эндпоинт: `/metrics` (Prometheus text format)

**Доступные метрики:**

- `materials_agent_requests_total` - Всего запросов
- `materials_agent_requests_successful_total` - Успешные запросы
- `materials_agent_response_time_avg` - Среднее время ответа
- `materials_agent_llm_calls_total{model}` - LLM вызовы по моделям
- `materials_agent_cost_total` - Общая стоимость (USD)
- `materials_agent_tool_calls_total{tool}` - Tool вызовы
- `materials_agent_documents_filtered_total` - Отфильтровано документов

### Database Metrics

**Таблица `material_generations`:**

- ToT метрики (iterations, explored_nodes, dead_end_nodes, best_path_depth)
- Tool usage (tools_used, tool_call_counts)
- LLM usage (gigachat2_max_calls, gigachat3_calls, estimated_cost_usd)
- Результаты (success, final_completeness_score, documents_collected)
- Производительность (generation_time_seconds)

**Таблица `tot_node_logs`:**

- Детальные логи каждого узла ToT
- Scores (promise_score, completeness_score, relevance_score, quality_score)
- Статус узла (pending, executing, promising, dead_end, goal_reached)

**Таблица `content_guard_logs`:**

- Логи фильтрации документов
- Toxicity scores, policy compliance, quality checks

### Analytics API

**Эндпоинт:** `/api/v1/analytics/generation-stats`

Агрегированная статистика по генерациям:

- Success rate
- Средний completeness score
- Среднее количество итераций
- Среднее время генерации
- Общая стоимость
- Tool usage distribution

---

## 🔄 Fallback Chains

Система устойчива к сбоям благодаря многоуровневым fallback стратегиям:

### LLM Fallback

1. **Primary:** GigaChat-2-Max/GigaChat3
2. **Retry:** Exponential backoff (3 попытки)
3. **Fallback:** Rule-based heuristics
4. **Last resort:** Cached response (если доступен)

### Database Fallback

1. **Primary:** SQLite (async)
2. **Fallback:** JSON файлы в `./backups/generations/`
3. **Recovery:** Автоматическая миграция при восстановлении БД

### ChromaDB Fallback

1. **Primary:** ChromaDB
2. **Fallback:** Pickle файлы в `./data/fallback/chromadb/`
3. **Search:** Keyword-based matching (Jaccard similarity)

### Content Guard Fallback

1. **Primary:** GigaChat3 batch check
2. **Fallback:** Individual checks
3. **Last resort:** Rule-based filter (blacklist words, patterns)
4. **Default:** Assume safe (log warning for manual review)

---

## 🧪 Тестирование и разработка

### Pre-commit Hooks

```
uv tool install pre-commit --with pre-commit-uv --force-reinstall
uv tool run pre-commit install
```

### Форматирование кода

Проект использует **just** для автоматизации задач.

**Установка just:**

```
# macOS
brew install just

# Linux
cargo install just

# Windows
scoop install just
```

**Запуск форматирования:**

```
just format
```

### Измерение эффективности верификации

```
uv run python -m src.scripts.measure_secondary_verification --test-data test_data.json
```

---

## 📖 Примеры использования

### Генерация материала через ToT

```
import requests

response = requests.post(
    "http://localhost:8001/api/v1/materials/generate-material",
    json={
        "user_id": "user123",
        "topic": "Быстрая сортировка",
        "user_level": "intermediate"
    }
)

result = response.json()

print(f"Material: {result['material'][:500]}...")
print(f"ToT Iterations: {result['metadata']['tot_iterations']}")
print(f"Documents Collected: {result['metadata']['documents_collected']}")
print(f"Tools Used: {result['metadata']['tools_used']}")
print(f"Completeness: {result['metadata']['final_completeness']}")
print(f"Cost: ${result['metadata']['estimated_cost_usd']:.4f}")
```

### Проверка здоровья системы

```
response = requests.get("http://localhost:8001/health/detailed")
health = response.json()

print(f"Overall Status: {health['status']}")
print(f"GigaChat-2-Max: {health['components']['gigachat2_max']['status']}")
print(f"GigaChat3: {health['components']['gigachat3']['status']}")
print(f"ChromaDB: {health['components']['chromadb']['status']}")
```

### Подробные примеры

Все примеры использования API находятся в файле [api-examples-report.md](api-examples-report.md)

---

## 🤝 Вклад в проект

Проект создан в учебных целях для демонстрации продвинутых паттернов работы с LLM.

**Ключевые паттерны:**

- Tree-of-Thoughts reasoning
- Cost optimization через распределение моделей
- Content safety через multi-stage guards
- Adaptive/Corrective RAG
- Procedural memory для обучения на опыте
- Fallback chains для устойчивости

---

## 📄 Лицензия

Проприетарнаe ПО.

---

**Версия:** 2.0.0
**Дата обновления:** Декабрь 2025
