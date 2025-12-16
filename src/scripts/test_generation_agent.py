"""
Скрипт для тестового запуска Test Generation Agent с автоматическим запуском сервера.

Тестирует все эндпоинты:
- POST /api/v1/tests/generate - генерация теста
- POST /api/v1/tests/generate-task - генерация задачи
- POST /api/v1/tests/submit-for-verification - отправка на верификацию
- GET /api/v1/tests/{testid} - получение теста по ID
- GET /api/v1/tests/user/{userid}/completed - получение завершенных тестов

Использование:
    uv run python -m src.scripts.test_generation_agent_manual
"""

import asyncio
import json
import os
import signal
import subprocess
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import aiohttp
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.tree import Tree


class ServerManager:
    """Менеджер для управления запуском и остановкой сервера."""

    def __init__(self, host: str = "127.0.0.1", port: int = 8001) -> None:
        """
        Инициализация менеджера сервера.

        Args:
            host: Хост для запуска сервера
            port: Порт для запуска сервера
        """
        self.host: str = host
        self.port: int = port
        self.process: subprocess.Popen | None = None
        self.console: Console = Console()

    async def start(self, timeout: int = 30) -> bool:
        """
        Запустить сервер и дождаться его готовности.

        Args:
            timeout: Максимальное время ожидания в секундах

        Returns:
            True если сервер успешно запущен, False иначе
        """
        self.console.print("\n[cyan]🚀 Запуск сервера...[/cyan]")

        # Запускаем сервер как subprocess
        try:
            # Формируем команду для запуска
            python_exe = sys.executable
            cmd = [
                python_exe,
                "-m",
                "uvicorn",
                "src.main:app",
                "--host",
                self.host,
                "--port",
                str(self.port),
                "--log-level",
                "warning",
            ]

            # Запускаем процесс
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )

            self.console.print(f"[dim]Команда: {' '.join(cmd)}[/dim]")
            self.console.print(f"[dim]Process ID: {self.process.pid}[/dim]")

            # Ждем запуска сервера
            start_time = time.time()
            url = f"http://{self.host}:{self.port}/health"

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=self.console,
            ) as progress:
                progress.add_task("Ожидание готовности сервера...", total=None)

                while time.time() - start_time < timeout:
                    try:
                        async with aiohttp.ClientSession() as session:
                            async with session.get(
                                url, timeout=aiohttp.ClientTimeout(total=2)
                            ) as response:
                                if response.status == 200:
                                    progress.stop()
                                    self.console.print(
                                        f"[green]✅ Сервер успешно запущен на http://{self.host}:{self.port}[/green]"
                                    )
                                    return True
                    except (TimeoutError, aiohttp.ClientError):
                        pass

                    # Проверяем, что процесс еще жив
                    if self.process.poll() is not None:
                        progress.stop()
                        self.console.print("[red]❌ Процесс сервера завершился неожиданно[/red]")
                        # Выводим stderr если есть
                        if self.process.stderr:
                            stderr = self.process.stderr.read()
                            if stderr:
                                self.console.print(f"[red]Ошибка: {stderr}[/red]")
                        return False

                    await asyncio.sleep(0.5)

            self.console.print(f"[red]❌ Timeout: сервер не запустился за {timeout} секунд[/red]")
            return False

        except Exception as e:
            self.console.print(f"[red]❌ Ошибка запуска сервера: {e}[/red]")
            return False

    def stop(self) -> None:
        """Остановить сервер."""
        if self.process is None:
            return

        self.console.print("\n[cyan]🛑 Остановка сервера...[/cyan]")

        try:
            # Отправляем SIGTERM
            if sys.platform == "win32":
                self.process.terminate()
            else:
                os.kill(self.process.pid, signal.SIGTERM)

            # Ждем завершения
            try:
                self.process.wait(timeout=5)
                self.console.print("[green]✅ Сервер остановлен[/green]")
            except subprocess.TimeoutExpired:
                # Если не завершился, убиваем принудительно
                self.process.kill()
                self.process.wait()
                self.console.print("[yellow]⚠️  Сервер принудительно остановлен[/yellow]")

        except Exception as e:
            self.console.print(f"[red]❌ Ошибка при остановке сервера: {e}[/red]")
        finally:
            self.process = None


class TestGenerationAgentTester:
    """Класс для тестирования Test Generation Agent."""

    def __init__(self, base_url: str = "http://127.0.0.1:8001") -> None:
        """
        Инициализация тестера.

        Args:
            base_url: Базовый URL API сервера
        """
        self.base_url: str = base_url.rstrip("/")
        self.console: Console = Console()
        self.test_user_id: str = f"test_user_{uuid.uuid4().hex[:8]}"
        self.results: dict[str, Any] = {}

    def print_header(self, text: str) -> None:
        """Вывод заголовка раздела."""
        self.console.print()
        self.console.print(Panel(text, style="bold cyan"))

    def print_success(self, text: str) -> None:
        """Вывод успешного результата."""
        self.console.print(f"✅ {text}", style="green")

    def print_error(self, text: str) -> None:
        """Вывод ошибки."""
        self.console.print(f"❌ {text}", style="red")

    def print_info(self, text: str) -> None:
        """Вывод информационного сообщения."""
        self.console.print(f"ℹ️  {text}", style="blue")

    def print_json(self, data: dict[str, Any], title: str = "Response") -> None:
        """Красивый вывод JSON."""
        self.console.print(f"\n[bold]{title}:[/bold]")
        self.console.print_json(json.dumps(data, ensure_ascii=False, indent=2))

    async def test_generate_test(
        self,
        topic: str = "Сортировка массивов",
        difficulty: str = "medium",
        question_count: int = 5,
    ) -> dict[str, Any] | None:
        """
        Тест эндпоинта генерации теста.

        Args:
            topic: Тема теста
            difficulty: Сложность (easy, medium, hard)
            question_count: Количество вопросов

        Returns:
            Ответ сервера или None в случае ошибки
        """
        self.print_header("📝 Тест 1: Генерация теста (POST /api/v1/tests/generate)")

        payload = {
            "topic": topic,
            "difficulty": difficulty,
            "questioncount": question_count,
            "language": "ru",
        }

        self.print_info(
            f"Генерация теста: тема='{topic}', сложность='{difficulty}', вопросов={question_count}"
        )

        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=self.console,
            ) as progress:
                progress.add_task("Генерация теста...", total=None)

                async with (
                    aiohttp.ClientSession() as session,
                    session.post(
                        f"{self.base_url}/api/v1/tests/generate",
                        json=payload,
                        timeout=aiohttp.ClientTimeout(total=60),
                    ) as response,
                ):
                    if response.status == 200:
                        data = await response.json()
                        self.print_success("Тест успешно сгенерирован!")
                        self.results["generate_test"] = data

                        # Вывод структурированной информации
                        table = Table(title="Сгенерированный тест", show_header=True)
                        table.add_column("Параметр", style="cyan")
                        table.add_column("Значение", style="green")

                        table.add_row("Test ID", data.get("testid", "N/A"))
                        table.add_row(
                            "Количество вопросов",
                            str(len(data.get("questions", []))),
                        )
                        table.add_row(
                            "Ожидаемая длительность",
                            f"{data.get('expectedduration', 0)} мин",
                        )

                        self.console.print(table)

                        # Вывод вопросов
                        questions = data.get("questions", [])
                        if questions:
                            tree = Tree("[bold]📚 Вопросы теста[/bold]")
                            for i, q in enumerate(questions, 1):
                                q_branch = tree.add(f"[cyan]Вопрос {i}[/cyan]")
                                q_branch.add(f"ID: {q.get('questionid', 'N/A')}")
                                q_branch.add(f"Текст: {q.get('questiontext', 'N/A')[:100]}...")
                                q_branch.add(
                                    f"Ожидаемый ответ: {q.get('expectedanswer', 'N/A')[:50]}..."
                                )
                                keypoints = q.get("keypoints", [])
                                if keypoints:
                                    kp_branch = q_branch.add("[yellow]Ключевые моменты:[/yellow]")
                                    for kp in keypoints[:3]:  # Первые 3
                                        kp_branch.add(f"• {kp}")

                            self.console.print(tree)

                        return data
                    self.print_error(f"Ошибка: статус {response.status}")
                    text = await response.text()
                    self.console.print(text)
                    return None

        except TimeoutError:
            self.print_error("Timeout при генерации теста (60 секунд)")
            return None
        except Exception as e:
            self.print_error(f"Исключение при генерации теста: {e}")
            return None

    async def test_generate_task(
        self,
        topic: str = "Бинарный поиск",
        difficulty: str = "medium",
        task_type: str = "implementation",
    ) -> dict[str, Any] | None:
        """
        Тест эндпоинта генерации задачи.

        Args:
            topic: Тема задачи
            difficulty: Сложность
            task_type: Тип задачи

        Returns:
            Ответ сервера или None в случае ошибки
        """
        self.print_header("🎯 Тест 2: Генерация задачи (POST /api/v1/tests/generate-task)")

        payload = {
            "topic": topic,
            "difficulty": difficulty,
            "tasktype": task_type,
            "language": "ru",
        }

        self.print_info(f"Генерация задачи: тема='{topic}', тип='{task_type}'")

        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=self.console,
            ) as progress:
                progress.add_task("Генерация задачи...", total=None)

                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        f"{self.base_url}/api/v1/tests/generate-task",
                        json=payload,
                        timeout=aiohttp.ClientTimeout(total=60),
                    ) as response:
                        if response.status == 200:
                            data = await response.json()
                            self.print_success("Задача успешно сгенерирована!")
                            self.results["generate_task"] = data

                            # Вывод информации о задаче
                            task = data.get("task", {})
                            table = Table(title="Сгенерированная задача", show_header=True)
                            table.add_column("Параметр", style="cyan")
                            table.add_column("Значение", style="green")

                            table.add_row("Task ID", str(task.get("taskid", "N/A")))
                            table.add_row("Тема", task.get("topic", "N/A"))
                            table.add_row("Сложность", task.get("difficulty", "N/A"))
                            table.add_row("Тип задачи", task.get("tasktype", "N/A"))
                            table.add_row(
                                "Описание",
                                task.get("description", "N/A")[:100] + "...",
                            )

                            self.console.print(table)

                            # Вывод подсказок
                            hints = data.get("solutionhints", [])
                            if hints:
                                tree = Tree("[bold]💡 Подсказки к решению[/bold]")
                                for hint in hints:
                                    tree.add(
                                        f"Уровень {hint.get('hintlevel', 0)}: {hint.get('hinttext', 'N/A')[:80]}..."
                                    )
                                self.console.print(tree)

                            return data
                        self.print_error(f"Ошибка: статус {response.status}")
                        text = await response.text()
                        self.console.print(text)
                        return None

        except TimeoutError:
            self.print_error("Timeout при генерации задачи (60 секунд)")
            return None
        except Exception as e:
            self.print_error(f"Исключение при генерации задачи: {e}")
            return None

    async def test_get_test_by_id(self, test_id: str) -> dict[str, Any] | None:
        """
        Тест эндпоинта получения теста по ID.

        Args:
            test_id: ID теста

        Returns:
            Ответ сервера или None в случае ошибки
        """
        self.print_header(f"🔍 Тест 3: Получение теста по ID (GET /api/v1/tests/{test_id})")

        try:
            async with (
                aiohttp.ClientSession() as session,
                session.get(
                    f"{self.base_url}/api/v1/tests/{test_id}",
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as response,
            ):
                if response.status == 200:
                    data = await response.json()
                    self.print_success(f"Тест с ID '{test_id}' успешно получен!")
                    self.results["get_test_by_id"] = data

                    test_info = data.get("test", {})
                    table = Table(title="Информация о тесте", show_header=True)
                    table.add_column("Параметр", style="cyan")
                    table.add_column("Значение", style="green")

                    table.add_row("Test ID", test_info.get("testid", "N/A"))
                    table.add_row("Тема", test_info.get("topic", "N/A"))
                    table.add_row("Сложность", test_info.get("difficulty", "N/A"))
                    table.add_row(
                        "Вопросов",
                        str(len(test_info.get("questions", []))),
                    )
                    table.add_row(
                        "Длительность",
                        f"{test_info.get('expectedduration', 0)} мин",
                    )

                    self.console.print(table)

                    metadata = data.get("metadata", {})
                    if metadata:
                        self.console.print(
                            f"\n[dim]Создан: {metadata.get('createdat', 'N/A')}[/dim]"
                        )

                    return data
                if response.status == 404:
                    self.print_error(f"Тест с ID '{test_id}' не найден")
                    return None
                self.print_error(f"Ошибка: статус {response.status}")
                text = await response.text()
                self.console.print(text)
                return None

        except Exception as e:
            self.print_error(f"Исключение при получении теста: {e}")
            return None

    async def test_submit_test(
        self, test_id: str, answers: list[dict[str, Any]]
    ) -> dict[str, Any] | None:
        """
        Тест эндпоинта отправки теста на верификацию.

        Args:
            test_id: ID теста
            answers: Список ответов

        Returns:
            Ответ сервера или None в случае ошибки
        """
        self.print_header(
            "📤 Тест 4: Отправка теста на верификацию (POST /api/v1/tests/submit-for-verification)"
        )

        payload = {
            "testid": test_id,
            "userid": self.test_user_id,
            "answers": answers,
        }

        self.print_info(
            f"Отправка ответов: {len(answers)} ответов от пользователя {self.test_user_id}"
        )

        try:
            async with (
                aiohttp.ClientSession() as session,
                session.post(
                    f"{self.base_url}/api/v1/tests/submit-for-verification",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as response,
            ):
                if response.status == 200:
                    data = await response.json()
                    self.print_success("Ответы успешно отправлены на верификацию!")
                    self.results["submit_test"] = data

                    table = Table(title="Результат отправки", show_header=True)
                    table.add_column("Параметр", style="cyan")
                    table.add_column("Значение", style="green")

                    table.add_row("Verification ID", data.get("verificationid", "N/A"))
                    table.add_row("Статус", data.get("status", "N/A"))

                    self.console.print(table)
                    return data
                self.print_error(f"Ошибка: статус {response.status}")
                text = await response.text()
                self.console.print(text)
                return None

        except Exception as e:
            self.print_error(f"Исключение при отправке теста: {e}")
            return None

    async def test_get_completed_tests(self, user_id: str) -> dict[str, Any] | None:
        """
        Тест эндпоинта получения завершенных тестов пользователя.

        Args:
            user_id: ID пользователя

        Returns:
            Ответ сервера или None в случае ошибки
        """
        self.print_header(
            f"📊 Тест 5: Получение завершенных тестов (GET /api/v1/tests/user/{user_id}/completed)"
        )

        try:
            async with (
                aiohttp.ClientSession() as session,
                session.get(
                    f"{self.base_url}/api/v1/tests/user/{user_id}/completed",
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as response,
            ):
                if response.status == 200:
                    data = await response.json()
                    self.print_success(f"Завершенные тесты пользователя '{user_id}' получены!")
                    self.results["get_completed_tests"] = data

                    completed = data.get("completedtests", [])
                    stats = data.get("statistics", {})

                    table = Table(title="Статистика", show_header=True)
                    table.add_column("Метрика", style="cyan")
                    table.add_column("Значение", style="green")

                    table.add_row(
                        "Всего завершено тестов",
                        str(stats.get("totalcompleted", 0)),
                    )

                    self.console.print(table)

                    if completed:
                        tests_table = Table(title="Завершенные тесты", show_header=True)
                        tests_table.add_column("Test ID", style="cyan")
                        tests_table.add_column("Тема", style="yellow")
                        tests_table.add_column("Сложность", style="magenta")
                        tests_table.add_column("Дата сдачи", style="green")

                        for test in completed[:10]:  # Показываем первые 10
                            tests_table.add_row(
                                test.get("testid", "N/A")[:20] + "...",
                                test.get("topic", "N/A"),
                                test.get("difficulty", "N/A"),
                                test.get("submittedat", "N/A")[:19],
                            )

                        self.console.print(tests_table)
                    else:
                        self.console.print("[dim]Завершенных тестов пока нет[/dim]")

                    return data
                self.print_error(f"Ошибка: статус {response.status}")
                text = await response.text()
                self.console.print(text)
                return None

        except Exception as e:
            self.print_error(f"Исключение при получении завершенных тестов: {e}")
            return None

    def print_summary(self) -> None:
        """Вывод итогового отчета."""
        self.print_header("📋 Итоговый отчет тестирования")

        summary_table = Table(title="Результаты тестов", show_header=True)
        summary_table.add_column("№", style="cyan", width=5)
        summary_table.add_column("Тест", style="yellow", width=40)
        summary_table.add_column("Статус", style="green", width=15)

        tests = [
            ("Генерация теста", "generate_test"),
            ("Генерация задачи", "generate_task"),
            ("Получение теста по ID", "get_test_by_id"),
            ("Отправка на верификацию", "submit_test"),
            ("Получение завершенных тестов", "get_completed_tests"),
        ]

        for i, (name, key) in enumerate(tests, 1):
            status = "✅ Пройден" if self.results.get(key) else "❌ Не пройден"
            summary_table.add_row(str(i), name, status)

        self.console.print(summary_table)

        # Статистика
        passed = sum(1 for _, key in tests if self.results.get(key))
        total = len(tests)

        self.console.print(f"\n[bold]Успешных тестов: {passed}/{total}[/bold]")

        if passed == total:
            self.console.print("\n[bold green]🎉 Все тесты пройдены успешно![/bold green]")
        else:
            self.console.print(
                f"\n[bold yellow]⚠️  Пройдено {passed} из {total} тестов[/bold yellow]"
            )

    async def run_all_tests(self) -> None:
        """Запуск всех тестов последовательно."""
        start_time = datetime.now()

        self.console.print(
            Panel.fit(
                "[bold cyan]Test Generation Agent - Ручное тестирование[/bold cyan]\n"
                f"User ID: {self.test_user_id}\n"
                f"Base URL: {self.base_url}\n"
                f"Время запуска: {start_time.strftime('%Y-%m-%d %H:%M:%S')}",
                border_style="cyan",
            )
        )

        # Тест 1: Генерация теста
        test_data = await self.test_generate_test(
            topic="Алгоритмы сортировки",
            difficulty="medium",
            question_count=3,
        )

        test_id = None
        if test_data:
            test_id = test_data.get("testid")

        # Тест 2: Генерация задачи
        await self.test_generate_task(
            topic="Динамическое программирование",
            difficulty="hard",
            task_type="optimization",
        )

        # Тест 3: Получение теста по ID
        if test_id:
            retrieved_test = await self.test_get_test_by_id(test_id)

            # Тест 4: Отправка теста на верификацию
            if retrieved_test:
                questions = retrieved_test.get("test", {}).get("questions", [])
                if questions:
                    # Создаем тестовые ответы
                    answers = [
                        {
                            "questionid": q.get("questionid", i),
                            "answer": "Это тестовый ответ на вопрос",
                        }
                        for i, q in enumerate(questions, 1)
                    ]
                    await self.test_submit_test(test_id, answers)

        # Тест 5: Получение завершенных тестов
        await self.test_get_completed_tests(self.test_user_id)

        # Итоговый отчет
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        self.print_summary()

        self.console.print(f"\n[dim]Общее время выполнения тестов: {duration:.2f} секунд[/dim]")


async def main() -> None:
    """Главная функция."""
    console = Console()

    # Проверяем, что мы в корне проекта
    if not Path("src/main.py").exists():
        console.print("[red]❌ Ошибка: запустите скрипт из корневой директории проекта[/red]")
        console.print("[yellow]Текущая директория:[/yellow]", Path.cwd())
        sys.exit(1)

    # Инициализируем менеджер сервера
    server_manager = ServerManager(host="127.0.0.1", port=8001)

    # Инициализируем тестер
    tester = TestGenerationAgentTester(base_url="http://127.0.0.1:8001")

    try:
        # Запускаем сервер
        if not await server_manager.start(timeout=30):
            console.print("[red]❌ Не удалось запустить сервер[/red]")
            sys.exit(1)

        # Небольшая задержка для стабилизации
        await asyncio.sleep(2)

        # Запускаем тесты
        await tester.run_all_tests()

    except KeyboardInterrupt:
        console.print("\n[yellow]⚠️  Тестирование прервано пользователем[/yellow]")
    except Exception as e:
        console.print(f"\n[red]❌ Критическая ошибка: {e}[/red]")
        import traceback

        traceback.print_exc()
    finally:
        # Всегда останавливаем сервер
        server_manager.stop()


if __name__ == "__main__":
    asyncio.run(main())
