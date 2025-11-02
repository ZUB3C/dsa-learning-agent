"""Скрипт для заполнения ChromaDB материалами из PDF."""

import argparse
import sys
from pathlib import Path

# Добавляем корневую директорию в путь
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ..data_processing import populate_from_pdf


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Заполнение ChromaDB материалами из PDF документа с оглавлением"
    )

    parser.add_argument(
        "--pdf",
        type=str,
        default="algobook.pdf",
        help="Путь к PDF файлу (по умолчанию: algobook.pdf)",
    )

    parser.add_argument(
        "--clear", action="store_true", help="Очистить существующую коллекцию перед заполнением"
    )

    parser.add_argument(
        "--chunk-size", type=int, default=1000, help="Размер чанка в символах (по умолчанию: 1000)"
    )

    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=200,
        help="Перекрытие чанков в символах (по умолчанию: 200)",
    )

    args = parser.parse_args()

    # Проверяем существование файла
    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        print(f"❌ Ошибка: PDF файл не найден: {pdf_path}")
        sys.exit(1)

    print(f"📚 Начало обработки PDF: {pdf_path}")
    print("🔧 Параметры:")
    print(f"   - Размер чанка: {args.chunk_size}")
    print(f"   - Перекрытие: {args.chunk_overlap}")
    print(f"   - Очистка БД: {'Да' if args.clear else 'Нет'}")
    print()

    # Заполняем БД
    try:
        result = populate_from_pdf(
            pdf_path=str(pdf_path),
            clear_existing=args.clear,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
        )

        if result["status"] == "success":
            print("\n✅ Успешно!")
            print("📊 Статистика:")
            print(f"   - Разделов: {result['total_sections']}")
            print(f"   - Документов: {result['total_documents']}")
            print(f"   - ID документов: {len(result['document_ids'])}")
        else:
            print("\n❌ Ошибка при заполнении БД:")
            print(f"   {result.get('error', 'Неизвестная ошибка')}")
            sys.exit(1)

    except Exception as e:
        print(f"\n❌ Критическая ошибка: {e}")
        import traceback  # noqa: PLC0415

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
