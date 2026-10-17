import csv
import os

def get_csv_element(file_path: str, row_index: int, col_index: int) -> str:
    """
    Возвращает элемент CSV-файла по индексу строки и столбца.
    
    :param file_path: Путь к CSV-файлу
    :param row_index: Индекс строки (0 - заголовок, 1 - первый вариант и т.д.)
    :param col_index: Индекс столбца (0 - variant_name, 1 - url, 2 - task_1_text и т.д.)
    :return: Содержимое ячейки или сообщение об ошибке
    """
    if not os.path.exists(file_path):
        return f"❌ Ошибка: Файл '{file_path}' не найден. Убедитесь, что он лежит в той же папке."
    
    try:
        with open(file_path, mode='r', encoding='utf-8-sig') as file:
            # Читаем весь файл в список списков
            reader = list(csv.reader(file))
            
            if not reader:
                return "❌ Ошибка: CSV файл пуст."
            
            # Проверка индекса строки
            if row_index < 0 or row_index >= len(reader):
                return f"❌ Ошибка: Индекс строки {row_index} вне диапазона. Доступно строк: {len(reader)} (от 0 до {len(reader)-1})."
            
            row = reader[row_index]
            
            # Проверка индекса столбца
            if col_index < 0 or col_index >= len(row):
                return f"❌ Ошибка: Индекс столбца {col_index} вне диапазона для строки {row_index}. Доступно столбцов: {len(row)} (от 0 до {len(row)-1})."
            
            return row[col_index]
            
    except Exception as e:
        return f"❌ Произошла ошибка при чтении файла: {e}"


def print_column_mapping():
    """Выводит справку по индексам столбцов для удобства."""
    print("📌 Справочник индексов столбцов:")
    print("  0 : variant_name         (Название варианта)")
    print("  1 : url                  (Ссылка на страницу)")
    print("  2 : task_1_text          (Текст для чтения)")
    print("  3 : task_2_questions     (Вопросы беседы по тексту)")
    print("  4 : task_3_questions     (Вопросы для описания фото)")
    print("  5 : task_3_photo_url     (Ссылка на фотографию)")
    print("-" * 50)


if __name__ == "__main__":
    csv_file = "sobesedovanie_30_variantov.csv"
    
    print_column_mapping()
    
    # Пример автоматического запроса (можно изменить индексы здесь)
    # row_index = 1 (первый вариант, т.к. 0 - это заголовки столбцов)
    # col_index = 2 (текст для чтения)
    test_row = 1
    test_col = 2
    
    print(f"\n🔍 Тестовый запрос: строка {test_row}, столбец {test_col}")
    result = get_csv_element(csv_file, test_row, test_col)
    
    if not result.startswith("❌"):
        print("✅ Найденный элемент:")
        print("=" * 50)
        print(result[:500] + "..." if len(result) > 500 else result) # Выводим первые 500 символов для краткости
        print("=" * 50)
    else:
        print(result)


print("\n🎮 Интерактивный режим (введите 'выход' для остановки)")
while True:
    try:
        r_idx = input("Введите индекс строки (например, 1): ")
        if r_idx.lower() == 'выход': break
        c_idx = input("Введите индекс столбца (0-5): ")
        if c_idx.lower() == 'выход': break
        
        res = get_csv_element(csv_file, int(r_idx), int(c_idx))
        print("\n" + "=" * 50)
        print(res)
        print("=" * 50 + "\n")
    except ValueError:
        print("⚠️ Пожалуйста, вводите целые числа.")
    except KeyboardInterrupt:
        print("\nПрограмма завершена.")
        break