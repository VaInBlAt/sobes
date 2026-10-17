import pandas as pd

# Читаем CSV (кодировка utf-8-sig убирает BOM в начале файла)
df = pd.read_csv("sobesedovanie_30_variantov.csv", encoding="utf-8-sig")

# Добавляем столбец-заглушку standard_answer
df["standard_answer"] = "эталонный ответ"

# Сохраняем результат
df.to_csv("sobesedovanie_30_variantov_with_answers.csv", index=False, encoding="utf-8-sig")

print("Готово. Столбцы:", list(df.columns))
print(df[["variant_name", "standard_answer"]].head())