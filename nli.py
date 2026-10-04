"""Утилиты для сравнения двух текстов с помощью NLI-модели RuBERT.

Модуль загружает токенизатор и модель `cointegrated/rubert-base-cased-nli-threeway`
из Hugging Face и предоставляет функции для получения вероятности NLI-класса,
сравнения ответов в строке таблицы и проверки результата по порогу.
"""

from dotenv import load_dotenv
import os

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


load_dotenv()

hf_token = os.getenv("HF_TOKEN")

model_name = "cointegrated/rubert-base-cased-nli-threeway"

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    token=hf_token,
)

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    token=hf_token,
)

if torch.cuda.is_available():
    model.cuda()


def compare_texts_nli(text1, text2):
    """Сравнить два текста и вернуть вероятность первого NLI-класса.

    Args:
        text1: Первый текст.
        text2: Второй текст.

    Returns:
        Вероятность класса с индексом 0 после применения softmax.
    """
    batch = tokenizer(text1, text2, return_tensors="pt")

    if torch.cuda.is_available():
        batch = {k: v.cuda() for k, v in batch.items()}

    with torch.no_grad():
        outputs = model(**batch)

        proba = torch.softmax(outputs.logits, -1).cpu().numpy()[0]

    return proba[0]


def compare_as_row_nli(row):
    """Сравнить эталонный и пользовательский ответы из строки.

    Args:
        row: Объект с полями ``correct_answer`` и ``answer1``.

    Returns:
        Вероятность первого NLI-класса для пары ответов.
    """
    return compare_texts_nli(row["correct_answer"], row["answer1"])


def is_good_answer_nli(row, threshold: float) -> bool:
    """Проверить, превышает ли NLI-оценка заданный порог.

    Args:
        row: Строка с полем ``compare_nli``.
        threshold: Порог, выше которого ответ считается подходящим.

    Returns:
        True, если ``compare_nli`` больше ``threshold``, иначе False.
    """
    return row["compare_nli"] > threshold


def compare_texts_nli_full(text1, text2):
    """Вернуть полный вектор вероятностей NLI-классов для двух текстов.

    Args:
        text1: Первый текст.
        text2: Второй текст.

    Returns:
        Массив вероятностей всех классов модели после softmax.
    """
    batch = tokenizer(text1, text2, return_tensors="pt")

    if torch.cuda.is_available():
        batch = {k: v.cuda() for k, v in batch.items()}

    with torch.no_grad():
        outputs = model(**batch)

        proba = torch.softmax(outputs.logits, -1).cpu().numpy()[0]

    return proba
