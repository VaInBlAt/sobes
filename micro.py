import re
import time

from nexara import Nexara

# --- Настройки ---
# Ключ API. Лучше задавать через переменную окружения NEXARA_API_KEY,
# но для совместимости с текущим кодом оставлен фолбэк.
import os
_NEXARA_API_KEY = os.environ.get("NEXARA_API_KEY", "nx-7coL3OXAQegat3WMKIlT8wms")
_NEXARA_MODEL = os.environ.get("NEXARA_MODEL", None)  # если API поддерживает выбор модели

# Глобальный клиент (ленивая инициализация)
_client = None


# Типичные галлюцинации Whisper на русском: титровые подписи и «концовки»
# с телевидения/YouTube. Появляются на тишине в конце записи.
_HALLUCINATION_PATTERNS = [
    re.compile(r"\bредактор\s+субтитров\b[^.!?\n]*", re.IGNORECASE),
    re.compile(r"\bкорректор\b[^.!?\n]*", re.IGNORECASE),
    re.compile(r"\bсубтитры\s+(?:сделал[а]?|подготовил[а]?|создал[а]?|и\s+озвучка)\b[^.!?\n]*", re.IGNORECASE),
    re.compile(r"\bпереводчик\b[^.!?\n]*", re.IGNORECASE),
    re.compile(r"\bтекст\s+читал[а]?\b[^.!?\n]*", re.IGNORECASE),
    re.compile(r"\bспасибо\s+за\s+(?:просмотр|внимание)\b[^.!?\n]*", re.IGNORECASE),
    re.compile(r"\bподписывайтесь\s+на\s+канал\b[^.!?\n]*", re.IGNORECASE),
    re.compile(r"\bставьте\s+лайки\b[^.!?\n]*", re.IGNORECASE),
    re.compile(r"\bпродолжение\s+следует\b[^.!?\n]*", re.IGNORECASE),
    re.compile(r"\bсубтитры\s+от\s+сообщества\b[^.!?\n]*", re.IGNORECASE),
    re.compile(r"\bсделано\s+в\s+программе\b[^.!?\n]*", re.IGNORECASE),
]


def _strip_hallucinations(text: str) -> str:
    """Убирает из расшифровки известные «концовки»-галлюцинации."""
    if not text:
        return text
    cleaned = text
    for pat in _HALLUCINATION_PATTERNS:
        cleaned = pat.sub("", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    cleaned = re.sub(r"\s+([,.!?;:])", r"\1", cleaned)
    return cleaned.strip()


def _get_client() -> Nexara:
    """Ленивая инициализация клиента Nexara."""
    global _client
    if _client is None:
        if not _NEXARA_API_KEY or _NEXARA_API_KEY == "nx-":
            raise RuntimeError(
                "[micro] Не задан API-ключ Nexara. "
                "Установите переменную окружения NEXARA_API_KEY."
            )
        _client = Nexara(api_key=_NEXARA_API_KEY)
    return _client


def transcribe(
    audio_path: str,
    language: str = "ru",
    beam_size: int = 5,
    max_seconds_budget: float = 20.0,
) -> str:
    """
    Транскрибирует аудиофайл в текст через API Nexara.

    :param audio_path: путь к аудиофайлу (mp3, ogg, wav и т.д.).
    :param language: код языка ("ru" по умолчанию). Передаётся, если API это поддерживает.
    :param beam_size: сохранён для обратной совместимости; API Nexara может его игнорировать.
    :param max_seconds_budget: ожидаемый бюджет времени в секундах (для совместимости).
    :return: распознанный текст.
    """
    client = _get_client()
    start = time.perf_counter()

    try:
        params = {"file": audio_path}
        # Пробрасываем язык/модель, только если API их принимает.
        # Если ваш Nexara SDK не поддерживает эти аргументы — просто уберите их.
        if language:
            params["language"] = language
        if _NEXARA_MODEL:
            params["model"] = _NEXARA_MODEL

        response = client.transcriptions.create(**params)
        text = (response.text or "").strip()
        text = _strip_hallucinations(text)

    except Exception as e:
        print(f"[micro] ОШИБКА при транскрибации: {e}")
        raise

    elapsed = time.perf_counter() - start
    print(f"[micro] Транскрибация завершена за {elapsed:.2f} сек.")

    return text


# Блок для автономного тестирования модуля
if __name__ == "__main__":
    test_audio = "records/test/Anya/1.ogg"
    print(f"Тестовый запуск micro.py на файле: {test_audio}")
    try:
        result = transcribe(test_audio)
        print("\n--- РЕЗУЛЬТАТ ---")
        print(result)
    except FileNotFoundError:
        print(f"Файл не найден: {test_audio}. Проверьте путь.")
    except Exception as e:
        print(f"Произошла ошибка: {e}")