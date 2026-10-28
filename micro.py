import time
import torch
from faster_whisper import WhisperModel

# --- Настройки ---
if torch.cuda.is_available():
    _MODEL_SIZE = "medium"
    _DEVICE = "cpu" #ВРЕМЕННО
    _COMPUTE_TYPE = "int8" 
else:
    _MODEL_SIZE = "medium"
    _DEVICE = "cpu"
    _COMPUTE_TYPE = "int8"     # Для CPU лучше int8 (квантование)

# Глобальная переменная для хранения модели (ленивая загрузка)
_model = None


def _get_model():
    """Ленивая загрузка модели: загружается только при первом вызове transcribe."""
    global _model
    if _model is None:
        start_load = time.perf_counter()
        _model = WhisperModel(
            _MODEL_SIZE,
            device=_DEVICE,
            compute_type=_COMPUTE_TYPE,
        )
        elapsed_load = time.perf_counter() - start_load
    return _model


def transcribe(
    audio_path: str,
    language: str = "ru",
    beam_size: int = 5,
    max_seconds_budget: float = 20.0,
) -> str:
    """
    Транскрибирует аудиофайл в текст.

    :param audio_path: путь к аудиофайлу (mp3, ogg, wav и т.д.).
    :param language: код языка ("ru" по умолчанию).
    :param beam_size: размер луча для beam search (выше = точнее, но медленнее).
    :param max_seconds_budget: ожидаемый бюджет времени в секундах;
                                если фактическое время превышено, выводится предупреждение.
    :return: распознанный текст (склеенные сегменты).
    """
    model = _get_model()
    start = time.perf_counter()

    
    try:
        segments, info = model.transcribe(
            audio_path,
            language=language,
            beam_size=beam_size,
            vad_filter=True,       # отрезаем тишину/паузы — быстрее и чище результат
            condition_on_previous_text=False,  # короткие ответы, контекст не нужен
        )

        # Собираем текст из сегментов (генератор выполняется здесь)
        text = " ".join(segment.text.strip() for segment in segments).strip()
        
    except Exception as e:
        print(f"[micro] ОШИБКА при транскрибации: {e}")
        raise

    elapsed = time.perf_counter() - start
    print(f"[micro] Транскрибация завершена за {elapsed:.2f} сек.")

    return text


# Блок для автономного тестирования модуля
if __name__ == "__main__":
    # Пример использования
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