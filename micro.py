import time
import torch
from faster_whisper import WhisperModel

if torch.cuda.is_available():
    _MODEL_SIZE = "medium"
    _DEVICE = "cpu"
    _COMPUTE_TYPE = "int8s"
else:
    _MODEL_SIZE = "medium"
    _DEVICE = "cpu"
    _COMPUTE_TYPE = "int8"  # квантование для скорости на CPU

_model = WhisperModel(
    _MODEL_SIZE,
    device=_DEVICE,
    compute_type=_COMPUTE_TYPE,
)


def transcribe(
    audio_path: str,
    language: str = "ru",
    beam_size: int = 5,
    max_seconds_budget: float = 20.0,
) -> str:
    """
    Транскрибирует mp3-файл в текст.

    :param audio_path: путь к mp3-файлу.
    :param language: код языка ("ru" по умолчанию, судя по датасету).
                      Явное указание языка ускоряет и повышает точность
                      по сравнению с авто-детектом.
    :param beam_size: размер луча для beam search (выше = точнее, но медленнее).
    :param max_seconds_budget: ожидаемый бюджет времени в секундах;
                                если фактическое время превышено, в лог
                                выводится предупреждение (для мониторинга).
    :return: распознанный текст (склеенные сегменты).
    """
    start = time.perf_counter()

    segments, _info = _model.transcribe(
        audio_path,
        language=language,
        beam_size=beam_size,
        vad_filter=True,       # отрезаем тишину/паузы — быстрее и чище результат
        condition_on_previous_text=False,  # короткие ответы, контекст не нужен
    )

    text = " ".join(segment.text.strip() for segment in segments).strip()

    elapsed = time.perf_counter() - start
    if elapsed > max_seconds_budget:
        print(
            f"[micro] Внимание: транскрипция заняла {elapsed:.2f}с, "
            f"что превышает бюджет {max_seconds_budget}с "
            f"(модель={_MODEL_SIZE}, устройство={_DEVICE})."
        )

    return text