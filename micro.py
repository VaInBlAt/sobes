import os
import time
import shutil
import tempfile
import threading
import subprocess

import requests

# --- Настройки VK Cloud Voice ---
# client_secret НЕ храните в коде — задайте переменную окружения VK_VOICE_CLIENT_SECRET
_VK_CLIENT_ID = os.getenv("VK_VOICE_CLIENT_ID", "mcs4809035818.voice.635QdTtSeU8wbGpeiHUnT")
_VK_CLIENT_SECRET = os.getenv("VK_VOICE_CLIENT_SECRET", "")  # обязательный!
_VK_OAUTH_URL = "https://mcs.mail.ru/auth/oauth/v1/token"
_VK_ASR_URL = "https://voice.mcs.mail.ru/asr"

_MAX_BYTES = 20 * 1024 * 1024        # 20 МБ
_MAX_DURATION = 5 * 60               # 5 мин
_CHUNK_SECONDS = 280                 # с запасом под лимит 5 мин

# --- Кэш токенов (потокобезопасный) ---
_token_lock = threading.Lock()
_access_token: str | None = None
_refresh_token: str | None = None
_token_expires_at: float = 0.0


def _fetch_token(force_refresh: bool = False) -> str:
    """Возвращает валидный access_token, при необходимости обновляя его."""
    global _access_token, _refresh_token, _token_expires_at

    if not _VK_CLIENT_SECRET:
        raise RuntimeError(
            "Не задан VK_VOICE_CLIENT_SECRET. "
            "Установите переменную окружения или впишите секрет в код."
        )

    with _token_lock:
        now = time.time()
        if not force_refresh and _access_token and now < _token_expires_at - 60:
            return _access_token

        if force_refresh and _refresh_token:
            payload = {
                "client_id": _VK_CLIENT_ID,
                "refresh_token": _refresh_token,
                "grant_type": "refresh_token",
            }
        else:
            payload = {
                "client_id": _VK_CLIENT_ID,
                "client_secret": _VK_CLIENT_SECRET,
                "grant_type": "client_credentials",
            }

        r = requests.post(_VK_OAUTH_URL, json=payload, timeout=30)
        r.raise_for_status()
        data = r.json()

        _access_token = data["access_token"]
        _refresh_token = data.get("refresh_token") or _refresh_token
        _token_expires_at = now + int(data.get("expired_in", 3600))
        return _access_token


def _ensure_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        raise RuntimeError(
            "Не найден ffmpeg/ffprobe. Установите ffmpeg и добавьте его в PATH."
        )


def _probe_duration(path: str) -> float | None:
    try:
        out = subprocess.check_output(
            [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=nw=1:nk=1", path,
            ],
            text=True,
        ).strip()
        return float(out)
    except Exception:
        return None


def _convert_to_wav(path: str) -> str:
    """Конвертирует входной файл в моно WAV 16 кГц (формат, который примет VK)."""
    fd, out_path = tempfile.mkstemp(suffix=".wav", prefix="vk_asr_")
    os.close(fd)
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", path,
        "-ac", "1", "-ar", "16000",
        "-f", "wav", out_path,
    ]
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError:
        os.remove(out_path)
        raise
    return out_path


def _split_wav(path: str) -> list[str]:
    """Режет WAV на куски ≤ _CHUNK_SECONDS. Возвращает список путей."""
    dur = _probe_duration(path) or 0
    if dur <= _CHUNK_SECONDS:
        return [path]

    out_dir = tempfile.mkdtemp(prefix="vk_asr_chunks_")
    pattern = os.path.join(out_dir, "chunk_%03d.wav")
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", path,
        "-f", "segment", "-segment_time", str(_CHUNK_SECONDS),
        "-c", "copy", pattern,
    ]
    subprocess.run(cmd, check=True)
    return sorted(
        os.path.join(out_dir, f) for f in os.listdir(out_dir) if f.endswith(".wav")
    )


def _recognize_file(path: str) -> str:
    """Отправляет один WAV в VK ASR. Возвращает текст."""
    token = _fetch_token()
    with open(path, "rb") as f:
        data = f.read()

    if len(data) > _MAX_BYTES:
        raise RuntimeError(f"Файл {path} больше 20 МБ после конвертации")

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "audio/wave",
    }

    r = requests.post(_VK_ASR_URL, headers=headers, data=data, timeout=180)
    if r.status_code == 401:
        # токен истёк — обновляем и пробуем ещё раз
        token = _fetch_token(force_refresh=True)
        headers["Authorization"] = f"Bearer {token}"
        r = requests.post(_VK_ASR_URL, headers=headers, data=data, timeout=180)

    r.raise_for_status()
    payload = r.json()

    texts = payload.get("result", {}).get("texts", [])
    if not texts:
        return ""
    best = max(texts, key=lambda t: t.get("confidence", 0.0))
    return (best.get("punctuated_text") or best.get("text") or "").strip()


def transcribe(
    audio_path: str,
    language: str = "ru",              # VK определяет язык сам; оставлен для совместимости
    beam_size: int = 5,                 # не используется
    max_seconds_budget: float = 20.0,   # не используется
) -> str:
    """
    Транскрибирует аудиофайл в текст через VK Cloud Voice ASR.
    Сигнатура сохранена для обратной совместимости с server.py.
    """
    _ensure_ffmpeg()

    start = time.perf_counter()
    wav_path: str | None = None
    chunks: list[str] = []

    try:
        wav_path = _convert_to_wav(audio_path)
        chunks = _split_wav(wav_path)

        parts = [_recognize_file(c) for c in chunks]
        text = " ".join(p for p in parts if p).strip()

    except Exception as e:
        print(f"[micro] ОШИБКА при транскрибации через VK: {e}")
        raise

    finally:
        if wav_path and os.path.exists(wav_path):
            try:
                os.remove(wav_path)
            except OSError:
                pass
        for c in chunks:
            if c != wav_path and os.path.exists(c):
                try:
                    os.remove(c)
                except OSError:
                    pass

    elapsed = time.perf_counter() - start
    print(f"[micro] Транскрибация (VK Cloud Voice) за {elapsed:.2f} сек.")
    return text


# Автономный тест
if __name__ == "__main__":
    test_audio = "records/test/Anya/1.ogg"
    print(f"Тест micro.py (VK) на файле: {test_audio}")
    try:
        print("\n--- РЕЗУЛЬТАТ ---")
        print(transcribe(test_audio))
    except FileNotFoundError:
        print(f"Файл не найден: {test_audio}")
    except Exception as e:
        print(f"Ошибка: {e}")