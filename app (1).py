import os
import uuid

import pandas as pd
from flask import Flask, jsonify, render_template, request, session

import nli
import paraphrase
from micro import transcribe

app = Flask(__name__)
# В проде замените на переменную окружения / случайный секрет
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret-key-change-me")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "test.csv")
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads_audio")
os.makedirs(UPLOAD_DIR, exist_ok=True)

NLI_THRESHOLD = 0.8
PARAPHRASE_THRESHOLD = 0.8

df = pd.read_csv(CSV_PATH, sep=";")


def get_question(index: int):
    if index < 0 or index >= len(df):
        return None
    row = df.iloc[index]
    return {
        "index": index,
        "question": str(row["question"]),
        "correct_answer": str(row["correct_answer"]),
    }


@app.route("/")
def index():
    session.setdefault("q_index", 0)
    q = get_question(session["q_index"])
    if q is None:
        return render_template("done.html", total=len(df))
    return render_template("index.html", question=q, total=len(df))


@app.route("/answer", methods=["POST"])
def answer():
    q_index = session.get("q_index", 0)
    q = get_question(q_index)
    if q is None:
        return jsonify({"error": "Вопросы закончились"}), 400

    audio_file = request.files.get("audio")
    if audio_file is None:
        return jsonify({"error": "Аудио не получено"}), 400

    filename = f"{uuid.uuid4().hex}.webm"
    filepath = os.path.join(UPLOAD_DIR, filename)
    audio_file.save(filepath)

    try:
        recognized_text = transcribe(filepath)
    finally:
        try:
            os.remove(filepath)
        except OSError:
            pass

    correct_answer = q["correct_answer"]

    # Сигнатура: compare_texts_*(эталонный_ответ, ответ_пользователя)
    nli_score = float(nli.compare_texts_nli(correct_answer, recognized_text))
    paraphrase_score = float(
        paraphrase.compare_texts_paraphrase(correct_answer, recognized_text)
    )

    good_nli = nli_score > NLI_THRESHOLD
    good_paraphrase = paraphrase_score > PARAPHRASE_THRESHOLD

    return jsonify(
        {
            "question": q["question"],
            "correct_answer": correct_answer,
            "recognized_text": recognized_text,
            "nli_score": round(nli_score, 4),
            "paraphrase_score": round(paraphrase_score, 4),
            "good_nli": good_nli,
            "good_paraphrase": good_paraphrase,
            "is_last": q_index + 1 >= len(df),
        }
    )


@app.route("/next", methods=["POST"])
def next_question():
    session["q_index"] = session.get("q_index", 0) + 1
    q = get_question(session["q_index"])
    if q is None:
        return jsonify({"done": True, "total": len(df)})
    return jsonify({"done": False, "question": q})


@app.route("/restart", methods=["POST"])
def restart():
    session["q_index"] = 0
    return jsonify({"ok": True})


if __name__ == "__main__":
    # host=0.0.0.0 чтобы телефон в той же сети мог достучаться до сервера.
    #
    # ВАЖНО: navigator.mediaDevices.getUserMedia доступен в браузере ТОЛЬКО
    # в защищённом контексте — https:// или http://localhost. Обычный
    # http://<локальный-IP>:5000 с телефона работать не будет (mediaDevices
    # будет undefined).
    #
    # Раскомментируйте ssl_context ниже для самоподписанного HTTPS
    # (потребуется `pip install pyopenssl`; браузер покажет предупреждение
    # о недоверенном сертификате — нужно подтвердить "продолжить").
    # Либо используйте HTTPS-туннель: `ngrok http 5000` / `cloudflared tunnel --url http://localhost:5000`.

    USE_ADHOC_SSL = os.environ.get("USE_ADHOC_SSL", "0") == "1"

    if USE_ADHOC_SSL:
        app.run(debug=True, host="0.0.0.0", port=5000, ssl_context="adhoc")
    else:
        app.run(debug=True, host="0.0.0.0", port=5000)
