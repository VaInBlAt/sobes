import os
import tempfile

from flask import Flask, jsonify, redirect, request

from micro import transcribe  # готовая функция распознавания
from questions import load_questions
from nli import compare_texts_nli_full, is_good_answer_nli, model as nli_model  # готовая NLI-модель

# Раздаём html/css из этой же папки, чтобы страница и API были на одном адресе
app = Flask(__name__, static_folder=".", static_url_path="")

# Вопросы, абзацы и эталонные ответы (см. questions.py — там же место для автоматизации)
QUESTIONS = load_questions()
BY_ID = {q["id"]: q for q in QUESTIONS}

# Порог для вердикта «ответ подходит» (по вероятности entailment), подберите по своим данным
NLI_THRESHOLD = 0.5


@app.after_request
def add_cors(resp):
    # разрешаем запросы со страницы, лежащей на другом домене (хостинг)
    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Access-Control-Allow-Headers"] = "*"
    resp.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    # Chrome требует это, когда https-страница обращается к localhost
    resp.headers["Access-Control-Allow-Private-Network"] = "true"
    return resp


@app.route("/")
def index():
    return redirect("/record.html")


@app.route("/questions")
def questions_route():
    # эталонные ответы браузеру не отдаём — они приходят только после анализа
    return jsonify([{"id": q["id"], "passage": q["passage"], "question": q["question"]} for q in QUESTIONS])


@app.route("/transcribe", methods=["POST"])
def transcribe_route():
    audio = request.files.get("audio")
    if audio is None:
        return jsonify(error="Файл 'audio' не передан"), 400

    # браузер присылает webm/ogg — ffmpeg внутри faster-whisper это читает
    suffix = os.path.splitext(audio.filename or "")[1] or ".webm"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        audio.save(tmp.name)
        path = tmp.name

    try:
        text = transcribe(path)
        return jsonify(text=text)
    except Exception as e:
        return jsonify(error=str(e)), 500
    finally:
        os.remove(path)


@app.route("/compare", methods=["POST"])
def compare_route():
    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()
    try:
        qid = int(data.get("question_id"))
    except (TypeError, ValueError):
        return jsonify(error="Не передан question_id"), 400

    q = BY_ID.get(qid)
    if q is None:
        return jsonify(error="Неизвестный вопрос"), 404
    reference = q["reference"]
    if not text:
        return jsonify(error="Пустой ответ ученика"), 400

    try:
        proba = compare_texts_nli_full(reference, text)  # [entailment, contradiction, neutral]
        labels = [nli_model.config.id2label[i].lower() for i in range(len(proba))]
        probs = {labels[i]: float(proba[i]) for i in range(len(proba))}
        entail = float(proba[0])
        good = bool(is_good_answer_nli({"compare_nli": entail}, NLI_THRESHOLD))
        return jsonify(reference=reference, probs=probs, entailment=entail, good=good, threshold=NLI_THRESHOLD)
    except Exception as e:
        return jsonify(error=str(e)), 500


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000)
