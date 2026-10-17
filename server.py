import os
import tempfile

import pandas as pd
from flask import Flask, jsonify, redirect, request

from micro import transcribe  # готовая функция распознавания
from nli import compare_texts_nli_full, is_good_answer_nli, model as nli_model  # готовая NLI-модель

# Раздаём html/css из этой же папки, чтобы страница и API были на одном адресе
app = Flask(__name__, static_folder=".", static_url_path="")

CSV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "sobesedovanie_30_variantov.csv")

# Порог для вердикта «ответ подходит» (по вероятности entailment)
NLI_THRESHOLD = 0.5


def _clean(v):
    if v is None:
        return ""
    try:
        if pd.isna(v):
            return ""
    except (TypeError, ValueError):
        pass
    return str(v).strip()


def load_variants():
    """Читаем CSV один раз при старте и складываем в удобную структуру."""
    df = pd.read_csv(CSV_PATH, encoding="utf-8-sig")
    variants = []
    for i, row in df.iterrows():
        vid = i + 1
        raw_q = _clean(row.get("task_2_questions"))
        questions = [q.strip() for q in raw_q.split("\n") if q.strip()]
        variants.append({
            "id": vid,
            "name": _clean(row.get("variant_name")) or f"Вариант {vid}",
            "url": _clean(row.get("url")),
            "passage": _clean(row.get("task_1_text")),
            "questions": questions,
            "reference": _clean(row.get("standard_answer")),
        })
    return variants


VARIANTS = load_variants()
BY_ID = {v["id"]: v for v in VARIANTS}


def _qid(variant_id, q_idx):
    """Составной id вопроса: variant*100 + индекс вопроса (0-based)."""
    return variant_id * 100 + q_idx


# Карта: id вопроса -> (вариант, индекс вопроса). Позволяет /compare найти эталон.
QUESTIONS_BY_ID = {}
for _v in VARIANTS:
    for _i, _q in enumerate(_v["questions"]):
        QUESTIONS_BY_ID[_qid(_v["id"], _i)] = (_v, _i)


@app.after_request
def add_cors(resp):
    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Access-Control-Allow-Headers"] = "*"
    resp.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    resp.headers["Access-Control-Allow-Private-Network"] = "true"
    return resp


@app.route("/")
def index():
    return redirect("/record.html")


@app.route("/variants")
def variants_route():
    """Список вариантов для numofvar.html (id + подпись «Вариант X»)."""
    return jsonify([
        {"id": v["id"], "label": f"Вариант {v['id']}", "name": v["name"]}
        for v in VARIANTS
    ])


@app.route("/questions")
def questions_route():
    """Данные варианта: текст задания 1 + список вопросов задания 2."""
    try:
        vid = int(request.args.get("v") or request.args.get("variant") or 1)
    except (TypeError, ValueError):
        return jsonify(error="Некорректный номер варианта"), 400

    v = BY_ID.get(vid)
    if v is None:
        return jsonify(error="Неизвестный вариант"), 404

    return jsonify({
        "variant_id": v["id"],
        "name": v["name"],
        "passage": v["passage"],
        "questions": [
            {"id": _qid(v["id"], i), "question": q}
            for i, q in enumerate(v["questions"])
        ],
    })


@app.route("/transcribe", methods=["POST"])
def transcribe_route():
    audio = request.files.get("audio")
    if audio is None:
        return jsonify(error="Файл 'audio' не передан"), 400

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

    entry = QUESTIONS_BY_ID.get(qid)
    if entry is None:
        return jsonify(error="Неизвестный вопрос"), 404

    v, _q_idx = entry
    reference = v["reference"]
    if not text:
        return jsonify(error="Пустой ответ ученика"), 400

    try:
        proba = compare_texts_nli_full(reference, text)  # [entailment, contradiction, neutral]
        labels = [nli_model.config.id2label[i].lower() for i in range(len(proba))]
        probs = {labels[i]: float(proba[i]) for i in range(len(proba))}
        entail = float(proba[0])
        good = bool(is_good_answer_nli({"compare_nli": entail}, NLI_THRESHOLD))
        return jsonify(reference=reference, probs=probs, entailment=entail,
                       good=good, threshold=NLI_THRESHOLD)
    except Exception as e:
        return jsonify(error=str(e)), 500


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000)