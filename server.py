import os
import tempfile

import pandas as pd
from flask import Flask, jsonify, redirect, request

from micro import transcribe  # готовая функция распознавания
from giga_eval import evaluate_student_answer, evaluate_task3, evaluate_task3_logic
from ru_syntax import count_grammatical_stems  # единственный источник правды по основам

# Раздаём html/css из этой же папки, чтобы страница и API были на одном адресе
app = Flask(__name__, static_folder=".", static_url_path="")

CSV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "sobesedovanie_30_variantov.csv")


def _clean(v):
    if v is None:
        return ""
    try:
        if pd.isna(v):
            return ""
    except (TypeError, ValueError):
        pass
    return str(v).strip()


def _clean_questions(raw):
    """Разбивает поле с вопросами (в т.ч. task_3_questions) на отдельные строки."""
    out = []
    for line in (raw or "").split("\n"):
        s = line.strip().lstrip("•·▪-—–").strip()
        if s:
            out.append(s)
    return out


def load_variants():
    """Читаем CSV один раз при старте и складываем в удобную структуру."""
    df = pd.read_csv(CSV_PATH, encoding="utf-8-sig")
    variants = []
    for i, row in df.iterrows():
        vid = i + 1
        raw_q = _clean(row.get("task_2_questions"))
        questions = [q.strip() for q in raw_q.split("\n") if q.strip()]
        photo_description = (
            _clean(row.get("photo_content"))
            or _clean(row.get("task_3_photo_description"))
        )
        variants.append({
            "id": vid,
            "name": _clean(row.get("variant_name")) or f"Вариант {vid}",
            "url": _clean(row.get("url")),
            "passage": _clean(row.get("task_1_text")),
            "questions": questions,
            "reference": _clean(row.get("standard_answer")),
            "photo_url": _clean(row.get("task_3_photo_url")),
            "t3_questions": _clean_questions(row.get("task_3_questions")),
            "photo_description": photo_description,
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


# Маркеры «мягкой» придирки. Если модель обнулила факты, но в заметке
# явно что-то из этого списка — считаем, что фактической ошибки нет.
# Страховка от того самого случая с «кофтой» и «двумя годами».
_SOFT_FACTS_MARKERS = (
    "возможно",
    "предположительн",
    "незначительн",
    "небольш",
    "мелк",
    "неточный возраст",
    "обобщ",
    "оттенок",
    "не критичн",
    "не существенн",
    "допустим",
    "приемлем",
    "скорее всего",
    "вероятно",
)


def _is_soft_facts_note(note: str) -> bool:
    note_l = (note or "").lower()
    return any(m in note_l for m in _SOFT_FACTS_MARKERS)


@app.after_request
def add_cors(resp):
    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Access-Control-Allow-Headers"] = "*"
    resp.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    resp.headers["Access-Control-Allow-Private-Network"] = "true"
    return resp


@app.route("/")
def index():
    return redirect("/index.html")


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


@app.route("/task3")
def task3_route():
    """Данные задания 3: ссылка на фото + вопросы-опоры."""
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
        "photo_url": v["photo_url"],
        "questions": v["t3_questions"],
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
    """Оценивает ответ ученика через GigaChat по критериям ФИПИ (задание 2)."""
    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()
    try:
        qid = int(data.get("question_id"))
    except (TypeError, ValueError):
        return jsonify(error="Не передан question_id"), 400

    entry = QUESTIONS_BY_ID.get(qid)
    if entry is None:
        return jsonify(error="Неизвестный вопрос"), 404

    v, q_idx = entry
    question_text = v["questions"][q_idx]
    passage = v["passage"]
    reference = v["reference"]

    if not text:
        return jsonify(error="Пустой ответ ученика"), 400

    try:
        score = evaluate_student_answer(text, question_text, passage)
        return jsonify(
            reference=reference,
            question=question_text,
            score=score,
            good=bool(score == 1),
        )
    except Exception as e:
        return jsonify(error=str(e)), 500


@app.route("/compare3", methods=["POST"])
def compare3_route():
    """Оценивает монолог (задание 3): М1 (основы + факты по описанию) и М2 (логика)."""
    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()
    try:
        vid = int(data.get("variant_id"))
    except (TypeError, ValueError):
        return jsonify(error="Не передан variant_id"), 400

    v = BY_ID.get(vid)
    if v is None:
        return jsonify(error="Неизвестный вариант"), 404
    if not text:
        return jsonify(error="Пустой ответ ученика"), 400

    questions = "\n".join(v["t3_questions"])
    description = v["photo_description"]

    # Основы считаем локально через spaCy — модели этот подсчёт не доверяем.
    # Она систематически занижает (в нашем случае вернула «3» вместо «10+»).
    stems = count_grammatical_stems(text)

    if description:
        # GigaChat проверяет ТОЛЬКО факты и логику, отдельными запросами.
        try:
            verdict = evaluate_task3(text, questions, description)
        except Exception as e:
            return jsonify(error=f"Ошибка GigaChat: {e}"), 500

        facts_ok = bool(verdict.get("facts_ok", True))
        logic_ok = bool(verdict.get("logic_ok", True))
        note = verdict.get("note", "")

        # Страховка: если модель всё же обнулила факты по «мягкой»
        # придирке (возраст, обобщение, оттенок) — не обнуляем.
        if not facts_ok and _is_soft_facts_note(note):
            facts_ok = True
            note = ""
    else:
        # Описание не заполнено: считаем основы через spaCy, факты не проверяем.
        facts_ok = True
        try:
            logic_ok = bool(evaluate_task3_logic(text, questions))
        except Exception as e:
            return jsonify(error=f"Ошибка GigaChat: {e}"), 500
        note = "описание фото не задано — фактическая точность не проверялась"

    # М1: 2 при ≥10 основах и без фактических ошибок, 1 при 5–9, иначе 0.
    if facts_ok:
        m1 = 2 if stems >= 10 else (1 if stems >= 5 else 0)
    else:
        m1 = 0

    # М2 оценивается только при М1 ≥ 1 (как в спецификации ФИПИ).
    m2 = 1 if (m1 >= 1 and logic_ok) else 0

    return jsonify(
        variant_id=v["id"],
        stems=stems,
        facts_ok=facts_ok,
        logic_ok=logic_ok,
        note=note,
        m1=m1,
        m2=m2,
        total=m1 + m2,
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)