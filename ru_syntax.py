"""Подсчёт грамматических основ в тексте.

Основной путь — spaCy (модель ``ru_core_news_sm``). Резервный путь —
эвристика по финитным глаголам и противительным союзам (на однородных
сказуемых возможны погрешности, но сервер не падает).
"""

_nlp = None
_spacy_failed = False


def _get_nlp():
    global _nlp, _spacy_failed
    if _nlp is not None:
        return _nlp
    if _spacy_failed:
        return None
    try:
        import spacy
        _nlp = spacy.load("ru_core_news_sm")
    except Exception:
        _spacy_failed = True
    return _nlp


def count_grammatical_stems(text: str) -> int:
    """Считает грамматические основы (предикативные центры).

    Простое предложение = 1 основа. Сложное = по числу частей.
    Однородные сказуемые при общем подлежащем = одна основа.

    Отличия от старой версии: считаются номинальные сказуемые
    («Врач — женщина») и составные номинальные клаузы
    («Врач — женщина, а ребёнку два года» — две основы).
    """
    if not text or not text.strip():
        return 0
    nlp = _get_nlp()
    if nlp is not None:
        try:
            return _count_with_spacy(nlp, text)
        except Exception:
            pass
    return _count_with_heuristics(text)


# --- spaCy ---

# Зависимости, которые вводят отдельную клаузу.
# xcomp и acl НЕ включены: «хочу спать», «человек, читающий книгу»
# — это одна основа в школьном смысле.
_CLAUSE_DEPS = {
    "advcl", "ccomp", "relcl", "parataxis",
    "csubj", "csubj:pass",
}

# Противительные союзы, которые почти всегда разделяют клаузы.
_CONTRASTIVE_CC = {"а", "но", "однако", "зато"}


def _has_subject(tok) -> bool:
    return any(c.dep_ in ("nsubj", "nsubj:pass") for c in tok.children)


def _shares_subject_with_head(tok) -> bool:
    """True, если у conj-сказуемого нет своего подлежащего,
    но оно есть у ближайшего предка в цепочке однородности."""
    if _has_subject(tok):
        return False
    h = tok.head
    while h is not None and h is not h.head:
        if _has_subject(h):
            return True
        if h.dep_ != "conj":
            break
        h = h.head
    return False


def _introduces_nominal_clause(tok) -> bool:
    """True, если conj-номинал (сущ./прил.) вводит отдельную
    номинальную клаузу: свой субъект, копула или противительный союз.
    Ловит «а ребёнку два года», «а он — врач»."""
    for c in tok.children:
        if c.dep_ in ("nsubj", "nsubj:pass", "cop", "iobj"):
            return True
        if c.dep_ == "cc" and c.lower_ in _CONTRASTIVE_CC:
            return True
    return False


def _count_with_spacy(nlp, text: str) -> int:
    doc = nlp(text)
    total = 0
    for sent in doc.sents:
        stems = 1  # ROOT — всегда как минимум одна основа
        for tok in sent:
            if tok.dep_ == "ROOT":
                continue
            if tok.dep_ in _CLAUSE_DEPS:
                stems += 1
                continue
            if tok.dep_ == "conj":
                # Однородное сказуемое-глагол
                if tok.pos_ in ("VERB", "AUX"):
                    if not _shares_subject_with_head(tok):
                        stems += 1
                    continue
                # Однородная номинальная клауза: «X — Y, а Z — W»
                if _introduces_nominal_clause(tok):
                    stems += 1
        total += stems
    return total


# --- Резервная эвристика без spaCy ---

def _count_with_heuristics(text: str) -> int:
    """Если spaCy недоступен: число финитных глаголов + противительные
    союзы (для номинальных клауз). Однородные сказуемые при общем
    подлежащем могут завышать счёт — известная плата за простоту."""
    import re

    sentences = [s for s in re.split(r"[.!?…]+", text) if s.strip()]
    if not sentences:
        return 0

    verb_re = re.compile(
        r"\b[а-яё]+(?:ет|ит|ют|ут|ат|ят|ешь|ишь|ем|им|ете|ите|у|ю|ся|сь)\b",
        re.IGNORECASE,
    )
    contrastive_re = re.compile(r",\s*(?:а|но|однако|зато)\b", re.IGNORECASE)

    total = 0
    for sent in sentences:
        verbs = len(verb_re.findall(sent.lower()))
        contrastive = len(contrastive_re.findall(sent))
        if verbs <= 1:
            # Предложение с номинальным сказуемым: 1 + противительные клаузы
            total += 1 + contrastive
        else:
            total += verbs
    return total