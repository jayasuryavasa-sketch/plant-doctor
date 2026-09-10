"""Free question helper grounded only in approved disease-guide data."""


def answer_question(question: str, disease: dict | None) -> str:
    if disease is None:
        return "Please upload another clear leaf image or open the Disease Guide first."
    q = question.lower().strip()
    if not q:
        return "Ask about symptoms, causes, spread, disadvantages, cure or treatment, organic options, prevention, or care."
    if any(x in q for x in ("symptom", "sign", "look like")):
        return _answer("Common symptoms", disease["symptoms"])
    if any(x in q for x in ("cause", "why", "reason")):
        return _answer("Possible causes", disease["causes"])
    if any(x in q for x in ("spread", "transfer", "contagious")):
        return "How it can spread: " + disease["spread"]
    if any(x in q for x in ("disadvantage", "damage", "harm", "effect", "loss", "risk")):
        return "Possible impact: " + disease["description"] + " One photo cannot estimate severity or yield loss."
    if any(x in q for x in ("organic", "natural")):
        return _answer("Organic or natural options", disease["organic"])
    if any(x in q for x in ("avoid", "don't", "do not")):
        return _answer("What to avoid", disease["avoid"])
    if any(x in q for x in ("prevent", "prevention", "stop")):
        return _answer("Prevention and care", disease["prevention"] + disease["care"])
    if any(x in q for x in ("cure", "treat", "medicine", "fix", "help", "do")):
        return _answer("What you can do now", disease["immediate_steps"] + disease["treatment"])
    return "Ask about symptoms, causes, spread, disadvantages, treatment, organic options, prevention, or care."


def _answer(title: str, items: list[str]) -> str:
    return title + ": " + " • ".join(items)
