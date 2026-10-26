from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import torch
from flask import Flask, jsonify, render_template, request

from model_loader import DEVICE, LoadedModel, load_models


app = Flask(__name__)
MODELS: dict[str, LoadedModel] = {}
MODEL_LOAD_ERRORS: dict[str, str] = {}


def ensure_models_loaded() -> None:
    global MODELS, MODEL_LOAD_ERRORS
    if MODELS or MODEL_LOAD_ERRORS:
        return
    try:
        MODELS, MODEL_LOAD_ERRORS = load_models()
    except Exception as exc:  # pragma: no cover - defensive fallback.
        MODEL_LOAD_ERRORS = {"system": f"{type(exc).__name__}: {exc}"}
        MODELS = {}


@dataclass
class AssessmentState:
    question_count: int = 0
    grammar_score: int | None = None
    vocabulary_score: int | None = None
    errors: list[str] = field(default_factory=list)
    improvements: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    is_active: bool = False
    history: list[dict[str, str]] = field(default_factory=list)


ASSESSMENT = AssessmentState()
QUESTION_BANK = [
    "What do you usually do after school?",
    "Describe your favorite place to study and explain why you like it.",
    "How has technology changed the way people communicate?",
    "What are the advantages and disadvantages of learning English online?",
    "Tell me about a memorable trip you have taken.",
]


def _generate_question(index: int) -> str:
    return QUESTION_BANK[index % len(QUESTION_BANK)]


def _normalize_text(value: str) -> str:
    return " ".join(value.strip().split())


def _extract_errors(answer: str) -> tuple[list[str], list[str], list[str]]:
    text = _normalize_text(answer)
    grammar_issues: list[str] = []
    vocabulary_issues: list[str] = []
    tips: list[str] = []

    if not text:
        return grammar_issues, vocabulary_issues, tips

    common_errors = {
        "i am": "I am",
        "i is": "I am",
        "she go": "She goes",
        "he go": "He goes",
        "do you likes": "Do you like",
        "i has": "I have",
        "the informations": "the information",
        "informations": "information",
        "this are": "These are",
        "many informations": "much information",
    }
    lowered = text.lower()
    for wrong, suggestion in common_errors.items():
        if wrong in lowered:
            grammar_issues.append(f"Consider changing '{wrong}' to '{suggestion}'.")

    if re.search(r"\b(i|you|he|she|we|they)\s+(am|is|are)\s+\w+\s+\w+\s+\w+\b", text, re.I):
        grammar_issues.append("The sentence structure may need clearer word order to sound more natural.")

    vocabulary_list = [
        "study",
        "improve",
        "benefit",
        "enjoy",
        "conversation",
        "experience",
        "opportunity",
        "travel",
        "technology",
    ]
    for token in vocabulary_list:
        if token.lower() in lowered:
            vocabulary_issues.append(f"You can expand the idea behind '{token}' with a more specific example.")

    if len(text.split()) < 8:
        tips.append("Your answer could be longer to show more detail and confidence.")
    if "because" in lowered:
        tips.append("Try adding a concrete example after 'because' to make your explanation stronger.")
    if not grammar_issues and not vocabulary_issues:
        tips.append("Your answer is already strong. Keep using clear connectors and specific examples.")

    return grammar_issues, vocabulary_issues, tips


def _rule_based_correction(text: str) -> str:
    normalized = _normalize_text(text)
    replacements = {
        "she go": "She goes",
        "he go": "He goes",
        "i am": "I am",
        "i has": "I have",
        "do you likes": "Do you like",
        "the informations": "the information",
        "informations": "information",
    }

    for wrong, correct in replacements.items():
        if re.search(re.escape(wrong), normalized, flags=re.IGNORECASE):
            normalized = re.sub(re.escape(wrong), correct, normalized, count=1, flags=re.IGNORECASE)
            break

    if not normalized.endswith((".", "!", "?")):
        normalized = f"{normalized}."
    return normalized


def _run_grammar_correction(model: LoadedModel, text: str) -> str:
    if model is None:
        return _rule_based_correction(text)
    try:
        encoded = model.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=model.max_input_tokens,
        )
        encoded = {key: value.to(DEVICE) for key, value in encoded.items()}
        with torch.inference_mode():
            output_ids = model.model.generate(
                **encoded,
                max_length=min(model.max_input_tokens, 128),
                num_beams=4,
                do_sample=False,
            )
        return model.tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
    except Exception:
        return _rule_based_correction(text)


def _generate_reference(answer: str) -> str:
    lowered = answer.lower()
    if "school" in lowered:
        return "A strong answer could be: 'I usually go to school by bus and I study with my classmates after class.'"
    if "travel" in lowered:
        return "A strong answer could be: 'Traveling helps me learn new vocabulary and understand different cultures.'"
    if "technology" in lowered:
        return "A strong answer could be: 'Technology makes communication faster, but it can also reduce face-to-face interaction.'"
    return "A strong answer could be: 'I think this topic is important because it helps me improve my speaking and writing skills in real life.'"


def _set_new_chat_session() -> dict[str, Any]:
    global ASSESSMENT
    ASSESSMENT = AssessmentState(is_active=True)
    first_question = _generate_question(0)
    model_note = ""
    if not MODELS:
        model_note = "\n\nThe AI model is unavailable in this environment, so the app is using a safe fallback evaluation mode."
    ASSESSMENT.history = [
        {
            "role": "assistant",
            "text": f"Hello! I’m your English Learning Assistant. {first_question}{model_note}",
        }
    ]
    return {
        "messages": ASSESSMENT.history,
        "status": "ready",
        "question": first_question,
        "summary": {
            "question_count": 0,
            "grammar_score": "—",
            "vocabulary_score": "—",
            "errors": [],
            "improvements": [],
            "recommendations": [],
        },
        "assessment_ready": False,
    }


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/health", methods=["GET"])
def health_check():
    ensure_models_loaded()
    return jsonify(
        {
            "status": "ok",
            "device": str(DEVICE),
            "models_loaded": len(MODELS),
            "model_errors": MODEL_LOAD_ERRORS,
        }
    )


@app.route("/api/new-chat", methods=["POST"])
def new_chat():
    ensure_models_loaded()
    return jsonify(_set_new_chat_session())


@app.route("/api/submit-answer", methods=["POST"])
def submit_answer():
    ensure_models_loaded()
    data = request.get_json(silent=True) or {}
    answer = _normalize_text(data.get("answer", ""))
    if not answer:
        return jsonify({"error": "Please enter an answer before sending."}), 400

    if not ASSESSMENT.is_active:
        return jsonify({"error": "The assessment has not started yet. Please create a new chat."}), 400

    grammar_errors, vocab_errors, tips = _extract_errors(answer)
    grammar_score = max(60, 100 - len(grammar_errors) * 12)
    vocabulary_score = max(60, 100 - len(vocab_errors) * 10)

    ASSESSMENT.question_count += 1
    ASSESSMENT.grammar_score = grammar_score
    ASSESSMENT.vocabulary_score = vocabulary_score
    ASSESSMENT.errors = grammar_errors
    ASSESSMENT.improvements = vocab_errors
    ASSESSMENT.recommendations = tips

    if "grammar" in MODELS:
        try:
            corrected = _run_grammar_correction(MODELS["grammar"], answer)
        except Exception as exc:
            corrected = f"The model could not correct this answer: {type(exc).__name__}."
    else:
        corrected = _rule_based_correction(answer)

    reference = _generate_reference(answer)
    feedback = [
        "Your answer is understandable, and there is room to improve the grammar and vocabulary structure.",
        f"Grammar score: {grammar_score}/100",
        f"Vocabulary score: {vocabulary_score}/100",
    ]
    feedback.extend(grammar_errors[:3])
    feedback.extend(vocab_errors[:3])
    feedback.extend(tips[:2])

    next_question = _generate_question(ASSESSMENT.question_count)
    ASSESSMENT.history.append({"role": "user", "text": answer})
    ASSESSMENT.history.append(
        {
            "role": "assistant",
            "text": "\n\n".join(
                [
                    *feedback,
                    "Suggested correction:\n" + corrected,
                    "Reference answer:\n" + reference,
                    "Next question:\n" + next_question,
                ]
            ),
        }
    )

    summary = {
        "question_count": ASSESSMENT.question_count,
        "grammar_score": grammar_score,
        "vocabulary_score": vocabulary_score,
        "errors": grammar_errors,
        "improvements": vocab_errors,
        "recommendations": tips,
    }

    return jsonify(
        {
            "messages": ASSESSMENT.history,
            "status": "ready",
            "question": next_question,
            "summary": summary,
            "assessment_ready": True,
        }
    )


@app.route("/api/chat", methods=["GET"])
def chat_status():
    return jsonify({"messages": ASSESSMENT.history, "status": "ready"})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
