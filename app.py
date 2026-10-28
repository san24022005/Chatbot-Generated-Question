from __future__ import annotations

import json
import logging
import os
import re
from typing import Any
from uuid import uuid4

from flask import Flask, jsonify, render_template, request, session

from conversation_manager import (
    ConversationManager,
    ConversationManagerRegistry,
    ConversationState,
    QuestionGenerationError,
)

from model_loader import (
    DEVICE,
    LoadedModel,
    check_grammar,
    generate_questions,
    load_models,
)


app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("FLASK_SECRET_KEY", "development-only-secret")
LOGGER = logging.getLogger(__name__)
MODELS: dict[str, LoadedModel] = {}
MODEL_LOAD_ERRORS: dict[str, str] = {}
CONVERSATION_REGISTRY = ConversationManagerRegistry()


def ensure_models_loaded() -> None:
    global MODELS, MODEL_LOAD_ERRORS
    if MODELS or MODEL_LOAD_ERRORS:
        return
    try:
        MODELS, MODEL_LOAD_ERRORS = load_models()
    except Exception as exc:  # pragma: no cover - defensive fallback.
        MODEL_LOAD_ERRORS = {"system": f"{type(exc).__name__}: {exc}"}
        MODELS = {}


QUESTION_BANK = [
    "What do you usually do after school?",
    "Describe your favorite place to study and explain why you like it.",
    "How has technology changed the way people communicate?",
    "What are the advantages and disadvantages of learning English online?",
    "Tell me about a memorable trip you have taken.",
]


def _generate_question(context: dict[str, Any]) -> str:
    """Adapter: serialize the explicit manager context for the existing T5 API."""
    prompt = json.dumps(context, ensure_ascii=False)
    if MODELS.get("question") is None:
        raise RuntimeError("The question model is not available.")
    return generate_questions(prompt)


CONVERSATION_MANAGER = ConversationManager(
    question_generator=_generate_question,
    fallback_questions=QUESTION_BANK,
)


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
        return check_grammar(text)
    except Exception:
        LOGGER.exception("Grammar inference failed")
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


def _messages_for(state: ConversationState) -> list[dict[str, str]]:
    messages = [
        {
            "role": "assistant",
            "text": f"Hello! I’m your English Learning Assistant. {state.current_question}",
        }
    ]
    for turn in state.history:
        messages.append({"role": "user", "text": turn.answer})
        messages.append({"role": "assistant", "text": _format_turn(state, turn)})
    return messages


def _display_summary(state: ConversationState) -> dict[str, Any]:
    summary = CONVERSATION_MANAGER.summary(state)
    latest = state.latest_analysis
    summary.update(
        {
            "grammar_score": latest.get("grammar_score", "—"),
            "vocabulary_score": latest.get("vocabulary_score", "—"),
            "errors": list(state.grammar_errors),
            "improvements": list(state.vocabulary_issues),
            "recommendations": summary["recommendations"],
            "difficulty": state.difficulty,
        }
    )
    return summary


def _format_turn(state: ConversationState, turn: Any) -> str:
    analysis = turn.analysis
    feedback = [
        "Your answer is understandable, and there is room to improve your grammar and vocabulary.",
        f"Grammar score: {analysis['grammar_score']}/100",
        f"Vocabulary score: {analysis['vocabulary_score']}/100",
    ]
    feedback.extend(analysis["grammar_errors"][:3])
    feedback.extend(analysis["vocabulary_issues"][:3])
    feedback.append("Suggested correction:\n" + (analysis["corrected_text"] or "No correction was returned."))
    feedback.append("Reference answer:\n" + _generate_reference(turn.answer))
    if state.status == "completed":
        feedback.append("Assessment complete. You can start a new chat anytime.")
    elif turn.strategy != "finish":
        feedback.append(f"Next question:\n{state.current_question}")
    return "\n\n".join(feedback)


def _set_new_chat_session(data: dict[str, Any] | None = None) -> dict[str, Any]:
    data = data or {}
    session_id = str(uuid4())
    state = CONVERSATION_MANAGER.create_session(
        session_id=session_id,
        learner_id=data.get("learner_id"),
        topic=str(data.get("topic") or "English communication"),
        assessment_goal=str(data.get("assessment_goal") or "both"),
        difficulty=int(data.get("difficulty") or 2),
        max_turns=int(data.get("max_turns") or 8),
    )
    CONVERSATION_REGISTRY.create(state)
    session["conversation_session_id"] = session_id
    session["ui_messages"] = []
    try:
        first_question = CONVERSATION_MANAGER.start(state)
    except QuestionGenerationError as exc:
        CONVERSATION_MANAGER.stop(state, reason="question_generation_failed")
        return {"messages": [], "status": "completed", "question": "", "summary": _display_summary(state), "assessment_ready": True, "error": str(exc)}

    model_note = ""
    if not MODELS:
        model_note = "\n\nThe AI model is unavailable in this environment, so the app is using a safe fallback evaluation mode."
    messages = [{"role": "assistant", "text": f"Hello! I’m your English Learning Assistant. {first_question}{model_note}"}]
    session["ui_messages"] = messages
    return {
        "messages": messages,
        "status": "ready",
        "question": first_question,
        "summary": _display_summary(state),
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
    data = request.get_json(silent=True) or {}
    return jsonify(_set_new_chat_session(data if isinstance(data, dict) else {}))


@app.route("/api/submit-answer", methods=["POST"])
def submit_answer():
    ensure_models_loaded()
    data = request.get_json(silent=True) or {}
    answer = _normalize_text(data.get("answer", "") if isinstance(data, dict) else "")
    if not answer:
        return jsonify({"error": "Please enter an answer before sending."}), 400

    session_id = session.get("conversation_session_id")
    try:
        state = CONVERSATION_REGISTRY.get(session_id)
    except KeyError:
        return jsonify({"error": "The assessment has not started yet. Please create a new chat."}), 400

    grammar_errors, vocab_errors, tips = _extract_errors(answer)
    try:
        corrected_text = (
            _run_grammar_correction(MODELS["grammar"], answer)
            if "grammar" in MODELS
            else _rule_based_correction(answer)
        )
    except Exception:
        LOGGER.exception("Grammar analysis adapter failed")
        corrected_text = _rule_based_correction(answer)
    analysis = {
        "grammar_errors": grammar_errors,
        "vocabulary_issues": vocab_errors,
        "grammar_score": max(60, 100 - len(grammar_errors) * 12),
        "vocabulary_score": max(60, 100 - len(vocab_errors) * 10),
        "corrected_text": corrected_text,
        "confidence": 75,
    }

    try:
        result = CONVERSATION_MANAGER.process_answer(
            state,
            answer,
            analysis,
            stop_requested=bool(data.get("end_session", False)),
        )
    except (TypeError, ValueError) as exc:
        LOGGER.warning("Invalid assessment answer: %s", exc)
        return jsonify({"error": str(exc)}), 400

    messages = session.get("ui_messages", [])
    messages.append({"role": "user", "text": answer})
    messages.append({"role": "assistant", "text": _format_turn(state, state.history[-1])})
    session["ui_messages"] = messages

    return jsonify(
        {
            "messages": messages,
            "status": result["status"] if result["status"] == "completed" else "ready",
            "question": result["question"],
            "summary": _display_summary(state),
            "assessment_ready": True,
            "strategy": result["strategy"],
            "difficulty": result["difficulty"],
            "error": result.get("error"),
        }
    )


@app.route("/api/end-chat", methods=["POST"])
def end_chat():
    session_id = session.get("conversation_session_id")
    try:
        state = CONVERSATION_REGISTRY.get(session_id)
    except KeyError:
        return jsonify({"error": "No active assessment session."}), 400

    summary = CONVERSATION_MANAGER.stop(state, reason="learner_requested")
    return jsonify(
        {
            "status": state.status,
            "summary": summary,
            "messages": session.get("ui_messages", []),
        }
    )


@app.route("/api/check-grammar", methods=["POST"])
def check_grammar_endpoint():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"success": False, "error": "Request body must be a JSON object."}), 400

    text = _normalize_text(data.get("text", ""))
    if not text:
        return jsonify({"success": False, "error": "Please enter text to correct."}), 400

    try:
        corrected = check_grammar(text)
    except Exception as exc:
        LOGGER.exception("Grammar endpoint inference failed")
        return jsonify(
            {
                "success": False,
                "error": f"The grammar model could not process the text: {type(exc).__name__}.",
            }
        ), 503

    if not corrected:
        return jsonify({"success": False, "error": "The grammar model returned no result."}), 502
    return jsonify({"success": True, "original_text": text, "corrected_text": corrected})


@app.route("/api/generate-questions", methods=["POST"])
def generate_questions_endpoint():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"success": False, "error": "Request body must be a JSON object."}), 400

    context = _normalize_text(data.get("context", ""))
    if not context:
        return jsonify({"success": False, "error": "Please enter context to generate questions."}), 400

    try:
        generated = generate_questions(context)
    except Exception as exc:
        LOGGER.exception("Question endpoint inference failed")
        return jsonify(
            {
                "success": False,
                "error": f"The question model could not process the text: {type(exc).__name__}.",
            }
        ), 503

    if not generated:
        return jsonify({"success": False, "error": "The question model returned no result."}), 502
    questions = [line.strip() for line in generated.splitlines() if line.strip()]
    return jsonify({"success": True, "context": context, "questions": questions or [generated]})


@app.route("/api/chat", methods=["GET"])
def chat_status():
    session_id = session.get("conversation_session_id")
    try:
        state = CONVERSATION_REGISTRY.get(session_id)
    except KeyError:
        return jsonify({"messages": [], "status": "not_started", "summary": None}), 404
    return jsonify(
        {
            "messages": session.get("ui_messages", []),
            "status": state.status,
            "question": state.current_question,
            "summary": _display_summary(state),
        }
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
