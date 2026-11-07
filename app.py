from __future__ import annotations

import logging
import os
import random
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
    generate_contextual_text,
    generate_questions,
    load_models,
)


app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("FLASK_SECRET_KEY", "development-only-secret")
LOGGER = logging.getLogger(__name__)
MODELS: dict[str, LoadedModel] = {}
MODEL_LOAD_ERRORS: dict[str, str] = {}
CONVERSATION_REGISTRY = ConversationManagerRegistry()
INITIAL_QUESTIONS = (
    "What do you like to do in your free time?",
    "What is your favorite food, and why do you like it?",
    "Where do you like to go on weekends?",
    "Who do you usually spend your free time with?",
    "What is something new you would like to learn?",
)
CORRECT_ANSWER_ENCOURAGEMENT = "Great job! Your answer is grammatically correct."


def ensure_models_loaded() -> None:
    global MODELS, MODEL_LOAD_ERRORS
    if MODELS or MODEL_LOAD_ERRORS:
        return
    try:
        MODELS, MODEL_LOAD_ERRORS = load_models()
    except Exception as exc:  # pragma: no cover - defensive fallback.
        MODEL_LOAD_ERRORS = {"system": f"{type(exc).__name__}: {exc}"}
        MODELS = {}


def _generate_question(context: dict[str, Any]) -> str:
    history = context.get("recent_history", [])
    history_lines = []
    if isinstance(history, list):
        for turn in history[-4:]:
            if not isinstance(turn, dict):
                continue
            analysis = turn.get("analysis", {})
            corrected = analysis.get("corrected_text", "") if isinstance(analysis, dict) else ""
            history_lines.append(
                f"Question: {turn.get('question', '')}\n"
                f"Learner answer: {turn.get('answer', '')}\n"
                f"Corrected answer: {corrected}\n"
                f"Related context: {turn.get('expanded_text', '')}"
            )

    previous_questions = context.get("previous_questions", [])
    prompt = "\n".join(
        [
            "Create one natural, open-ended English follow-up question.",
            f"Topic: {context.get('topic', '')}",
            f"Learner level: {context.get('learner_level', 'beginner')}",
            f"Learning strategy: {context.get('strategy', 'follow_up')}",
            f"Latest original answer: {context.get('latest_answer', '')}",
            f"Latest corrected answer: {context.get('latest_corrected_answer', '')}",
            f"New related context: {context.get('latest_expanded_text', '')}",
            "Recent conversation:\n" + "\n\n".join(history_lines),
            "Earlier questions:\n" + "\n".join(previous_questions)
            if isinstance(previous_questions, list)
            else "",
            "New question:",
        ]
    )
    if MODELS.get("question") is None:
        raise RuntimeError("The question model is not available.")
    return generate_questions(prompt)


CONVERSATION_MANAGER = ConversationManager(
    question_generator=_generate_question,
    fallback_questions=[
        "What do you usually do after school?",
        "Describe your favorite place to study and explain why you like it.",
        "How has technology changed the way people communicate?",
        "What are the advantages and disadvantages of learning English online?",
        "Tell me about a memorable trip you have taken.",
        "How do you usually spend your weekends?",
        "What skill would you most like to improve, and why?",
        "What is something new you learned recently?",
        "How do you prepare when you have an important challenge?",
        "What would you like to achieve in the next few years?",
    ],
)


def _normalize_text(value: str) -> str:
    return " ".join(value.strip().split())


def _missing_models() -> list[str]:
    return [name for name in ("grammar", "contextual", "question") if name not in MODELS]


def _comparison_text(text: str) -> str:
    return " ".join(re.sub(r"[^\w']+", " ", text).split())


def _analyze_answer(answer: str) -> tuple[dict[str, object], str]:
    corrected = _normalize_text(check_grammar(answer))
    if not corrected:
        raise RuntimeError("The grammar model returned no correction.")

    is_correct = _comparison_text(answer) == _comparison_text(corrected)
    corrected_answer = answer if is_correct else corrected
    grammar_errors = [] if is_correct else ["The grammar model suggested a correction."]
    grammar_score = 100 if is_correct else 75

    stop_words = {
        "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
        "i", "in", "is", "it", "me", "of", "on", "or", "the", "to", "you", "your",
    }
    words = [
        word.casefold()
        for word in re.findall(r"\b[a-zA-Z']+\b", answer)
        if len(word) > 2 and word.casefold() not in stop_words
    ]
    unique_words = set(words)
    vocabulary_score = round(100 * len(unique_words) / len(words)) if words else 100
    vocabulary_issues = []
    if len(words) >= 4 and vocabulary_score < 75:
        vocabulary_issues.append("Try varying repeated words to express your idea more precisely.")

    analysis = {
        "grammar_errors": grammar_errors,
        "vocabulary_issues": vocabulary_issues,
        "grammar_score": grammar_score,
        "vocabulary_score": vocabulary_score,
        "corrected_text": corrected_answer,
        "confidence": 100 if is_correct else 75,
        "is_correct": is_correct,
    }
    return analysis, corrected_answer


def _learner_level(difficulty: int) -> str:
    if difficulty <= 2:
        return "beginner"
    if difficulty <= 4:
        return "intermediate"
    return "advanced"


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
            "topic": state.topic,
            "learner_level": _learner_level(state.difficulty),
        }
    )
    return summary


def _format_turn(state: ConversationState, turn: Any) -> str:
    analysis = turn.analysis
    feedback = []
    if analysis.get("is_correct"):
        feedback.append(CORRECT_ANSWER_ENCOURAGEMENT)
    else:
        feedback.append(f"Did you mean: {analysis['corrected_text']}?")
    if state.status != "completed" and turn.strategy != "finish":
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
        first_question = CONVERSATION_MANAGER.start(
            state,
            initial_question=random.choice(INITIAL_QUESTIONS),
        )
    except QuestionGenerationError:
        CONVERSATION_MANAGER.stop(state, reason="question_generation_failed")
        CONVERSATION_REGISTRY.delete(session_id)
        session.pop("conversation_session_id", None)
        raise

    messages = [{"role": "assistant", "text": first_question}]
    session["ui_messages"] = messages
    return {
        "messages": messages,
        "status": "ready",
        "question": first_question,
        "summary": _display_summary(state),
        "assessment_ready": True,
    }


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/health", methods=["GET"])
def health_check():
    ensure_models_loaded()
    missing = _missing_models()
    return jsonify(
        {
            "status": "ok" if not missing else "degraded",
            "device": str(DEVICE),
            "models_loaded": len(MODELS),
            "models": {name: name in MODELS for name in ("grammar", "contextual", "question")},
            "model_errors": MODEL_LOAD_ERRORS,
        }
    )


@app.route("/api/new-chat", methods=["POST"])
def new_chat():
    ensure_models_loaded()
    missing = _missing_models()
    if missing:
        return jsonify(
            {
                "error": "The chat is unavailable because required models could not be loaded.",
                "missing_models": missing,
                "model_errors": MODEL_LOAD_ERRORS,
            }
        ), 503
    data = request.get_json(silent=True) or {}
    try:
        return jsonify(_set_new_chat_session(data if isinstance(data, dict) else {}))
    except QuestionGenerationError as exc:
        LOGGER.exception("Initial question generation failed")
        return jsonify({"error": f"Could not start the chat: {exc}"}), 503
    except (TypeError, ValueError) as exc:
        return jsonify({"error": str(exc)}), 400


@app.route("/api/submit-answer", methods=["POST"])
def submit_answer():
    ensure_models_loaded()
    missing = _missing_models()
    if missing:
        return jsonify(
            {
                "error": "The chat is unavailable because required models could not be loaded.",
                "missing_models": missing,
                "model_errors": MODEL_LOAD_ERRORS,
            }
        ), 503
    data = request.get_json(silent=True) or {}
    raw_answer = data.get("answer", "") if isinstance(data, dict) else ""
    if not isinstance(raw_answer, str):
        return jsonify({"error": "Answer must be text."}), 400
    answer = _normalize_text(raw_answer)
    if not answer:
        return jsonify({"error": "Please enter an answer before sending."}), 400

    session_id = session.get("conversation_session_id")
    try:
        state = CONVERSATION_REGISTRY.get(session_id)
    except KeyError:
        return jsonify({"error": "The assessment has not started yet. Please create a new chat."}), 400

    try:
        analysis, corrected_answer = _analyze_answer(answer)
        history = [
            {"question": turn.question, "answer": turn.analysis.get("corrected_text", turn.answer)}
            for turn in state.history[-3:]
        ]
        history.append({"question": state.current_question, "answer": corrected_answer})
        expanded_text = generate_contextual_text(
            topic=state.topic,
            conversation_history=history,
            corrected_answer=corrected_answer,
            learner_level=_learner_level(state.difficulty),
        ).strip()
        if not expanded_text:
            raise RuntimeError("The contextual text model returned no result.")
    except Exception as exc:
        LOGGER.exception("The answer-processing models failed")
        return jsonify(
            {
                "error": f"The answer could not be processed: {type(exc).__name__}. Please try again.",
            }
        ), 503

    try:
        result = CONVERSATION_MANAGER.process_answer(
            state,
            answer,
            analysis,
            stop_requested=bool(data.get("end_session", False)),
            expanded_text=expanded_text,
        )
    except (TypeError, ValueError) as exc:
        LOGGER.warning("Invalid assessment answer: %s", exc)
        return jsonify({"error": str(exc)}), 400

    messages = session.get("ui_messages", [])
    messages.append({"role": "user", "text": answer})
    messages.append({"role": "assistant", "text": _format_turn(state, state.history[-1])})
    session["ui_messages"] = messages
    response = {
        "messages": messages,
        "status": result["status"] if result["status"] == "completed" else "ready",
        "question": result["question"],
        "summary": _display_summary(state),
        "assessment_ready": True,
        "strategy": result["strategy"],
        "difficulty": result["difficulty"],
        "error": result.get("error"),
    }
    if result.get("error"):
        return jsonify(response), 503
    return jsonify(response)


@app.route("/api/end-chat", methods=["POST"])
def end_chat():
    session_id = session.get("conversation_session_id")
    try:
        state = CONVERSATION_REGISTRY.get(session_id)
    except KeyError:
        return jsonify({"error": "No active assessment session."}), 400

    CONVERSATION_MANAGER.stop(state, reason="learner_requested")
    summary = _display_summary(state)
    latest = state.latest_analysis
    summary_text = (
        f"Session summary: {summary['question_count']} turns completed. "
        f"Grammar {latest.get('grammar_score', '—')}/100, "
        f"vocabulary {latest.get('vocabulary_score', '—')}/100."
    )
    messages = session.get("ui_messages", [])
    messages.append({"role": "assistant", "text": summary_text})
    session["ui_messages"] = messages
    return jsonify(
        {
            "status": state.status,
            "summary": summary,
            "messages": messages,
        }
    )


@app.route("/api/check-grammar", methods=["POST"])
def check_grammar_endpoint():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"success": False, "error": "Request body must be a JSON object."}), 400

    raw_text = data.get("text", "")
    if not isinstance(raw_text, str):
        return jsonify({"success": False, "error": "Text must be a string."}), 400
    text = _normalize_text(raw_text)
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

    raw_context = data.get("context", "")
    if not isinstance(raw_context, str):
        return jsonify({"success": False, "error": "Context must be a string."}), 400
    context = _normalize_text(raw_context)
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
