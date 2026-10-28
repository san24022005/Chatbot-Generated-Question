from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
from difflib import SequenceMatcher
import re
from threading import RLock
from typing import Callable, Iterable, Mapping


STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
    "in", "is", "it", "me", "of", "on", "or", "the", "to", "you", "your",
}


def normalize_question(text: str) -> str:
    text = str(text or "")
    text = re.sub(r"[^\w\s]", "", text.lower())
    return " ".join(text.split())


def questions_are_similar(first: str, second: str, threshold: float = 0.86) -> bool:
    normalized_first = normalize_question(first)
    normalized_second = normalize_question(second)
    if not normalized_first or not normalized_second:
        return False
    if normalized_first == normalized_second:
        return True

    # A length-sensitive check catches paraphrases without treating short questions
    # with a shared opening phrase as duplicates by accident.
    if abs(len(normalized_first.split()) - len(normalized_second.split())) > 3:
        return False
    return SequenceMatcher(None, normalized_first, normalized_second).ratio() >= threshold


@dataclass
class StrategyRules:
    strong_score: int = 85
    weak_score: int = 70
    consecutive_weak_turns: int = 2
    consecutive_strong_turns: int = 2
    topic_turns: int = 4
    similarity_threshold: float = 0.86
    max_question_retries: int = 3
    minimum_assessment_turns: int = 3


@dataclass
class ConversationTurn:
    question: str
    answer: str
    analysis: dict[str, object] = field(default_factory=dict)
    strategy: str = "follow_up"


@dataclass
class ConversationState:
    session_id: str
    learner_id: str
    topic: str = "English communication"
    assessment_goal: str = "both"
    difficulty: int = 2
    max_turns: int = 8
    turn_count: int = 0
    status: str = "active"
    stop_reason: str = ""
    current_question: str = ""
    used_questions: list[str] = field(default_factory=list)
    history: list[ConversationTurn] = field(default_factory=list)
    grammar_errors: list[str] = field(default_factory=list)
    vocabulary_issues: list[str] = field(default_factory=list)
    strengths: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    topic_turn_count: int = 0
    difficulty_history: list[int] = field(default_factory=list)
    latest_analysis: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["history"] = [asdict(turn) for turn in self.history]
        return data


class QuestionGenerationError(RuntimeError):
    pass


class ConversationManager:
    """Coordinates assessment sessions without loading or defining any AI model."""

    def __init__(
        self,
        question_generator: Callable[[dict[str, object]], str],
        fallback_questions: Iterable[str] = (),
        rules: StrategyRules | None = None,
    ) -> None:
        self.question_generator = question_generator
        self.fallback_questions = [str(q).strip() for q in fallback_questions if str(q).strip()]
        self.rules = rules or StrategyRules()

    def create_session(
        self,
        session_id: str,
        learner_id: str | None = None,
        topic: str = "English communication",
        assessment_goal: str = "both",
        difficulty: int = 2,
        max_turns: int = 8,
    ) -> ConversationState:
        if assessment_goal not in {"grammar", "vocabulary", "both"}:
            raise ValueError("assessment_goal must be 'grammar', 'vocabulary', or 'both'.")
        if max_turns < 1:
            raise ValueError("max_turns must be at least 1.")
        return ConversationState(
            session_id=str(session_id),
            learner_id=str(learner_id or session_id),
            topic=str(topic).strip() or "English communication",
            assessment_goal=assessment_goal,
            difficulty=self._bounded_difficulty(difficulty),
            max_turns=int(max_turns),
            difficulty_history=[self._bounded_difficulty(difficulty)],
        )

    def start(self, state: ConversationState) -> str:
        if state.status != "active":
            raise ValueError("Only an active session can start a question.")
        context = self.build_context(state, strategy="follow_up")
        question = self._generate_valid_question(state, context)
        state.current_question = question
        state.used_questions.append(question)
        return question

    def process_answer(
        self,
        state: ConversationState,
        answer: str,
        analysis: Mapping[str, object] | None,
        stop_requested: bool = False,
    ) -> dict[str, object]:
        normalized_answer = " ".join(str(answer or "").strip().split())
        if not normalized_answer:
            raise ValueError("Answer cannot be empty.")
        if state.status != "active":
            raise ValueError("This assessment session is no longer active.")

        state.turn_count += 1
        safe_analysis = self._normalize_analysis(analysis)
        state.latest_analysis = safe_analysis
        self._update_findings(state, safe_analysis)

        strategy = self.choose_strategy(state, safe_analysis, stop_requested)
        if stop_requested:
            state.status = "completed"
            state.stop_reason = "learner_requested"
        elif state.turn_count >= state.max_turns:
            state.status = "completed"
            state.stop_reason = "maximum_turns_reached"

        self._adjust_difficulty(state, safe_analysis)
        state.history.append(
            ConversationTurn(
                question=state.current_question,
                answer=normalized_answer,
                analysis=safe_analysis,
                strategy=strategy,
            )
        )

        if state.status == "completed":
            state.current_question = ""
            return self._result(state, strategy, next_question="")

        context = self.build_context(state, strategy=strategy)
        try:
            next_question = self._generate_valid_question(state, context)
        except QuestionGenerationError as exc:
            state.status = "completed"
            state.stop_reason = "question_generation_failed"
            state.current_question = ""
            return self._result(state, "finish", next_question="", error=str(exc))

        state.current_question = next_question
        state.used_questions.append(next_question)
        state.topic_turn_count += 1
        return self._result(state, strategy, next_question=next_question)

    def choose_strategy(
        self,
        state: ConversationState,
        analysis: Mapping[str, object],
        stop_requested: bool = False,
    ) -> str:
        if stop_requested:
            return "finish"
        if state.turn_count >= state.max_turns:
            return "finish"

        score = self._mean_score(analysis)
        recent_scores = [self._score_of_turn(turn) for turn in state.history[-2:]]
        recent_scores.append(score)

        if len(recent_scores) >= self.rules.consecutive_weak_turns:
            weak_count = sum(value < self.rules.weak_score for value in recent_scores[-self.rules.consecutive_weak_turns:])
            if weak_count == self.rules.consecutive_weak_turns:
                return "simplify_question"

        strong_count = sum(value >= self.rules.strong_score for value in recent_scores[-self.rules.consecutive_strong_turns:])
        if strong_count == self.rules.consecutive_strong_turns:
            return "increase_challenge"

        grammar_errors = analysis.get("grammar_errors", [])
        vocabulary_issues = analysis.get("vocabulary_issues", [])
        if grammar_errors:
            return "follow_up"
        if vocabulary_issues:
            return "explore_vocabulary"

        if state.turn_count >= self.rules.minimum_assessment_turns:
            return "finish" if self._has_sufficient_information(state, analysis) else "change_topic"
        if state.topic_turn_count >= self.rules.topic_turns:
            return "change_topic"
        if len(recent_scores) >= 3 and min(recent_scores) >= self.rules.strong_score:
            return "increase_challenge"
        return "explore_vocabulary"

    def _has_sufficient_information(
        self,
        state: ConversationState,
        current_analysis: Mapping[str, object],
    ) -> bool:
        prior_count = self.rules.minimum_assessment_turns - 1
        recent = state.history[-prior_count:] if prior_count else []
        if len(recent) != prior_count:
            return False
        for analysis in [*(turn.analysis for turn in recent), current_analysis]:
            if state.assessment_goal in {"grammar", "both"} and "grammar_score" not in analysis:
                return False
            if state.assessment_goal in {"vocabulary", "both"} and "vocabulary_score" not in analysis:
                return False
        return True

    def build_context(self, state: ConversationState, strategy: str) -> dict[str, object]:
        latest = asdict(state.history[-1]) if state.history else {}
        weak_spots = list(dict.fromkeys([*state.grammar_errors, *state.vocabulary_issues]))
        return {
            "topic": state.topic,
            "assessment_goal": state.assessment_goal,
            "difficulty": state.difficulty,
            "latest_answer": latest.get("answer", ""),
            "latest_analysis": state.latest_analysis,
            "weak_spots": weak_spots[-12:],
            "recent_history": [asdict(turn) for turn in state.history[-4:]],
            "strategy": strategy,
            "previous_questions": list(state.used_questions),
        }

    def stop(self, state: ConversationState, reason: str = "learner_requested") -> dict[str, object]:
        if state.status == "active":
            state.status = "completed"
            state.stop_reason = reason
            state.current_question = ""
        return self.summary(state)

    def summary(self, state: ConversationState) -> dict[str, object]:
        return {
            "status": state.status,
            "stop_reason": state.stop_reason,
            "question_count": state.turn_count,
            "difficulty_reached": state.difficulty,
            "difficulty_history": list(state.difficulty_history),
            "grammar_strengths": list(dict.fromkeys(
                item for item in state.strengths if item == "Clear grammar structures"
            )),
            "vocabulary_strengths": list(dict.fromkeys(
                item for item in state.strengths if item == "Specific and varied vocabulary"
            )),
            "grammar_weaknesses": list(state.grammar_errors),
            "vocabulary_weaknesses": list(state.vocabulary_issues),
            "common_errors": [error for error, _ in Counter(state.grammar_errors).most_common(5)],
            "recommendations": list(dict.fromkeys(state.recommendations)),
            "session_id": state.session_id,
            "learner_id": state.learner_id,
        }

    def _generate_valid_question(self, state: ConversationState, context: dict[str, object]) -> str:
        for attempt in range(1, self.rules.max_question_retries + 1):
            try:
                candidate = str(self.question_generator(dict(context))).strip()
            except Exception as exc:
                if attempt == self.rules.max_question_retries:
                    raise QuestionGenerationError(f"Question model failed: {exc}") from exc
                continue

            if candidate and not self._is_duplicate(state, candidate):
                return candidate

        for fallback in self.fallback_questions:
            if fallback and not self._is_duplicate(state, fallback):
                return fallback
        raise QuestionGenerationError(
            f"No unique valid question could be generated after {self.rules.max_question_retries} retries."
        )

    def _is_duplicate(self, state: ConversationState, candidate: str) -> bool:
        return any(
            questions_are_similar(question, candidate, self.rules.similarity_threshold)
            for question in state.used_questions
        )

    def _normalize_analysis(self, analysis: Mapping[str, object] | None) -> dict[str, object]:
        if not isinstance(analysis, Mapping):
            analysis = {}
        invalid_fields = set()
        grammar_errors = self._string_list(analysis.get("grammar_errors", []))
        vocabulary_issues = self._string_list(analysis.get("vocabulary_issues", []))
        grammar_score = self._bounded_score(analysis.get("grammar_score"), invalid_fields, "grammar_score")
        vocabulary_score = self._bounded_score(analysis.get("vocabulary_score"), invalid_fields, "vocabulary_score")
        corrected = analysis.get("corrected_text")
        if not isinstance(corrected, str):
            invalid_fields.add("corrected_text")
            corrected = ""
        return {
            "grammar_errors": grammar_errors,
            "vocabulary_issues": vocabulary_issues,
            "grammar_score": grammar_score,
            "vocabulary_score": vocabulary_score,
            "corrected_text": corrected.strip(),
            "confidence": self._bounded_score(analysis.get("confidence"), invalid_fields, "confidence", default=50),
            "invalid_fields": sorted(invalid_fields),
        }

    @staticmethod
    def _string_list(value: object) -> list[str]:
        if not isinstance(value, list):
            return []
        return list(dict.fromkeys(str(item).strip() for item in value if str(item).strip()))

    @staticmethod
    def _bounded_score(value: object, invalid_fields: set[str], field_name: str, default: int = 60) -> int:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            invalid_fields.add(field_name)
            return default
        score = int(value)
        if not 0 <= score <= 100:
            invalid_fields.add(field_name)
        return min(100, max(0, score))

    def _update_findings(self, state: ConversationState, analysis: Mapping[str, object]) -> None:
        state.grammar_errors.extend(str(item) for item in analysis["grammar_errors"])
        state.vocabulary_issues.extend(str(item) for item in analysis["vocabulary_issues"])
        grammar_score = int(analysis["grammar_score"])
        vocabulary_score = int(analysis["vocabulary_score"])
        if grammar_score >= self.rules.strong_score:
            state.strengths.append("Clear grammar structures")
        if vocabulary_score >= self.rules.strong_score:
            state.strengths.append("Specific and varied vocabulary")
        if grammar_score < self.rules.weak_score:
            state.recommendations.append("Practice the identified grammar patterns in full sentences")
        if vocabulary_score < self.rules.weak_score:
            state.recommendations.append("Use more specific words and explain ideas with examples")
        if analysis["invalid_fields"]:
            state.recommendations.append("The model output was incomplete, so this turn should be reviewed conservatively")

    def _score_of_turn(self, turn: ConversationTurn) -> int:
        analysis = turn.analysis
        return self._mean_score(analysis)

    @staticmethod
    def _mean_score(analysis: Mapping[str, object]) -> int:
        return int(round((int(analysis.get("grammar_score", 60)) + int(analysis.get("vocabulary_score", 60))) / 2))

    def _adjust_difficulty(self, state: ConversationState, analysis: Mapping[str, object]) -> None:
        recent = [self._score_of_turn(turn) for turn in state.history[-2:]]
        recent.append(self._mean_score(analysis))
        change = 0
        if len(recent) >= self.rules.consecutive_strong_turns:
            strong_window = recent[-self.rules.consecutive_strong_turns:]
            if all(value >= self.rules.strong_score for value in strong_window):
                change = 1
        if len(recent) >= self.rules.consecutive_weak_turns:
            weak_window = recent[-self.rules.consecutive_weak_turns:]
            if all(value < self.rules.weak_score for value in weak_window):
                change = -1
        state.difficulty = self._bounded_difficulty(state.difficulty + change)
        state.difficulty_history.append(state.difficulty)

    @staticmethod
    def _bounded_difficulty(value: int) -> int:
        try:
            number = int(value)
        except (TypeError, ValueError):
            number = 2
        return max(1, min(5, number))

    def _result(
        self,
        state: ConversationState,
        strategy: str,
        next_question: str,
        error: str = "",
    ) -> dict[str, object]:
        result = {
            "session_id": state.session_id,
            "status": state.status,
            "strategy": strategy,
            "question": next_question,
            "turn_count": state.turn_count,
            "difficulty": state.difficulty,
            "analysis": state.latest_analysis,
            "summary": self.summary(state),
        }
        if error:
            result["error"] = error
        return result


class ConversationManagerRegistry:
    """In-memory, isolated session storage with one lock protecting all operations."""

    def __init__(self) -> None:
        self._sessions: dict[str, ConversationState] = {}
        self._lock = RLock()

    def create(self, state: ConversationState) -> ConversationState:
        with self._lock:
            self._sessions[state.session_id] = state
            return state

    def get(self, session_id: str) -> ConversationState:
        with self._lock:
            session = self._sessions.get(str(session_id))
            if session is None:
                raise KeyError(session_id)
            return session

    def delete(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(str(session_id), None)
