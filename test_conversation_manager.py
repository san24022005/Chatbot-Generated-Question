from __future__ import annotations

import unittest

import app as app_module
from conversation_manager import (
    ConversationManager,
    ConversationManagerRegistry,
    QuestionGenerationError,
    StrategyRules,
)


def analysis(grammar_score: int = 90, vocabulary_score: int = 90, grammar_errors=None, vocabulary_issues=None):
    return {
        "grammar_errors": grammar_errors or [],
        "vocabulary_issues": vocabulary_issues or [],
        "grammar_score": grammar_score,
        "vocabulary_score": vocabulary_score,
        "corrected_text": "A corrected answer.",
    }


class ConversationManagerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.generated = iter(["How do you learn English?"])
        self.manager = ConversationManager(
            question_generator=lambda context: next(self.generated),
            fallback_questions=["What do you usually do after school?"],
            rules=StrategyRules(minimum_assessment_turns=10),
        )

    def state(self, max_turns: int = 8, difficulty: int = 2):
        return self.manager.create_session(
            "s1", learner_id="learner-1", max_turns=max_turns, difficulty=difficulty
        )

    def test_correct_answers_raise_difficulty(self):
        state = self.state()
        self.manager.start(state)
        self.generated = iter(["Why do people study?", "Compare online and classroom learning."])
        self.manager.question_generator = lambda context: next(self.generated)

        first = self.manager.process_answer(state, "I study every day.", analysis())
        second = self.manager.process_answer(state, "I study carefully and ask questions.", analysis())

        self.assertEqual(first["strategy"], "explore_vocabulary")
        self.assertEqual(second["strategy"], "increase_challenge")
        self.assertEqual(state.difficulty, 3)

    def test_grammar_error_uses_follow_up_not_simplify(self):
        state = self.state()
        self.manager.start(state)
        self.generated = iter(["Can you explain that sentence again?"])
        self.manager.question_generator = lambda context: next(self.generated)

        result = self.manager.process_answer(
            state,
            "She go to school.",
            analysis(80, 80, ["The verb should agree with the subject"]),
        )

        self.assertEqual(result["strategy"], "follow_up")
        self.assertEqual(state.difficulty, 2)

    def test_consecutive_weak_answers_simplify(self):
        state = self.state(difficulty=3)
        self.manager.start(state)
        self.generated = iter(["A simple question?", "Another simple question?"])
        self.manager.question_generator = lambda context: next(self.generated)

        self.manager.process_answer(state, "I try.", analysis(50, 55))
        result = self.manager.process_answer(state, "Not sure.", analysis(45, 52))

        self.assertEqual(result["strategy"], "simplify_question")
        self.assertEqual(state.difficulty, 2)

    def test_duplicate_question_retries_then_uses_fallback(self):
        state = self.state()
        used = ["How do you usually learn English?"]
        state.used_questions.extend(used)
        manager = ConversationManager(
            question_generator=lambda context: used[0],
            fallback_questions=["What is your favorite place to study?"],
        )

        question = manager._generate_valid_question(state, manager.build_context(state, "follow_up"))

        self.assertEqual(question, "What is your favorite place to study?")

    def test_maximum_turns_finishes(self):
        state = self.state(max_turns=2)
        self.manager.start(state)
        self.generated = iter(["Question two?"])
        self.manager.question_generator = lambda context: next(self.generated)

        self.manager.process_answer(state, "First answer.", analysis())
        result = self.manager.process_answer(state, "Second answer.", analysis())

        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["summary"]["stop_reason"], "maximum_turns_reached")

    def test_learner_requested_stop(self):
        state = self.state()
        self.manager.start(state)
        result = self.manager.process_answer(state, "Please stop.", analysis(), stop_requested=True)

        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["summary"]["stop_reason"], "learner_requested")

    def test_invalid_model_fields_are_normalized(self):
        state = self.state()
        self.manager.start(state)
        safe = self.manager._normalize_analysis(
            {"grammar_errors": "bad", "vocabulary_issues": [None, "word choice"], "grammar_score": 140,
             "vocabulary_score": True, "corrected_text": None}
        )

        self.assertEqual(safe["grammar_errors"], [])
        self.assertEqual(safe["grammar_score"], 100)
        self.assertEqual(safe["vocabulary_score"], 60)
        self.assertIn("grammar_score", safe["invalid_fields"])
        self.assertIn("corrected_text", safe["invalid_fields"])

    def test_registry_keeps_sessions_independent(self):
        registry = ConversationManagerRegistry()
        first = self.manager.create_session("first")
        second = self.manager.create_session("second")
        registry.create(first)
        registry.create(second)
        first.topic = "Travel"

        self.assertEqual(registry.get("first").topic, "Travel")
        self.assertEqual(registry.get("second").topic, "English communication")

    def test_question_generation_failure_has_safe_result(self):
        def fail(context):
            raise RuntimeError("model unavailable")

        manager = ConversationManager(
            question_generator=fail,
            fallback_questions=[],
            rules=StrategyRules(max_question_retries=1),
        )
        state = manager.create_session("failure")
        with self.assertRaises(QuestionGenerationError):
            manager.start(state)
        self.assertEqual(state.status, "active")


class FlaskConversationTests(unittest.TestCase):
    def setUp(self) -> None:
        app_module.app.config.update(TESTING=True)
        app_module.ensure_models_loaded = lambda: None
        app_module.MODELS = {}
        app_module.MODEL_LOAD_ERRORS = {}
        app_module.CONVERSATION_REGISTRY = ConversationManagerRegistry()
        app_module.CONVERSATION_MANAGER = ConversationManager(
            question_generator=lambda context: "What do you usually do after school?",
            fallback_questions=app_module.QUESTION_BANK,
        )
        self.client = app_module.app.test_client()

    def test_http_history_is_preserved_across_requests(self):
        created = self.client.post("/api/new-chat", json={"max_turns": 4})
        self.assertEqual(created.status_code, 200)
        first_question = created.get_json()["question"]

        answered = self.client.post("/api/submit-answer", json={"answer": "I study English every day because it helps me communicate."})
        payload = answered.get_json()
        self.assertEqual(answered.status_code, 200)
        self.assertNotEqual(payload["question"], "")
        self.assertEqual(payload["summary"]["question_count"], 1)

        status = self.client.get("/api/chat").get_json()
        self.assertEqual(len(status["messages"]), 3)
        self.assertIn("Hello!", status["messages"][0]["text"])
        self.assertNotEqual(status["question"], first_question)

    def test_http_learner_stop(self):
        self.client.post("/api/new-chat", json={"max_turns": 4})
        stopped = self.client.post("/api/end-chat")

        self.assertEqual(stopped.status_code, 200)
        self.assertEqual(stopped.get_json()["status"], "completed")
        self.assertEqual(stopped.get_json()["summary"]["stop_reason"], "learner_requested")


if __name__ == "__main__":
    unittest.main()
