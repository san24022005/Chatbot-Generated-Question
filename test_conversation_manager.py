from __future__ import annotations

import unittest
from unittest.mock import patch

import app as app_module
import model_loader
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

    def test_non_question_generation_uses_fallback(self):
        manager = ConversationManager(
            question_generator=lambda _context: "Do not repeat the previous question",
            fallback_questions=["What do you enjoy about learning English?"],
        )
        state = manager.create_session("invalid-question")

        with self.assertLogs("conversation_manager", level="WARNING"):
            question = manager.start(state)

        self.assertEqual(question, "What do you enjoy about learning English?")
        self.assertEqual(state.used_questions, [question])

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

    def test_start_uses_predefined_question_without_generating_one(self):
        manager = ConversationManager(
            question_generator=lambda _context: self.fail("Question generation should not run.")
        )
        state = manager.create_session("predefined-question")

        question = manager.start(state, initial_question="What do you like to do in your free time?")

        self.assertEqual(question, "What do you like to do in your free time?")
        self.assertEqual(state.used_questions, [question])


class ModelPathTests(unittest.TestCase):
    def test_all_model_and_tokenizer_paths_exist(self):
        self.assertEqual(set(model_loader.MODEL_PATHS), {"grammar", "question", "contextual"})
        for paths in model_loader.MODEL_PATHS.values():
            model_dir = paths["model_dir"]
            tokenizer_dir = paths["tokenizer_dir"]
            self.assertTrue((model_dir / "config.json").is_file())
            self.assertTrue((model_dir / "model.safetensors").is_file())
            self.assertTrue((tokenizer_dir / "tokenizer.json").is_file())
            self.assertTrue((tokenizer_dir / "tokenizer_config.json").is_file())

    def test_load_models_loads_all_models_once(self):
        with (
            patch.dict(model_loader._LOADED_MODELS, {}, clear=True),
            patch.dict(model_loader._LOAD_ERRORS, {}, clear=True),
            patch("model_loader.load_local_t5_model", return_value=object()) as load_model,
        ):
            loaded, errors = model_loader.load_models()
            cached, cached_errors = model_loader.load_models()

        self.assertEqual(set(loaded), {"grammar", "question", "contextual"})
        self.assertEqual(loaded, cached)
        self.assertEqual(errors, {})
        self.assertEqual(cached_errors, {})
        self.assertEqual(load_model.call_count, 3)


class FlaskConversationTests(unittest.TestCase):
    def setUp(self) -> None:
        app_module.app.config.update(TESTING=True)
        app_module.ensure_models_loaded = lambda: None
        app_module.MODELS = {
            "grammar": object(),
            "contextual": object(),
            "question": object(),
        }
        app_module.MODEL_LOAD_ERRORS = {}
        app_module.check_grammar = lambda text: text
        app_module.generate_contextual_text = lambda **kwargs: "A short related passage."
        app_module.CONVERSATION_REGISTRY = ConversationManagerRegistry()
        questions = iter(
            [
                "What do you usually do after school?",
                "How do you make time for your English studies?",
                "What has helped you improve most?",
            ]
        )
        app_module.CONVERSATION_MANAGER = ConversationManager(
            question_generator=lambda _context: next(questions),
        )
        self.client = app_module.app.test_client()

    def test_question_prompt_avoids_echoing_instruction(self):
        context = {
            "topic": "Travel",
            "learner_level": "beginner",
            "strategy": "follow_up",
            "latest_answer": "I visited Da Lat.",
            "latest_corrected_answer": "I visited Da Lat.",
            "latest_expanded_text": "Da Lat is known for its cool weather.",
            "recent_history": [],
            "previous_questions": ["Where did you go?"],
        }
        with patch.object(app_module, "generate_questions", return_value="What did you enjoy?") as generate:
            question = app_module._generate_question(context)

        prompt = generate.call_args.args[0]
        self.assertEqual(question, "What did you enjoy?")
        self.assertIn("New question:", prompt)
        self.assertNotIn("Do not repeat a previous question", prompt)

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
        self.assertEqual(status["messages"][0]["text"], first_question)
        self.assertNotEqual(status["question"], first_question)
        self.assertEqual(
            status["messages"][-1]["text"],
            f"{app_module.CORRECT_ANSWER_ENCOURAGEMENT}\n\nNext question:\n{status['question']}",
        )

    def test_new_chat_randomly_chooses_predefined_initial_question(self):
        generated_contexts = []
        app_module.CONVERSATION_MANAGER = ConversationManager(
            question_generator=lambda context: generated_contexts.append(context) or "Generated question?"
        )

        selected_question = app_module.INITIAL_QUESTIONS[1]
        with patch.object(app_module.random, "choice", return_value=selected_question) as choose_question:
            response = self.client.post("/api/new-chat", json={})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["question"], selected_question)
        choose_question.assert_called_once_with(app_module.INITIAL_QUESTIONS)
        self.assertEqual(generated_contexts, [])

    def test_invalid_generated_questions_fall_back_without_failing_submission(self):
        app_module.CONVERSATION_MANAGER = ConversationManager(
            question_generator=lambda _context: "Do not repeat the previous question",
            fallback_questions=[
                "What do you enjoy about learning English?",
                "Which place would you like to visit next?",
            ],
        )

        with self.assertLogs("conversation_manager", level="WARNING"):
            created = self.client.post("/api/new-chat", json={})
            answered = self.client.post(
                "/api/submit-answer",
                json={"answer": "I once visited a city with my family."},
            )

        self.assertEqual(created.status_code, 200)
        self.assertIn(created.get_json()["question"], app_module.INITIAL_QUESTIONS)
        self.assertEqual(answered.status_code, 200)
        self.assertEqual(answered.get_json()["question"], "What do you enjoy about learning English?")

    def test_http_learner_stop(self):
        self.client.post("/api/new-chat", json={"max_turns": 4})
        stopped = self.client.post("/api/end-chat")

        self.assertEqual(stopped.status_code, 200)
        self.assertEqual(stopped.get_json()["status"], "completed")
        self.assertEqual(stopped.get_json()["summary"]["stop_reason"], "learner_requested")
        self.assertIn("Session summary:", stopped.get_json()["messages"][-1]["text"])

    def test_pipeline_passes_corrected_and_expanded_context_to_question_model(self):
        contexts = []
        grammar_inputs = []
        contextual_inputs = []
        app_module.check_grammar = lambda text: grammar_inputs.append(text) or "I visit museums every weekend."
        app_module.generate_contextual_text = lambda **kwargs: contextual_inputs.append(kwargs) or (
            "Museums preserve art and history for future generations."
        )
        app_module.CONVERSATION_MANAGER = ConversationManager(
            question_generator=lambda context: contexts.append(context)
            or ("What do you enjoy learning at museums?" if context["recent_history"] else "What do you enjoy?")
        )

        self.client.post("/api/new-chat", json={"topic": "Museums"})
        response = self.client.post(
            "/api/submit-answer",
            json={"answer": "I visits museums every weekends."},
        )
        payload = response.get_json()

        self.assertEqual(response.status_code, 200)
        self.assertEqual(grammar_inputs, ["I visits museums every weekends."])
        self.assertEqual(contextual_inputs[0]["corrected_answer"], "I visit museums every weekend.")
        self.assertEqual(contextual_inputs[0]["topic"], "Museums")
        self.assertEqual(contexts[-1]["latest_corrected_answer"], "I visit museums every weekend.")
        self.assertEqual(
            contexts[-1]["latest_expanded_text"],
            "Museums preserve art and history for future generations.",
        )
        assistant_text = payload["messages"][-1]["text"]
        self.assertEqual(
            assistant_text,
            f"Did you mean: I visit museums every weekend.?\n\nNext question:\n{payload['question']}",
        )
        self.assertNotIn(app_module.CORRECT_ANSWER_ENCOURAGEMENT, assistant_text)
        self.assertNotIn("Grammar:", assistant_text)
        self.assertNotIn("More to explore:", assistant_text)

    def test_answer_processing_failure_does_not_append_partial_turn(self):
        app_module.check_grammar = lambda text: (_ for _ in ()).throw(RuntimeError("grammar unavailable"))
        self.client.post("/api/new-chat", json={})

        with self.assertLogs(app_module.LOGGER, level="ERROR"):
            response = self.client.post("/api/submit-answer", json={"answer": "I like reading."})

        self.assertEqual(response.status_code, 503)
        self.assertIn("could not be processed", response.get_json()["error"])
        with self.client.session_transaction() as browser_session:
            session_id = browser_session["conversation_session_id"]
        state = app_module.CONVERSATION_REGISTRY.get(session_id)
        self.assertEqual(state.history, [])

    def test_new_chat_requires_all_three_models(self):
        app_module.MODELS = {"grammar": object(), "question": object()}

        response = self.client.post("/api/new-chat", json={})

        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.get_json()["missing_models"], ["contextual"])

    def test_standalone_model_apis_remain_available(self):
        app_module.check_grammar = lambda text: "Corrected text."
        app_module.generate_questions = lambda context: "What do you enjoy?"

        grammar = self.client.post("/api/check-grammar", json={"text": "Text."})
        questions = self.client.post("/api/generate-questions", json={"context": "A short story."})

        self.assertEqual(grammar.status_code, 200)
        self.assertEqual(grammar.get_json()["corrected_text"], "Corrected text.")
        self.assertEqual(questions.status_code, 200)
        self.assertEqual(questions.get_json()["questions"], ["What do you enjoy?"])


if __name__ == "__main__":
    unittest.main()
