from __future__ import annotations

from pathlib import Path

from model_loader import LoadedModel, generate_text, load_local_t5_model


MODEL_DIR = Path(__file__).resolve().parent / "fine_tuned_model"


class WoWTextGenerator:
    def __init__(self, model_path: str | Path = MODEL_DIR) -> None:
        self.model_path = Path(model_path).resolve()
        self._model: LoadedModel | None = None

    def _get_model(self) -> LoadedModel:
        if self._model is None:
            self._model = load_local_t5_model(self.model_path, self.model_path)
        return self._model

    def expand_answer(
        self,
        topic: str,
        history: list[str],
        learner_answer: str,
        knowledge_text: str,
        max_input_len: int = 256,
    ) -> str:
        prompt = (
            "expand answer:\n"
            f"topic: {topic}\n"
            f"history: {' | '.join(history[-4:])}\n"
            f"answer: {learner_answer}\n"
            f"knowledge: {knowledge_text}"
        )
        return generate_text(
            self._get_model(),
            prompt,
            max_output_tokens=128,
            max_input_tokens=max_input_len,
        )


if __name__ == "__main__":
    generator = WoWTextGenerator()
    result = generator.expand_answer(
        topic="Music",
        history=["Learner: I love listening to piano music.", "Chatbot: Piano music is beautiful."],
        learner_answer="Yes, it helps me focus.",
        knowledge_text="Piano music can help people relax and concentrate.",
    )
    print("\n--- Direct model test ---")
    print(f"Expanded text: {result}")
