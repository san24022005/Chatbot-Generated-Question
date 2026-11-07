import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

def generate_question(context: str, conversation_history: list, max_new_tokens: int = 64) -> str:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = T5Tokenizer.from_pretrained('./model')
    model = T5ForConditionalGeneration.from_pretrained('./model').to(device)

    history_parts = []
    for turn in conversation_history:
        history_parts.append(f"Question: {turn['question']} Answer: {turn['answer']}")
    history_str = " ".join(history_parts)

    input_text = f"generate next question: context: {context} history: {history_str}"
    inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True).to(device)

    with torch.no_grad():
        outputs = model.generate(
            inputs["input_ids"],
            max_length=max_new_tokens,
            num_beams=4,
            repetition_penalty=2.5,
            early_stopping=True
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
