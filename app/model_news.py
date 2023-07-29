import re
import os
import torch
from transformers import GPT2Tokenizer


MODEL_PATH = os.path.join("..", "models", "model.pth")

tokenizer = GPT2Tokenizer.from_pretrained('sberbank-ai/rugpt3small_based_on_gpt2')
model = torch.load(MODEL_PATH)
device = torch.device("cpu")


def generate_news(start_words):
    start_words = tokenizer.encode(start_words, return_tensors='pt').to(device)
    out = model.generate(
        input_ids=start_words,
        max_length=50,
        num_beams=5,
        do_sample=True,
        temperature=1.,
        top_k=55,
        top_p=0.6,
        no_repeat_ngram_size=4,
        num_return_sequences=1,
        ).cpu().numpy()
    return re.sub('\n', '', tokenizer.decode(out[0]))


if __name__ == "__main__":
    print(generate_news(input()))
