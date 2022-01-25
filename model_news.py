from bs4 import BeautifulSoup
# import requests
import re
# import pandas as pd
import numpy as np
import random
import torch
# from tqdm.notebook import tqdm
from transformers import GPT2Tokenizer

if torch.cuda.is_available():    
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

tokenizer = GPT2Tokenizer.from_pretrained('sberbank-ai/rugpt3small_based_on_gpt2')

model = torch.load('news_gpt3.pt')


def generate_news(start_words):
  # start_words = start_words
  start_words = tokenizer.encode(start_words, return_tensors='pt').to(device)
  out = model.generate(
    input_ids=start_words,
    max_length=50,
    num_beams=5,
    do_sample=True,
    temperature=7.,
    top_k=55,
    top_p=0.6,
    no_repeat_ngram_size=4,
    num_return_sequences=1,
    ).cpu().numpy()
  return re.sub('\n', '', tokenizer.decode(out[0]))


