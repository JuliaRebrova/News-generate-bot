from ast import pattern
from cgi import test
import os
import pandas as pd
import numpy as np
import re
from numpy.random import RandomState
from transformers import GPT2Tokenizer


# DATA_PATH = os.path.join("..", "data")
# NEWS_TYPES = os.listdir(DATA_PATH)
# NEWS_TYPES = ["econ", "polit", "sc", "sp"]

class Dataset:
    def __init__(self, data_path) -> None:
        self.data_path = data_path
        self.news_types = os.listdir(self.data_path)
        self.texts = self._read_all_news()


    def _extract_clean_text(self, row_text) -> str:
        n_pattern = re.compile(r"\n")
        html_pattern = re.compile(r"(.+>)(.+)(<.+)")
        xa_pattern = re.compile(r"\xa0")

        cleaned_text = re.sub(n_pattern, " ", row_text)
        cleaned_text = re.sub(html_pattern, "\\2", cleaned_text)
        cleaned_text = re.sub(xa_pattern, " ", cleaned_text)
        return cleaned_text.strip()


    def _read_one_type(self, news_type: str) -> list:
        news_path = os.path.join(self.data_path, news_type)
        news_df = pd.read_csv(
            news_path,
            names=["indexes", "texts"],
            usecols=["texts"],
            skiprows=[0]
        )
        news_list = news_df["texts"].apply(lambda x: self._extract_clean_text(x)).tolist()
        return news_list

    def _read_all_news(self) -> str:
        texts_list = [self._read_one_type(news_type) for news_type in self.news_types]
        texts_flatten = []
        [texts_flatten.extend(x) for x in texts_list]
        random_state = RandomState(42)
        random_state.shuffle(texts_flatten)
        return texts_flatten


class Tokenizer:
    def __init__(self) -> None:
        self.tokenizer = GPT2Tokenizer.from_pretrained('sberbank-ai/rugpt3small_based_on_gpt2')

    def tokenize_train_test(self, texts):
        train_texts, test_texts = self._train_test_split(texts)
        train_tokens = self._tokenize(train_texts)
        test_tokens = self._tokenize(test_texts)
        print(f"len train_tokens = {len(train_tokens)}", f"len test_tokens = {len(test_tokens)}", sep="\n")
        return train_tokens, test_tokens

    def _tokenize(self, texts: str) -> list:
        tokens = self.tokenizer.encode(" ".join(texts), add_special_tokens=True)
        tokens = np.array(tokens)
        return tokens

    def _train_test_split(self, shuffled_texts) -> tuple:
        len_dataset = len(shuffled_texts)
        trash_hold = int(0.7 * len_dataset)
        return shuffled_texts[:trash_hold], shuffled_texts[trash_hold:]
       

# print(Dataset(DATA_PATH).texts)