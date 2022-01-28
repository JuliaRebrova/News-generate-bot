# News-generate-bot
A telegram bot generating fake news.

Base model: sberbank-ai/rugpt3small_based_on_gpt2

Subsequent training on Ria News (https://ria.ru)

# Example:

<img src="https://sun9-62.userapi.com/impg/ExLZkYM5sY_-GYH41ydYZ3aeGzKYlFGZfr6phw/qWtpfWqP8P4.jpg?size=828x1165&quality=95&sign=38cb1968afce0bd13ca99421537306ac&type=album" alt="sketch" width=200 height=300> <img src="https://sun9-1.userapi.com/impg/Rh8zhzBrBkKuN-WzylQ6vlRJAlJUY51QOAwUbg/AjTLsOV9aRI.jpg?size=591x1280&quality=95&sign=26f15af564ade3f4bb5a34337cc7ee44&type=album" alt="sketch" width=200 height=400>

# How to run
Firstable you should run the train_model_for_news_bot.ipynb, train the model on data from the folder news_1 and load it. Then run main.py.
