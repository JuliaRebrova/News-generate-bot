
import logging 
from aiogram import Bot, Dispatcher, executor, types
from token_for_bot import at
import numpy as np

from aiogram.types import ReplyKeyboardRemove, \
    ReplyKeyboardMarkup, KeyboardButton \
    # InlineKeyboardMarkup, InlineKeyboardButton

at = at

# Configure logging
logging.basicConfig(level=logging.INFO)

from model_news import generate_news, model            

bot = Bot(token=at)
dp = Dispatcher(bot) 

from aiogram.types import ReplyKeyboardRemove, \
    ReplyKeyboardMarkup, KeyboardButton, \
    InlineKeyboardMarkup, InlineKeyboardButton

button_start = KeyboardButton('/start')
start_kb = ReplyKeyboardMarkup()
start_kb.add(button_start)     # doesn t work

greating = 'Привет!\n Я умею генерировать новости. Для этого нужно отправить мне несколько слов на русском языке.'
help_text = 'Для получения новсти отправь мне несколько слов на русском языке.'

@dp.message_handler(commands=['help']) 
async def send_welcome(message): 
    await message.reply(help_text)

# import keyboards_custom

@dp.message_handler(commands=['start'])
async def process_start_command(message: types.Message):
    button_news = KeyboardButton('Новость')
    greet_kb = ReplyKeyboardMarkup()
    greet_kb.add(button_news)
    await message.reply(greating, 
        reply_markup=greet_kb)

@dp.message_handler(regexp='Новость')
async def send_welcome(message): 
    await message.reply('Введите первые слова новости')

@dp.message_handler()  # news generator
async def echo(message: types.Message):
    answer_1 = generate_news(message['text'])
    await message.answer(answer_1)


if __name__ == '__main__':
    executor.start_polling(dp,skip_updates=True)

