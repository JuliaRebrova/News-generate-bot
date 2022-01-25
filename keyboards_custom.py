from aiogram.types import ReplyKeyboardRemove, \
    ReplyKeyboardMarkup, KeyboardButton, \
    InlineKeyboardMarkup, InlineKeyboardButton


def add_button(text):
    button_hi = KeyboardButton(text)
    greet_kb = ReplyKeyboardMarkup()
    return greet_kb.add(button_hi)
