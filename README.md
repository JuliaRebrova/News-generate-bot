# News-generate-bot
A telegram bot generating fake news.  
Base model: sberbank-ai/rugpt3small_based_on_gpt2  
Subsequent training on Ria News (https://ria.ru)  

## Project structure
```
.
├── app
│   ├── .env                       # secret token
│   ├── model_news.py              # generates texts
│   └── telegram_bot.py            # runs telegram bot
├── data
│   ├── polit                      # political news
│   ├── econ                       # economical news
│   ├── sc                         # science news
│   └── sp                         # sport news
├── models
│   └── model.pth                  # torch model after training
├── train
│   ├── prepare_dataset.py         # reads all files in ../data shuffles them and tokenizes
│   └── train_model.py             # main train module 
├── ui
│   └── interface.py               # simple UI to check the model 
├── .gitignore
├── README.md
└── requirements.txt
```

## Installing dependencies
To install the dependencies run:
```
$ pip install -r requirements.txt
```

## Training model
You can train model on current dataset or replace it with your own one.  
To train model on existing dataset run:  
```
$ cd ./train
$ python3 ./train_model.py
```
As a result a new file `model.pth` will be created in the path `./models`

## Run interface
The repo has demo interface to check the model work. To run it on your local machine run:
```
$ cd ./ui
$ python3 ./interface.py
```
<img width="1279" alt="Снимок экрана 2023-07-30 в 01 55 08" src="https://github.com/JuliaRebrova/News-generate-bot/assets/90173032/377251be-b049-4213-aa0d-65a7d8a437c1">

It's possible to check the model from console:  
```
$ cd ./app
$ python3 ./model_news.py
```
## Run the telegram bot
For running a telegram bot it's nessesary to get a bot token from Bot Father.  
Create a file `.env` instead of the `.env_example` and write your token in it. 
Then run the bot:  
```
$ cd ./app
$ python3 ./telegram_bot.py
```

## Example:
<img width="692" alt="Снимок экрана 2023-07-30 в 02 23 03" src="https://github.com/JuliaRebrova/News-generate-bot/assets/90173032/03b315dc-9132-43d1-8077-c53e9ff62743">

