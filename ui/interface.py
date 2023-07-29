import gradio as gr
import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 
from app.model_news import generate_news


iface = gr.Interface(fn=generate_news, inputs=["text"], outputs="text").launch()