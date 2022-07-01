import torch 
import pandas as pd
from datahandler import scrape_text_from_file
from clean_encode import encoder, padding, token_encoder, clean_text, preprocessing, collate
from torchtext.vocab import  FastText
from model import NaiveClassifier

def predict(path):
    scraped_text = scrape_text_from_file(path)
    text = ''.join(scraped_text)
    text_cleaned = clean_text(text)
    df = pd.DataFrame([text_cleaned])
    df = df.rename(columns={0: 'text'})
    
    return df




predict('files/464_raw_html.txt')