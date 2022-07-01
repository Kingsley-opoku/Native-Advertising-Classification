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
    sequences = padding(encoder(preprocessing(text_cleaned), FastText('simple')), 32)
    sequences_t = torch.tensor(sequences)
    # sequences_s = torch.stack(sequences_t)
    # flat_input = torch.flatten(sequences_t)

    
    model = NaiveClassifier(2, 32, 128, 300)
    model.load_state_dict(torch.load('saved_models_accuracy/saved_model_acc_99_epochs.pt'))
    print(sequences_t.shape)
    hidden = model.init_hidden(24)


    output, _ = model(sequences_t, hidden)
        
    # print(type(sequences_t))
    # print(sequences_t)









    # return prediction



predict('files/464_raw_html.txt')