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
    


    vec = FastText("simple")
    vec.vectors[1] = -torch.ones(vec.vectors[1].shape[0]) # replacing the vector associated with 1 (padded value) to become a vector of -1.
    vec.vectors[0] = torch.zeros(vec.vectors[0].shape[0]) # replacing the vector associated with 0 (unknown) to become zeros
    vectorizer = lambda x: vec.vectors[x]
    # torch.stack([vectorizer(token) for token in text_cleaned])
    # df = df.rename(columns={'0':})
    sequences = padding(encoder(preprocessing(text_cleaned), FastText('simple')), 32)
    # sequences_t = torch.tensor(sequences)
    input = torch.stack([vectorizer(token) for token in sequences])
    # flat_input = torch.flatten(sequences_t)
    # print(df)

    
    model = NaiveClassifier(2, 32, 128, 300)
    model.load_state_dict(torch.load('saved_models_accuracy/saved_model_acc_99_epochs.pt'))
    # print(sequences_t.shape)
    hidden = model.init_hidden(1)

    # print(sequences)
    # print(input.shape)
    output, _ = model(input.unsqueeze(0), hidden)

    return output > 0.5



# if predict('files/464_raw_html.txt'):
#     print('Sponsored')
# else:
#     print('Not Sponsored')