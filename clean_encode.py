import pandas as pd
import numpy as np
import re
import string
import spacy
import torch
from torch.utils.data import Dataset
from collections import Counter
from torchtext.vocab import  FastText

nlp = spacy.load('en_core_web_sm')


    

def  clean_text(text):
    #text =  text.lower()
    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"\r", "", text)
    text = re.sub(r"he's", "he is", text)
    text = re.sub(r"she's", "she is", text)
    text = re.sub(r"it's", "it is", text)
    text = re.sub(r"that's", "that is", text)
    text = re.sub(r"what's", "that is", text)
    text = re.sub(r"where's", "where is", text)
    text = re.sub(r"how's", "how is", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can't", "cannot", text)
    text = re.sub(r"n't", " not", text)
    text = re.sub(r"n'", "ng", text)
    text = re.sub(r"'bout", "about", text)
    text = re.sub(r"'til", "until", text)
    text = re.sub(r"[-()\"#/@;:<>{}`+=~|.!?,]", "", text)
    text = text.translate(str.maketrans('', '', string.punctuation)) 
    text = re.sub("(\\W)"," ",text) 
    text = re.sub('\S*\d\S*\s*','', text)
    
    return text

def train_test_split(filename: str, train_size=0.8):
    df=pd.read_csv(filename, index_col='id', nrows=30000)
    df['text'] = df['text'].apply(lambda x:clean_text(x))
    df=df.drop(columns='title')
    df_idx = [i for i in range(len(df))]
    np.random.shuffle(df_idx)
    len_train = int(len(df) * train_size)
    df_train = df.iloc[:len_train].reset_index(drop=True)
    df_test = df.iloc[len_train:].reset_index(drop=True)
    return df_train, df_test

def preprocessing(sentence):
    """
    Get rid of special symbols, lower case the words.
    return token list
    """
    doc = nlp(sentence)
    tokens = [token.lemma_ for token in doc if not token.is_punct and not token.is_space and not token.is_stop]
    return tokens



def token_encoder(token, vec):
    if token == "<pad>":
        return 1
    else:
        try:
            return vec.stoi[token]
        except:
            return 0


def encoder(tokens, vec):
    return [token_encoder(token, vec) for token in tokens]


def padding(list_of_idx, max_seq_len, padding_index=1):
    output = list_of_idx + (max_seq_len - len(list_of_idx))*[padding_index]
    return output[:max_seq_len]


class TrainData(Dataset):
    def __init__(self, df, max_seq_len=32): # df is the input df, max_seq_len is the max lenght allowed to a sentence before cutting or padding
        self.max_seq_len = max_seq_len
        
        counter = Counter()
        train_iter = iter(df['text'].values)
        self.vec = FastText("simple")
        self.vec.vectors[1] = -torch.ones(self.vec.vectors[1].shape[0]) # replacing the vector associated with 1 (padded value) to become a vector of -1.
        self.vec.vectors[0] = torch.zeros(self.vec.vectors[0].shape[0]) # replacing the vector associated with 0 (unknown) to become zeros
        self.vectorizer = lambda x: self.vec.vectors[x]
        self.labels = df.drop(columns='text').values
        sequences = [padding(encoder(preprocessing(sequence), self.vec), max_seq_len) for sequence in df['text'].tolist()]
        self.sequences = sequences
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, i):
        assert len(self.sequences[i]) == self.max_seq_len
        return self.sequences[i], self.labels[i]


def collate(batch, vectorizer):
    inputs = torch.stack([torch.stack([vectorizer(token) for token in sentence[0]]) for sentence in batch])
    target = torch.LongTensor(np.array([item[1] for item in batch]))
    return inputs, target



