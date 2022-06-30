from spacy import Vocab
from model import NaiveClassifier
import torch.nn as nn
import torch
import numpy as np
from clean_encode import TrainData, train_test_split ,collate
from torch.utils.data import DataLoader
from functools import partial


#  obtaining the data and spliting into train and test
df_train, df_test=train_test_split('outputs/data_labels.csv')


train_set, valid_set=TrainData(df_train), TrainData(df_test)


batch_size = 24
train_loader = DataLoader(
                        train_set, batch_size=batch_size, 
                        collate_fn=partial(collate, vectorizer=train_set.vectorizer),
                        shuffle=True,
                        drop_last=True
                        )


valid_loader= DataLoader(
                        valid_set, batch_size=batch_size, 
                        collate_fn=partial(collate, vectorizer=valid_set.vectorizer),
                        # drop_last=True
                        )




# function to predict accuracy
def accuracy_score(pred,label):
    pred = torch.round(pred.squeeze())
    return torch.sum(pred == label.squeeze()).item()

# vocab=len(train_loader)+1
# setting a device for the training

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# loss and optimization functions
model=NaiveClassifier(2, 32, 128, 300)

model.to(device)

lr = 0.001
# loss and optimization functions
criterion = nn.BCELoss()

optimizer = torch.optim.Adam(model.parameters(), lr=lr)


clip = 5
epochs = 100
valid_loss_min = np.Inf
# train for some number of epochs
epoch_tr_loss,epoch_vl_loss = [],[]
epoch_tr_acc,epoch_vl_acc = [],[]

for epoch in range(epochs):
    train_losses = []
    train_acc = 0.0
    model.train()
    # initialize hidden state 
    h = model.init_hidden(batch_size)
    for inputs, labels in train_loader:
        # print(type(inputs))
        inputs, labels = inputs.to(device), labels.to(device)  

        # Creating new variables for the hidden state, otherwise
        # we'd backprop through the entire training history
        h = tuple([each.data for each in h])

        model.zero_grad()
        output,h = model(inputs,h)
        
        # calculate the loss and perform backprop
        loss = criterion(output.squeeze(), labels.float().squeeze())
        loss.backward()
        train_losses.append(loss.item())
        # calculating accuracy
        accuracy = accuracy_score(output,labels)
        train_acc += accuracy
        #`clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

    val_h = model.init_hidden(batch_size)
    val_losses = []
    val_acc = 0.0
    model.eval()
    with torch.no_grad():
        for inputs, labels in valid_loader:
            val_h = tuple([each.data for each in val_h])
            
        
            # inputs = torch.tensor(inputs).to(device).long()

            inputs, labels = inputs.to(device), labels.to(device)

            output, val_h = model(inputs, val_h)
            val_loss = criterion(output.squeeze(), labels.float().squeeze())
            
            val_losses.append(val_loss.item())
            
            accuracy = accuracy_score(output,labels)
            val_acc += accuracy
    model.train()
    epoch_train_loss = sum(train_losses)/len(train_losses)
    epoch_val_loss = sum(val_losses)/len(val_losses)
    epoch_train_acc = train_acc/len(train_loader.dataset)
    epoch_val_acc = val_acc/len(valid_loader.dataset)
    epoch_tr_loss.append(epoch_train_loss)
    epoch_vl_loss.append(epoch_val_loss)
    epoch_tr_acc.append(epoch_train_acc)
    epoch_vl_acc.append(epoch_val_acc)
    print(f'Epoch {epoch+1}') 
    print(f'train_loss : {epoch_train_loss} val_loss : {epoch_val_loss}')
    print(f'train_accuracy : {epoch_train_acc} val_accuracy : {epoch_val_acc}')
    if epoch_val_loss <= valid_loss_min:
        torch.save(model.state_dict(), f'outputs/state_dict_{epoch}_epochs.pt')
        torch.save(model.state_dict(), f'outputs/state_dict_{epoch}_epochs.pth')
        print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min,epoch_val_loss))
        valid_loss_min = epoch_val_loss
    print(25*'==')
    