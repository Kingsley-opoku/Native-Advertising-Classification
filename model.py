import torch
import torch.nn as nn
import torch.nn.functional as F



class NaiveClassifier(nn.Module):
    def __init__(self,no_layers,vocab_size,hidden_dim,embedding_dim,drop_prob=0.5):
        super(NaiveClassifier,self).__init__()
 
        # self.output_dim = output_dim
        self.hidden_dim = hidden_dim
 
        self.no_layers = no_layers
        self.vocab_size = vocab_size
        self.embedding_dim=embedding_dim
    
        # embedding and LSTM layers
        # self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)

        #lstm
        self.lstm = nn.LSTM(input_size=embedding_dim,hidden_size=self.hidden_dim,
                           num_layers=no_layers, batch_first=True)
        
        
        # dropout layer
        self.dropout = nn.Dropout(0.3)
    
        # linear and sigmoid layer

        self.fc1=nn.Linear(self.hidden_dim, 200)
        self.fc2 = nn.Linear(200, 1)
        self.sig = nn.Sigmoid()


    def forward(self,x,hidden):
        # print(x.shape)
        batch_size = x.size(0)
        # x = torch.tensor(x).to(torch.int64)
        # embeddings and lstm_out
        # embeds = self.embedding(x) 
        # print(embeds)
        
        lstm_out, hidden = self.lstm(x, hidden)
        
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim) 
        
        # dropout and fully connected layer
        out = self.dropout(lstm_out)
        out = self.fc1(out)
        act = F.relu(self.dropout(out))
        out = self.fc2(act)
        
        # sigmoid function
        sig_out = self.sig(out)
        
        # reshape to be batch_size first
        sig_out = sig_out.view(batch_size, -1)

        sig_out = sig_out[:, -1] # get last batch of labels
        
        # return last sigmoid output and hidden state
        return sig_out, hidden


    def init_hidden(self, batch_size):
        ''' Initializes hidden state '''
        # Create two new tensors with sizes n_layers x batch_size x hidden_dim,
        # initialized to zero, for hidden state and cell state of LSTM
        h_1 = torch.zeros((self.no_layers,batch_size,self.hidden_dim))
        h_2= torch.zeros((self.no_layers,batch_size,self.hidden_dim))
       
        return h_1, h_2


