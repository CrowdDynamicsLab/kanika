import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

class BiLSTM(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, label_size, use_gpu, batch_size, dropout=0.3):
        super(BiLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.use_gpu = use_gpu
        self.batch_size = batch_size
        self.dropout = torch.nn.Dropout(dropout)
        self.embeddings = torch.nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, bidirectional=True)
        self.cos = nn.CosineSimilarity(dim=1)
        self.conv = nn.Conv1d(200, 50, 3, stride=1)
        self.tanh = nn.Tanh()

    def init_hidden(self):
        # first is the hidden h
        # second is the cell c
        if self.use_gpu:
            return (Variable(torch.zeros(2, self.batch_size, self.hidden_dim).cuda()),
                    Variable(torch.zeros(2, self.batch_size, self.hidden_dim).cuda()))
        else:
            return (Variable(torch.zeros(2, self.batch_size, self.hidden_dim)),
                    Variable(torch.zeros(2, self.batch_size, self.hidden_dim)))

    def forward(self, sentence1, sentence2):
        q = self.embeddings(sentence1)
        ans = self.embeddings(sentence2)
        #import pdb; pdb.set_trace()
        q, _ = self.lstm(q)
        ans, _ = self.lstm(ans)
        #import pdb; pdb.set_trace()

        #q = self.dropout(q)
        #ans = self.dropout(ans)

        q = self.tanh(self.conv(q.permute(1,2,0)))
        ans = self.tanh(self.conv(ans.permute(1,2,0)))

        q = torch.max(q.permute(2,0,1), 0)[0] # (bs, 2H)
        ans = torch.max(ans.permute(2,0,1), 0)[0] # (bs, 2H)
        q_orig = q
        ans_orig = ans
        #q = self.dropout(q)
        #ans = self.dropout(ans)
        #q = torch.max(q, 0)[0] # (bs, 2H)
        #ans = torch.max(ans, 0)[0] # (bs, 2H)

        q = self.dropout(q)
        ans = self.dropout(ans)
        return self.cos(q, ans), q_orig, ans_orig
