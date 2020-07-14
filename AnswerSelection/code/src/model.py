import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn.utils
from torch.nn.parameter import Parameter
from layers import GraphConvolution
import numpy as np
from scipy import sparse
import math
import os
USE_CUDA = False

class FeedForward(nn.Module):
    def __init__(self, num_users, hidden_size, num_classes, vocab_size, embeddings=None, freeze_embeddings=False):

        super(FeedForward, self).__init__()
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.num_users = num_users

        self.textembed = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        if embeddings is not None:
            self.textembed.weight = nn.Parameter(embeddings)
        if freeze_embeddings:
            self.textembed.weight.requires_grad = False

        self.embed = nn.Embedding(num_users, hidden_size)
        self.layer1 = nn.Linear(26, hidden_size)
        self.layer2 = nn.Linear(hidden_size, num_classes)
        #self.init_weights()

    def forward(self, users, features, text, user_gcn_embed):

        seq_len = len(users)
        #user_embedded = self.embed(users)
        #embedded = user_embedded.view(user_embedded.size(0),-1)
        #print "User embedding", user_gcn_embed.shape, users.shape
        users = users.contiguous().view(1,-1).int().tolist()
        #print "Users reshaped", users
        user_embedded = user_gcn_embed[users]
        embedded = user_embedded.view(-1, 2*user_embedded.size(1))


        #print "Embedded", embedded

        #print "TEXT", text.sum()
        textEmbedded = self.textembed(text)
        #print "Shape of textEmbedding", textEmbedded.shape
        #print "Embedded", textEmbedded

        ####Concatenate embed output with features
        features = Variable(features.float(), requires_grad=False)

        #concatOutput = torch.cat((embedded, features), 1)
        #output_1 = self.layer1(concatOutput)

        output_1 = F.relu(self.layer1(features))

        out = self.layer2(output_1)
        #out = output_1
        return out

    def init_weights(self):
        initrange = 0.2
        self.embed.weight.data.uniform_(-initrange, initrange)
        #self.layer1.bias.data.fill_(0)
        self.layer1.weight.data.uniform_(-initrange, initrange)
        #self.layer1.bias.requires_grad = False

        #self.layer2.bias.data.fill_(0)
        #self.layer2.weight.data.uniform_(-initrange, initrange)
        #self.layer2.bias.requires_grad = False

class GCN(nn.Module):
    def __init__(self, nFeat, nhid, nclass, dropout=0.5):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nFeat, nhid, bias=True)
        self.gc2 = GraphConvolution(2*nhid, nhid, bias=True)
        self.dropout = dropout
        self.dense = nn.Linear(nhid, nclass, bias=1)

    def forward(self, x, adj):

        #x = self.gc1(x,adj)
        x = F.relu(self.gc1(x,adj))
        #x = F.dropout(x, self.dropout, training=self.training)
        #x = F.relu(self.gc2(x, adj))

        x = self.dense(x)
        #print x
        #x = self.dense(x)
        return x
