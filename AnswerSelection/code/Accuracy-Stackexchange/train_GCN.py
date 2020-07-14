import torch
import os
import argparse
import numpy as np
import pandas as pd
import torch.nn as nn
from test import evaluate_GCN
import torch.nn.functional as F
from utils import getPostContexts
from sklearn import preprocessing
from torch.autograd import Variable
from model import GCN_individual
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from data import QADataset, SubjObjDataset, get_GCN_inputs, get_GCN_inputs2, get_GCN_inputs3, normalize, sparse_mx_to_torch_sparse_tensor
from sklearn.metrics import accuracy_score

#torch.set_num_threads(12)

# parser
parser = argparse.ArgumentParser(description='PyTorch Credibility Prediction Model')
parser.add_argument('--use_cuda', dest='use_cuda', default=False, action='store_true')
parser.add_argument('-lr', '--learning_rate', type=float, default=0.001,\
                    help='learning rate for ensemble')
parser.add_argument('--weight_decay', type=float, default=5e-3)
parser.add_argument('--num_epochs', type=int, default=400)
parser.add_argument('--batch_size', type=int, default=500)
parser.add_argument('--root', type=str, default='../movie-data')

args = parser.parse_args()
seed = 1234734614
torch.manual_seed(seed)
if args.use_cuda:
	torch.cuda.manual_seed(seed)

# dataset
dataset = QADataset(dataset='StackExchange',questionFile = 'QuestionFeatures.tsv', answerFile = 'AnswerFeatures.tsv', userFile = 'UserFeatures.tsv',
                                    rootFolder= args.root)
print ("Dataset read", len(dataset))

PosClass = dataset.trainPairs_WFeatures[dataset.trainPairs_WFeatures['Credible'] == '1']
NegClass = dataset.trainPairs_WFeatures[dataset.trainPairs_WFeatures['Credible'] == '0']
print ("Positive samples",len(PosClass))
questions = dataset.trainPairs['QuestionId'].unique()

if len(PosClass) > len(NegClass):
    NegClass_Sample = NegClass
else:
    NegClass_Sample = NegClass.sample(n=len(PosClass))

X = dataset.trainPairs_WFeatures


## PreProcess the dataset
cols = X.columns.drop('QTags')
X[cols] = X[cols].apply(pd.to_numeric, errors='coerce')
X.fillna(0, inplace=True)
X["PairId"] = X["PairId"].apply(lambda x: int(x)-1)
data,X_Tags_Feature2 = getPostContexts(X, dataset)
X_Tags_Feature = Variable(data, requires_grad=False)

QuestionId = X.loc[:,'QuestionId'].drop_duplicates()
QuestionId = QuestionId[QuestionId>0]

# get data_loader
Pairs_train, Pairs_test, Y_train, Y_test = train_test_split(QuestionId[QuestionId!=0], QuestionId[QuestionId!=0], test_size=0.2, random_state=1234)
trainset = torch.utils.data.TensorDataset(torch.LongTensor(Pairs_train.values), torch.torch.LongTensor(Y_train.values))
testset = torch.utils.data.TensorDataset(torch.LongTensor(Pairs_test.values), torch.torch.LongTensor(Y_test.values))

train_loader = torch.utils.data.DataLoader(dataset=trainset,
                                                batch_size=args.batch_size)

test_loader = torch.utils.data.DataLoader(dataset=testset,
                                          batch_size=len(Pairs_test),
                                          shuffle=False)

# get adj
edges = dataset.getAdjList_clique()
edges2 = dataset.getAdjList_Similarity1()
edges3 = dataset.getAdjList_Similarity2()
edges4 = dataset.getAdjList_Similarity3(Pairs_train.values.tolist())
edges5 = dataset.getAdjMatrix_Identity(len(X))
Adj, rowsum = get_GCN_inputs(edges, len(X))
Adj2, rowsum = get_GCN_inputs3(edges2, len(X))
Adj3, rowsum = get_GCN_inputs3(edges3, len(X))
Adj4, rowsum = get_GCN_inputs3(edges4, len(X))
Adj5, rowsum = get_GCN_inputs2(edges5, len(X))

"""
# Random Forest
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(max_depth=5, random_state=0)
X_RF = []
y_RF = []
X_RF_test = []
y_RF_test = []
for i, (X_train, Y_train) in enumerate(train_loader):
    X_train = X[X["QuestionId"].isin(X_train)]['PairId'].values
    Y_train = X[X["QuestionId"].isin(Y_train)]['Credible'].values.tolist()
    X_RF += data.numpy()[X_train].tolist()
    y_RF += Y_train
clf.fit(X_RF, y_RF)
for i, (X_test, Y_test) in enumerate(test_loader):
    X_test = X[X["QuestionId"].isin(X_test)]['PairId'].values
    Y_test = X[X["QuestionId"].isin(Y_test)]['Credible'].values.tolist()
    X_RF_test += data.numpy()[X_test].tolist()
    y_RF_test += Y_test
result = clf.predict(X_RF_test)
accuracy = sum([1 if i==j else 0 for i,j in zip(result,y_RF_test)])/float(len(result))
print "accuracy for random forest is: ",accuracy
"""

# setting of GCN
nComm = 2
nHid1 = 50
nHid2 = 30
nHid3 = 20
nHid4 = 10
nFeat = X_Tags_Feature.shape[1]
gcn_model = GCN_individual(nFeat, nHid1, nHid2, nHid3, nHid4, nComm)
#gcn_model.load_state_dict(torch.load('gcn_complete2.pkl'))
criterion = nn.CrossEntropyLoss()
gcn_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad,gcn_model.parameters()), lr=args.learning_rate, weight_decay=args.weight_decay)

# train and test
try:
    for epoch in range(1, args.num_epochs + 1):

        train_acc = 0
        train_acc2 = 0
        train_acc3 = 0
        train_acc4 = 0
        train_acc5 = 0
        train_acc6 = 0
        length = 0
        loss_train = 0


        for i, (X_train, Y_train) in enumerate(train_loader):
            X_train = torch.tensor(X[X["QuestionId"].isin(X_train)]['PairId'].values)
            Y_train = torch.tensor(X[X["QuestionId"].isin(Y_train)]['Credible'].values)
            gcn_model.train(True)
            gcn_optimizer.zero_grad()
            user_gcn_embed,user_gcn_embed2,user_gcn_embed3,user_gcn_embed4 = gcn_model(X_Tags_Feature, Adj, Adj2, Adj3, Adj4, Adj5)

            predicted = torch.max(user_gcn_embed.squeeze()[X_train], 1)[1].data
            train_acc+= (Y_train.numpy()==predicted.numpy()).sum()
            length+= len(Y_train)

            predicted2 = torch.max(user_gcn_embed2.squeeze()[X_train], 1)[1].data
            train_acc2+= (Y_train.numpy()==predicted2.numpy()).sum()

            predicted3 = torch.max(user_gcn_embed3.squeeze()[X_train], 1)[1].data
            train_acc3+= (Y_train.numpy()==predicted3.numpy()).sum()

            predicted4 = torch.max(user_gcn_embed4.squeeze()[X_train], 1)[1].data
            train_acc4+= (Y_train.numpy()==predicted4.numpy()).sum()


            Y_train = Variable(Y_train, requires_grad=False)
            print(user_gcn_embed[X_train].shape)
            loss = criterion(user_gcn_embed[X_train], Y_train)+criterion(user_gcn_embed2[X_train], Y_train)+criterion(user_gcn_embed3[X_train], Y_train)+criterion(user_gcn_embed4[X_train], Y_train)
            print(Y_train.shape)
            loss_train+=loss.item()
            loss.backward(retain_graph=True)
            gcn_optimizer.step()

        if epoch > 0:
            #print "GCN Weight", gcn_model.gc1.weight.data
            print("Total loss", epoch, loss_train)
            currcorrect,currcorrect2,currcorrect3,currcorrect4= evaluate_GCN(gcn_model, test_loader, Adj, Adj2, Adj3, Adj4, Adj5, X_Tags_Feature, rowsum, X, epoch)

            print ("Epoch ", epoch, "'s accuracy is", currcorrect, currcorrect2, currcorrect3 , currcorrect4)
            train_acc = train_acc/float(length)
            train_acc2 = train_acc2/float(length)
            train_acc3 = train_acc3/float(length)
            train_acc4 = train_acc4/float(length)
            print ("Train Accuracy", train_acc, train_acc2, train_acc3 , train_acc4)
            print (args.root)

except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')

#torch.save(gcn_model.state_dict(), 'gcnmodel_movie.pt')
