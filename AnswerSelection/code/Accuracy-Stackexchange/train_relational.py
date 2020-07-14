import torch
import os
import argparse
import numpy as np
import pandas as pd
import torch.nn as nn
from test import evaluate_relational
import torch.nn.functional as F
from utils import getPostContexts
from sklearn import preprocessing
from torch.autograd import Variable
from model import GCN_relational
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from data import QADataset, SubjObjDataset, get_GCN_inputs, get_GCN_inputs2, get_GCN_inputs3, normalize, sparse_mx_to_torch_sparse_tensor
from sklearn.metrics import accuracy_score
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#torch.set_num_threads(12)

# parser
parser = argparse.ArgumentParser(description='PyTorch Credibility Prediction Model')
parser.add_argument('--use_cuda', dest='use_cuda', default=False, action='store_true')
parser.add_argument('-lr', '--learning_rate', type=float, default=0.001,\
                    help='learning rate for ensemble')
parser.add_argument('--weight_decay', type=float, default=5e-3)
parser.add_argument('--num_epochs', type=int, default=300)
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
print "Dataset read", len(dataset)

PosClass = dataset.trainPairs_WFeatures[dataset.trainPairs_WFeatures['Credible'] == '1']
NegClass = dataset.trainPairs_WFeatures[dataset.trainPairs_WFeatures['Credible'] == '0']
print "Positive samples",len(PosClass)
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
Target =  np.array([[2*i-1] for i in X['Credible'].values.tolist()])
Target = Variable(torch.FloatTensor(Target),requires_grad=False).cuda()
QuestionId = X.loc[:,'QuestionId'].drop_duplicates()
QuestionId = QuestionId[QuestionId>0]

# get data_loader
from sklearn.model_selection import KFold
kf = KFold(n_splits=5, shuffle=True)
Dataset = np.array(QuestionId[QuestionId!=0])

# get adj
edges = dataset.getAdjList_clique()
edges2 = dataset.getAdjList_Similarity1()
edges3 = dataset.getAdjList_Similarity2()
#edges4 = dataset.getAdjList_Similarity3(Pairs_train.values.tolist())
edges5 = dataset.getAdjMatrix_Identity(len(X))
Adj, rowsum = get_GCN_inputs(edges, len(X))
Adj2, rowsum = get_GCN_inputs3(edges2, len(X))
Adj3, rowsum = get_GCN_inputs3(edges3, len(X))
#Adj4, rowsum = get_GCN_inputs3(edges4, len(X))
Adj5, rowsum = get_GCN_inputs2(edges5, len(X))
Adj = Adj.to(device)
Adj2 = Adj2.to(device)
Adj3 = Adj3.to(device)
Adj5 = Adj5.to(device)
# setting of GCN
nComm = 1
nHid1 = 50
nHid2 = 30
nHid3 = 20
nHid4 = 10
#gcn_model.load_state_dict(torch.load('gcn_complete2.pkl'))


result_list = list()
count = 0
# train and test
try:
    for Pairs_train, Pairs_test in kf.split(Dataset):
        count += 1
        if count==4:
            print result_list
            print sum(result_list)/len(result_list)

            torch.save(gcn_model.state_dict(), '/home/junting/Downloads/GCN/UserCredibility/Accuracy-Stackexchange/relational/gcnmodel_'+args.root.split('/')[-1]+'_relational'+'.pt')
            exit()

        edges4 = dataset.getAdjList_Similarity3(Pairs_train.tolist())
        Adj4, rowsum = get_GCN_inputs3(edges4, len(X))
        Adj4 = Adj4.to(device)
        X["Rating"] = dataset.Rating["Rating"]
        data,X_Tags_Feature2 = getPostContexts(X, dataset)
        X_Tags_Feature = Variable(data, requires_grad=False).cuda()

        # setting of GCN
        nComm = 1
        nHid1 = 50
        nHid2 = 10
        nHid3 = 10
        nHid4 = 5
        nFeat = X_Tags_Feature.shape[1]

        gcn_model = GCN_relational(nFeat, nHid1, nHid2, nHid3, nHid4, nComm).to(device)
        criterion = nn.MSELoss()
        gcn_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad,gcn_model.parameters()), lr=args.learning_rate, weight_decay=args.weight_decay)
        #gcn_model.load_state_dict(torch.load('gcn_complete2.pkl'))

        trainset = torch.utils.data.TensorDataset(torch.LongTensor(Dataset[Pairs_train]), torch.torch.LongTensor(Dataset[Pairs_train]))
        testset = torch.utils.data.TensorDataset(torch.LongTensor(Dataset[Pairs_test]), torch.torch.LongTensor(Dataset[Pairs_test]))

        train_loader = torch.utils.data.DataLoader(dataset=trainset,
                                                        batch_size=len(Pairs_train))

        test_loader = torch.utils.data.DataLoader(dataset=testset,
                                                  batch_size=len(Pairs_test),
                                                  shuffle=False)
        best_accuracy = 0
        # train and test
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
                user_gcn_embed = gcn_model(X_Tags_Feature, Adj, Adj2, Adj3, Adj4, Adj5)

                predicted = user_gcn_embed.squeeze()[X_train]>0
                train_acc+= (Y_train.numpy()==predicted.cpu().numpy()).sum()
                length+= len(Y_train)

                Y_train = Variable(Y_train, requires_grad=False).cuda()
                loss = torch.sum(torch.exp(-torch.mul(user_gcn_embed[X_train],Target[X_train])))

                loss_train+=loss.item()
                loss.backward(retain_graph=True)
                gcn_optimizer.step()

            if epoch > 0:
                #print "GCN Weight", gcn_model.gc1.weight.data
                print("Total loss", epoch, loss_train)
                currcorrect= evaluate_relational(gcn_model, test_loader, Adj, Adj2, Adj3, Adj4, Adj5, X_Tags_Feature, rowsum, X, epoch)

                if currcorrect>best_accuracy:
                    best_accuracy = currcorrect
                print "Epoch ", epoch, "'s accuracy is", currcorrect
                print "best "+str(count)+" accuracy is", best_accuracy
                train_acc = train_acc/float(length)
                print "Train Accuracy", train_acc
                print args.root
        result_list.append(best_accuracy)

except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')

print result_list
print sum(result_list)/len(result_list)

#torch.save(gcn_model.state_dict(), 'gcnmodel_'+args.root.split('/')[-1]+'_relational'+'.pt')
