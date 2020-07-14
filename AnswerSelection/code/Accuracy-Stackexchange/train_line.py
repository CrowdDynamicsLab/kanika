import torch
import os
import argparse
import numpy as np
import pandas as pd
import torch.nn as nn
from test import evaluate_line
import torch.nn.functional as F
from utils import getPostContexts
from sklearn import preprocessing
from torch.autograd import Variable
from model import FeedForward, GCN
from vectorizers import IndexVectorizer
from glove_utils import load_glove_embeddings
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from data import QADataset, SubjObjDataset, get_GCN_inputs, get_GCN_inputs2, normalize, sparse_mx_to_torch_sparse_tensor
import matplotlib.pyplot as plt
import seaborn as sns



parser = argparse.ArgumentParser(description='PyTorch Credibility Prediction Model')
parser.add_argument('--use_cuda', dest='use_cuda', default=False, action='store_true')
parser.add_argument('-lr', '--learning_rate', type=float, default=0.01,\
                    help='learning rate for FeedForward')
parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--num_epochs', type=int, default=1000)
parser.add_argument('--batch_size', type=int, default=1000)
parser.add_argument('--root', type=str, default='/home/github/UserCredibility/movie-data')
glove_path = '/Users/kanika/Documents/glove.6B/glove.6B.50d.txt'
args = parser.parse_args()
seed = 1234734614
torch.manual_seed(seed)
if args.use_cuda:
	torch.cuda.manual_seed(seed)

dataset = QADataset(dataset='StackExchange',questionFile = 'QuestionFeatures.tsv', answerFile = 'AnswerFeatures.tsv', userFile = 'UserFeatures.tsv',
                                    rootFolder= args.root)


print "Dataset read", len(dataset)

###SAMPLE THE DATASET
PosClass = dataset.trainPairs_WFeatures[dataset.trainPairs_WFeatures['Credible'] == '1']
NegClass = dataset.trainPairs_WFeatures[dataset.trainPairs_WFeatures['Credible'] == '0']
print "Positive samples",len(PosClass)
questions = dataset.trainPairs['QuestionId'].unique()

if len(PosClass) > len(NegClass):
    NegClass_Sample = NegClass
else:
    NegClass_Sample = NegClass.sample(n=len(PosClass))

#X = pd.concat([PosClass, NegClass_Sample])
X = dataset.trainPairs_WFeatures
#print X
## PreProcess the dataset
cols = X.columns.drop('QTags')
X[cols] = X[cols].apply(pd.to_numeric, errors='coerce')
X.fillna(0, inplace=True)
X["PairId"] = X["PairId"].apply(lambda x: int(x)-1)
print len(X)
#print X
#edges =dataset.getAdjList_tags()
#edges = dataset.getAdjList_allTags()
#edges = dataset.getAdjList_lineGraph()
edges = dataset.getAdjList_clique()
#edges = dataset.getAdjMatrix_Identity(len(X))
#edges = np.concatenate((edges1,edges2))
Adj, rowsum = get_GCN_inputs(edges, len(X))
#Adj2 = get_GCN_inputs(edges2, len(X))
#Adj, rowsum = normalize(Adj+Adj2)
#Adj = sparse_mx_to_torch_sparse_tensor(Adj)

#print "Adjacency Graph", Adj
#print X
X_Tags_Feature = Variable(getPostContexts(X, dataset), requires_grad=False)
#X_Tags_Feature = X_Tags_Feature[:,[0,3,4,5,6,7,8,9,10,13,14,15,16,17,18,19,26,27,28,29,30,31]]
#exit()


Pairs_train, Pairs_test, Y_train, Y_test = train_test_split(X.loc[:,'PairId'], X.loc[:,'Credible'], test_size=0.2, random_state=1234)
trainset = torch.utils.data.TensorDataset(torch.LongTensor(Pairs_train.values), torch.torch.LongTensor(Y_train.values))
testset = torch.utils.data.TensorDataset(torch.LongTensor(Pairs_test.values), torch.torch.LongTensor(Y_test.values))

#exit()

# Data loader (input pipeline)
train_loader = torch.utils.data.DataLoader(dataset=trainset,
                                                batch_size=args.batch_size)

test_loader = torch.utils.data.DataLoader(dataset=testset,
                                          batch_size=len(Pairs_test),
                                          shuffle=False)

# X_train = torch.LongTensor(X_train.values)
# X_test = (torch.LongTensor(X_test.values))
# Y_train = Variable(torch.LongTensor(Y_train.values), requires_grad=False)
# Y_test = torch.LongTensor(Y_test.values)
# print Y_train

## Neural Model
# Loss and Optimizer
nComm = 2
nHid1 = 50
nHid2 = 20
nHid3 = 20
nHid4 = 10
nFeat = X_Tags_Feature.shape[1] #len(Tags_Features)
gcn_model = GCN(nFeat, nHid1, nComm)
#gcn_model = GCN(nFeat, nHid1, nHid2, nHid3, nHid4, nComm)
criterion = nn.CrossEntropyLoss()
gcn_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad,gcn_model.parameters()), lr=args.learning_rate, weight_decay=args.weight_decay)

epoch_loss = []
epoch_accuracy = []
train_accuracy = []
epochs = []

correct = 0
try:
    for epoch in range(1, args.num_epochs + 1):

        train_acc = 0
        loss_train = 0

        for i, (X_train, Y_train) in enumerate(train_loader):
            gcn_model.train(True)
            gcn_optimizer.zero_grad()

            user_gcn_embed = gcn_model(X_Tags_Feature, Adj)
            user_gcn_embed.squeeze_()
            #print "GCN EMbeddings", user_gcn_embed
            predicted = torch.max(user_gcn_embed[X_train], 1)[1].data
            train_acc+= ( Y_train.numpy()==predicted).sum()
            if epoch == 3000:
                indexes = np.where(np.not_equal(Y_train,predicted))
                print "Wrong at ", indexes, len(indexes[0])
                np.savetxt('Trainerrors.txt', np.array(rowsum[indexes]).T, fmt = '%3d')
                #with open('Trainerrors.txt', 'w') as f:
                #    f.write(indexes)
                plt.close("all")
                sns.distplot(rowsum[indexes])
                plt.xscale('log')
                #plt.yscale('log')
                plt.savefig("figures/TrainDegreeDist.png")
                sns.distplot(rowsum[indexes],hist_kws={'log':True})
                plt.xscale('log')
                #plt.yscale('log')
                plt.savefig("figures/TrainDegreeDist_ylog.png")
                #exit()

            Y_train = Variable(Y_train, requires_grad=False)
            loss = criterion(user_gcn_embed[X_train], Y_train)
            loss_train+=loss.item()
            loss.backward()
            gcn_optimizer.step()

        if epoch > 0:
            #print "GCN Weight", gcn_model.gc1.weight.data
            print("Total loss", epoch, loss_train)
            currcorrect, degrees = evaluate_line(gcn_model, test_loader, Adj, X_Tags_Feature, rowsum)
            print "Epoch ", epoch, "'s accuracy is", currcorrect
            train_acc = train_acc.numpy()/float(len(Pairs_train))
            print "Train Accuracy", train_acc
            epochs.append(epoch)
            epoch_loss.append(loss.item())
            epoch_accuracy.append(currcorrect)
            train_accuracy.append(train_acc)

            if currcorrect > correct:
                correct = currcorrect
                torch.save(gcn_model.state_dict(), 'gcnmodel_linegraph.pt')

            if epoch == 3000 :
                np.savetxt('Testerrors.txt', np.array(degrees).T, fmt = '%3d')
                plt.close("all")
                sns.distplot(degrees)
                plt.xscale('log')
                plt.savefig("figures/TestDegreeDist.png")
                sns.distplot(degrees,hist_kws={'log':True})
                plt.xscale('log')
                plt.savefig("figures/TestDegreeDist_ylog.png")
            	#plt.show()
            	#exit()

except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')


print("Best accuracy is", correct)
if True:
    torch.save(gcn_model.state_dict(), 'random_all.pkl')
    sns.set()
    plt.figure(figsize=(12,4))
    ax1 = plt.subplot(121)
    plt.plot(epochs, epoch_loss)
    plt.ylabel("Total Loss")
    plt.xlabel("Epochs")
    ax1.set_title("GCN Loss")
    #plt.show()

    ax2 = plt.subplot(122)
    plt.plot(epochs, epoch_accuracy, 'r', label='Test')
    plt.plot(epochs, train_accuracy, 'b', label='Train')
    plt.ylabel("Train & Test accuracy")
    plt.xlabel("Epochs")
    plt.legend(frameon=True)
    ax2.set_title("GCN Accuracy")
    plt.show()
