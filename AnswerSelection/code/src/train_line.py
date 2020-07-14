import torch
import os
import argparse
#import seaborn as sns
#import matplotlib
#matplotlib.use('Agg')
#import matplotlib.pyplot as plt
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
from data import QADataset, SubjObjDataset, get_GCN_inputs



parser = argparse.ArgumentParser(description='PyTorch Credibility Prediction Model')
parser.add_argument('--use_cuda', dest='use_cuda', default=False, action='store_true')
parser.add_argument('-lr', '--learning_rate', type=float, default=0.001,\
                    help='learning rate for FeedForward')
parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--num_epochs', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=1000)
parser.add_argument('--root', type=str, default='/home/knarang2/StackExchange')
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

# NegClass_List = []
# for id in questions:
#     Sample = NegClass[NegClass['QuestionId'] == id]
#     #print Sample, id
#     if not Sample.empty:
#         n_sample = 1
#         if len(Sample) < n_sample:
#             dict1 = Sample.to_dict('list')
#         else:
#             dict1 = Sample.sample(n=n_sample).to_dict('list')
#         #print dict1
#         for n in range(len(dict1['PairId'])):
#             dict2 = {}
#             for key in dict1:
#                 dict2[key] = str(dict1[key][n])
#             NegClass_List.append(dict2)
#         #print dict1
#         #exit()
#         #NegClass_List.append(dict1)
#         #NegClass_Sample.append(Sample.sample(n=1))

#NegClass_Sample = pd.DataFrame(NegClass_List, columns=PosClass.columns)
if len(PosClass) > len(NegClass):
    NegClass_Sample = NegClass
else:
    NegClass_Sample = NegClass.sample(n=len(PosClass))

#X = pd.concat([PosClass, NegClass_Sample])
X = dataset.trainPairs_WFeatures

## PreProcess the dataset
cols = X.columns.drop('QTags')
X[cols] = X[cols].apply(pd.to_numeric, errors='coerce')
X.fillna(0, inplace=True)
X["PairId"] = X["PairId"].apply(lambda x: int(x)-1)
#print X

edges = dataset.getAdjList_clique()
#edges = dataset.getAdjList_lineGraph()
#edges = dataset.getAdjList_tags()
#edges = dataset.getAdjList_allTags()

print "Length of edges returned", len(edges)

Adj, rowsum = get_GCN_inputs(edges, len(X))
print "Adjacency Graph", Adj
#print X

X_Tags_Feature = Variable(getPostContexts(X, dataset), requires_grad=False)
print len(X_Tags_Feature[0])



Pairs_train, Pairs_test, Y_train, Y_test = train_test_split(X.loc[:,'PairId'], X.loc[:,'Credible'], test_size=0.2, random_state=1234)
trainset = torch.utils.data.TensorDataset(torch.LongTensor(Pairs_train.values), torch.torch.LongTensor(Y_train.values))
testset = torch.utils.data.TensorDataset(torch.LongTensor(Pairs_test.values), torch.torch.LongTensor(Y_test.values))


# Data loader (input pipeline)
train_loader = torch.utils.data.DataLoader(dataset=trainset,
                                                batch_size=args.batch_size)

test_loader = torch.utils.data.DataLoader(dataset=testset,
                                          batch_size=len(Pairs_test),
                                          shuffle=False)


## Neural Model
# Loss and Optimizer
nComm = 2
nHid = 9
nFeat = X_Tags_Feature.shape[1]
print "FEATURE COUNT ", nFeat
gcn_model = GCN(nFeat, nHid, nComm)
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
            if epoch == 200:
                indexes = np.where(np.not_equal(Y_train,predicted))
                print "Wrong at ", indexes, len(indexes[0])
                with open('Trainerrors_tags.txt', 'w') as fout:
                    for index in indexes[0]:
                        fout.write(str(index)+"\t"+str(rowsum[index]) + "\n")

                #np.savetxt('Trainerrors.txt', np.array(indexes), np.array(rowsum[indexes]).T, fmt = '%3d')
                #with open('Trainerrors.txt', 'w') as f:
                #    f.write(indexes)
                #plt.close("all")
                #sns.distplot(rowsum[indexes])
                #plt.xscale('log')
                #plt.yscale('log')
                #plt.savefig("figures/TrainDegreeDist.png")
                #sns.distplot(rowsum[indexes],hist_kws={'log':True})
                #plt.xscale('log')
                #plt.yscale('log')
                #plt.savefig("figures/TrainDegreeDist_ylog.png")
                #exit()

            Y_train = Variable(Y_train, requires_grad=False)
            loss = criterion(user_gcn_embed[X_train], Y_train)
            loss_train+=loss.data[0]
            loss.backward()
            gcn_optimizer.step()

        if epoch > 0:
            #print "GCN Weight", gcn_model.gc1.weight.data
            print("Total loss", epoch, loss_train)
            currcorrect, indexes = evaluate_line(gcn_model, test_loader, Adj, X_Tags_Feature, rowsum)
            print "Epoch ", epoch, "'s accuracy is", currcorrect
            train_acc = train_acc/float(len(Pairs_train))
            print "Train Accuracy", train_acc
            epochs.append(epoch)
            epoch_loss.append(loss.data[0])
            epoch_accuracy.append(currcorrect)
            train_accuracy.append(train_acc)
            if currcorrect > correct:
                correct = currcorrect
                torch.save(gcn_model.state_dict(), 'gcnmodel_linegraph.pt')

            if epoch == 200 :

                with open('Testerrors_asker.txt', 'w') as fout:
                    for index in indexes[0]:
                        fout.write(str(index)+"\t"+str(rowsum[index]) + "\n")
                #np.savetxt('Testerrors.txt', np.array(degrees).T, fmt = '%3d')
                #plt.close("all")
                #sns.distplot(degrees)
                #plt.xscale('log')
                #plt.savefig("figures/TestDegreeDist.png")
                #sns.distplot(degrees,hist_kws={'log':True})
                #plt.xscale('log')
                #plt.savefig("figures/TestDegreeDist_ylog.png")
            	#plt.show()
            	#exit()

except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')


print("Best accuracy is", correct)
if True:
    ext= 'cred_line'
    plt.close("all")
    plt.plot(epochs, epoch_loss)
    plt.ylabel("Total Loss")
    plt.xlabel("Epochs")
    plt.title("GCN model")
    plt.savefig('figures/epoch_loss_'+ext+'.png')
    #plt.show()
    plt.clf()

    plt.plot(epochs, epoch_accuracy, 'r', label='Test')
    plt.plot(epochs, train_accuracy, 'b', label='Train')
    plt.ylabel("Train & Test accuracy")
    plt.xlabel("Epochs")
    plt.legend()
    plt.title("GCN model")
    plt.savefig('figures/epoch_acc_'+ext+'.png')
    #plt.show()
    plt.clf()
