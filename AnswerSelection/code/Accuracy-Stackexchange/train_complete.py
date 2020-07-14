import torch
import os
import argparse
import numpy as np
import pandas as pd
import torch.nn as nn
from test import evaluate_line4
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
from data import QADataset, SubjObjDataset, get_GCN_inputs, get_GCN_inputs2, normalize, sparse_mx_to_torch_sparse_tensor, makeAdj3
import matplotlib.pyplot as plt
import seaborn as sns
from layers import SparseMM
from sklearn.metrics import accuracy_score



parser = argparse.ArgumentParser(description='PyTorch Credibility Prediction Model')
parser.add_argument('--use_cuda', dest='use_cuda', default=False, action='store_true')
parser.add_argument('-lr', '--learning_rate', type=float, default=0.001,\
                    help='learning rate for FeedForward')
parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--num_epochs', type=int, default=2)
parser.add_argument('--batch_size', type=int, default=500)
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


################ clique size staff ########
clique_staff = dataset.trainPairs[["PairId","QuestionId","CommentId"]]
clique_staff["PairId"] = clique_staff["PairId"].apply(lambda x: str(int(x)-1))
clique_staff_group = clique_staff.groupby("QuestionId")
clique_size = dict()
for i,j in clique_staff_group:
    clique_size[i] = len(j)
clique_staff["Size"] = clique_staff["QuestionId"]
# key---PairId value---size
clique_staff["Size"] = clique_staff["Size"].apply(lambda x: clique_size[x])
# key---QuestionId value---size
clique_staff2 = clique_staff[["QuestionId","Size"]].drop_duplicates()
##########################################


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
edges2 = dataset.getAdjList_clique()
edges = dataset.getAdjMatrix_Identity(len(X))
#edges = np.concatenate((edges1,edges2))
Adj, rowsum = get_GCN_inputs2(edges, len(X))
Adj2, rowsum = get_GCN_inputs(edges2, len(X))
Adj3 = makeAdj3(edges2, len(X))

#Adj2 = get_GCN_inputs(edges2, len(X))
#Adj, rowsum = normalize(Adj+Adj2)
#Adj = sparse_mx_to_torch_sparse_tensor(Adj)

#print "Adjacency Graph", Adj
#print X
data,X_Tags_Feature2 = getPostContexts(X, dataset)
#data = SparseMM(Adj)(data)
X_Tags_Feature = Variable(data, requires_grad=False)

#X_Tags_Feature = X_Tags_Feature[:,[0,3,4,5,6,7,8,9,10,13,14,15,16,17,18,19,26,27,28,29,30,31]]
#exit()

Pairs_train, Pairs_test, Y_train, Y_test = train_test_split(X[X['Label']==1].loc[:,'QuestionId'].drop_duplicates(), X[X['Label']==1].loc[:,'QuestionId'].drop_duplicates(), test_size=0.2, random_state=1234)

trainset = torch.utils.data.TensorDataset(torch.LongTensor(Pairs_train), torch.torch.LongTensor(Y_train))
testset = torch.utils.data.TensorDataset(torch.LongTensor(Pairs_test), torch.torch.LongTensor(Y_test))
"""
count = 0

### change credible label
for i,j in X.groupby('QuestionId'):
    if 1 not in j['Credible'].values:
        X.loc[j['AScore'].argmax(),'Credible'] = 1
        count += 1
        print count,
    else:
        pass
"""
# Data loader (input pipeline)
train_loader = torch.utils.data.DataLoader(dataset=trainset,
                                                batch_size=args.batch_size)

test_loader = torch.utils.data.DataLoader(dataset=testset,
                                          batch_size=len(testset),
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
#gcn_model.load_state_dict(torch.load('gcn_complete2.pkl'))
criterion = nn.CrossEntropyLoss()
criterion2 = nn.MSELoss()
gcn_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad,gcn_model.parameters()), lr=args.learning_rate, weight_decay=args.weight_decay)

epoch_loss = []
epoch_accuracy = []
train_accuracy = []
epochs = []

correct = 0
try:
    for epoch in range(1, args.num_epochs + 1):

        train_acc = 0
        length = 0
        loss_train = 0

        for i, (X_train, Y_train) in enumerate(train_loader):
            X_train = torch.tensor(X[X["QuestionId"].isin(X_train)]['PairId'].values)
            Y_train = torch.tensor(X[X["QuestionId"].isin(Y_train)]['Credible'].values)
            gcn_model.train(True)
            gcn_optimizer.zero_grad()

            user_gcn_embed = gcn_model(X_Tags_Feature, Adj2)
            user_gcn_embed.squeeze_()
            #print "GCN EMbeddings", user_gcn_embed
            predicted = torch.max(user_gcn_embed[X_train], 1)[1].data
            train_acc+= (Y_train.numpy()==predicted.numpy()).sum()
            length+= len(Y_train)

            Y_train = Variable(Y_train, requires_grad=False)
            Y_train2 = Variable(torch.FloatTensor(np.ones(len(Y_train))), requires_grad=False)
            predicted = torch.max(user_gcn_embed, 1)[1].data
            loss = criterion(user_gcn_embed[X_train], Y_train)#+criterion2(SparseMM(Adj3)(torch.unsqueeze(predicted, dim=1).float()).squeeze()[X_train], Y_train2)
            loss.backward()
            gcn_optimizer.step()

        if epoch > 0:
            #print "GCN Weight", gcn_model.gc1.weight.data
            print("Total loss", epoch, loss_train)
            currcorrect, degrees = evaluate_line4(gcn_model, test_loader, Adj2, X_Tags_Feature, rowsum, X)
            print "Epoch ", epoch, "'s accuracy is", currcorrect
            train_acc = train_acc/float(length)
            print "Train Accuracy", train_acc
            print args.root
            epochs.append(epoch)
            epoch_loss.append(loss.item())
            epoch_accuracy.append(currcorrect)
            train_accuracy.append(train_acc)

        if epoch > args.num_epochs:
            # utils to capture error
            # utils to capture error
            predict_list2 = []
            y_list2= []
            predict_list3 = []
            y_list3= []
            X_list = []
            #record the QuestionId
            for i, (X_test, Y_test) in enumerate(test_loader):
                #X_test_unique = X_test.unique()
                for i, (j,k) in enumerate(zip(X_test, Y_test)):
                    X_test1 = torch.tensor(X[X["QuestionId"]==j]["PairId"].values)
                    Y_test1 = torch.tensor(X[X["QuestionId"]==j]['Credible'].values)
                    if len(X_test1)==2:
                        predicted = torch.max(user_gcn_embed[X_test1], 1)[1].data
                        predict_list2 += predicted.numpy().tolist()
                        y_list2 += Y_test1.numpy().tolist()
                    else:
                        predicted = torch.max(user_gcn_embed[X_test1], 1)[1].data
                        predict_list3 += predicted.numpy().tolist()
                        y_list3 += Y_test1.numpy().tolist()
            print "size2", accuracy_score(predict_list2,y_list2)
            print "size3", accuracy_score(predict_list3,y_list3)


            """
            # build dictionary
            # size_id: key----size, value----id list
            size_id = dict()
            # clique_real_size: key----questionId, value----# of Pairs in test
            clique_real_size = dict()
            for i,(j,k,l) in enumerate(zip(predict_list,y_list,X_list)):
                qId = clique_staff[clique_staff["PairId"]==str(l)]["QuestionId"].values.tolist()[0]
                size = clique_staff[clique_staff["PairId"]==str(l)]["Size"].values.tolist()[0]
                #print size,size[0]
                try:
                    clique_real_size[qId] += 1
                except:
                    clique_real_size[qId] = 1
                try:
                    if qId not in size_id[size]:
                        size_id[size].append(qId)
                except:
                    size_id[size] = [qId]
                if j!=k:
                    error_list.append(clique_staff[clique_staff["PairId"]==str(l)]["QuestionId"].values.tolist()[0])
                if j==1 and k==0:
                    error_list_0to1.append(clique_staff[clique_staff["PairId"]==str(l)]["QuestionId"].values.tolist()[0])
                if j==0 and k==1:
                    error_list_1to0.append(clique_staff[clique_staff["PairId"]==str(l)]["QuestionId"].values.tolist()[0])
                    #error_list.append(str(X_list[i]))
            print len(error_list)
            print len(error_list_0to1)
            print len(error_list_1to0)

            # error_clique: key----questionId, value----# of wrong Pairs in test
            error_clique = {}.fromkeys(clique_real_size.keys())
            for i in error_clique.keys():
                error_clique[i] = 0
            # error_clique_0_to_1: key----questionId, value----# of FP Pairs in test
            error_clique_0_to_1 = {}.fromkeys(clique_real_size.keys())
            for i in error_clique_0_to_1.keys():
                error_clique_0_to_1[i] = 0
            # error_clique_1_to_0: key----questionId, value----# of FN Pairs in test
            error_clique_1_to_0 = {}.fromkeys(clique_real_size.keys())
            for i in error_clique_1_to_0.keys():
                error_clique_1_to_0[i] = 0
            print len(X_list)
            for i in error_list:
                error_clique[i] += 1
            for i in error_list_0to1:
                error_clique_0_to_1[i] += 1
            for i in error_list_1to0:
                error_clique_1_to_0[i] += 1
            print len(error_clique)
            print len(error_clique_0_to_1)
            print len(error_clique_1_to_0)
            print sum(error_clique_0_to_1.values())
            print sum(error_clique_1_to_0.values())
            print len(error_list)


            # p(e)
            p_e = sum(error_clique.values())/float(len(X_list))
            print "p(e)",p_e
            p_e_k = dict()
            p_k = dict()
            n_k = dict()
            final_dict_0_to_1 = dict()
            final_dict_1_to_0 = dict()
            for i in size_id.keys():
                p_e_k[i] = sum([error_clique[j] for j in size_id[i]])/float(sum([clique_real_size[j] for j in size_id[i]]))
                final_dict_0_to_1[i] = sum([error_clique_0_to_1[j] for j in size_id[i]])
                final_dict_1_to_0[i] = sum([error_clique_1_to_0[j] for j in size_id[i]])
                p_k[i] = sum([clique_real_size[j] for j in size_id[i]])/float(len(X_list))
                n_k[i] = sum([clique_real_size[j] for j in size_id[i]])
            print p_e_k
            print p_k
            print n_k

            # x,y,z
            x = []
            y = []
            z = []
            error0to1 = []
            error1to0 = []
            m2 = []
            pic2 = []
            for i in p_k.keys():
                x.append(i)
                y.append(n_k[i])
                z.append(p_e_k[i])
                error0to1.append(final_dict_0_to_1[i])
                error1to0.append(final_dict_1_to_0[i])
                pic2.append((p_e-p_e_k[i])*p_k[i]/p_e)
            data = pd.DataFrame()
            data["x"] = x;
            data["y"] = y;
            data["z"] = z;
            data["error0to1"] = error0to1
            data["error1to0"] = error1to0
            data["proportion"] = pic2
            data.to_csv("../csv_data/test_GCN_FF34_"+args.root.split('/')[-2]+".csv")
            """
            """
            sns.set()
            plt.figure(figsize=(15,4))
            ax1 = plt.subplot(131)
            my_x_ticks = np.arange(0, 20, 1)
            ax1.set_xticks(my_x_ticks)
            plt.bar(x,pic2)
            ax1.set_ylabel(r"$\frac{(p(e) - p(e|k)) p(k)}{p(e)}$")
            ax1.set_xlabel("Size of clique k")
            ax1.set_title("Clique Error Distribution 1")

            ax2 = plt.subplot(132)
            #plt.xlim((0, 20))
            my_x_ticks = np.arange(0, 20, 1)
            ax2.set_xticks(my_x_ticks)
            line = plt.scatter(x=x, y=z, c="r",s=[np.log(i+1)*20 for i in y])
            ax2.set_ylabel("p(e|k)")
            ax2.set_xlabel("Size of clique k")
            #ax.set_zlabel("p(e)-p(e|k)")
            ax2.set_title("Clique Error Distribution 2")
            plt.legend(("Maker size: log(S(k)+1)",))
            plt.legend(frameon=True)
            #plt.savefig('/home/chaoqi/Desktop/7.23/Identity_clique1v2.png')

            ax3 = plt.subplot(133)
            width = 0.4
            #plt.xlim((0, 20))
            #ax3.set_yticks(my_y_ticks)
            plt.bar(data["x"]-width/2, data["error0to1"],label="FP", width = width)
            plt.bar(data["x"]+width/2, data["error1to0"],label="FN", width = width)
            ax3.set_ylabel("Number")
            ax3.set_xlabel("Size of clique k")
            ax3.set_xticks(my_x_ticks)
            #ax.set_zlabel("p(e)-p(e|k)")
            ax3.set_title("Clique Error Distribution 3")
            plt.legend()
            plt.legend(frameon=True)
            plt.show()
            """

except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')


print("Best accuracy is", correct)
if True:
    #torch.save(gcn_model.state_dict(), 'complete_bigger_2.pkl')
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
