import torch
import os
import argparse
import numpy as np
import pandas as pd
import torch.nn as nn
from test import evaluate_basic
import torch.nn.functional as F
from utils import getPostContexts, getPostContentEmbed
from sklearn import preprocessing
from torch.autograd import Variable
from model import GCN_basic
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from data import QADataset, SubjObjDataset, get_GCN_inputs, get_GCN_inputs2, get_GCN_inputs3, normalize, sparse_mx_to_torch_sparse_tensor
from sklearn.metrics import accuracy_score
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
#torch.set_num_threads(12)
print("Device", device)
# parser
parser = argparse.ArgumentParser(description='PyTorch Credibility Prediction Model')
parser.add_argument('--use_cuda', dest='use_cuda', default=False, action='store_true')
parser.add_argument('--use_content', dest='use_content', default=False, action='store_true')
parser.add_argument('-lr', '--learning_rate', type=float, default=0.001,
                    help='learning rate for ensemble')
parser.add_argument('--weight_decay', type=float, default=5e-3)
parser.add_argument('--num_epochs', type=int, default=300)
parser.add_argument('--batch_size', type=int, default=500)
parser.add_argument('--root', type=str, default='/scratch/uc_data/data')
parser.add_argument('--dataset', type=str, default='ai')

args = parser.parse_args()
seed = 1234734614
torch.manual_seed(seed)
if args.use_cuda:
	torch.cuda.manual_seed(seed)

if args.use_content:
    print("Using content embeddings")
# dataset
dataset = QADataset(dataset=args.dataset,questionFile = 'QuestionFeatures.tsv', answerFile = 'AnswerFeatures.tsv', userFile = 'UserFeatures.tsv',
                                    rootFolder= os.path.join(args.root, args.dataset))
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
"""
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
# key---QuestionId value---size300
clique_staff2 = clique_staff[["QuestionId","Size"]].drop_duplicates()
key = clique_staff["QuestionId"].keys()
#print(clique_staff)
##########################################
#print("Length of dict", clique_size)
#print(clique_staff)
#print(clique_staff["Size"][5])
"""
## PreProcess the dataset
cols = X.columns.drop('QTags')
X[cols] = X[cols].apply(pd.to_numeric, errors='coerce')
X.fillna(0, inplace=True)
X["PairId"] = X["PairId"].apply(lambda x: int(x)-1)
#data,X_Tags_Feature2 = getPostContexts(X, dataset)
#X_Tags_Feature = Variable(data, requires_grad=False)
Target =  np.array([[2*i-1] for i in X['Credible'].values.tolist()])
Target = Variable(torch.FloatTensor(Target),requires_grad=False).to(device)

# unique, counts = np.unique(Target, return_counts=True)
# di = dict(zip(unique, counts))
# print("Total length, ", len(Target))
# print("Positive Sample, ", di[1])
# print("Negative Sample, ",di[-1])
# print("Positive/Negative, ", float(float(di[-1])/(di[-1]+di[1])))

QuestionId = X.loc[:,'QuestionId'].drop_duplicates()
QuestionId = QuestionId[QuestionId>0]
#exit()
#print(Target.shape)
# get data_loader
from sklearn.model_selection import KFold
kf = KFold(n_splits=5, shuffle=True, random_state=1234734614)
Dataset = np.array(QuestionId[QuestionId!=0])



#Pairs_train, Pairs_test, Y_train, Y_test = train_test_split(QuestionId[QuestionId!=0], QuestionId[QuestionId!=0], test_size=0.2, random_state=1234)


# get adj
edges = dataset.getAdjList_clique()
#edges2 = dataset.getAdjList_Similarity1()
edges3 = dataset.getAdjList_Similarity2()
#edges4 = dataset.getAdjList_Similarity3(Pairs_train.values.tolist())
edges5 = dataset.getAdjMatrix_Identity(len(X))
edges6 = dataset.getAdjList_ContentSimilarity()
Adj, rowsum = get_GCN_inputs(edges, len(X))
#Adj2, rowsum = get_GCN_inputs3(edges2, len(X))
Adj3, rowsum = get_GCN_inputs3(edges3, len(X))
#Adj4, rowsum = get_GCN_inputs3(edges4, len(X))
Adj5, rowsum = get_GCN_inputs2(edges5, len(X))
Adj6, rowsum = get_GCN_inputs3(edges6, len(X))
Adj = Adj.to(device)
#Adj2 = Adj2.to(device)
Adj2 = None
Adj3 = Adj3.to(device)
Adj5 = Adj5.to(device)
Adj6 = Adj6.to(device)

count1 = 0
acc = []
"""
Pairs_train, Pairs_test, Y_train, Y_test = train_test_split(QuestionId[QuestionId!=0], QuestionId[QuestionId!=0], test_size=0.2, random_state=1234)
trainset = torch.utils.data.TensorDataset(torch.LongTensor(list(Pairs_train)), torch.LongTensor(list(Pairs_train)))
testset = torch.utils.data.TensorDataset(torch.LongTensor(list(Pairs_test)), torch.LongTensor(list(Pairs_test)))

train_loader = torch.utils.data.DataLoader(dataset=trainset,
                                                batch_size=len(Pairs_train))

test_loader = torch.utils.data.DataLoader(dataset=testset,
                                          batch_size=len(Pairs_test),
                                          shuffle=False)
print(len(test_loader))
#count1 += 1
#if count1==5:
#    print "accuracy for random forest is: ",acc
#    print "Average accuracy: ", sum(acc)/4
#    exit()
data,X_Tags_Feature2 = getPostContexts(X, dataset)
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
acc.append(accuracy)

print "accuracy for random forest is: ",acc
#print "Average accuracy: ", sum(acc)/5
exit()
"""

result_list = list()
count = 0
try:
    for Pairs_train, Pairs_test in kf.split(Dataset):
        #print(Dataset[Pairs_test])
        #(len(Pairs_train))
        count += 1
        if count==6:
            print result_list
            print sum(result_list)/len(result_list)
            """
            with open ("/home/jtwang/GCN/UserCredibility/Accuracy-Stackexchange/Basic_contrasive/result/" + args.root.split('/')[-1] +".txt", "w+") as f:
                f.write(''.join(str(e) for e in result_list)+"\n")
                f.write(str(sum(result_list)/len(result_list)))
            f.close()
            exit()
            torch.save(gcn_model.state_dict(), '/home/jtwang/GCN/UserCredibility/Accuracy-Stackexchange/Basic_contrasive/gcnmodel_'+args.root.split('/')[-1]+'_basic'+'.pt')
            """
            exit()

        edges4 = dataset.getAdjList_Similarity3(Pairs_train.tolist())
        Adj4, rowsum = get_GCN_inputs3(edges4, len(X))
        Adj4 = Adj4.to(device)
        X["Rating"] = dataset.Rating["Rating"]

        if args.use_content:
            data,X_Tags_Feature2 = getPostContentEmbed(X, args.dataset)
            data2,X_Tags_Feature2 = getPostContexts(X, dataset)
            data = torch.cat([data, data2], 1)
        else:
            data,X_Tags_Feature2 = getPostContexts(X, dataset)

        X_Tags_Feature = Variable(data, requires_grad=False).to(device)

        # setting of GCN
        nComm = 1
        nHid1 = 50
        if args.use_content:
            nHid2 = 50
        else:
            nHid2 = 10
        nHid3 = 10
        nHid4 = 5
        nFeat = X_Tags_Feature.shape[1]
        gcn_model = GCN_basic(nFeat, nHid1, nHid2, nHid3, nHid4, nComm,0).to(device)
        #gcn_model.load_state_dict(torch.load('gcn_complete2.pkl'))
        criterion = nn.CrossEntropyLoss()
        gcn_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad,gcn_model.parameters()), lr=args.learning_rate, weight_decay=args.weight_decay)

        trainset = torch.utils.data.TensorDataset(torch.LongTensor(Dataset[Pairs_train]).to(device), torch.torch.LongTensor(Dataset[Pairs_train]).to(device))
        testset = torch.utils.data.TensorDataset(torch.LongTensor(Dataset[Pairs_test]).to(device), torch.torch.LongTensor(Dataset[Pairs_test]).to(device))
        #print(Dataset[Pairs_test])
        train_loader = torch.utils.data.DataLoader(dataset=trainset,
                                                        batch_size=len(Pairs_train))

        test_loader = torch.utils.data.DataLoader(dataset=testset,
                                                  batch_size=len(Pairs_test),
                                                  shuffle=False)

        best_accuracy = 0
        # train and test
        final = []
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
                X_train = torch.tensor(X[X["QuestionId"].isin(X_train)]['PairId'].values).to(device)
                #print(X_train)
                Y_train = torch.tensor(X[X["QuestionId"].isin(Y_train)]['Credible'].values).to(device)
                gcn_model.train(True)
                gcn_optimizer.zero_grad()
                user_gcn_embed = gcn_model(X_Tags_Feature, Adj, Adj2, Adj3, Adj4, Adj5, Adj6)
                #print(user_gcn_embed.shape)
                predicted = user_gcn_embed.squeeze()[X_train]>0
                train_acc+= (Y_train.cpu().numpy()==predicted.cpu().numpy()).sum()
                length+= len(Y_train)

                Y_train = Variable(Y_train, requires_grad=False).to(device)
                loss = torch.sum(torch.exp(-torch.mul(user_gcn_embed[X_train],Target[X_train])))

                loss_train+=loss.item()
                loss.backward(retain_graph=True)
                gcn_optimizer.step()

            if epoch > 0:
                #print "GCN Weight", gcn_model.gc1.weight.data
                print("Total loss", epoch, loss_train)
                currcorrect = evaluate_basic(gcn_model, test_loader, Adj, Adj2, Adj3, Adj4, Adj5, Adj6, X_Tags_Feature, rowsum, X)
                #print(len(user_gcn_embed))
                if currcorrect>best_accuracy:
                    best_accuracy = currcorrect
                print "Epoch ", epoch, "'s accuracy is", currcorrect
                print "best accuracy is", best_accuracy
                train_acc = train_acc/float(length)
                print "Train Accuracy", train_acc
                print args.root
                """
            if epoch == args.num_epochs:
                user = gcn_model(X_Tags_Feature, Adj, Adj2, Adj3, Adj4, Adj5,epoch)
                pre = user.squeeze()>0
                #print(Target.squeeze().shape)
                ##print(Target.unique)
                #print(pre.shape)
                #print(pre.unique)
                #print(sum(pre))
                Target = Target>0
                #print(np.unique(Target.cpu().numpy()))
                #print(sum(Target))
                #print(accuracy_score(Target.squeeze().cpu().numpy(),pre.cpu().numpy()))
                #print((Target.squeeze().cpu().numpy()==pre.cpu().numpy()))
                resultblah = (Target.squeeze().cpu().numpy()==pre.cpu().numpy()).astype(int)
                #print("Number of correct = ", sum(resultblah))
                #print(user_gcn_embed)
                clid = {2:[],3:[],4:[],5:[],6:[],7:[],8:[],9:[],10:[]}
                #3print(res)
                f = open("/home/junting/Downloads/GCN/UserCredibility/Accuracy-Stackexchange/clique/basic_"+args.root.split('/')[-1] +".txt", "w+")
                for i in range(0,len(key)):
                    blah = key[i]
                    si = clique_staff["Size"][blah]
                    #print(si)
                    if si >1 and si < 11:
                        clid[si].append(resultblah[i])
                for k in clid.keys():
                    print(len(clid[k]))
                    f.write("Clique size "+ str(k)+":"+"\n")
                    f.write("Number of elements in clique: "+ str(len(clid[k]))+"\n")
                    print("Accuracy for clique size "+str(k)+":  ", float(sum(clid[k]))/len(clid[k]))
                    f.write(str(float(sum(clid[k]))/len(clid[k]))+"\n""\n")
                f.close()
                """
        result_list.append(best_accuracy)


except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')

print result_list
print sum(result_list)/len(result_list)

#torch.save(gcn_model.state_dict(), 'gcnmodel_'+args.root.split('/')[-1]+'_basic'+'.pt')
