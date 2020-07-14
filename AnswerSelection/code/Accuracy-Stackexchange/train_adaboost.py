import torch
import os
import argparse
import numpy as np
import pandas as pd
import torch.nn as nn
from test import evaluate_adaboost
import torch.nn.functional as F
from utils import getPostContexts, getPostContentEmbed, getPostUserReliability
from sklearn import preprocessing
from torch.autograd import Variable
from model import GCN_adaboost
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from data import QADataset, SubjObjDataset, get_GCN_inputs, get_GCN_inputs2, get_GCN_inputs3, normalize, sparse_mx_to_torch_sparse_tensor
from sklearn.metrics import accuracy_score
import random
#torch.set_num_threads(12)

# parser
parser = argparse.ArgumentParser(description='PyTorch Credibility Prediction Model')
parser.add_argument('--use_cuda', dest='use_cuda', default=False, action='store_true')
parser.add_argument('--use_rel', dest='use_rel', default=False, action='store_true')
parser.add_argument('--use_content', dest='use_content', default=False, action='store_true')
parser.add_argument('-lr', '--learning_rate', type=float, default=0.001,
                    help='learning rate for ensemble')
parser.add_argument('--weight_decay', type=float, default=5e-3)
parser.add_argument('--num_epochs', type=int, default=500)
parser.add_argument('--batch_size', type=int, default=400)
parser.add_argument('--root', type=str, default='/scratch/uc_data/data')
parser.add_argument('--dataset', type=str, default='ai')

args = parser.parse_args()
seed = 1234734614
torch.manual_seed(seed)
if args.use_cuda:
	torch.cuda.manual_seed(seed)

if args.use_content:
    print("Using content embeddings")

device = torch.device("cuda:0" if args.use_cuda else "cpu")
print device
# dataset
dataset = QADataset(dataset=args.dataset,questionFile = 'QuestionFeatures.tsv', answerFile = 'AnswerFeatures.tsv', userFile = 'UserFeatures.tsv',
                                    rootFolder= os.path.join(args.root, args.dataset))
print "Dataset read", len(dataset), args.dataset

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
print(X.shape)
## PreProcess the dataset
"""
cols = X.columns.drop('QTags')
X[cols] = X[cols].apply(pd.to_numeric, errors='coerce')
X.fillna(0, inplace=True)
X["PairId"] = X["PairId"].apply(lambda x: int(x)-1)
Target =  np.array([[2*i-1] for i in X['Credible'].values.tolist()])
Target = Variable(torch.FloatTensor(Target),requires_grad=False).to(device)
#Target[Target<0]=0
QuestionId = X.loc[:,'QuestionId'].drop_duplicates()
QuestionId = QuestionId[QuestionId>0]

# get data_loader
from sklearn.model_selection import KFold
#kf = KFold(n_splits=5, shuffle=True, random_state=1234734614)
kf = KFold(n_splits=5, shuffle=True)
Dataset = np.array(QuestionId[QuestionId!=0])
# get adj
edges = dataset.getAdjList_clique()
#edges2 = dataset.getAdjList_Similarity1()
edges3 = dataset.getAdjList_Similarity2()
#edges4 = dataset.getAdjList_Similarity3(Pairs_train.values.tolist())
edges5 = dataset.getAdjMatrix_Identity(len(X))
Adj, rowsum = get_GCN_inputs(edges, len(X))
#Adj2, rowsum = get_GCN_inputs3(edges2, len(X))
Adj3, rowsum = get_GCN_inputs3(edges3, len(X))
#Adj4, rowsum = get_GCN_inputs3(edges4, len(X))
Adj5, rowsum = get_GCN_inputs2(edges5, len(X))

Adj = Adj.to(device)
#Adj2 = Adj2.to(device)
Adj2 = None
Adj3 = Adj3.to(device)
Adj5 = Adj5.to(device)
#print(Adj.shape)
#X = X.ix[selection]
#Target = torch.index_select(Target,0,tensor_select)
#Adj = sparse_mx_to_torch_sparse_tensor(Adj.tocsr()[selection,:]).to(device)
#Adj2 = sparse_mx_to_torch_sparse_tensor(Adj2.tocsr()[selection,:]).to(device)
#Adj3 = sparse_mx_to_torch_sparse_tensor(Adj3.tocsr()[selection,:]).to(device)
#Adj5 = sparse_mx_to_torch_sparse_tensor(Adj5.tocsr()[selection,:]).to(device)
#a,b = kf.split(Dataset)
#print(a.shape)
#Dataset = np.take(Dataset,selection)
# setting of GCN
nComm = 2
nHid1 = 50
nHid2 = 30
nHid3 = 20
nHid4 = 10
#gcn_model.load_state_dict(torch.load('gcn_complete2.pkl'))

def rampup(epoch, scaled_unsup_weight_max, exp=5.0, rampup_length=80):
    if epoch < rampup_length:
        p = max(0.0, float(epoch)) / float(rampup_length)
        p = 1.0 - p
        return math.exp(-p * p * exp) * scaled_unsup_weight_max
    else:
        return 1.0 * scaled_unsup_weight_max


def get_scaled_unsup_weight_max(num_labels=2, X_train_shape=13, unsup_weight_max=100.0):
    return unsup_weight_max * 1.0 * num_labels / X_train_shape

lambda1, lambda2 = 0.5, 0.01
result_list = list()
count = 0
#f = open("/home/junting/Downloads/GCN/UserCredibility/Accuracy-Stackexchange/Sparsity/sparse_result_1"+ args.root.split('/')[-1]+".txt" ,"w+")
# train and test
ARR = [0.8,0.4,0.2,0.1,0.05,0.025,0.01,0.005,0.0025,0.001]
count1 = 0
try:
    #for i in :
    for Pairs_train, Pairs_test in kf.split(Dataset):
        count += 1
        if count==6:
            print result_list
            print sum(result_list)/len(result_list)
            #torch.save(gcn_model.state_dict(), '/home/junting/Downloads/GCN/UserCredibility/Accuracy-Stackexchange/Ablation/neigh_'+args.root.split('/')[-1]+'_www'+'.pt')
            #with open ("/home/junting/Downloads/GCN/UserCredibility/Accuracy-Stackexchange/Ablation/R_" + args.root.split('/')[-1] +".txt", "w+") as f:
            with open ("/home/knarang2/induced_results/R_" + args.root.split('/')[-1] +".txt", "w+") as f:
                f.write(''.join(str(e) for e in result_list)+"\n")
                f.write(str(sum(result_list)/len(result_list)))
            f.close()

            exit() # Running for one epoch
            #torch.save(gcn_model.state_dict(), '/home/junting/Downloads/GCN/UserCredibility/Accuracy-Stackexchange/Adaboost/gcnmodel_'+args.root.split('/')[-1]+str(args.learning_rate)+'_adaboost'+'.pt')


        edges4 = dataset.getAdjList_Similarity3(Pairs_train.tolist())
        Adj4, rowsum = get_GCN_inputs3(edges4, len(X))
        X["Rating"] = dataset.Rating["Rating"]

        if args.use_content:
            data,X_Tags_Feature2 = getPostContentEmbed(X, args.dataset)
            data2,X_Tags_Feature2 = getPostContexts(X, dataset)
            data = torch.cat([data, data2], 1)
        elif args.use_rel:
            data,X_Tags_Feature2 = getPostUserReliability(X, args.dataset)
            data2,X_Tags_Feature2 = getPostContexts(X, dataset)
            data = torch.cat([data, data2], 1)
        else:
            data,X_Tags_Feature2 = getPostContexts(X, dataset)

        X_Tags_Feature = Variable(data, requires_grad=False).to(device)
        Adj4 = Adj4.to(device)
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

        gcn_model = GCN_adaboost(nFeat, nHid1, nHid2, nHid3, nHid4, nComm).to(device)
        criterion = nn.MSELoss()
        #criterion =  nn.CrossEntropyLoss()
        gcn_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad,gcn_model.parameters()), lr=args.learning_rate, weight_decay=args.weight_decay)
        #gcn_model.load_state_dict(torch.load('gcn_complete2.pkl'))
        #scheduler = torch.optim.lr_scheduler.StepLR(gcn_optimizer, step_size=50, gamma=0.01)
        trainset = torch.utils.data.TensorDataset(torch.LongTensor(Dataset[Pairs_train]).to(device), torch.torch.LongTensor(Dataset[Pairs_train]).to(device))
        testset = torch.utils.data.TensorDataset(torch.LongTensor(Dataset[Pairs_test]).to(device), torch.torch.LongTensor(Dataset[Pairs_test]).to(device))

        train_loader = torch.utils.data.DataLoader(dataset=trainset,
                                                        batch_size=len(Pairs_train))

        test_loader = torch.utils.data.DataLoader(dataset=testset,
                                                  batch_size=len(Pairs_test),

                                                  shuffle=False)
        best_accuracy = 0
        # train and test
        for epoch in range(1, args.num_epochs + 1):
            #scheduler.step()
            train_acc = 0
            train_acc2 = 0
            train_acc3 = 0
            train_acc4 = 0
            train_acc5 = 0
            train_acc6 = 0
            length = 0
            loss_train = 0

            decay = float(np.exp(-epoch/20.0))

            for i, (X_train, Y_train) in enumerate(train_loader):
                X_train = torch.tensor(X[X["QuestionId"].isin(X_train)]['PairId'].values).to(device)
                Y_train = torch.tensor(X[X["QuestionId"].isin(Y_train)]['Credible'].values).to(device)
                gcn_model.train(True)
                gcn_optimizer.zero_grad()
                sim1, sim2, part2_dense = gcn_model(X_Tags_Feature, Adj, Adj2, Adj3, Adj4, Adj5, Target, X_train)

                length+= len(Y_train)

                #predicted = torch.max(part2_dense[X_train], 1)[1].data
                predicted = part2_dense.squeeze()[X_train]>0
                train_acc+= (Y_train.cpu().numpy()==predicted.cpu().numpy()).sum()

                Y_train = Variable(Y_train, requires_grad=False).to(device)

                all_params_gc1 = torch.cat([x.view(-1) for x in gcn_model.gc1.parameters()])
                all_params_gc2 = torch.cat([x.view(-1) for x in gcn_model.gc2.parameters()])
                all_params_gc3 = torch.cat([x.view(-1) for x in gcn_model.gc3.parameters()])
                all_params_gc4 = torch.cat([x.view(-1) for x in gcn_model.gc4.parameters()])
                all_params_gc5 = torch.cat([x.view(-1) for x in gcn_model.gc5.parameters()])
                all_params_gc6 = torch.cat([x.view(-1) for x in gcn_model.gc6.parameters()])
                all_params_gc10 = torch.cat([x.view(-1) for x in gcn_model.gc10.parameters()])
                all_params_gc11 = torch.cat([x.view(-1) for x in gcn_model.gc11.parameters()])
                all_params_gc12 = torch.cat([x.view(-1) for x in gcn_model.gc12.parameters()])
                all_params_dense1 = torch.cat([x.view(-1) for x in gcn_model.dense1.parameters()])
                all_params_dense2 = torch.cat([x.view(-1) for x in gcn_model.dense2.parameters()])
                all_params_dense3 = torch.cat([x.view(-1) for x in gcn_model.dense3.parameters()])
                all_params_dense4 = torch.cat([x.view(-1) for x in gcn_model.dense4.parameters()])
                all_params_densesim = torch.cat([x.view(-1) for x in gcn_model.simdense.parameters()])
                all_params = torch.cat((all_params_gc1,all_params_gc2,all_params_gc3,all_params_gc4,all_params_gc5,all_params_gc6,all_params_gc10,all_params_gc11,all_params_gc12,all_params_dense1,all_params_dense2,all_params_dense3,all_params_dense4,all_params_densesim))

                if epoch%2==1:
                    loss =torch.add(torch.sum(torch.exp(-torch.mul(sim1[X_train],Target[X_train]))),torch.sum(torch.exp(-torch.mul(sim2[X_train],Target[X_train]))))
                    sim2 = Variable(sim2, requires_grad=False)
                    loss += criterion(sim1[X_train],sim2[X_train])+lambda1 * torch.norm(torch.cat((all_params_gc4,all_params_gc5,all_params_gc6,all_params_densesim)), 1)+lambda2 * torch.norm(torch.cat((all_params_gc4,all_params_gc5,all_params_gc6,all_params_densesim)), 2)
                    loss = torch.mul(loss,decay)+torch.sum(torch.exp(-torch.mul(part2_dense[X_train],Target[X_train])))+lambda1 * torch.norm(all_params, 1)+lambda2 * torch.norm(all_params, 2)
                else:
                    loss = torch.sum(torch.exp(-torch.mul(part2_dense[X_train],Target[X_train])))+lambda1 * torch.norm(all_params, 1)+lambda2 * torch.norm(all_params, 2)

                #print(Y_train.unique())
                #print(part2_dense[X_train].shape)
                #loss = criterion(part2_dense[X_train], Y_train)
                loss_train+=loss.item()
                loss.backward(retain_graph=True)
                #loss.backward(retain_graph=True)
                gcn_optimizer.step()

            if epoch > 0:
                #print "GCN Weight", gcn_model.gc1.weight.data
                print("Total loss", epoch, loss_train)
                currcorrect= evaluate_adaboost(gcn_model, test_loader, Adj, Adj2, Adj3, Adj4, Adj5, X_Tags_Feature, rowsum, X, epoch, Target)

                if currcorrect>best_accuracy:
                    best_accuracy = currcorrect
                #if len(result_list)!=0 and best_accuracy<result_list[-1]-0.02 and epoch>350:
                    #print "########################"

                print "Epoch ", epoch, "'s accuracy is", '{0:.3f}'.format(currcorrect)#, currcorrect2
                print "best accuracy @"+ str(count) +" is", "{:.4f}".format(best_accuracy)
                train_acc = train_acc/float(length)
                #train_acc2 = train_acc2/float(length)
                print "Train Accuracy", train_acc#, train_acc2
                print args.root
                """
            if epoch == args.num_epochs:
                a,b,user = gcn_model(X_Tags_Feature, Adj, Adj2, Adj3, Adj4, Adj5, Target, X_train)
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
                f = open("/home/junting/Downloads/GCN/UserCredibility/Accuracy-Stackexchange/clique/adaboost_"+args.root.split('/')[-1]+".txt", "w+")
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

#torch.save(gcn_model.state_dict(), 'gcnmodel_'+args.root.split('/')[-1]+str(args.learning_rate)+'_adaboost'+'.pt')
