import torch
import os
import argparse
import numpy as np
import pandas as pd
import torch.nn as nn
from test import evaluate_adaboost
import torch.nn.functional as F
from utils import getPostContexts
from sklearn import preprocessing
from torch.autograd import Variable
from model import GCN_adaboost
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from data import QADataset, SubjObjDataset, get_GCN_inputs, get_GCN_inputs2, get_GCN_inputs3, normalize, sparse_mx_to_torch_sparse_tensor
from sklearn.metrics import accuracy_score
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print device
#torch.set_num_threads(12)

# parser
parser = argparse.ArgumentParser(description='PyTorch Credibility Prediction Model')
parser.add_argument('--use_cuda', dest='use_cuda', default=False, action='store_true')
parser.add_argument('-lr', '--learning_rate', type=float, default=0.001,\
                    help='learning rate for ensemble')
parser.add_argument('--weight_decay', type=float, default=5e-3)
parser.add_argument('--num_epochs', type=int, default=500)
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
# train and test
try:
    for Pairs_train, Pairs_test in kf.split(Dataset):
        count += 1
        if count==3:

            print result_list
            print sum(result_list)/len(result_list)
            with open ("Adaboost/result/" + args.root.split('/')[-1] + str(args.learning_rate) +".txt", "w+") as f:
                f.write(''.join(str(e) for e in result_list)+"\n")
                f.write(str(sum(result_list)/len(result_list)))
            f.close()

            torch.save(gcn_model.state_dict(), 'Adaboost/MRRmodel_'+args.root.split('/')[-1]+'_adaboost'+'.pt')
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

        gcn_model = GCN_adaboost(nFeat, nHid1, nHid2, nHid3, nHid4, nComm).to(device)
        criterion = nn.MSELoss()
        gcn_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad,gcn_model.parameters()), lr=args.learning_rate, weight_decay=args.weight_decay)
        #gcn_model.load_state_dict(torch.load('gcn_complete2.pkl'))

        trainset = torch.utils.data.TensorDataset(torch.LongTensor(Dataset[Pairs_train]).cuda(), torch.torch.LongTensor(Dataset[Pairs_train]).cuda())
        testset = torch.utils.data.TensorDataset(torch.LongTensor(Dataset[Pairs_test]).cuda(), torch.torch.LongTensor(Dataset[Pairs_test]).cuda())

        train_loader = torch.utils.data.DataLoader(dataset=trainset,
                                                        batch_size=len(Pairs_train))

        test_loader = torch.utils.data.DataLoader(dataset=testset,
                                                  batch_size=1,
                                                  shuffle=False)
        best_MRR = 0
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

            decay = float(np.exp(-epoch/20.0))

            for i, (X_train, Y_train) in enumerate(train_loader):
                X_train = torch.tensor(X[X["QuestionId"].isin(X_train)]['PairId'].values)
                Y_train = torch.tensor(X[X["QuestionId"].isin(Y_train)]['Credible'].values)
                gcn_model.train(True)
                gcn_optimizer.zero_grad()
                sim1, sim2, part2_dense = gcn_model(X_Tags_Feature, Adj, Adj2, Adj3, Adj4, Adj5, Target, X_train)

                length+= len(Y_train)

                predicted = part2_dense.squeeze()[X_train]>0
                train_acc+= (Y_train.numpy()==predicted.cpu().numpy()).sum()


                Y_train = Variable(Y_train, requires_grad=False).cuda()

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

                loss_train+=loss.item()
                loss.backward()
                gcn_optimizer.step()

            if epoch > 0:
                #print "GCN Weight", gcn_model.gc1.weight.data
                print("Total loss", epoch, loss_train)
                MRR, ran= evaluate_adaboost(gcn_model, test_loader, Adj, Adj2, Adj3, Adj4, Adj5, X_Tags_Feature, rowsum, X, epoch, Target)

                if MRR>best_MRR:
                    best_MRR = MRR
                print "Epoch ", epoch, "'s MRR is", MRR, "random is:", ran
                print "best MRR is", best_MRR
                print args.root
        result_list.append(best_MRR)

except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')

print result_list
print sum(result_list)/len(result_list)

torch.save(gcn_model.state_dict(), 'MRRmodel_'+args.root.split('/')[-1]+'_adaboost'+'.pt')
