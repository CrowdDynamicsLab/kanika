import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from data import QADataset, SubjObjDataset, get_GCN_inputs
from model import FeedForward, GCN
from test import evaluate
from utils import getPostContexts
from vectorizers import IndexVectorizer
from glove_utils import load_glove_embeddings
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn import preprocessing
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
import numpy as np
import pandas as pd
import os

parser = argparse.ArgumentParser(description='PyTorch Credibility Prediction Model')
parser.add_argument('--use_cuda', dest='use_cuda', default=False, action='store_true')
parser.add_argument('-lr', '--learning_rate', type=float, default=0.001,\
                    help='learning rate for FeedForward')
parser.add_argument('--num_epochs', type=int, default=100)
parser.add_argument('--root', type=str, default='/home/knarang2/StackExchange')
glove_path = '/Users/kanika/Documents/glove.6B/glove.6B.50d.txt'
args = parser.parse_args()
seed = 1234734614
torch.manual_seed(seed)
if args.use_cuda:
	torch.cuda.manual_seed(seed)

dataset = QADataset(dataset='StackExchange',questionFile = 'QuestionFeatures.tsv', answerFile = 'AnswerFeatures.tsv', userFile = 'UserFeatures.tsv',
                                    rootFolder= args.root)

vectorizer = IndexVectorizer(min_frequency=10)
##CHANGED HERE
textDataset = SubjObjDataset(os.path.join(args.root, "pairText_merge_processed.tsv"), vectorizer)
word2idx = textDataset.vectorizer.word2idx
embeddings = load_glove_embeddings(glove_path, word2idx)
#print embeddings
print "#WORDS in the Vocabulary", len(word2idx)



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

X = pd.concat([PosClass, NegClass_Sample])

## PreProcess the dataset
cols = X.columns.drop('QTags')
X[cols] = X[cols].apply(pd.to_numeric, errors='coerce')
X.fillna(0, inplace=True)
X["PairId"] = X["PairId"].apply(lambda x: int(x)-1)

X_Features_scaled = getPostContexts(X, dataset)
X.iloc[:,6:] = X_Features_scaled

print X
exit()
#Text_Features_df = dataset.getTextFeatures(X)
#X_Text = X.merge(Text_Features_df, on='PairId')
#dtype = dict(PairId=int)


# #print X.dtypes
# Posters = X.AskerId.unique()
# Commenters = X.CommenterId.unique()
# Users = set(Posters).union(set(Commenters))
#
# #Create UserIndex
# UserIndex = dict()
# count = 0
# for user in sorted(list(Users)):
#     UserIndex[user] = count
#     count+=1
#
# Users = [k for k in range(len(UserIndex))]
# print "#POSTERS, #Commenters, #AllUSers", len(Posters), len(Commenters), len(Users) #, len(UserIndex)
# X["AskerId"].replace(UserIndex, inplace=True)
# X["CommenterId"].replace(UserIndex, inplace=True)
# #print X["AskerId"]
# #User_Feature = dataset.userFeatures.loc[dataset.userFeatures['Id'].isin(UserIndex.keys())]
# #print "Users loaded", len(User_Feature)
# #User_Feature["Id"].replace(UserIndex, inplace=True)
# #print User_Feature, dataset.userFeatures["Id"]
#
# Commenter_Feature = X[["CommenterId", "Reputation", "NetworkAge" ,"Upvotes", "DownVotes", "Gold" , "Silver", "Bronze", "NQuestions", "NAnswers"]]
# Commenter_Feature = Commenter_Feature.rename(index=str, columns={"CommenterId":"Id"})
# Asker_Feature =  X[["AskerId", "AReputation", "ANetworkAge","AUpvotes", "ADownVotes", "AGold" , "ASilver", "ABronze", "ANQuestions", "ANAnswers"]]
# Asker_Feature = Asker_Feature.rename(index=str, columns={"AskerId":"Id", "AReputation":"Reputation", "ANetworkAge":"NetworkAge","AUpvotes":"Upvotes", "ADownVotes":"DownVotes", "AGold":"Gold" , "ASilver":"Silver", "ABronze":"Bronze", "ANQuestions":"NQuestions", "ANAnswers":"NAnswers"})
# User_Feature = pd.concat([Commenter_Feature, Asker_Feature])
# User_Feature.drop_duplicates(subset=["Id"], inplace=True)
# User_Feature.set_index('Id', inplace=True)
# User_Feature.sort_index(inplace=True)
# #min_max_scaler = preprocessing.MinMaxScaler()
# #User_Features_scaled = min_max_scaler.fit_transform(User_Feature.as_matrix().astype(float))
#
# User_Features_scaled = User_Feature.values
# User_Features_scaled = (User_Features_scaled - User_Features_scaled.mean(axis=0))/User_Features_scaled.std(axis=0)
# #print User_Features_scaled
# print "#PAIRS loaded for training", len(X)
#
# #exit()
# ## GET BASIC TEXT FEATURES
# X = X.drop(['Age', 'AAge', 'QTags'], axis = 1)
# #X = X.dropna(axis=0, how='any')
# #X_Features = pd.concat([X.iloc[:, 6:25], X.iloc[:,26]])
#
# X["PairId"] = X["PairId"].apply(lambda x: int(x)-1)
# Text_Features_df = dataset.getTextFeatures(X)
# X_Text = X.merge(Text_Features_df, on='PairId')
# dtype = dict(PairId=int)
#
#
# edges = dataset.getAdjMatrix(X, 'interaction')
# #edges = dataset.getAdjMatrix_half(len(Users))
# Adj, rowsum = get_GCN_inputs(edges, len(Users))
# print Adj
# #Features = Variable(torch.FloatTensor(np.eye(len(Users))), requires_grad=False)
# Features = Variable(torch.FloatTensor(User_Feature.values), requires_grad=False)
#
# ##MERGING WITH GLOVE EMBEDDINGS
# X_Text_WordEmbed = X_Text.astype(dtype).merge(textDataset.corpus[list(['PairId','vectors'])].astype(dtype), on='PairId', how='left')
# #print "Text Features", X_Text_WordEmbed.keys(), len(X_Text_WordEmbed) , len(textDataset.corpus[['PairId', 'vectors']])
# #X_Text_WordEmbed = X
#
# ##NORMALIZE FEATURES
# X_Features_Text = X_Text_WordEmbed.iloc[:, 6:-1]
#
# print X_Features_Text


#min_max_scaler = preprocessing.MinMaxScaler()
#X_Features_scaled = min_max_scaler.fit_transform(X_Features_Text.values)
#X_Features_scaled = X_Features_Text.values
#X_Features_scaled = (X_Features_scaled - X_Features_scaled.mean(axis=0))/X_Features_scaled.std(axis=0)
#X_Text_WordEmbed[X_Features_Text.columns] = X_Features_scaled
#print X_Text_WordEmbed
exit()
X_train, X_test, Y_train, Y_test = train_test_split(X_Text_WordEmbed, X_Text_WordEmbed.iloc[:,5], test_size=0.2, random_state=1234)

X_train = X_train.iloc[:,:-1]
X_train_text = X_train.iloc[:,-1]
X_train.fillna(-1, inplace=True)
X_test = X_test.iloc[:,:-1]
X_test_text = X_test.iloc[:,-1]
X_test.fillna(-1, inplace=True)

model = LogisticRegression()
model.fit(X_train.iloc[:,6:], Y_train)
Y_pred = model.predict(X_test.iloc[:, 6:])
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(model.score(X_test.iloc[:,6:], Y_test)))
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(accuracy_score( Y_test, Y_pred)))
print "confusion_matrix", confusion_matrix(Y_test, Y_pred)
print(model.coef_)

#exit()
X_train = torch.FloatTensor(X_train.values)
X_train_text = torch.LongTensor(X_train_text.values)
X_test = (torch.FloatTensor(X_test.values))
X_test_text = torch.LongTensor(X_test_text.values)
Y_train = Variable(torch.LongTensor(Y_train.values), requires_grad=False)
Y_test = torch.LongTensor(Y_test.values)

print Y_train
## Neural Model
# Loss and Optimizer
nComm = 20
nHid = 50
nFeat = 9
model = FeedForward(len(Users), nHid, 2, textDataset.vectorizer.vocabulary_size, embeddings, freeze_embeddings=True)
gcn_model = GCN(nFeat, nHid, nComm)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad,model.parameters()), lr=args.learning_rate , weight_decay=5e-4)
gcn_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad,gcn_model.parameters()), lr=args.learning_rate, weight_decay=5e-4)

if args.use_cuda:
    model.cuda()
    gcn_model.cuda()
    X_train = X_train.cuda()
    X_test = X_test.cuda()
    Y_train = Y_train.cuda()
    Y_test = Y_test.cuda()


epoch_loss = []
epoch_accuracy = []
train_accuracy = []
epochs = []

correct = 0
try:
    for epoch in range(1, args.num_epochs + 1):

        optimizer.zero_grad()
        gcn_optimizer.zero_grad()
        #print X_train[:,2:]
        user_gcn_embed = gcn_model(Features, Adj)
        #print "GCN EMbeddings", user_gcn_embed
        #exit()
        output = model(X_train[:,1:3], X_train[:,6:], X_train_text, user_gcn_embed)
        output.squeeze_()
        #print output
        #predicted = np.rint(F.sigmoid(output).data.numpy())
        _, predicted = torch.max(output.data, 1)
        #print predicted, Y_train.data.numpy()
        train_acc = accuracy_score( Y_train.data.numpy(), predicted)
        print "Train Accuracy", train_acc
        #print X_train[:,6:]
        #print X_train[:,6:].numpy()
        #print model.layer1.weight.numpy()
        #print "Unscaled output", np.multiply(X_train[:,6:].numpy(), model.layer1.weight.numpy())
        loss = criterion(output, Y_train)
        loss.backward()
        optimizer.step()
        gcn_optimizer.step()

        if epoch > 0:
            print("Total loss", epoch, loss.data[0])
            currcorrect = evaluate(model, gcn_model, X_test, X_test_text, Y_test, Adj, Features)
            print "Epoch ", epoch, "'s accuracy is", currcorrect

            epochs.append(epoch)
            epoch_loss.append(loss.data[0])
            epoch_accuracy.append(currcorrect)
            train_accuracy.append(train_acc)

            if currcorrect > correct:
                correct = currcorrect
                torch.save(model.state_dict(), 'feedmodel.pt')



except KeyboardInterrupt:
	print('-' * 89)
	print('Exiting from training early')

#evaluate(model, X_test, Y_test)
#torch.save(model.state_dict(), 'feedmodel.pt')
print("Best accuracy is", correct)
print model.layer1.weight.data
#confusion_matrix = confusion_matrix(Y_test, Y_pred)
#print(confusion_matrix)

if True:
    ext= 'cred'
    plt.close("all")
    plt.plot(epochs, epoch_loss)
    plt.ylabel("Total Loss")
    plt.xlabel("Epochs")
    plt.title("LR_GCN model")
    plt.savefig('figures/epoch_loss_'+ext+'.png')
    #plt.show()
    plt.clf()

    plt.plot(epochs, epoch_accuracy, 'r', label='Test')
    plt.plot(epochs, train_accuracy, 'b', label='Train')
    plt.ylabel("Train & Test accuracy")
    plt.xlabel("Epochs")
    plt.legend()
    plt.title("RNN_GCN model")
    plt.savefig('figures/epoch_acc_'+ext+'.png')
    #plt.show()
    plt.clf()
