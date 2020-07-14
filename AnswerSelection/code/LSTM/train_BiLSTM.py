import torch
import os
import argparse
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from sklearn import preprocessing
from torch.autograd import Variable
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from data import QADataset, SubjObjDataset, get_GCN_inputs, get_GCN_inputs2, get_GCN_inputs3, normalize, sparse_mx_to_torch_sparse_tensor
from sklearn.metrics import accuracy_score
from torch import optim
import time, random
import os
from tqdm import tqdm
from lstm import LSTMSentiment
from bilstm import BiLSTM
from torchtext import data
import gensim

# parser
parser = argparse.ArgumentParser(description='PyTorch Credibility Prediction Model')
parser.add_argument('--use_cuda', dest='use_cuda', default=False, action='store_true')
parser.add_argument('-lr', '--learning_rate', type=float, default=0.01,\
                    help='learning rate for ensemble')
parser.add_argument('--weight_decay', type=float, default=5e-3)
parser.add_argument('--num_epochs', type=int, default=50)
parser.add_argument('--batch_size', type=int, default=200)
#parser.add_argument('--root', type=str, default='../arduino-data')
parser.add_argument('--root', type=str, default='/scratch/uc_data/data')
parser.add_argument('--m', dest='model', default='bilstm', help='specify the mode to use (default: bilstm)')
parser.add_argument('--dataset', type=str, default='ai')
parser.add_argument('--gpu', type=str, default='0')

# parameters
args = parser.parse_args()
EPOCHS = 20
#USE_GPU = torch.cuda.is_available()
USE_GPU = args.use_cuda
EMBEDDING_DIM = 300
HIDDEN_DIM = 100
BARSCORE = 0.7
dataset_NLP = args.dataset

BATCH_SIZE = args.batch_size
timestamp = str(int(time.time()))
seed = 999
torch.manual_seed(seed)
if args.use_cuda:
	torch.cuda.manual_seed(seed)

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
# torch.device object used throughout this script
DEVICE = torch.device("cuda:0" if args.use_cuda else "cpu")


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


## PreProcess the dataset
cols = X.columns.drop('QTags')
X[cols] = X[cols].apply(pd.to_numeric, errors='coerce')
X.fillna(0, inplace=True)
X["PairId"] = X["PairId"].apply(lambda x: int(x)-1)
#data,X_Tags_Feature2 = getPostContexts(X, dataset)
#X_Tags_Feature = Variable(data, requires_grad=False)
QuestionId = X.loc[:,'QuestionId'].drop_duplicates()
QuestionId = QuestionId[QuestionId>0]

# 5-fold cross_validation
from sklearn.model_selection import KFold
#kf = KFold(n_splits=5, shuffle=True, random_state=1234734614)
kf = KFold(n_splits=5, shuffle=True)
Dataset = np.array(QuestionId[QuestionId!=0])

def load_sst(text_field, label_field, batch_size, dataset_NLP, X):
    #train, dev, test = data.TabularDataset.splits(path='./data/SST2/', train=dataset+'-NLP2.tsv',
    #                                              validation='dev2.tsv', test=dataset+'-NLP2.tsv', format='tsv',
    #                                              fields=[('Q', text_field), ('A', text_field), ('label', label_field)])
    #Pairs_train, Pairs_test, Y_train, Y_test = train_test_split(QuestionId[QuestionId!=0], QuestionId[QuestionId!=0], test_size=0.2, random_state=1234)
    #Pairs_train, Pairs_test = train, test = train_test_split(df, test_size=0.2)
    """
    Read or filter and write the content file
    #Read tsv file, filter it and write it
    #if existing file, read it
    """

    traindata = data.TabularDataset(path=os.path.join('../../StackExchangeText', dataset_NLP,'pairText_merge_label.tsv'),
                                                  format='tsv', fields=[('P', label_field), ('Q', text_field), ('A', text_field), ('label', label_field)])

    pairIds = list(X["PairId_original"].values)
    traindata_filter = []
    count = 0
    newID_list = []
    for exampledata in traindata:
        id = int(exampledata.P)
        if id in pairIds:
            #newID = X[X["PairId_original"] == id]["PairId"]
            newID = pairIds.index(id)
            newID_list.append(newID)
            exampledata.P = str(int(newID))
            traindata_filter.append(exampledata)


    #print("PairIDs of training data", newID_list)
    traindata_filter = data.Dataset(traindata_filter, fields=[('P', label_field), ('Q', text_field), ('A', text_field), ('label', label_field)])
    text_field.build_vocab(traindata_filter)
    label_field.build_vocab(traindata_filter)
    #train_iter, dev_iter, test_iter = data.BucketIterator.splits((train, dev, test),
    #            batch_sizes=(batch_size, len(dev), len(test)), sort_key=lambda x: len(x.text), repeat=False, device=-1)
    #for GPU run

    #train_iter, dev_iter, test_iter = data.BucketIterator.splits((train, dev, test),
    #            batch_sizes=(batch_size, batch_size, batch_size), repeat=False, device=None)

    #return train_iter, dev_iter, test_iter
    return traindata_filter.examples

def load_bin_vec(fname, vocab):
    """
    Loads 300x1 word vecs from Google (Mikolov) word2vec
    """
    word_vecs = {}
    with open(fname, "rb") as f:
        header = f.readline()
        print(header)
        vocab_size, layer1_size = map(int, header.split())
        binary_len = np.dtype('float32').itemsize * layer1_size
        for line in range(vocab_size):
            word = []
            while True:
                ch = f.read(1).decode('latin-1')
                if ch == ' ':
                    word = ''.join(word)
                    break
                if ch != '\n':
                    word.append(ch)
            if word in vocab:
               word_vecs[word] = np.fromstring(f.read(binary_len), dtype='float32')
            else:
                f.read(binary_len)
    return word_vecs

# load dataset
text_field = data.Field(lower=True, fix_length=250)
label_field = data.Field(sequential=False)
#train_iter, dev_iter, test_iter = load_sst(text_field,label_field, BATCH_SIZE, dataset_NLP)
traindata = load_sst(text_field,label_field, BATCH_SIZE, dataset_NLP, X)
traindata = np.array(traindata)

# word2vector
word_to_idx = text_field.vocab.stoi
pretrained_embeddings = np.random.uniform(-0.25, 0.25, (len(text_field.vocab), 300))
pretrained_embeddings[0] = 0

print "Initialize embeddings..."
#word2vec = load_bin_vec('../../GoogleNews-vectors-negative300.bin', word_to_idx)
#for word, vector in word2vec.items():
#    pretrained_embeddings[word_to_idx[word]-1] = vector
##CHeck if vocab file is not there
embedding_file = os.path.join('../../embedding_data', args.dataset + '_embedding_filter.npy')

if os.path.exists(embedding_file):
    print("Loaded from file")
    pretrained_embeddings = np.load(embedding_file)
else:


    #from gensim.scripts.glove2word2vec import glove2word2vec
    #glove_input_file = '/home/knarang2/glove.840B.300d.txt'
    #word2vec_output_file = '/home/knarang2/glove.840B.300d.txt.word2vec'
    #glove2word2vec(glove_input_file, word2vec_output_file)

    model = gensim.models.KeyedVectors.load_word2vec_format('../../GoogleNews-vectors-negative300.bin', binary=True)
    #model = gensim.models.KeyedVectors.load_word2vec_format('/home/knarang2/glove.840B.300d.txt.word2vec', binary=False)
    #weights = torch.FloatTensor(model.vectors)
    for word, id in word_to_idx.items():
        word = word.encode('ascii','ignore')
        if word in model:
            pretrained_embeddings[id] = model[word]

    np.save(embedding_file, pretrained_embeddings)



def get_accuracy(truth, pred):
    assert len(truth) == len(pred)
    right = 0
    for i in range(len(truth)):
        temp = 0
        if pred[i]>BARSCORE:
            temp = 1
        if truth[i] == temp:
            right += 1.0
    return right / len(truth)

def writeContentEmbeddings(model):
    model.load_state_dict(torch.load("/home/knarang2/UserCredibility/models/"+ dataset_NLP +"_LSTM.pt"))
    model.eval()

    embeddings = np.zeros((len(X), HIDDEN_DIM))
    testset = torch.utils.data.TensorDataset(torch.LongTensor(Dataset), torch.torch.LongTensor(Dataset))
    test_loader = torch.utils.data.DataLoader(dataset=testset,
                                              batch_size=BATCH_SIZE,
                                              shuffle=False)

    for i, (X_test_1, Y_test_1) in enumerate(test_loader):
        X_test = torch.tensor(X[X["QuestionId"].isin(X_test_1)]['PairId'].values)
        Y_test = torch.tensor(X[X["QuestionId"].isin(Y_test_1)]['Credible'].values)

        Ques_Pair = X[X["QuestionId"].isin(X_test_1)][['QuestionId','PairId', 'Credible']].groupby(['QuestionId']).agg(lambda x:x.tolist()).reset_index().values
        #X[X["QuestionId"].isin(X_test_1)][['QuestionId', 'PairId']].groupby(['QuestionId']).agg(lambda x:x.tolist())
        test = traindata[X_test]
        fields=[('P', label_field), ('Q', text_field), ('A', text_field), ('label', label_field)]
        test_iter = data.BucketIterator(data.Dataset(test, fields), batch_size=len(test), repeat=False, device=DEVICE)

        for batch in test_iter:
            Q, A, label = batch.Q, batch.A, batch.label
            #Q, A, label = test.Q, test.A, test.label
            label.data.sub_(1)
            model.batch_size = len(label.data)
            model.hidden = model.init_hidden()
            _, q, ans = model(Q,A)
            #import pdb; pdb.set_trace()
            embeddings[X_test, :] = np.concatenate((q.cpu().detach().numpy(), ans.cpu().detach().numpy()), axis=1)

    dir = "/home/knarang2/StackExchangeDataEmbed/" + dataset_NLP
    if not os.path.exists(dir):
        os.makedirs(dir)
    np.save(dir + "/dataEmbeddings.npy", embeddings)

    print("Embeddings written of the best model")


result_list = list()
result_list_acc = list()
count = 0
try:
    for Ques_train, Ques_test in kf.split(Dataset):

        count+=1
        print("Fold ", count)
        if count == 2:
            #writeContentEmbeddings(model)
            break
        # define model
        model = BiLSTM(embedding_dim=EMBEDDING_DIM, hidden_dim=HIDDEN_DIM, vocab_size=len(text_field.vocab), label_size=len(label_field.vocab)-1,\
                              use_gpu=USE_GPU, batch_size=BATCH_SIZE)
        if USE_GPU:
            model = model.to(DEVICE)
        model.embeddings.weight.data.copy_(torch.from_numpy(pretrained_embeddings))
        optimizer = optim.Adam(model.parameters(), lr=1e-3)

        trainset = torch.utils.data.TensorDataset(torch.LongTensor(Dataset[Ques_train]), torch.torch.LongTensor(Dataset[Ques_train]))
        testset = torch.utils.data.TensorDataset(torch.LongTensor(Dataset[Ques_test]), torch.torch.LongTensor(Dataset[Ques_test]))

        train_loader = torch.utils.data.DataLoader(dataset=trainset,
                                                        batch_size=64)

        test_loader = torch.utils.data.DataLoader(dataset=testset,
                                                  batch_size=BATCH_SIZE,
                                                  shuffle=False)

        MRR = []
        best_MRR = 0
        best_accuracy = 0
        prev_train_acc = 0.0
        # train and test
        for epoch in range(1, args.num_epochs + 1):

            truth_res = []
            pred_res = []
            model.train(True)

            for i, (X_train_index, Y_train_index) in enumerate(train_loader):
                #print i,"out of",len(train_loader)
                X_train = torch.tensor(X[X["QuestionId"].isin(X_train_index)]['PairId'].values)
                #Y_train = torch.tensor(X[X["QuestionId"].isin(X_train_index)]['Credible'].values)

                #train = [traindata[X_train] for x in Pairs_train]
                train = traindata[X_train]
                fields=[('P', label_field), ('Q', text_field), ('A', text_field), ('label', label_field)]
                train_iter = data.BucketIterator(data.Dataset(train, fields), batch_size=len(train), repeat=False, device=DEVICE)
                avg_loss = 0.0

                for batch in train_iter:
                    Q, A, label = batch.Q, batch.A, batch.label
                    #Q = [traindata[x].Q for x in X_train]
                    #A = [traindata[x].A for x in X_train]
                    #label = [traindata[x].label for x in X_train]

                    label.data.sub_(1)
                    truth_res += label.cpu().data.numpy().tolist()
                    model.batch_size = len(label.data)
                    model.hidden = model.init_hidden()
                    pred, _, _ = model(Q,A)
                    pred_label = pred.cpu().data.numpy()
                    pred_res += [x for x in pred_label]
                    optimizer.zero_grad()
                    Target = np.array([2*i-1 for i in label.cpu().data.numpy().tolist()])
                    Target = torch.FloatTensor(Target).to(DEVICE)
                    loss = torch.sum(torch.exp(-torch.mul(pred,Target)))
                    #avg_loss += loss.data[0]
                    loss.backward()
                    optimizer.step()
                    #avg_loss /= len(train_iter)
                    #print("Batch")
                    #print truth_res, pred_res
            train_acc = get_accuracy(truth_res, pred_res)

            if epoch%2 == 0 or epoch==1:
                MRR = []
                model.eval()
                truth_res = []
                pred_res = []

                for i, (X_test_1, Y_test_1) in enumerate(test_loader):
                    X_test = torch.tensor(X[X["QuestionId"].isin(X_test_1)]['PairId'].values)
                    #Y_test = torch.tensor(X[X["QuestionId"].isin(X_test_1)]['Credible'].values)

                    Ques_Pair = X[X["QuestionId"].isin(X_test_1)][['QuestionId','PairId', 'Credible']].groupby(['QuestionId']).agg(lambda x:x.tolist()).reset_index().values
                    #X[X["QuestionId"].isin(X_test_1)][['QuestionId', 'PairId']].groupby(['QuestionId']).agg(lambda x:x.tolist())
                    test = traindata[X_test]
                    fields=[('P', label_field), ('Q', text_field), ('A', text_field), ('label', label_field)]
                    test_iter = data.BucketIterator(data.Dataset(test, fields), batch_size=len(test), repeat=False, device=DEVICE)

                    for batch in test_iter:
                        Q, A, label = batch.Q, batch.A, batch.label
                        #Q, A, label = test.Q, test.A, test.label
                        label.data.sub_(1)
                        truth_res += label.cpu().data.numpy().tolist()
                        model.batch_size = len(label.data)
                        model.hidden = model.init_hidden()
                        pred, _, _ = model(Q,A)
                        pred_res += pred.cpu().data.numpy().tolist()
                        for entry in Ques_Pair:
                            ques = entry[0]
                            pairs = entry[1]
                            acc_labels = entry[2]
                            pairs_index = [X_test.tolist().index(x) for x in pairs]
                            values = [pred_res[x] for x in pairs_index]
                            temp = np.array(values).argsort()
                            ranks = len(values) - temp  #Descending
                            #print(acc_labels, entry)
                            groundT = acc_labels.index(1)
                            MRR.append(1.0/ranks[groundT])

                test_mrr = sum(MRR)/len(MRR)
                #print("mrr at epoch", epoch, test_mrr)
                if best_MRR < test_mrr:
                    best_MRR = test_mrr

                #import pdb; pdb.set_trace()
                acc = get_accuracy(truth_res, pred_res)
                if best_accuracy < acc:
                    best_accuracy = acc
                    torch.save(model.state_dict(), "/home/knarang2/UserCredibility/models/"+ dataset_NLP +"_LSTM.pt")
                #print("Test Accuracy at epoch", epoch, acc)

                print("Training and Test Accuracy and MRR at epoch", epoch, '{0:.3f}'.format(train_acc), '{0:.3f}'.format(acc), '{0:.3f}'.format(test_mrr))
            if (train_acc - prev_train_acc) < 0.01:
                print("Exiting training")
                break
        print("best_mrr and accuracy", best_MRR, best_accuracy)
        result_list.append(best_MRR)
        result_list_acc.append(best_accuracy)

except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')


writeContentEmbeddings(model)

print result_list
print "Average Best MRR", '{0:.2f}'.format(sum(result_list)/len(result_list))
print result_list_acc
print "Average Best Accuracy", '{0:.2f}'.format(sum(result_list_acc)/len(result_list_acc))

# def train_epoch_progress(model, train_iter, loss_function, optimizer, epoch):
#     model.train(True)
#     avg_loss = 0.0
#     truth_res = []
#     pred_res = []
#     count = 0
#     for batch in tqdm(train_iter, desc='Train epoch '+str(epoch+1)):
#         Q, A, label = batch.Q, batch.A, batch.label
#         label.data.sub_(1)
#         truth_res += label.cpu().data.numpy().tolist()
#         model.batch_size = len(label.data)
#         model.hidden = model.init_hidden()
#         pred = model(Q,A)
#         pred_label = pred.cpu().data.numpy()
#         pred_res += [x for x in pred_label]
#         optimizer.zero_grad()
#         Target = np.array([2*i-1 for i in label.cpu().data.numpy().tolist()])
#         Target = Variable(torch.cuda.FloatTensor(Target),requires_grad=False)
#         loss = torch.sum(torch.exp(-torch.mul(pred,Target)))
#         #avg_loss += loss.data[0]
#         loss.backward()
#         optimizer.step()
#     #avg_loss /= len(train_iter)
#     #print truth_res, pred_res
#     acc = get_accuracy(truth_res, pred_res)
#     return acc
#
#
#
#
# def train_epoch(model, train_iter, loss_function, optimizer):
#     model.train()
#     avg_loss = 0.0
#     truth_res = []
#     pred_res = []
#     count = 0
#     for batch in train_iter:
#         Q, A, label = batch.Q, batch.A, batch.label
#         label.data.sub_(1)
#         truth_res += list(label.data)
#         model.batch_size = len(label.data)
#         model.hidden = model.init_hidden()
#         pred = model(Q,A)
#         pred_label = pred.data.numpy()
#         pred_res += [x for x in pred_label]
#         model.zero_grad()
#         Target = np.array([2*i-1 for i in label.data.numpy().tolist()])
#         Target = Variable(torch.FloatTensor(Target),requires_grad=False)
#         loss = torch.sum(torch.exp(-torch.mul(pred,Target)))
#         #avg_loss += loss.data[0]
#         count += 1
#         loss.backward()
#         optimizer.step()
#     #avg_loss /= len(train_iter)
#     acc = get_accuracy(truth_res, pred_res)
#     return acc
#
#
# def evaluate(model, data, loss_function, name):
#     model.eval()
#     avg_loss = 0.0
#     truth_res = []
#     pred_res = []
#     for batch in data:
#         Q, A, label = batch.Q, batch.A, batch.label
#         label.data.sub_(1)
#         truth_res += label.cpu().data.numpy().tolist()
#         model.batch_size = len(label.data)
#         model.hidden = model.init_hidden()
#         pred = model(Q,A)
#         pred_label = pred.cpu().data.numpy()
#         pred_res += [x for x in pred_label]
#         Target = np.array([2*i-1 for i in label.cpu().data.numpy().tolist()])
#         Target = Variable(torch.cuda.FloatTensor(Target),requires_grad=False)
#         loss = torch.sum(torch.exp(-torch.mul(pred,Target)))
#         #avg_loss += loss.data[0]
#         #avg_loss += loss.data[0]
#     #avg_loss /= len(data)
#     acc = get_accuracy(truth_res, pred_res)
#     print(name + ': loss %.2f acc %.1f' % (avg_loss, acc*100) +"%")
#     return acc
#
# def evaluate_mrr(test_iter):
#     for batch in test_iter:
#         Q, A, label = batch.Q, batch.A, batch.label
#         #Q, A, label = test.Q, test.A, test.label
#         label.data.sub_(1)
#         truth_res += label.cpu().data.numpy().tolist()
#         model.batch_size = len(label.data)
#         model.hidden = model.init_hidden()
#         pred = model(Q,A)
#         pred_label = pred.cpu().data.numpy()
#         pred_res += [x for x in pred_label]
#         groundT = Y_test.numpy().tolist().index(1)
#         temp = np.array(pred_res).argsort()
#         ranks = np.empty_like(temp)
#         ranks[temp] = len(pred_res) - np.arange(len(pred_res))
#         MRR.append(1.0/ranks[groundT])

# def adjust_learning_rate(learning_rate, optimizer, epoch):
#     lr = learning_rate * (0.1 ** (epoch // 10))
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = lr
#     return optimizer
