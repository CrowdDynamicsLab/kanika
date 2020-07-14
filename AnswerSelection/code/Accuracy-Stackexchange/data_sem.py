import torch
from torch.autograd import Variable
import numpy as np
import scipy.sparse as sp
import pandas as pd
import pickle
import os
import re
from torch.utils.data import Dataset, DataLoader
from itertools import combinations
##from gensim.models.doc2vec import Doc2Vec, TaggedDocument

class QADataset(Dataset):

    def __init__(self, dataset, rootFolder):
        self.dataset = dataset
        self.rootFolder = rootFolder
        self.trainPairs = None
        self.testPairs = None
        self.embedding = None
        self.Rating = None
        self.createPairs_separate2()

    def createPairs_separate2(self):
        fname = self.rootFolder + "/"+ 'New_SemEval.csv'
        print "Read pairs from file"
        with open(fname, 'r') as f:
            self.trainPairs = pd.read_csv(fname, sep='\t', quotechar="\"", encoding = 'utf-8', dtype = 'str')
            self.trainPairs['Q_id'] = self.trainPairs['Q_id'].apply(lambda x: change(x))
            print "# of Pairs",len(self.trainPairs)
        return

    def __len__(self):
        return len(self.trainPairs)

    # text embedding
    def getEmbeding(self):
        model = Doc2Vec.load("../doc2Vec/doc2vec_model")
        self.embedding = self.pairText
        self.embedding["QBody"] = self.embedding["QBody"].apply(lambda x: model.infer_vector(str(x).split(' ')))
        self.embedding["ABody"] = self.embedding["ABody"].apply(lambda x: model.infer_vector(str(x).split(' ')))
        temp = np.array([np.concatenate((i[0],i[1]), axis=0) for i in self.embedding[["QBody","ABody"]].values])
        temp = (temp - temp.mean(axis=0))/temp.std(axis=0)
        self.embedding = torch.FloatTensor(temp)
        print self.embedding[:10]


    # group by A_user
    def getAdjList_Similarity1(self):

    	AdjList = []
        ext = ''

        trainPairs = pd.merge(self.trainPairs, self.pairFeatures, on='PairId')
        trainPairs["PairId"] = trainPairs["PairId"].apply(lambda x: int(x)-1)
        firstPart = trainPairs.groupby(["A_user"])
        secondPart = trainPairs.groupby(["A_user"])

    	# group by A_user
        for index in secondPart:
            if len(index[1])>=5:
                x = index[1].sort_values(by=["A_user"] , ascending=True)["PairId"].tolist()
                for j in range(len(x)-5):
                    for i in combinations(x[j:j+3],2):
                            AdjList.append(i)


        AdjList = pd.DataFrame(data=AdjList, columns=["Nid1","Nid2"])
        AdjList = AdjList.drop_duplicates()
        print "similarity1:", len(AdjList)
        #AdjList = AdjList.sample(500)

        edges = AdjList.sort_values(by=["Nid1","Nid2"] , ascending=True).values

        with open(self.rootFolder + "/"+ 'edges_' + 'line' + '.pkl', 'w') as f:
            pickle.dump( edges, f)
    	return edges

    # arrival similarity
    def getAdjList_Similarity2(self):

    	AdjList = []
        ext = ''
        coeff = 0.5

        trainPairs = self.trainPairs
        #trainPairs["PairId"] = trainPairs["PairId"].apply(lambda x: int(x)-1)
        self.trainPairs["ArrivalRank"] =  self.trainPairs["QASpan"]
        import scipy.stats as ss
        for i,j in self.trainPairs.groupby('Q_id'):
            temp = self.trainPairs.loc[self.trainPairs['Q_id']==i,"ArrivalRank"]
            self.trainPairs.at[self.trainPairs['Q_id']==i,"ArrivalRank"] = ss.rankdata(temp)

        trainPairs["Link"] = [0 for i in range(len(trainPairs))]
        step2 = self.trainPairs.groupby(["Q_id"])
        idxList1 = []
        idxList2 = []
        for i,j in step2:
            tempScore = sorted(j["QASpan"].values.tolist())
            if tempScore[-1]-tempScore[-2]>=(j["QASpan"].values.max()-j["QASpan"].values.min())*coeff:
                idxList1.append(j["QASpan"].idxmax())
            if tempScore[1]-tempScore[0]>=(j["QASpan"].values.max()-j["QASpan"].values.min())*coeff:
                idxList2.append(j["QASpan"].idxmin())
        trainPairs.loc[idxList1,"Link"] = 1
        trainPairs.loc[idxList2,"Link"] = 2
        step3 = trainPairs[trainPairs["Link"].isin([1,2])].groupby(["A_user","Link"])

        for i,j in step3:
            if len(j)>=2:
                for i in combinations(j["PairId"].tolist(),2):
                        AdjList.append(i)

        AdjList = pd.DataFrame(data=AdjList, columns=["Nid1","Nid2"])
        AdjList = AdjList.drop_duplicates()

        print "similarity2:", len(AdjList)
        #AdjList = AdjList.sample(500)

        edges = AdjList.sort_values(by=["Nid1","Nid2"] , ascending=True).values

        with open(self.rootFolder + "/"+ 'edges_' + 'line' + '.pkl', 'w') as f:
            pickle.dump( edges, f)
    	return edges

    # TrueSkill similarity
    def getAdjList_Similarity3(self,QuestionList):

        from trueskill import Rating, rate
    	AdjList = []
        ext = ''
        margin = 4

        trainPairs = self.trainPairs
        #trainPairs["PairId"] = trainPairs["PairId"].apply(lambda x: int(x)-1)

        # compute trueSkill value
        RatingDict = dict()
        for i in trainPairs["A_user"].drop_duplicates().values.tolist():
            RatingDict[i] = (Rating(),)
        step1 = trainPairs.groupby(["Q_id"])
        for i,j in step1:
            if int(i) in QuestionList:
                IdList = j["A_user"].values.tolist()
                rating = j["Credible"].apply(lambda x: 1-int(x)).values.tolist()
                for i,j in zip(IdList,rate([RatingDict[x] for x in IdList],ranks=rating)):
                    RatingDict[i] = j
        trainPairs["Rating"] = trainPairs["A_user"]
        trainPairs["Rating"] = trainPairs["Rating"].apply(lambda x: RatingDict[x][0].mu-3*RatingDict[x][0].sigma)
        trainPairs["Link"] = [0 for i in range(len(trainPairs))]

        self.Rating = trainPairs[["PairId","Rating"]]

        # create clique using trueSkill value
        step2 = trainPairs.groupby(["Q_id"])
        idxList1 = []
        idxList2 = []
        for i,j in step2:
            tempScore = sorted(j["Rating"].values.tolist())
            if tempScore[-1]-tempScore[-2]>=margin:
                idxList1.append(j["Rating"].idxmax())
            if tempScore[1]-tempScore[0]>=margin/2:
                idxList2.append(j["Rating"].idxmin())
        trainPairs.loc[idxList1,"Link"] = 1
        trainPairs.loc[idxList2,"Link"] = 2
        trainPairs["Credible"] = trainPairs["Credible"].apply(lambda x: int(x))
        step3 = trainPairs[trainPairs["Link"].isin([1,2])].groupby(["A_user","Link"])
        for i,j in step3:
            if len(j)>=2:
                for i in combinations(j["PairId"].tolist(),2):
                        AdjList.append(i)
        #exit()
        AdjList = pd.DataFrame(data=AdjList, columns=["Nid1","Nid2"])
        AdjList = AdjList.drop_duplicates()

        #AdjList = AdjList.sample(500)
        print "similarity3:", len(AdjList)
        edges = AdjList.sort_values(by=["Nid1","Nid2"] , ascending=True).values

        with open(self.rootFolder + "/"+ 'edges_' + 'line' + '.pkl', 'w') as f:
            pickle.dump( edges, f)
    	return edges

    # line graph trial
    def getAdjList_clique(self):

    	# clique list for line graph
        ext = ""
    	edge_clique = []
        print "####",len(self.trainPairs)
        print self.trainPairs
        trainPairs = self.trainPairs[["PairId","Q_id","A_id"]]
        print trainPairs
        #trainPairs["PairId"] = trainPairs["PairId"].apply(lambda x: int(x)-1)
        Part = trainPairs.groupby("Q_id")

        # group by Q_id
        for index in Part:
            if index[0]!='0':# and len(index[1])==2:
        	    for i in combinations(index[1]["PairId"].tolist(),2):
                        edge_clique.append(i)
    	# group by A_user
        #for index in secondPart:
        #    for i in combinations(index[1]["PairId"].tolist(),2):
        #            AdjList.append(i)

        edge_clique = pd.DataFrame(data=edge_clique, columns=["Nid1","Nid2"])
        edge_clique = edge_clique.sort_values(by=["Nid1","Nid2"], ascending=True).values

        #edge_clique = edge_clique.sort_values(by=["Nid1","Nid2"] , ascending=True).values

        with open(self.rootFolder + "/"+ 'edges_' + 'line' + '.pkl', 'w') as f:
            pickle.dump( edge_clique, f)

    	return edge_clique

    def __getitem__(self, idx):
        return np.concatenate(self.trainPairs.iloc[index,:], self.pairFeatures.iloc[index,:])



    def getAdjMatrix(self, X, type):

        fname = self.rootFolder + '/'+ 'edges_'+ type + '.pkl'
        print fname
        if os.path.isfile(fname):
            print "Read adj from file"
            with open(fname, 'r') as f:
                edges = pickle.load(f)
                #self.edges = edges
            print "Edges Read", edges
            return edges


        Feature_edges = dict()
        if type == "interaction":
            Users = X[['Q_user', 'A_user']]
            for index in range(len(Users)):
                row = Users.iloc[index,:]
                asker = str(row['Q_user'])
                commenter = str(row['A_user'])
                if asker not in Feature_edges:
                    Feature_edges[asker] = []
                    #Feature_edges[asker].append(asker)

                Feature_edges[asker].append(commenter)

        Total_edges = 0
        for key in Feature_edges:
            n = len(Feature_edges[key])
            Total_edges+=((n*(n-1))/2)


        edges = np.zeros((Total_edges,2))
        values = np.zeros((Total_edges,1))
        count = 0
        for key in sorted(Feature_edges.keys()):
            users = sorted(Feature_edges[key])
            #temp = []
            print "Starting ", key
            for i in range(len(users)):
                for j in range(i+1, len(users)):
                    edges[count][0] = users[i]
                    edges[count][1] = users[j]
                    values[count] = 1
                    count+=1
        print count
        #edges=edges[:count,:]
        if 'interaction' in type:
            with open(self.rootFolder + "/"+ 'edges_' + type + '.pkl', 'w') as f:
            	pickle.dump( edges, f)
        else:
            with open(self.rootFolder + "/"+ 'edges_' + type + '.pkl', 'w') as f:
            	pickle.dump( edges, f)

        return edges

    def getAdjMatrix_Identity(self, num_users):

        edges = []
        #for i in range(0, 10):
        #    edges = edges + [(i, j) for j in range(i, num_users/2)]

        #for i in range(0, num_users):
        edges = edges + [(i, i) for i in range(0, num_users)]
            #edges = edges + [(i,i)]

        edges = np.array(edges)
        edges = list(edges)
        return edges

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row,
                                          sparse_mx.col))).long()
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx, rowsum

def makeAdj(edges, N):
    edges = np.array(edges)
    #print edges
    #print N
    adj = sp.coo_matrix((np.ones(len(edges)),(edges[:, 0], edges[:, 1])),shape=(N, N),dtype=np.float32)


    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    adj, rowsum = normalize(adj)
    adj = sp.eye(adj.shape[0]) - adj
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    return adj, rowsum

def makeAdj2(edges, N):
    edges = np.array(edges)
    adj = sp.coo_matrix((np.ones(len(edges)),(edges[:, 0], edges[:, 1])),shape=(N, N),dtype=np.float32)


    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    adj, rowsum = normalize(adj)
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    return adj, rowsum

def makeAdj3(edges, N):
    edges = np.array(edges)
    adj = sp.coo_matrix((np.ones(len(edges)),(edges[:, 0], edges[:, 1])),shape=(N, N),dtype=np.float32)


    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    adj, rowsum = normalize(adj)
    adj = sp.eye(adj.shape[0]) + adj
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    return adj, rowsum

def get_GCN_inputs( edges, num_users):

    adj, rowsum = makeAdj(edges, num_users)
    #print "adjacency matrix", adj, adj_item
    return adj, rowsum

def get_GCN_inputs2( edges, num_users):

    adj, rowsum = makeAdj2(edges, num_users)
    #print "adjacency matrix", adj, adj_item
    return adj, rowsum

def get_GCN_inputs3( edges, num_users):

    adj, rowsum = makeAdj3(edges, num_users)
    #print "adjacency matrix", adj, adj_item
    return adj, rowsum


def parseHTML(Text):
    soup = BeautifulSoup(Text,"html5lib")
    codetags = soup.find_all('code')
    for codetag in codetags:
        codetag.extract()
    Text = re.sub(' +',' ',soup.get_text())
    Text = Text.replace("\t", " ")
    Text = Text.encode('ascii', 'ignore').decode('ascii')
    return Text

def parseHTMLText(inputFile, outputFile):

    with open(inputFile, "r") as f:
        with open(outputFile, "w") as fout:
            for line in f:
                id, QText, AText = line.strip().split("\t")
                QText = parseHTML(QText)
                AText = parseHTML(AText)
                fout.write(id + "\t" + str(QText) + "\t" + str(AText) + "\n")

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x, axis =1).reshape((x.shape[0],1)))
    return e_x / e_x.sum(axis=1).reshape((e_x.shape[0],1))

class SubjObjDataset(Dataset):

    def __init__(self, path, vectorizer, tokenizer=None, stopwords=None):
        self.corpus = pd.read_csv(path, sep='\t', quotechar="\"", encoding = 'utf-8', dtype= 'str')
        self.tokenizer = tokenizer
        self.vectorizer = vectorizer
        self.stopwords = stopwords
        self.pairIds = enumerate(sorted(np.unique(self.corpus['PairId'])))
        self._tokenize_corpus()
        if self.stopwords:
            self._remove_stopwords()
        self._vectorize_corpus()

    def _remove_stopwords(self):
        stopfilter = lambda doc: [word for word in doc if word not in self.stopwords]
        self.corpus['tokens'] = self.corpus['tokens'].apply(stopfilter)

    def _tokenize_corpus(self):
        #print self.corpus[['QBody', 'ABody']]
        #self.corpus['Body'] = self.corpus[['QBody', 'ABody']].apply(lambda x: ''.join(str(x)), axis=1)
        self.corpus['Body'] = self.corpus['QBody'].map(str) + self.corpus['ABody'].map(str)
        #print self.corpus['Body']
        if self.tokenizer:
            self.corpus['tokens'] = self.corpus['Body'].apply(self.tokenizer)
        else:
            #print self.corpus['ABody']
            self.corpus['tokens'] = self.corpus['Body'].apply(lambda x: str(x).lower().strip().split())
        #print self.corpus['tokens'] #, self.corpus['QBody'].iloc[1,:], self.corpus['ABody'].iloc[1,:]

    def _vectorize_corpus(self):
        if not self.vectorizer.vocabulary:
            self.vectorizer.fit(self.corpus['tokens'])
        self.corpus['vectors'] = self.corpus['tokens'].apply(self.vectorizer.transform_document)

    def __getitem__(self, index):
        sentence = self.corpus['vectors'].iloc[index]
        #target = [self.class2idx[self.corpus['labels'].iloc[index]]]
        return torch.LongTensor(sentence) #, torch.LongTensor(target)

    def __len__(self):
        return len(self.corpus)

def change(x):
    temp = x.split('_')
    return int(temp[0][1:]+temp[1][1:])
