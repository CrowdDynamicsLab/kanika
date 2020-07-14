import torch
from torch.autograd import Variable
import numpy as np
import scipy.sparse as sp
import pandas as pd
import pickle
import os
from bs4 import BeautifulSoup
import re
from torch.utils.data import Dataset, DataLoader
from itertools import combinations

class QADataset(Dataset):

    def __init__(self, dataset, rootFolder, questionFile, answerFile, userFile):
        self.dataset = dataset
        self.rootFolder = rootFolder
        self.questionFile = questionFile
        self.answerFile = answerFile
        self.userFile = userFile
        self.trainPairs = None
        self.testPairs = None
        self.ext = ''
        self.createPairs_separate()


    def load(self):
        print os.path.join(self.rootFolder, self.answerFile)
        self.questionFeatures = pd.read_csv(os.path.join(self.rootFolder, self.questionFile), sep='\t', lineterminator='\n', dtype= 'str', error_bad_lines=False, nrows=120)
        self.questionFeatures.dropna(axis=0, how='any', inplace=True)
        print("[#] QUESTION Loaded [#] Total number of question features : {}".format(
            len(self.questionFeatures)))
        print self.questionFeatures.keys()

        self.answerFeatures = pd.read_csv(os.path.join(self.rootFolder, self.answerFile), sep='\t', lineterminator='\n', dtype = 'str', error_bad_lines=False)
        self.answerFeatures.dropna(axis=0, how='any', inplace=True)
        print("[#] ANSWER Loaded [#] Total number of answer features : {}".format(
            len(self.answerFeatures)))
        print self.answerFeatures.keys()

        self.userFeatures = pd.read_csv(os.path.join(self.rootFolder, self.userFile), sep='\t', quotechar="\"", dtype = 'str')
        print("[#] USER Loaded [#] Total number of user features : {}".format(
            len(self.userFeatures)))
        print self.userFeatures.keys()

        #self.filterData(threshold=2)
        self.concatFeatures = pd.merge(self.questionFeatures, self.answerFeatures, left_on = 'Id', right_on = 'ParentId', suffixes=('_a', '_c')).merge(self.userFeatures, left_on = 'OwnerUserId_c', right_on = 'Id', suffixes=('_cx', '_cy')).merge(self.userFeatures, left_on = 'OwnerUserId_a', right_on = 'Id', suffixes=('_ax', '_ay'))
        print("[#] PAIRS Loaded [#] Total number of QA pairs : {}".format(
            len(self.concatFeatures)))
        print self.concatFeatures.keys()

    def __len__(self):
        print self.trainPairs.keys()
        return len(self.trainPairs)


    def filterData(self, threshold):

        self.questionFeatures.dropna(axis=0, how='any', inplace=True)
        NQuestion_asker = self.questionFeatures.groupby(['OwnerUserId']).size().reset_index(name='counts')
        NQuestion_asker = NQuestion_asker[NQuestion_asker['counts']>threshold]
        #print NQuestion_asker, NQuestion_asker['OwnerUserId'].tolist()
        Askers = NQuestion_asker['OwnerUserId'].tolist()
        self.questionFeatures = self.questionFeatures[self.questionFeatures['OwnerUserId'].isin(Askers)]  ##CHANGE HERE
        QuestionIds = self.questionFeatures['Id'].tolist()
        #print self.questionFeatures

        #exit()
        #Remove answers where ID == AskerId
        self.answerFeatures = self.answerFeatures[self.answerFeatures['ParentId'].isin(QuestionIds)]
        self.answerFeatures.dropna(axis=0, how='any', inplace=True)

        #Remove answerers with less than threshold
        NAnswer_comment = self.answerFeatures.groupby(['OwnerUserId']).size().reset_index(name='counts')
        #print NAnswer_comment
        NAnswer_comment = NAnswer_comment[NAnswer_comment['counts']>threshold]
        #print NAnswer_comment
        Commenters = NAnswer_comment['OwnerUserId'].tolist()
        self.answerFeatures = self.answerFeatures[self.answerFeatures['OwnerUserId'].isin(Commenters)]  ##CHANGE HERE
        #print self.answerFeatures

        Counts = self.answerFeatures.groupby(['ParentId']).size().reset_index(name='counts')['counts']
        #print np.mean(Counts.values)

    def createPairs_separate(self):

        #_filter_2

        fname = self.rootFolder + "/"+ 'pairs_merge' + self.ext + '.tsv'
        fname1 = self.rootFolder + "/"+ 'pairFeatures_merge' + self.ext +'.tsv'
        fname2 = self.rootFolder + "/"+ 'pairText_merge' + self.ext + '.tsv'
        fname3 = self.rootFolder + "/"+ 'pairText_merge_processed' + self.ext + '.tsv'
        #print fname

        if os.path.isfile(fname):
           print "Read adj from file"
           with open(fname, 'r') as f:
               self.trainPairs = pd.read_csv(fname, sep='\t', quotechar="\"", encoding = 'utf-8', dtype = 'str')
               self.pairFeatures = pd.read_csv(fname1, sep='\t', quotechar="\"", encoding = 'utf-8', dtype = 'str')
               #if not os.path.isfile(fname3):
            #       parseHTMLText(fname2, fname3)
               self.pairText = pd.read_csv(fname2, sep='\t', dtype = 'str', lineterminator='\n', error_bad_lines=False)
               #print self.pairText
               self.trainPairs_WFeatures = pd.merge(self.trainPairs, self.pairFeatures, on='PairId')
               #print self.trainPairs_WFeatures.keys()
               self.pairText.fillna('', inplace=True)
           return

        self.load()
        self.userFeatures.set_index('Id', inplace=True)

        count = 1
        fout = open(fname, "w")
        fout1 = open(fname1, "w")
        fout2 = open(fname2, "w")
        fout1.write("PairId\tReputation\tNetworkAge\tAge\tUpvotes\tDownVotes\tGold\tSilver\tBronze\tNQuestions\tNAnswers\tAReputation\tANetworkAge\tAAge\tAUpvotes\tADownVotes\tAGold\tASilver\tABronze\tANQuestions\tANAnswers\tQScore\tQTags\tAScore\n")
        fout.write("PairId\tAskerId\tCommenterId\tQuestionId\tCommentId\tCredible\n")
        fout2.write("PairId\tQBody\tABody\n")

        for idx in range(len(self.concatFeatures)):
            pair = self.concatFeatures.iloc[idx,:]
            if idx % 100 == 0:
                print idx, "records written"
            asker = pair['OwnerUserId_a']
            commenter = pair['OwnerUserId_c']
            quesId = pair['Id_a']
            commentId = pair['Id_c']
            accepted_answer = pair['AcceptedAnswerId']

            fout.write(str(count)+"\t"+str(asker)+"\t"+str(commenter)+"\t"+str(quesId)+"\t"+str(commentId)+"\t")
            fout2.write(str(count)+"\t"+pair["Body_a"]+"\t"+pair["Body_c"]+"\n")

            features = self.concatFeatures.iloc[idx,12:20].values
            features = [str(f) for f in features]

            features.append(str(len(self.concatFeatures.iloc[idx,20].split(","))) )
            features.append(str(len(self.concatFeatures.iloc[idx,21].split(","))) )
            feature_str='\t'.join(features)
            fout1.write(str(count)+"\t"+feature_str+"\t")

            #userFeatures = self.userFeatures[self.userFeatures["Id"] == str(asker)]
            userFeatures = self.userFeatures.loc[str(asker)]
            features = self.concatFeatures.iloc[idx,23:31].values
            features = [str(f) for f in features]

            Nques = len(self.concatFeatures.iloc[idx,31].split(","))
            features.append(str(Nques) if Nques > 1 else str(0))
            Nans = len(self.concatFeatures.iloc[idx,32].split(","))
            features.append(str(Nans) if Nans > 1 else str(0))
            feature_str='\t'.join(features)

            fout1.write(feature_str+"\t"+pair['Score_a']+"\t"+pair['Tags']+"\t"+pair['Score_c']+"\n")
            if str(pair['Id_c']) == str(accepted_answer):
                fout.write("1\n")
            else:
                fout.write("0\n")
            count+=1
            fout.flush()
            fout1.flush()
            fout2.flush()
            os.fsync(fout)
            os.fsync(fout1)
            os.fsync(fout2)
        fout.close()
        fout1.close()
        fout2.close()

        with open(fname, 'r') as f:
            self.trainPairs = pd.read_csv(fname, sep='\t', quotechar="\"", encoding = 'utf-8', dtype = 'str')
            self.pairFeatures = pd.read_csv(fname1, sep='\t', quotechar="\"", encoding = 'utf-8', dtype = 'str')
            self.pairText = pd.read_csv(fname2, sep='\t', dtype = 'str', lineterminator='\n', error_bad_lines=False)
            self.trainPairs_WFeatures = pd.merge(self.trainPairs, self.pairFeatures, on='PairId')
        return


    def getAdjList_lineGraph(self):

    	# adjacency list for line graph
    	AdjList = []


        # fname = self.rootFolder + '/'+ 'edges_'+ "line" + '.pkl'
        # print fname
        # if os.path.isfile(fname):
        #     print "Read adj from file"
        #     with open(fname, 'r') as f:
        #         edges = pickle.load(f)
        #         #self.edges = edges
        #     print "Edges Read", edges
        #     return edges

        trainPairs = self.trainPairs[["PairId","AskerId","CommenterId","Credible"]]
        trainPairs["PairId"] -= 1
        firstPart = trainPairs.groupby("AskerId")
        secondPart = trainPairs.groupby("CommenterId")
        # group by AskerId
        #for index in firstPart:
    	#    for i in combinations(index[1]["PairId"].tolist(),2):
        #            AdjList.append(i)

    	# group by CommenterId
        for index in secondPart:
            for i in combinations(index[1]["PairId"].tolist(),2):
                    AdjList.append(i)


        AdjList = pd.DataFrame(data=AdjList, columns=["Nid1","Nid2"])
        AdjList = AdjList.drop_duplicates()

        edges = AdjList.sort_values(by=["Nid1","Nid2"] , ascending=True).values

        with open(self.rootFolder + "/"+ 'edges_' + 'line' + self.ext + '.pkl', 'w') as f:
            pickle.dump( edges, f)
    	return edges

    def getAdjList_clique(self):

    	# clique list for line graph
    	edge_clique = []

        trainPairs = self.trainPairs[["PairId","QuestionId","CommentId"]]
        trainPairs["PairId"] -= 1
        Part = trainPairs.groupby("QuestionId")

        # group by QuestionId
        for index in Part:
    	    for i in combinations(index[1]["PairId"].tolist(),2):
                    edge_clique.append(i)
    	# group by CommenterId
        #for index in secondPart:
        #    for i in combinations(index[1]["PairId"].tolist(),2):
        #            AdjList.append(i)

        edge_clique = pd.DataFrame(data=edge_clique, columns=["Nid1","Nid2"]).values

        #edge_clique = edge_clique.sort_values(by=["Nid1","Nid2"] , ascending=True).values

        with open(self.rootFolder + "/"+ 'edges_' + 'line' + '.pkl', 'w') as f:
            pickle.dump( edge_clique, f)
    	return edge_clique


    def getAdjList_tags(self):

        fname = self.rootFolder + '/'+ 'edges_'+ "line_tags" + '.pkl'
        print fname
        if os.path.isfile(fname):
            print "Read adj from file"
            with open(fname, 'r') as f:
                edges = pickle.load(f)
                #self.edges = edges
            print "Edges Read", edges
            return edges
        data = self.pairFeatures[["PairId","QTags"]]
        data["PairId"] = data["PairId"].apply(lambda x: int(x)-1)
        data["QTag_vector"]=data["QTags"].apply(lambda x:x.split(","))
        QTags_data = dict()
        count = 0
        edges2 = []

        # construct dict
        for index, row in data.iterrows():
            QTags = row['QTag_vector']
            for tag in QTags:
                if tag not in QTags_data:
                    QTags_data[tag] = count
                    count+=1

        # construct the tag vector for each tag
        for index, row in data.iterrows():
            QTags = row['QTag_vector']
            for tag in QTags_data.keys():
                if tag in QTags:
                    try:
                        QTags_data[tag].append(row['PairId'])
                    except:
                        QTags_data[tag] = []
                        QTags_data[tag].append(row['PairId'])

        # give links
        for Tag in QTags_data.keys():
            for i in combinations(QTags_data[Tag],2):
                    edges2.append(map(int,i))

        edges2 = pd.DataFrame(data=edges2, columns=["Nid1","Nid2"])
        edges2 = edges2.drop_duplicates()

        edges2 = edges2.sort_values(by=["Nid1","Nid2"] , ascending=True).values
        with open(self.rootFolder + "/"+ 'edges_' + 'line_tags' + '.pkl', 'w') as f:
            pickle.dump( edges2, f)
        return edges2

    def getAdjList_allTags(self):

        fname = self.rootFolder + '/'+ 'edges_'+ "line_alltags" + '.pkl'
        print fname
        if os.path.isfile(fname):
            print "Read adj from file"
            with open(fname, 'r') as f:
                edges = pickle.load(f)
                #self.edges = edges
            print "Edges Read", edges
            return edges

        data = self.pairFeatures[["PairId","QTags"]]
        data["PairId"] = data["PairId"].apply(lambda x: int(x)-1)
        data["QTag_vector"] = data["QTags"].apply(lambda x:x.split(","))
        data["QTag_vector"] = data["QTag_vector"].apply(lambda x: ' '.join(sorted(list(x))))
        edges3 = []
        for index in data.groupby("QTag_vector"):
            for i in combinations(index[1]["PairId"].tolist(),2):
                    edges3.append(map(int,i))

        edges3 = pd.DataFrame(data=edges3, columns=["Nid1","Nid2"])
        edges3 = edges3.drop_duplicates()

        edges3 = edges3.sort_values(by=["Nid1","Nid2"] , ascending=True).values
        with open(self.rootFolder + "/"+ 'edges_' + 'line_alltags' + '.pkl', 'w') as f:
            pickle.dump( edges3, f)
        return edges3


    def __getitem__(self, idx):
        return np.concatenate(self.trainPairs.iloc[index,:], self.pairFeatures.iloc[index,:])

    def getTextFeatures(self, X):

        ##IF THERE IS A FILE JUST READ IT
        Train_Pairs = list(X['PairId'].values)
        Text_Features = np.zeros((len(Train_Pairs), 7))
        count = 0
        self.pairText["PairId"] = self.pairText["PairId"].apply(lambda x: int(x)-1)
        self.pairText.set_index('PairId', inplace=True)

        #print self.pairText.iloc[0,:]
        for pair in Train_Pairs:
            row = self.pairText.iloc[int(pair),:]

            QText = str(row['QBody'])
            AText = str(row['ABody'])

            Text_Features[count][0] = int(pair)
            Text_Features[count][1] = QText.count('<p>')
            Text_Features[count][2] = QText.count('<code>')
            Text_Features[count][3] = len(QText.split())

            Text_Features[count][4] = AText.count('<p>')
            Text_Features[count][5] = AText.count('<code>')
            Text_Features[count][6] = len(AText.split())
            count+=1


        Text_Features_df = pd.DataFrame(Text_Features, columns=['PairId', '#ParaCount', 'Code', '#Words', '#AParaCount', 'ACode', '#AWords'], dtype = 'float')

        #cols = Text_Features_df.columns
        #Text_Features_df[cols] = Text_Features_df[cols].apply(pd.to_numeric, errors='coerce')
        return Text_Features_df


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
            Users = X[['AskerId', 'CommenterId']]
            for index in range(len(Users)):
                row = Users.iloc[index,:]
                asker = str(row['AskerId'])
                commenter = str(row['CommenterId'])
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
        #print "Length of item edges are", len(edges)
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
    #print mx
    #print sp.eye(mx.shape[0]) - mx
    mx = sp.eye(mx.shape[0]) - mx
    #exit()
    return mx, rowsum

def makeAdj(edges, N):
    edges = np.array(edges)
    adj = sp.coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])),
                        shape=(N, N),
                        dtype=np.float32)


    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    #print "With their method", adj_[0]
    #adj = adj + adj.T
    #adj = np.ones((num_users, num_users))
    #adj, rowsum = normalize(adj + sp.eye(adj.shape[0]))
    adj, rowsum = normalize(adj)
    #print np.sum(adj, axis=1)
    #print adj.sum(1)
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    return adj, rowsum


def get_GCN_inputs( edges, num_users):

    adj, rowsum = makeAdj(edges, num_users)
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
