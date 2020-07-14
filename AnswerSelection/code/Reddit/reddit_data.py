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

    def __init__(self, dataset, rootFolder, questionFile, answerFile, userFile):
        self.dataset = dataset
        self.rootFolder = rootFolder
        self.questionFile = questionFile
        self.answerFile = answerFile
        self.userFile = userFile
        self.trainPairs = None
        self.testPairs = None
        self.embedding = None
        self.Rating = None
        #self.createPairs_separate()
        self.createPairs_separate2()

    def createPairs_separate2(self):
        fname = self.rootFolder + "/"+ self.rootFolder+'Pair2.tsv'
        print "Read pairs from file"
        with open(fname, 'r') as f:
            self.trainPairs = pd.read_csv(fname, sep='\t', quotechar="\"", dtype = 'str')
            self.trainPairs = self.trainPairs.drop(["Unnamed: 0","created_utc_account"],axis=1)
            self.trainPairs = self.trainPairs.astype("int")
            self.trainPairs = self.trainPairs[self.trainPairs['QASpan']>0]
        know_list = []
        for i,j in self.trainPairs.groupby('Qid'):
            if 1 in j['Credible'].values and len(j['Credible'].values)>=2:# and len(j['Credible'].values)<=3:
                know_list.append(i)
            else:
                pass
        self.trainPairs = self.trainPairs[self.trainPairs['Qid'].isin(know_list)]
        new_ID = [str(i+1) for i in range(len(self.trainPairs))]
        self.trainPairs['PairId'] = new_ID

        print "# of Pairs",len(self.trainPairs)
        print "# of Qs", len(know_list)

        return

    def load(self):
        print os.path.join(self.rootFolder, self.answerFile)
        self.questionFeatures = pd.read_csv(os.path.join(self.rootFolder, self.questionFile), sep='\t', lineterminator='\n', dtype= 'str', error_bad_lines=False)
        print self.questionFeatures.keys()

        self.answerFeatures = pd.read_csv(os.path.join(self.rootFolder, self.answerFile), sep='\t', lineterminator='\n', dtype = 'str', error_bad_lines=False)
        print self.answerFeatures.keys()

        self.userFeatures = pd.read_csv(os.path.join(self.rootFolder, self.userFile), sep='\t', lineterminator='\n', dtype = 'str', error_bad_lines=False)

        print self.userFeatures.keys()

        self.concatFeatures = pd.merge(self.questionFeatures, self.answerFeatures, left_on = 'Id', right_on = 'ParentId', suffixes=('_a', '_c')).merge(self.userFeatures, left_on = 'OwnerUserId_c', right_on = 'Id', suffixes=('_cx', '_cy')).merge(self.userFeatures, left_on = 'OwnerUserId_a', right_on = 'Id', suffixes=('_ax', '_ay'))
        print self.concatFeatures.keys()


    def __len__(self):
        return len(self.trainPairs)

    def select(self):
        know_list = []
        for i,j in self.trainPairs.groupby('Qid'):
            if "1" in j['Credible'].values and len(j['Credible'].values)>=2:# and len(j['Credible'].values)<=3:
                know_list.append(i)
            else:
                pass

        print "# of question:",len(know_list)
        self.trainPairs = self.trainPairs[self.trainPairs['Qid'].isin(know_list)]
        # select the Pair
        Pair_list = self.trainPairs["PairId"].values
        ## select ###
        self.pairFeatures = self.pairFeatures[self.pairFeatures['PairId'].isin(Pair_list)]
        ## select ###
        self.pairText = self.pairText[self.pairText['PairId'].isin(Pair_list)]

    # remove the answer, which is later than the accepted one.
    def filter_time(self):
        know_pairList = []
        for i,j in self.trainPairs_WFeatures.groupby('Qid'):
            if "1" in j['Credible'].values:
                know_pairList += j[j['AnswerAge']>=j[j['Credible']=='1']['AnswerAge'].values[0]]["PairId"].values.tolist()
        self.trainPairs = self.trainPairs[self.trainPairs['PairId'].isin(know_pairList)]
        self.pairFeatures = self.pairFeatures[self.pairFeatures['PairId'].isin(know_pairList)]
        self.pairText = self.pairText[self.pairText['PairId'].isin(know_pairList)]
        self.trainPairs_WFeatures = self.trainPairs_WFeatures[self.trainPairs_WFeatures['PairId'].isin(know_pairList)]

    # if the question has an accepted answer, then the answer for the question will get label 1, and otherwise 0
    def add_label(self):
        know_list = []
        for i,j in self.trainPairs.groupby('Qid'):
            if "1" in j['Credible'].values:
                know_list.append(i)
            else:
                pass
        self.trainPairs['Label'] = self.trainPairs['Qid']
        self.trainPairs['Label'] = self.trainPairs['Label'].apply(lambda x: 1 if x in know_list else 0)

    # reindex the pairs
    def change_ID(self):
        new_ID = [str(i+1) for i in range(len(self.trainPairs))]
        self.trainPairs['PairId'] = new_ID
        self.mapping = self.trainPairs[["PairId","Qid","Aid"]]
        self.pairFeatures['PairId'] = new_ID
        self.pairText['PairId'] = new_ID
        self.trainPairs_WFeatures['PairId'] = new_ID

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


    # create negative sample
    def filterDataset(self):
        tempQuestionTime = self.pairFeatures[["PairId","QuestionAge"]]
        tempPairMerge = pd.merge(self.trainPairs, tempQuestionTime, on='PairId')
        QidList = []
        for Qer, content in tempPairMerge.groupby("Qer"):
            TempList = sorted([(i, j) for i,j in content.groupby("Qid")], key=lambda x: x[1]["QuestionAge"].values[0],reverse=True)
            TempList = [(i,"1" in j["Credible"].values) for i,j in TempList]
            for i,j in enumerate(TempList):
                if j[1]:
                    QidList += [x for x,y in TempList[i:]]
                    break
        QidList.append("0")
        # the remaining question
        #print QidList
        ## select ##
        self.trainPairs = self.trainPairs[self.trainPairs['Qid'].isin(QidList)]
        # select the Pair
        Pair_list = self.trainPairs["PairId"].values
        ## select ###
        self.pairFeatures = self.pairFeatures[self.pairFeatures['PairId'].isin(Pair_list)]
        ## select ###
        self.pairText = self.pairText[self.pairText['PairId'].isin(Pair_list)]


    def createPairs_separate(self):
        ext = ''
        fname = self.rootFolder + "/"+ 'pairs_merge' + ext + '.tsv'
        fname1 = self.rootFolder + "/"+ 'pairFeatures_merge' + ext +'.tsv'
        fname2 = self.rootFolder + "/"+ 'pairText_merge' + ext + '.tsv'
        if os.path.isfile(fname):
           print "Read pairs from file"
           with open(fname, 'r') as f:
               self.trainPairs = pd.read_csv(fname, sep='\t', quotechar="\"", encoding = 'utf-8', dtype = 'str')
               self.trainPairs.loc[len(self.trainPairs)] = u'0'
               self.trainPairs.loc[len(self.trainPairs)-1]["PairId"] = unicode(str(len(self.trainPairs)))

               self.pairFeatures = pd.read_csv(fname1, sep='\t', quotechar="\"", encoding = 'utf-8', dtype = 'str')
               self.pairFeatures.loc[len(self.pairFeatures)] = u'0'
               self.pairFeatures.loc[len(self.pairFeatures)-1]["PairId"] = unicode(str(len(self.pairFeatures)))

               self.pairText = pd.read_csv(fname2, sep='\t', dtype = 'str', lineterminator='\n', error_bad_lines=False)
               self.pairText.loc[len(self.pairText)] = ','
               self.pairText.loc[len(self.pairText)-1]["PairId"] = unicode(str(len(self.pairText)))
               self.pairText.fillna('', inplace=True)
               #self.filterDataset()
               self.select()
               #self.add_label()
               self.trainPairs_WFeatures = pd.merge(self.trainPairs, self.pairFeatures, on='PairId')

               ### change questionAge
               compensate = pd.read_csv(self.rootFolder+"/"+"QuestionFeatures.tsv", sep='\t', dtype = 'str', lineterminator='\n', error_bad_lines=False)
               questionAgeDict = dict()
               for i in range(len(compensate)):
                   questionAgeDict[compensate.loc[i,"Id"]] = compensate.loc[i,"QuestionAge"]
               self.trainPairs_WFeatures['QuestionAge'] = self.trainPairs_WFeatures['Qid'].apply(lambda x: int(questionAgeDict[x]))
               self.trainPairs_WFeatures['QuestionAge'] = self.trainPairs_WFeatures['QuestionAge'].apply(lambda x: int(x))

               ### change answerAge
               compensate = pd.read_csv(self.rootFolder+"/"+"AnswerFeatures.tsv", sep='\t', dtype = 'str', lineterminator='\n', error_bad_lines=False)
               answerAgeDict = dict()
               for i in range(len(compensate)):
                   answerAgeDict[compensate.loc[i,"Id"]] = compensate.loc[i,"AnswerAge"]
               self.trainPairs_WFeatures['AnswerAge'] = self.trainPairs_WFeatures['Aid'].apply(lambda x: int(answerAgeDict[x]))
               self.trainPairs_WFeatures['AnswerAge'] = self.trainPairs_WFeatures['AnswerAge'].apply(lambda x: int(x))

               self.trainPairs_WFeatures["QASpan"] = self.trainPairs_WFeatures['QuestionAge'] - self.trainPairs_WFeatures['AnswerAge']

               self.change_ID()
               print "# of Pairs",len(self.trainPairs)
           return

        self.load()
        self.userFeatures.set_index('Id', inplace=True)

        # write the files from original tsv
        count = 1
        fout = open(fname, "w")
        fout1 = open(fname1, "w")
        fout2 = open(fname2, "w")
        fout1.write("PairId\tReputation\tLastAccessDate\tViews\tNetworkAge\tAge\tUpvotes\tDownVotes\tGold\tSilver\tBronze\tNQuestions\tNAnswers\tAReputation\tALastAccessDate\tAViews\tANetworkAge\tAAge\tAUpvotes\tADownVotes\tAGold\tASilver\tABronze\tANQuestions\tANAnswers\tQScore\tQTags\tAScore\tQuestionAge\tQViewCount\tQAnswerCount\tQCommentCount\tAnswerAge\tACommentCount\n")
        fout.write("PairId\tQer\tAer\tQid\tAid\tCredible\n")
        fout2.write("PairId\tLocation\tAboutMe\tALocation\tAAboutMe\tQBody\tABody\tQTitle\n")

        for idx in range(len(self.concatFeatures)):
            pair = self.concatFeatures.iloc[idx,:]
            if idx % 100 == 0:
                print idx, "records written"
            asker = pair['OwnerUserId_a']
            commenter = pair['OwnerUserId_c']
            quesId = pair['Id_a']
            Aid = pair['Id_c']
            accepted_answer = pair['AcceptedAnswerId']
            location_asker = pair['Location_ax']
            location_commenter = pair['Location_ay']
            aboutme_asker = pair['AboutMe_ax']
            aboutme_commenter = pair['AboutMe_ay']
            title = pair['Title']

            fout.write(str(count)+"\t"+str(asker)+"\t"+str(commenter)+"\t"+str(quesId)+"\t"+str(Aid)+"\t")
            fout2.write(str(count)+"\t"+str(location_asker)+'\t'+str(aboutme_asker)+"\t"+str(location_commenter)+'\t'+str(aboutme_commenter)+'\t'+str(pair["Body_a"])+"\t"+ str(pair["Body_c"])+'\t'+str(title)+"\n")

            feature_columns = ["Reputation_ax","LastAccessDate_ax","Views_ax","NetworkAge_ax","Age_ax",
            "UpVotes_ax","DownVotes_ax","Gold_ax","Silver_ax","Bronze_ax","Questions_ax","Answers_ax",
            "Reputation_ay","LastAccessDate_ay","Views_ay","NetworkAge_ay","Age_ay","UpVotes_ay",
            "DownVotes_ay","Gold_ay","Silver_ay","Bronze_ay","Questions_ay","Answers_ay","Score_a","Tags",
            "Score_c","QuestionAge","ViewCount","AnswerCount","CommentCount_a","AnswerAge","CommentCount_c"]

            features = self.concatFeatures[feature_columns].iloc[idx]
            features["Questions_ax"] = len(features["Questions_ax"].split(","))
            features["Answers_ax"] = len(features["Answers_ax"].split(","))
            features["Questions_ay"] = len(features["Questions_ay"].split(","))
            features["Answers_ay"] = len(features["Answers_ay"].split(","))
            features = features.values
            features = [str(f) for f in features]

            feature_str='\t'.join(features)
            fout1.write(str(count)+"\t"+feature_str+"\n")

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
            self.pairText = pd.read_csv(fname2, sep='\t',  dtype = 'str', lineterminator='\n', error_bad_lines=False)
            self.select()
            self.trainPairs_WFeatures = pd.merge(self.trainPairs, self.pairFeatures, on='PairId')
            self.change_ID()
            print "#Pairs",len(self.trainPairs)
        return

    # group by Aer
    def getAdjList_Similarity1(self):

    	AdjList = []
        ext = ''
        secondPart = self.trainPairs.groupby(["Aer"])

    	# group by Aer
        for index in secondPart:
            if len(index[1])>=5:
                x = index[1].sort_values(by=["QASpan"] , ascending=True)["PairId"].tolist()
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
        coeff = 0.995

        self.trainPairs["ArrivalRank"] =  self.trainPairs["QASpan"]
        import scipy.stats as ss
        print 1
        for i,j in self.trainPairs.groupby('Qid'):
            temp = self.trainPairs.loc[self.trainPairs['Qid']==i,"ArrivalRank"]
            self.trainPairs.at[self.trainPairs['Qid']==i,"ArrivalRank"] = ss.rankdata(temp)


        self.trainPairs["Link"] = [0 for i in range(len(self.trainPairs))]
        step2 = self.trainPairs.groupby(["Qid"])
        idxList1 = []
        idxList2 = []
        print 2
        for i,j in step2:
            tempScore = sorted(j["QASpan"].values.tolist())
            if tempScore[-1]-tempScore[-2]>=(j["QASpan"].values.max()-j["QASpan"].values.min())*coeff:
                idxList1.append(j["QASpan"].idxmax())
            if tempScore[1]-tempScore[0]>=(j["QASpan"].values.max()-j["QASpan"].values.min())*coeff:
                idxList2.append(j["QASpan"].idxmin())
        self.trainPairs.loc[idxList1,"Link"] = 1
        self.trainPairs.loc[idxList2,"Link"] = 2
        print 3
        step3 = self.trainPairs[self.trainPairs["Link"].isin([1,2])].groupby(["Aer","Link"])

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
        # compute trueSkill value
        RatingDict = dict()
        for i in trainPairs["Aer"].drop_duplicates().values.tolist():
            RatingDict[i] = (Rating(),)
        step1 = trainPairs.groupby(["Qid"])
        for i,j in step1:
            if int(i) in QuestionList:
                IdList = j["Aer"].values.tolist()
                rating = j["Credible"].apply(lambda x: 1-int(x)).values.tolist()
                for i,j in zip(IdList,rate([RatingDict[x] for x in IdList],ranks=rating)):
                    RatingDict[i] = j
        trainPairs["Rating"] = trainPairs["Aer"]
        trainPairs["Rating"] = trainPairs["Rating"].apply(lambda x: RatingDict[x][0].mu-3*RatingDict[x][0].sigma)
        trainPairs["Link"] = [0 for i in range(len(trainPairs))]

        self.Rating = trainPairs[["PairId","Rating"]]

        # create clique using trueSkill value
        step2 = trainPairs.groupby(["Qid"])
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
        step3 = trainPairs[trainPairs["Link"].isin([1,2])].groupby(["Aer","Link"])
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
        trainPairs = self.trainPairs[["PairId","Qid","Aid"]]
        Part = trainPairs.groupby("Qid")

        # group by Qid
        for index in Part:
    	    for i in combinations(index[1]["PairId"].tolist(),2):
                    edge_clique.append(i)
    	# group by Aer
        #for index in secondPart:
        #    for i in combinations(index[1]["PairId"].tolist(),2):
        #            AdjList.append(i)

        edge_clique = pd.DataFrame(data=edge_clique, columns=["Nid1","Nid2"])
        edge_clique = edge_clique.sort_values(by=["Nid1","Nid2"], ascending=True).values

        #edge_clique = edge_clique.sort_values(by=["Nid1","Nid2"] , ascending=True).values

        with open(self.rootFolder + "/"+ 'edges_' + 'line' + '.pkl', 'w') as f:
            pickle.dump( edge_clique, f)

    	return edge_clique

    # graph, at least one tag should be the same
    def getAdjList_tags(self):

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
        with open(self.rootFolder + "/"+ 'edges_' + 'line' + '.pkl', 'w') as f:
            pickle.dump( edges2, f)
        return edges2

    # graph, all tags should be the same
    def getAdjList_allTags(self):

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
        with open(self.rootFolder + "/"+ 'edges_' + 'line' + '.pkl', 'w') as f:
            pickle.dump( edges3, f)
        return edges3

    def __getitem__(self, idx):
        return np.concatenate(self.trainPairs.iloc[index,:], self.pairFeatures.iloc[index,:])

    def getTextFeatures(self, X):

          Train_Pairs = list(X['PairId'].values)
          Text_Features = np.zeros((len(Train_Pairs), 10))
          count = 0
          try:
              self.pairText["PairId"] = self.pairText["PairId"].apply(lambda x: int(x)-1)
              self.pairText.set_index('PairId', inplace=True)
          except:
              pass

          #print self.pairText.iloc[0,:]
          for pair in Train_Pairs:
              row = self.pairText.iloc[int(pair),:]

              QText = str(row['QBody'])
              AText = str(row['ABody'])
              Title = str(row['QTitle'])
              Q_Aboutme = str(row['AboutMe'])
              A_Aboutme = str(row['AAboutMe'])

              Text_Features[count][0] = int(pair)
              Text_Features[count][1] = QText.count('<p>')
              Text_Features[count][2] = QText.count('<code>')
              Text_Features[count][3] = len(QText.split())

              Text_Features[count][4] = AText.count('<p>')
              Text_Features[count][5] = AText.count('<code>')
              Text_Features[count][6] = len(AText.split())

              Text_Features[count][7] = len(Title.split())
              Text_Features[count][8] = len(Q_Aboutme.split())
              Text_Features[count][9] = len(A_Aboutme.split())

              count+=1

          Text_Features_df = pd.DataFrame(Text_Features, columns=['PairId', '#ParaCount', 'Code', '#Words', '#AParaCount', 'ACode', '#AWords', "#Title", "#Q_Aboutme", "#A_Aboutme"], dtype = 'float')
          return Text_Features_df

    def getText(self, X):

          Train_Pairs = list(X['PairId'].values)
          Text_List = []
          self.pairText["PairId"] = self.pairText["PairId"].apply(lambda x: int(x)-1)
          self.pairText.set_index('PairId', inplace=True)

          #print self.pairText.iloc[0,:]
          for pair in Train_Pairs:
              row = self.pairText.iloc[int(pair),:]
              temp = []

              QText = str(row['QBody'])
              AText = str(row['ABody'])

              temp.append(int(pair))
              temp.append(QText)
              temp.append(AText)

              Text_List.append(temp)

          Text = pd.DataFrame(Text_List, columns=['PairId', 'Q', 'A'])
          return Text


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
            Users = X[['Qer', 'Aer']]
            for index in range(len(Users)):
                row = Users.iloc[index,:]
                asker = str(row['Qer'])
                commenter = str(row['Aer'])
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

def get_GCN_inputs(edges, num_users):

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
