import numpy as np
import scipy.sparse as sp
import pandas as pd
import pickle
import torch
from sklearn import preprocessing
import scipy.stats as ss

def getPostContexts(data, dataset):

    """zecheng
    file = open("../movie-data/struc_movie.txt",'r')
    embeding = file.readlines()
    embed_dict = dict()
    for line in embeding:
        emb = map(float,line[:-1].split(" "))
        embed_dict[int(emb[0])]=emb[1:]
    data['CommentId'] = data['CommentId'].apply(lambda x: embed_dict[int(x)])

    embed_vec = data["CommentId"]
    """

    # remove some columns
    #data = data.drop(['QScore','AScore','NetworkAge', 'ANetworkAge','Age', 'AAge', 'QTags', 'AskerId', 'CommenterId', 'CommentId','Reputation','LastAccessDate','Views','Upvotes','DownVotes','Gold','Silver','Bronze','NQuestions','NAnswers','AReputation','ALastAccessDate','AViews','AUpvotes','ADownVotes','AGold','ASilver','ABronze','ANQuestions','ANAnswers','QuestionAge','AnswerAge','QAnswerCount',"ArrivalRank"], axis = 1)
    data = data.drop(['QScore','AScore','NetworkAge', 'ANetworkAge','Age', 'AAge', 'QTags', 'AskerId', 'CommenterId', 'CommentId','Reputation','LastAccessDate','Views','Upvotes','DownVotes','Gold','Silver','Bronze','NQuestions','NAnswers','AReputation','ALastAccessDate','AViews','AUpvotes','ADownVotes','AGold','ASilver','ABronze','ANQuestions','ANAnswers','QuestionAge','AnswerAge','QAnswerCount'], axis = 1)

    """
    # get text file
    get_Text = dataset.getText(data)
    data1 = data.merge(get_Text, on='PairId')
    file = open("dataset-NLP.tsv",'w')
    for i in range(len(data1)):
        row = data1.iloc[i,:]
        file.write(str(row["Q"])+'\t'+str(row["A"])+'\t'+str(row["Credible"])+'\n')
    file.close()
    exit()
    """



    Text_Features_df = dataset.getTextFeatures(data)
    data = data.merge(Text_Features_df, on='PairId')
    Feature_copy = data

    data = data.drop(['Credible'], axis = 1)
    data.set_index("PairId", inplace=True)

    """
    # normalize within the clique
    X_Text_scaled = data.values
    data.at[:,:] = X_Text_scaled
    import scipy.stats as ss
    for i,j in data.groupby('QuestionId'):
        print i,
        temp = data.loc[data['QuestionId']==i,"QASpan"]
        data.at[data['QuestionId']==i,"QASpan"] = ss.rankdata(temp)
    """

    print ("########### feature: ###############")
    print (data.iloc[0])

    data = data.drop(['QuestionId'], axis = 1)
    X_Text_scaled = data.values
    X_Text_scaled = np.array(X_Text_scaled)
    X_Text_scaled = (X_Text_scaled-X_Text_scaled.mean(axis=0))/X_Text_scaled.std(axis=0)
    Feature = torch.FloatTensor(X_Text_scaled)

    return Feature,Feature_copy

def getPostContentEmbed(data, dataset):

    # remove some columns
    #data = data.drop(['QScore','AScore','NetworkAge', 'ANetworkAge','Age', 'AAge', 'QTags', 'AskerId', 'CommenterId', 'CommentId','Reputation','LastAccessDate','Views','Upvotes','DownVotes','Gold','Silver','Bronze','NQuestions','NAnswers','AReputation','ALastAccessDate','AViews','AUpvotes','ADownVotes','AGold','ASilver','ABronze','ANQuestions','ANAnswers','QuestionAge','AnswerAge','QAnswerCount',"ArrivalRank"], axis = 1)
    data = data.drop(['QScore','AScore','NetworkAge', 'ANetworkAge','Age', 'AAge', 'QTags', 'AskerId', 'CommenterId', 'CommentId','Reputation','LastAccessDate','Views','Upvotes','DownVotes','Gold','Silver','Bronze','NQuestions','NAnswers','AReputation','ALastAccessDate','AViews','AUpvotes','ADownVotes','AGold','ASilver','ABronze','ANQuestions','ANAnswers','QuestionAge','AnswerAge','QAnswerCount'], axis = 1)

    dir = "/home/knarang2/StackExchangeDataEmbed/" + dataset
    embeddings = np.load(dir + "/dataEmbeddings.npy")
    print("Content Embeddings", embeddings.shape)

    #embeddings_new = embeddings[:,:50] *  embeddings[:,50:]
    #print("Content Embeddings", embeddings_new.shape)
    #import pdb; pdb.set_trace()
    #Text_Features_df = dataset.getTextFeatures(data)
    #data = data.merge(Text_Features_df, on='PairId')
    Feature_copy = data

    #data["contentembed"] = embeddings_new.tolist()
    data["contentembed"] = embeddings.tolist()
    data = data[["PairId", "contentembed"]]
    #data = data.drop(['Credible'], axis = 1)
    data.set_index("PairId", inplace=True)

    print ("########### feature: ###############")
    print (data.iloc[0])

    #X_Text_scaled = data.values
    #X_Text_scaled = np.array(X_Text_scaled)
    X_Text_scaled = np.array(data.values.tolist())[:,0,:]
    X_Text_scaled = (X_Text_scaled-X_Text_scaled.mean(axis=0))/X_Text_scaled.std(axis=0)
    Feature = torch.FloatTensor(X_Text_scaled)


    return Feature,Feature_copy


def getPostUserReliability(data, dataset):

    # remove some columns
    #data = data.drop(['QScore','AScore','NetworkAge', 'ANetworkAge','Age', 'AAge', 'QTags', 'AskerId', 'CommenterId', 'CommentId','Reputation','LastAccessDate','Views','Upvotes','DownVotes','Gold','Silver','Bronze','NQuestions','NAnswers','AReputation','ALastAccessDate','AViews','AUpvotes','ADownVotes','AGold','ASilver','ABronze','ANQuestions','ANAnswers','QuestionAge','AnswerAge','QAnswerCount',"ArrivalRank"], axis = 1)
    data = data.drop(['QScore','AScore','NetworkAge', 'ANetworkAge','Age', 'AAge', 'QTags', 'AskerId', 'CommentId','Reputation','LastAccessDate','Views','Upvotes','DownVotes','Gold','Silver','Bronze','NQuestions','NAnswers','AReputation','ALastAccessDate','AViews','AUpvotes','ADownVotes','AGold','ASilver','ABronze','ANQuestions','ANAnswers','QuestionAge','AnswerAge','QAnswerCount'], axis = 1)

    dir = "/home/knarang2/StackExchangeCrowdQM"
    relData = pd.read_csv(dir + "/" + dataset + "_user_com_weights.csv")
    print("Reliability", relData.shape)

    #import pdb; pdb.set_trace();

    data = data.merge(relData, how='left', left_on=['CommenterId', 'QuestionId'], right_on=['PrevUserID', 'QuestionId'])
    data = data.replace(np.nan, 0.000001)
    Feature_copy = data

    data = data[["PairId", "R_mn"]]
    #data = data.drop(['Credible'], axis = 1)
    data.set_index("PairId", inplace=True)

    print ("########### feature: ###############")
    print (data.iloc[0])
    #import pdb; pdb.set_trace()
    X_Text_scaled = np.array(data.values)
    #X_Text_scaled = (X_Text_scaled-X_Text_scaled.mean(axis=0))/X_Text_scaled.std(axis=0)
    Feature = torch.FloatTensor(X_Text_scaled)
    print(Feature)

    return Feature,Feature_copy
