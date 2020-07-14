#Read scratch data and then write comment and submission file
"""
fout1.write("PairId\tReputation\tLastAccessDate\tViews\tNetworkAge\tAge\tUpvotes\tDownVotes\tGold\tSilver\tBronze\tNQuestions\tNAnswers\tAReputation\tALastAccessDate\tAViews\tANetworkAge\tAAge\tAUpvotes\tADownVotes\tAGold\tASilver\tABronze\tANQuestions\tANAnswers\tQScore\tQTags\tAScore\tQuestionAge\tQViewCount\tQAnswerCount\tQCommentCount\tAnswerAge\tACommentCount\n")
fout.write("PairId\tAskerId\tCommenterId\tQuestionId\tCommentId\tCredible\n")
fout2.write("PairId\tLocation\tAboutMe\tALocation\tAAboutMe\tQBody\tABody\tQTitle\n")

"""

import os
import argparse
import numpy as np
import pandas as pd
import re

parser = argparse.ArgumentParser(description='Data export in CrowdQM format')
parser.add_argument('--root', type=str, default='/scratch/uc_data/data')
parser.add_argument('--dataset', type=str, default='ai')

args = parser.parse_args()

file = os.path.join(args.root, args.dataset)
fname = file + "/"+ 'pairs_merge.tsv'
fname1 = file + "/"+ 'pairFeatures_merge.tsv'
#fname2 = "/home/knarang2/StackExchangeText/" + args.dataset + "/"+ 'pairText_merge_label.tsv'
fname2 = file + "/"+ 'pairText_merge.tsv'

comment = open("/home/knarang2/StackExchangeCrowdQM/" + args.dataset + "_comments.csv","w")
submission = open("/home/knarang2/StackExchangeCrowdQM/" + args.dataset + "_submissions.csv", "w")

from HTMLParser import HTMLParser

class MLStripper(HTMLParser):
    def __init__(self):
        self.reset()
        self.strict = False
        self.convert_charrefs= True
        self.fed = []
    def handle_data(self, d):
        self.fed.append(d)
    def get_data(self):
        return ''.join(self.fed)

def strip_tags(html):
    s = MLStripper()
    s.feed(html)
    return s.get_data()

def remove_code(text):

	if pd.isna(text):
		return ""
	if "</code>" not in text:
		return text

	cleaned_text = ""

	code_li =  [m.start() for m in re.finditer('<code>', text)]
	code_li.sort()

	code_close_li = [m.start() for m in re.finditer('<\/code>', text)]

	code_close_li.sort()

	# print(code_li, code_close_li)
	idx_close = 0

	start_pointer = 0

	for idx, end_pointer in enumerate(code_li):
		# print(idx_close)

		while start_pointer >= end_pointer and idx_close < len(code_close_li):
			start_pointer = code_close_li[idx_close]+len("</code>")
			idx_close += 1

		if idx_close < len(code_close_li):
			cleaned_text = cleaned_text + " " + text[start_pointer:end_pointer]
			start_pointer = code_close_li[idx_close]+len("</code>")
			if idx!= len(code_li)-1:
				idx_close+=1

	if idx_close < len(code_close_li):
		start_pointer = code_close_li[idx_close]+len("</code>")
		cleaned_text = cleaned_text + " " + text[start_pointer:]

	return cleaned_text

def tokenize(s):
	tokens = []
	for t in sent_tokenize(s):
		tokens.extend(word_tokenize(t))
	return tokens

def is_ascii(s):
	return all(ord(c) < 128 for c in s)

def preprocess_old(text):

    # Remove Code
    text = remove_code(text)
    # Remove HTML Tags
    #text = text.decode('ascii').strip()
    text = text.decode('utf-8').encode('ascii', 'ignore')
    text = strip_tags(text)
    text = text.replace("\"", "")
    #text_ascii = ''
    #for token in text:
    #    if is_ascii(token):
    #        text += token
    return text

if os.path.isfile(fname):
    print ("Read pairs from file")
    with open(fname, 'r') as f:
        trainPairs = pd.read_csv(fname, sep='\t', quotechar="\"", encoding = 'utf-8', dtype = 'str')
        trainPairs.loc[len(trainPairs)] = u'0'
        trainPairs.loc[len(trainPairs)-1]["PairId"] = unicode(str(len(trainPairs)))

        pairFeatures = pd.read_csv(fname1, sep='\t', quotechar="\"", encoding = 'utf-8', dtype = 'str')
        pairFeatures.loc[len(pairFeatures)] = u'0'
        pairFeatures.loc[len(pairFeatures)-1]["PairId"] = unicode(str(len(pairFeatures)))

        pairText = pd.read_csv(fname2, sep='\t', dtype = 'str', lineterminator='\n', error_bad_lines=False)
        pairText.loc[len(pairText)] = ','
        pairText.loc[len(pairText)-1]["PairId"] = unicode(str(len(pairText)))
        pairText.fillna('', inplace=True)

        def select(trainPairs, pairFeatures, pairText):
            know_list = []
            for i,j in trainPairs.groupby('QuestionId'):
                if "1" in j['Credible'].values and len(j['Credible'].values)>=2:# and len(j['Credible'].values)<=3:
                    know_list.append(i)
                else:
                    pass

            print ("# of question:",len(know_list))
            trainPairs = trainPairs[trainPairs['QuestionId'].isin(know_list)]
            # select the Pair
            Pair_list = trainPairs["PairId"].values
            ## select ###
            pairFeatures = pairFeatures[pairFeatures['PairId'].isin(Pair_list)]
            ## select ###
            pairText = pairText[pairText['PairId'].isin(Pair_list)]
            return trainPairs, pairFeatures, pairText

        trainPairs, pairFeatures, pairText = select(trainPairs, pairFeatures, pairText) #Select only q with accepted answer and >2 answers
        pairText['QBody'] = pairText['QBody'].apply(lambda x :x.replace("\"", ""))
        pairText['ABody'] = pairText['ABody'].apply(lambda x :x.replace("\"", ""))
        pairText['QTitle'] = pairText['QTitle'].apply(lambda x :x.replace("\"", ""))
        pairText['QBody_strip'] = pairText['QBody'].apply(lambda x :preprocess_old((x)))
        pairText['ABody_strip'] = pairText['ABody'].apply(lambda x :preprocess_old((x)))
        print ("# of Pairs",len(trainPairs))

        questions = list()
        #comment.write("id,id,author,name,downs,ups,id.1,created_utc,is_submitter,parent_id,submission_id,score,author_flair_text,banned_by,removal_reason,body_html,body,numberOfPostComments\n")
        #submission.write("id,id,submission_author,submission_id,downs,ups,id.1,permalink,score,author_flair_text,title,created_utc,edited,link_flair_text,selftext,selftext_html\n")

        comment.write("author,name,downs,ups,id,created_utc,is_submitter,parent_id,link_id,score,author_flair_text,banned_by,removal_reason,body_html,body\n")
        submission.write("author,name,downs,ups,id,permalink,score,author_flair_text,title,created_utc,edited,link_flair_text,selftext,selftext_html\n")


        for index, row in trainPairs.iterrows():
            id = str(row["PairId"])
            quesid = str(row["QuestionId"])
            ansid = str(row["CommentId"])
            #import pdb; pdb.set_trace()
            row_feat = pairFeatures.loc[int(id)-1]
            down = str(row_feat["ADownVotes"])
            ups = str(row_feat["AUpvotes"])
            down_ques = str(row_feat["ADownVotes"])
            ups_ques = str(row_feat["AUpvotes"])
            date = str(row_feat["AnswerAge"])
            date_ques = str(row_feat["QuestionAge"])
            row_Text = pairText.loc[int(id)-1]
            qbody_strip = str(row_Text["QBody_strip"])
            qbody = str(row_Text["QBody"])
            qtitle = str(row_Text["QTitle"])
            if len(qbody_strip) <= 1:
                qbody_strip = "UNK"
                #print(qbody_strip)
            if len(qbody) <= 1:
                qbody = "UNK"
            if len(qtitle) <= 1:
                qtitle = "UNK"

            #qbody = preprocess_old(row_Text["QBody"])
            #print("ORIGINAL ANSWER", str(row_Text["ABody"]))
            #print("PROCESSED", row_Text["ABody_strip"])
            #print(str(row_Text["ABody_strip"].decode('utf-8').strip()))

            comment.write(str(row["CommenterId"]) + "," + ansid +"," + down + "," + ups  + "," + ansid +"," + date + ",False," + quesid + "," + quesid + "," + str(int(ups)-int(down)) + ",This user has not yet been verified.,\"\",\"\"," + "\"" + str(row_Text["ABody"]) + "\"" + "," + "\"" + str(row_Text["ABody_strip"]) + "\"" "\n")

            if quesid not in questions:
                submission.write(str(row["AskerId"]) + "," + quesid + "," + down_ques + "," + ups_ques + "," + quesid + ",," + str(int(ups_ques) - int(down_ques)) +",This user has not yet been verified.," + "\"" + qtitle + "\"" + "," + date_ques + "," + date_ques + ",," + '"' + qbody_strip + "\"" + "," + '"' + qbody + "\"" + "\n")
                questions.append(quesid)

        comment.close()
        submission.close()
else:
    print("Error! Input file does not exitsts.")
