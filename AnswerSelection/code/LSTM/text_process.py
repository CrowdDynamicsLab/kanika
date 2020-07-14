import codecs
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
### modify ###
from nltk.tokenize.stanford import StanfordTokenizer
from nltk.stem.porter import *
from stemming.porter2 import stem
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
# import datefinder
import sys
reload(sys)
sys.setdefaultencoding("utf-8")
import os
import pandas as pd
import re

#from html.parser import HTMLParser
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


def is_ascii(s):
	return all(ord(c) < 128 for c in s)

stop = []
def getStopWords():
	slist = []

	swords = codecs.open(stopWordPath,"r","utf-8")
	for words in swords:
		word = words.strip("\r\n ")
		slist.append(word)
	return slist


def getnltkstops():
	from nltk.corpus import stopwords
	return stopwords.words('english')

def stemWord(word):
	r = ""
	try:
		r = stem(word)
	except:
		pass
		# print(word)
	return r
	#return stem(word)
	#return word


def tokenize(s):
	tokens = []
	for t in sent_tokenize(s):
		tokens.extend(word_tokenize(t))
	return tokens
	# return s.split(' ')

def stanford_tokenize(s):
	return StanfordTokenizer().tokenize(s)


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

def preprocess_old(text):

    # Remove Code
    text = remove_code(text)
    # print(text)
    # Remove HTML Tags
    text = strip_tags(text)
    # print(text)
    # Remove Paths
    # text = re.sub(r'\/',' ',text)
    # text = re.sub(r' ((\/|\\|\\\\)?([a-zA-Z]|[0-9]|[_])*(\/|\\|\\\\)([a-zA-Z]|[0-9]|[_])*(\/|\\|\\\\)?)+ ', ' 0PATH1 ', text)
    text = re.sub(r'(?:\s+|^)(\/|\\|\\\\)?([a-zA-Z0-9_]*(\/|\\|\\\\)){2,}[a-zA-Z0-9_]*(?:\s+|$)', ' 0PATH1 ', text)
    # print(text)

    # Remove DateTime
    # matches = list(datefinder.find_dates(text, index = True, source = True))

    # for match in matches:
    # 	# print(match[1])
    # 	text = text[:match[2][0]]+" 0DATE1TIME2 "+text[match[2][1]:]
    # 	# text.replace(match[1], "0DATE1TIME2")

    # print(text)

    # Remove URLS
    text = re.sub(r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+","0URL1",text)

    # print(text)

    # Remove Special Characters
    text = re.sub(r'\W+', ' ', text)

    # print(text)

    # Clear Whitespaces
    text = ' '.join(text.split())

    # print(text)

    # Tokenize
    _tokens = tokenize(text)
    stripstring = ',.?;\':"[]{}()!<>/|-+_=@#$& '
    # print(_tokens)
    #tokens = [stemWord(token.strip(stripstring).lower()) for token in _tokens if token.strip(stripstring).lower() not in stop and len(token.strip(stripstring).lower()) > 2 and is_ascii(token)]
    tokens = [lemmatizer.lemmatize(token.strip(stripstring).lower()) for token in _tokens if token.strip(stripstring).lower() not in stop and len(token.strip(stripstring).lower()) > 2 and is_ascii(token)]
    #tokens = [token.strip(stripstring).lower() for token in _tokens if token.strip(stripstring).lower() not in stop and len(token.strip(stripstring).lower()) > 2 and is_ascii(token)]
    #tokens = [token for token in _tokens if len(token)>2] #Changed
    return ' '.join(tokens)


def init():
	global stop
	stop = getnltkstops()
	punc = list(".!?'\"$;:,()[]{}")
	stop.extend(punc)

def preprocess(text,_type="T"):
	# return texts
	tokens = preprocess_old(text)
	s = ' '.join(tokens)
	# print(s)
	return s.strip()

dataset = "physics"

count = 1

#Read file into pandas, process and write
root = '/scratch/uc_data/data/'
filename = os.path.join(root, dataset, 'pairs_merge.tsv')
filenameText = os.path.join(root, dataset, 'pairText_merge.tsv')
directory = os.path.join('/home/knarang2/StackExchangeText/', dataset)
if not os.path.exists(directory):
    os.makedirs(directory)
output = os.path.join(directory, 'pairText_merge_label.tsv')
pairs = pd.read_csv(filename, sep='\t', header=0, lineterminator='\n', dtype= 'str', error_bad_lines=False)
pairText = pd.read_csv(filenameText, sep='\t', header=0, lineterminator='\n', dtype= 'str', error_bad_lines=False)
print("read data")

pairs_merge = pairs.merge(pairText, on='PairId')
pairs_merge['QBody'] = pairs_merge['QBody'].apply(lambda x :preprocess_old((x)))
print("Questions tokenized")
pairs_merge['ABody'] = pairs_merge['ABody'].apply(lambda x :preprocess_old(x))
print("Answers tokenized")
pairs_merge[['PairId', 'QBody', 'ABody', 'Credible']].to_csv(output, sep='\t', index=False, header=False)
print("Files merged")

with open(output, 'r') as f:
    for line in f:
        tokens = line.strip().split("\t")
        print(tokens)
        break
exit()


file = open('data/SST2/'+dataset+'-NLP.tsv','r')
file2 = open('data/SST2/'+dataset+'-NLP2.tsv','w')
data = file.readlines()
for line in data:
    print count
    count += 1
    temp = line.split('\t')
    file2.write('\t'.join([preprocess_old(remove_code(strip_tags(temp[0]))),
            preprocess_old(remove_code(strip_tags(temp[1]))), temp[2]]))
file2.close()
file.close()

print "train done!"

file = open('data/SST2/'+dataset+'-test.tsv','r')
file2 = open('data/SST2/'+dataset+'-test2.tsv','w')
data = file.readlines()
for line in data:
    print count
    count += 1
    temp = line.split('\t')
    file2.write('\t'.join([preprocess_old(remove_code(strip_tags(temp[0]))),
            preprocess_old(remove_code(strip_tags(temp[1]))), temp[2]]))
file2.close()
file.close()

print "test done!"
