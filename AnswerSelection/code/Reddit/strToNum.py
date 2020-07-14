import pandas as pd
import argparse
parser = argparse.ArgumentParser(description='Reddit Propressing')
parser.add_argument('--root', type=str, default='Docs')
args = parser.parse_args()
dirname = args.root

data = pd.read_csv(dirname+"/"+dirname+"Pair.tsv", sep='\t', quotechar="\"", encoding = 'utf-8', dtype = 'str')

NameList = data["Aer"].tolist()
NameList += data["Qer"].tolist()
NameList = list(set(NameList))

NameDict = dict()
for i,j in enumerate(NameList):
    NameDict[j] = i

data["Qer"] = data["Qer"].apply(lambda x: NameDict[x])
data["Aer"] = data["Aer"].apply(lambda x: NameDict[x])

QList = data["Qid"].tolist()
QList = list(set(QList))

QDict = dict()
for i,j in enumerate(QList):
    QDict[j] = i

AList = data["Aid"].tolist()
AList = list(set(AList))

ADict = dict()
for i,j in enumerate(AList):
    ADict[j] = i

data["Qid"] = data["Qid"].apply(lambda x: QDict[x])
data["Aid"] = data["Aid"].apply(lambda x: ADict[x])
data["Credible"] = data["Areputation"]
data[data.columns] = data[data.columns].apply(pd.to_numeric, errors='coerce')
data.fillna(0, inplace=True)
data["QASpan"] = data["Acreated_utc"] - data["Qcreated_utc"]
data["Adowns"] = data["Adowns"].apply(lambda x: int(x))
data["Aups"] = data["Aups"].apply(lambda x: int(x))
data["Qdowns"] = data["Qdowns"].apply(lambda x: int(x))
data["Qups"] = data["Qups"].apply(lambda x: int(x))
data["#comment"] = data["#comment"].apply(lambda x: int(x))
data["#link"] = data["#link"].apply(lambda x: int(x))
data["QASpan"] = data["#comment"].apply(lambda x: int(x))
data = data.drop(["Unnamed: 0","Areputation","Acreated_utc","Qcreated_utc"],axis=1)
data.to_csv(dirname+"/"+dirname+"Pair2.tsv", sep="\t", index="PairId")
