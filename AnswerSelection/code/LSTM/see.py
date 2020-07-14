dataset = "bitcoin"
file = open("data/SST2/"+dataset+"-NLP.tsv",'r')
file2 = open("data/SST2/"+dataset+"-train.tsv",'w')
file3 = open("data/SST2/"+dataset+"-test.tsv",'w')

rate = 0.8
data = file.readlines()

for line in data[:int(len(data)*rate)]:
    file2.write(line)
for line in data[int(len(data)*rate):]:
    file3.write(line)

file.close()
file2.close()
file3.close()
