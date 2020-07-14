from time import sleep
file = open('../movie-data/check.txt','r')
temp = []
for item in file.readlines():
    temp.append(item[:-1].split(' ')[1:])
print temp[:10]
file.close()

file = open('../movie-data/movie_full_map.txt','r')
temp2 = []
for item in file.readlines():
    a = item[:-1].split('\t')[1]
    b = item[:-1].split('\t')[2]
    temp2.append([b,a])
print temp2[:10]
file.close()

print len(temp)
print len(temp2)

#count = 0
for i in temp:
    if i not in temp2:
        print i,
