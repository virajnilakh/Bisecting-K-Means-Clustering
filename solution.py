import operator
from math import exp, log
from random import randint
from sklearn.decomposition import TruncatedSVD
import time
import random
from pathlib import Path
import pickle

pickleFile = Path("./pickleFile.txt")
pickleFile1 = Path("./pickleFile1.txt")

# if my_file.is_file():
startTime = time.time()
print("Start time is:" + str(startTime))
with open('./train.dat', 'r') as fh:
    lines = fh.readlines()
train = []
min = 9999
max = 0
docFreq = {}
wordsInDoc = []
pointToClusterMap = {}

if pickleFile.is_file():
    with open("pickleFile.txt", "rb") as myFile:
        print("Loaing pickle!!")
        tfidf = pickle.load(myFile)
else:
    print("Creating pickle!!")
    print("Creating dictionary!!")
    for l in lines:
        l = l.split()
        wordsInDoc.append((len(l)) / 2)
        i = 0
        dict = {}
        for i in range(len(l) - 1):
            if (i + 1) % 2 == 0 and i != 0:
                continue
            if l[i] not in dict:
                if int(l[i]) > max:
                    max = int(l[i])
                if int(l[i]) < min:
                    min = int(l[i])
                dict[l[i]] = l[i + 1]
            else:
                dict[l[i]] += l[i + 1]

        train.append(dict)
    # print(train)
    for i in range(min, max):
        for l in train:
            if str(i) in l:
                if i in docFreq:
                    docFreq[i] += 1
                else:
                    docFreq[i] = 1
    tfidf = []
    j = 0
    print("Calculating tfidf!!")
    for l in train:
        list = []
        for i in range(min, max):
            if str(i) in l:
                tf = (int(l[str(i)]) / wordsInDoc[j])
                idf = log(len(train) / docFreq[i])
                list.append(tf / idf)
            else:
                list.append(0)
        tfidf.append(list)
        j += 1

    svd = TruncatedSVD(n_components=2, n_iter=7, random_state=42).fit(tfidf)
    tfidf = svd.transform(tfidf)
    '''with open("pickleFile.txt", "wb") as myFile:
        pickle.dump(tfidf, myFile)'''