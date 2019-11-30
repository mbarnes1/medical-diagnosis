# Python script for reading .csv files

import csv
import numpy
import scipy
import sklearn
from sklearn.feature_extraction.text import CountVectorizer

cc = []
nurse = []
md = []
diagnosis = []

with open('Test_Data.csv') as csvfile:
	reader = csv.reader(csvfile)
	# for row in reader:
	# 	print(row)

	# might be useful to have separate lists of each column
	for row in reader: 
		cc.append(row[0])
		nurse.append(row[1])
		md.append(row[2])
		diagnosis.append(row[3])

# one big list of all the entries - is there a better way to do this?
# each entry is one "document"
corpus = cc+nurse+md+diagnosis

vectorizer = CountVectorizer(analyzer = "word", tokenizer = None, preprocessor = None, stop_words = None)
wordcount = vectorizer.fit_transform(corpus)
print(wordcount.toarray())
# print(vectorizer.fit_transform(corpus).todense())

# vocabulary_ gives each word followed by a number - number isn't the count of how many times it appears; i think it's the word's ordinal number?
# print(vectorizer.vocabulary_)
words = vectorizer.get_feature_names()
print(words)
print(len(words))