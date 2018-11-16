import csv
import numpy
import scipy
import sklearn
from sklearn.feature_extraction.text import CountVectorizer

cc = []
diagnoses = []

with open('Test Data.csv') as csv_file:
    reader = csv.reader(csv_file, delimiter='\t')
    next(reader, None)

    for row in reader:
        cc.append(row[0])           # append each chief complaint to a list
        diagnoses.append(row[3])    # append each diagnosis to a list
    
    csv_file.close()

''' Convert the complaints and diagnoses to lower case words '''
for i in range(len(cc)):
    cc[i] = cc[i].lower()
    diagnoses[i] = diagnoses[i].lower()
    # cc[i] = cc[i].replace("abd", "abdominal")
    # we'll need to account for abbreviations

''' Get the sparse matrix for the 1-grams and 2-grams '''
vectorizer = CountVectorizer(analyzer="word", ngram_range=(1, 2))
sparse = vectorizer.fit_transform(cc)       # the sparse matrix where documents are rows, n-grams are columns

print(vectorizer.get_feature_names())
print()
print(sparse.toarray())


''' Figure out the rows that the diagnoses are in '''
# create a list, each element is a row in the diagnosis column
# create a dictionary where the key is the diagnosis, the value is a list of the rows that the diagnosis is in
# for i in range of the above list, update the dictionary's value list so that it contains i

diag_dict = dict()      # {diagnosis: list of rows that contain the diagnosis}
i = 0

for diagnosis in diagnoses:

    diags = diagnosis.split(', ')        # a patient can have multiple diagnoses, so split them up

    # for each diagnosis, add or update the dictionary with the row that has the diagnosis
    for d in diags:
        if d not in diag_dict.keys():
            diag_dict[d] = [i]
        else:
            diag_dict[d].append(i)
    
    i += 1

''' Create the cocluster input matrix '''

for key in diag_dict:
    # get the value which is a list of all the rows that the key, or diagnosis, is in
    # use the list elements to find the corresponding rows in the sparse matrix
    # add those rows of the sparse matrix together (by column) using numpy
    # put the new numpy array in the cocluster input matrix
    pass
