import csv
import string
import numpy as np
import scipy
# import sklearn
from sklearn.feature_extraction.text import CountVectorizer

cc = []
diagnoses = []

with open('Test Data.csv', 'r') as csv_file:
    reader = csv.reader(csv_file, delimiter='\t')
    next(reader, None)

    for row in reader:
        cc.append(row[0])           # append each chief complaint to a list
        diagnoses.append(row[3])    # append each diagnosis to a list
    
#csv_file.close()

''' Convert the complaints and diagnoses to lower case, change abbreviations, etc. '''
for i in range(len(cc)):
    cc[i] = cc[i].lower()
    # using .replace() for now, should be changed with string.translate()
    cc[i] = cc[i].replace('.', '')
    cc[i] = cc[i].replace('\n', '')
    cc[i] = cc[i].replace('-', '')
    cc[i] = cc[i].replace('"', '')
    cc[i] = cc[i].replace('  ', ' ')
    cc[i] = cc[i].replace('abd ', 'abdominal ')
    cc[i] = cc[i].replace('rt ', 'right ')

    diagnoses[i] = diagnoses[i].lower()
    diagnoses[i] = diagnoses[i].replace('.', '')
    diagnoses[i] = diagnoses[i].replace('\n', '')


''' Get the sparse matrix for the 1-grams and 2-grams '''
vectorizer = CountVectorizer(analyzer="word", ngram_range=(1, 2))
dense = vectorizer.fit_transform(cc)
sparse = dense.toarray()        # the sparse matrix where documents are rows, n-grams are columns

"""print(vectorizer.get_feature_names())
print()
print(sparse)"""


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

"""print()
print(diag_dict)"""

np_sums = []        # list of 1D arrays, each index is a summed row of the sparse matrix

''' Create the cocluster input matrix '''
j = 0
for key in diag_dict:
    # get the value which is a list of all the rows that the diagnosis is in
    # use the list elements to find the corresponding rows in the sparse matrix
    # add those rows of the sparse matrix together (by column) using numpy
    # put the new numpy array in the cocluster input matrix

    indices = diag_dict[key]        # the rows that a diagnosis corresponds to
    
    print("Indices: ", indices)
    to_add = np.take(sparse, indices)       # 2D array that represents only the rows needed to be added for this diagnosis
    summed_array = np.sum(to_add, axis=0)   # 1D array with the sum of the rows for this diagnosis

    print("To add: ", to_add)
    print("Summed array: ", summed_array)

    if j == 0:
        np_sums = summed_array              # initialize the shape of the input matrix
    else:
        concat1 = np.array([np_sums])
        concat2 = np.array([summed_array])
        np_sums = np.concatenate((concat1, concat2), axis=0)       # concatenate with a new summed row from the sparse matrix

    j += 1

print(np_sums)
"""for i in rows_to_add:
arrays = sparse[i]"""