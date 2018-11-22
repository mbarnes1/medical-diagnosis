import csv
import string
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import SpectralCoclustering

cc = []
diagnoses = []

with open('Test Data.csv', 'r') as csv_file:
    reader = csv.reader(csv_file, delimiter='\t')
    next(reader, None)

    for row in reader:
        cc.append(row[0])           # append each chief complaint to a list
        diagnoses.append(row[3])    # append each diagnosis to a list


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

#print(vectorizer.get_feature_names())
#print()
#print(sparse)


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

#print()
#print(diag_dict)


''' Create the cocluster input matrix '''
np_sums = []        # list of 1D arrays, each index is a summed row of the sparse matrix

for key in diag_dict:
    indices = diag_dict[key]        # the list of row indices from the sparse matrix that point to this diagnosis
    partial = np.zeros( ( len(sparse[0]) ), dtype=np.int )        # initialize partial solution to 0's of the necessary array length

    # print("For key: ", key)

    for i in indices:
        to_add = np.array(sparse[i])        # store the next row to add from the sparse matrix for this diagnosis
        stacked_arrays = np.stack((partial, to_add))        # stack the next row with the partial solution for the sum() input
        partial = np.sum(stacked_arrays, axis=0)       # sum the next row to add with the partial solution

        '''print("Row number to add: %d" % i)
        print("Array to add:")
        print(to_add)
        print()
        print("Stacked arrays:")
        print(stacked_arrays)
        print()
        print("Test sum:")
        print(partial)
        print()'''
    
    np_sums.append(partial)         # append the full solution to the list of summed rows


''' Perform the coclustering '''
x = np.array(np_sums)
# print(x)
clustering = SpectralCoclustering(n_clusters=4, random_state=0).fit(x)
print(clustering)
