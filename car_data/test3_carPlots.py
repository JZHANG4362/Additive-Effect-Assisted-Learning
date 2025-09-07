import pickle

# ######################
infile = open('subsetRes4.p', 'rb')
subsetRes = pickle.load(infile)
infile.close()

labelArray_label = subsetRes['labelArray_label123']

print(labelArray_label[2,[3,5]])

print(labelArray_label[4,[3,5]])

print(labelArray_label[5,[3,5]])