import numpy as np
import pickle

# possible p2 + p3 values
p23List = [8]
# number of vectors sent
K = 5
# Number of matrices to generate
nU = 6

for p23 in p23List:
    uArray = np.empty([nU, p23, K])
    for i in range(nU):
        # generate normal random values
        u_tmp = np.random.normal(loc = 0, scale = 1, size = (p23, K))
        # standardize each column
        L2Norms = np.sum(np.abs(u_tmp)**2,axis=0)**(1./2)
        uArray[i, :, :] = u_tmp/L2Norms
    pickle.dump(uArray, open("uArray_" + str(p23) + "_dic.p", "wb"))


# infile = open("uArray_" + str(p23) + '_dic.p', 'rb')
# new_dict = pickle.load(infile)
# infile.close()