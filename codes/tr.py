import scipy.io
import numpy
import gzip
import cPickle
f = gzip.open("../data/mnist.pkl.gz", 'rb')
train_set, valid_set, test_set = cPickle.load(f)
print type(train_set)
scipy.io.savemat("train.mat",{'trainData':train_set[0]})
scipy.io.savemat("trainlabel.mat",{'trainLabel':train_set[1]})
scipy.io.savemat("test.mat",{'testData':test_set[0]})
scipy.io.savemat("testlabel.mat",{'testLabel':test_set[1]})

