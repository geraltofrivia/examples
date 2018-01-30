import pickle
import numpy as np
from sklearn import svm
import time


np.random.seed(42)

start_time = time.time()

data = pickle.load(open('matrices.dat'))
clf = svm.SVC(decision_function_shape='ovo', verbose=True, max_iter=1, cache_size=3000)

load_time = time.time()
print("Loadtime: ", "--- %s seconds ---" % (load_time - start_time))

print("About to start training")

trainX = data['trainX']
trainY = data['trainY']

# Now select samples from it like a ba-mofo
indices = np.random.choice(np.arange(len(trainX)), len(trainX)/10)
trainX = [ trainX[i] for i in indices]
trainY = [ trainY[i] for i in indices]

split_time = time.time()
print("Splittime: ", "--- %s seconds ---" % (split_time - load_time))

clf.fit(trainX, trainY)

train_time = time.time()
print("Traintime: ", "--- %s seconds ---" % (train_time - split_time))

answers = clf.predict(data['testX'])

test_time = time.time()
print("Testtime: ", "--- %s seconds ---" % (test_time - train_time))

pickle.dump(answers, open('answers.dat', 'w+'))

pickle.dump(clf, open('classifier.svm', 'w+'))

dumptime = time.time()
print("Dumptime: ", "--- %s seconds ---" % (dumptime - test_time))