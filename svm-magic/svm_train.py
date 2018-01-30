import pickle
import numpy as np
from sklearn import svm

np.random.seed(42)

data = pickle.load(open('matrices.dat'))
clf = svm.SVC(decision_function_shape='ovo', verbose=True, max_iter=1, cache_size=3000)

print("About to start training")

trainX = data['trainX']
trainY = data['trainY']

# Now select samples from it like a ba-mofo
indices = np.random.choice(np.arange(len(trainX)), len(trainX)/10)
trainX = [ trainX[i] for i in indices]
trainY = [ trainY[i] for i in indices]


clf.fit(trainX, trainY)
answers = clf.predict(data['testX'])

pickle.dump(answers, open('answers.dat', 'w+'))

pickle.dump(clf, open('classifier.svm', 'w+'))