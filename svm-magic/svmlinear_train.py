import pickle
from sklearn import svm

data = pickle.load(open('matrices.dat'))
clf = svm.LinearSVC( verbose=True, max_iter=1)

print("About to start training")

clf.fit(data['trainX'], data['trainY'])
answers = clf.predict(data['testX'])

pickle.dump(answers, open('answerslinear.dat', 'w+'))

pickle.dump(clf, open('classifierlinear.svm', 'w+'))
