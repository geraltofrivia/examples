import pickle
from sklearn import svm

data = pickle.load(open('matrices.dat'))
clf = svm.SVC(decision_function_shape='ovo')

print("About to start training")

clf.fit(data['trainX'], data['trainY'])
answers = clf.predict(data['testX'])

pickle.dump(answers, open('answers.dat', 'w+'))

pickle.dump(clf, open('classifier.svm', 'w+'))