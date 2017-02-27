import numpy as np
from vectorizer import vect


clf = pickle.load(open(os.path.join('movieclassifier/pkl_objects', 'classifier.pkl'), 'rb'))
label = {0:'negative', 1:'positive'}


example = ['I love this movie']
X = vect.transform(example)
print('Prediction: {}\nProbability: {:.2f}%'.format(
    label[clf.predict(X)[0]],
    np.max(clf.predict_proba(X))*100))


def predict(comment):
    X = vect.transform([comment])
    print('Prediction: {} Probability: {:.2f}%'.format(
        label[clf.predict(X)[0]],
        np.max(clf.predict_proba(X))*100))

