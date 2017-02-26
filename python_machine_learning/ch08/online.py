from itertools import groupby, islice
import numpy as np
import re
import pyprind
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier

batch_size = 1000
train_size = 45000
test_size = 5000
stop = stopwords.words('english')


def tokenizer(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
    text = re.sub('[\W]+', ' ', text.lower()) + \
        ' '.join(emoticons).replace('-', '')
    tokenized = [w for w in text.split() if w not in stop]
    return tokenized


def stream_docs(path):
    with open(path, 'r', encoding='utf-8') as csv:
        # skip header
        next(csv)
        for line in csv:
            # text, label = line[:-3], int(line[-2])
            yield line[:-3], int(line[-2])


def get_minibatch(doc_stream, size):
    for _, group in groupby(enumerate(doc_stream), key=lambda e: e[0] // size):
        yield list(zip(*group))[1]


# main()

vect = HashingVectorizer(decode_error='ignore',
                         n_features=2**21,
                         preprocessor=None,
                         tokenizer=tokenizer)
clf = SGDClassifier(loss='log', random_state=1, n_iter=1)
doc_stream = stream_docs(path='./movie_data.csv')
classes = np.array([0, 1])
pbar = pyprind.ProgBar(int(train_size / batch_size))
for batch in get_minibatch(islice(doc_stream, train_size), size=batch_size):
    X_train, y_train = zip(*batch)
    X_train = vect.transform(X_train)
    clf.partial_fit(X_train, y_train, classes=classes)
    pbar.update()
pairs = next(get_minibatch(doc_stream, size=test_size))
X_test, y_test = zip(*pairs)
X_test = vect.transform(X_test)
print('\nAccuracy: {:.3f}'.format(clf.score(X_test, y_test)))
