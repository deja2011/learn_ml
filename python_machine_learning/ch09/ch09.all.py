Copyright (c) 2015, 2016 [Sebastian Raschka](sebastianraschka.com)

https://github.com/rasbt/python-machine-learning-book

[MIT License](https://github.com/rasbt/python-machine-learning-book/blob/master/LICENSE.txt)
# Python Machine Learning - Code Examples
# Chapter 9 - Embedding a Machine Learning Model into a Web Application
Note that the optional watermark extension is a small IPython notebook plugin that I developed to make the code reproducible. You can just skip the following line(s).
%load_ext watermark
%watermark -a 'Sebastian Raschka' -u -d -v -p numpy,pandas,matplotlib,nltk,sklearn
*The use of `watermark` is optional. You can install this IPython extension via "`pip install watermark`". For more information, please see: https://github.com/rasbt/watermark.*
<br>
<br>
### Overview
- [Chapter 8 recap - Training a model for movie review classification](#Chapter-6-recap---Training-a-model-for-movie-review-classification)

- [Serializing fitted scikit-learn estimators](#Serializing-fitted-scikit-learn-estimators)
- [Setting up a SQLite database for data storage Developing a web application with Flask](#Setting-up-a-SQLite-database-for-data-storage-Developing-a-web-application-with-Flask)
- [Our first Flask web application](#Our-first-Flask-web-application)
  - [Form validation and rendering](#Form-validation-and-rendering)
  - [Turning the movie classifier into a web application](#Turning-the-movie-classifier-into-a-web-application)
- [Deploying the web application to a public server](#Deploying-the-web-application-to-a-public-server)
  - [Updating the movie review classifier](#Updating-the-movie-review-classifier)
- [Summary](#Summary)
The code for the Flask web applications can be found in the following directories:

- `1st_flask_app_1/`: A simple Flask web app
- `1st_flask_app_2/`: `1st_flask_app_1` extended with flexible form validation and rendering
- `movieclassifier/`: The movie classifier embedded in a web application
- `movieclassifier_with_update/`: same as `movieclassifier` but with update from sqlite database upon start
To run the web applications locally, `cd` into the respective directory (as listed above) and execute the main-application script, for example,

    cd ./1st_flask_app_1
    python3 app.py

Now, you should see something like

     * Running on http://127.0.0.1:5000/
     * Restarting with reloader

in your terminal.
Next, open a web browsert and enter the address displayed in your terminal (typically http://127.0.0.1:5000/) to view the web application.
**Link to a live example application built with this tutorial: http://raschkas.pythonanywhere.com/**.
<br>
<br>
from IPython.display import Image
# Chapter 8 recap - Training a model for movie review classification
This section is a recap of the logistic regression model that was trained in the last section of Chapter 6. Execute the folling code blocks to train a model that we will serialize in the next section.
import numpy as np
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

stop = stopwords.words('english')
porter = PorterStemmer()

def tokenizer(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text.lower())
    text = re.sub('[\W]+', ' ', text.lower()) + ' '.join(emoticons).replace('-', '')
    tokenized = [w for w in text.split() if w not in stop]
    return tokenized

def stream_docs(path):
    with open(path, 'r') as csv:
        next(csv) # skip header
        for line in csv:
            text, label = line[:-3], int(line[-2])
            yield text, label
next(stream_docs(path='./movie_data.csv'))
### Note

The pickling-section may be a bit tricky so that I included simpler test scripts in this directory (pickle-test-scripts/) to check if your environment is set up correctly. Basically, it is just a trimmed-down version of the relevant sections from Ch08, including a very small movie_review_data subset.

Executing

    python pickle-dump-test.py

will train a small classification model from the `movie_data_small.csv` and create the 2 pickle files 

    stopwords.pkl
    classifier.pkl

Next, if you execute

    python pickle-load-test.py

You should see the following 2 lines as output:

    Prediction: positive
    Probability: 85.71%
<hr>
### Note

If you haven't created the `movie_data.csv` file in the previous chapter, you can find a download a zip archive at 
https://github.com/rasbt/python-machine-learning-book/tree/master/code/datasets/movie
<hr>
def get_minibatch(doc_stream, size):
    docs, y = [], []
    try:
        for _ in range(size):
            text, label = next(doc_stream)
            docs.append(text)
            y.append(label)
    except StopIteration:
        return None, None
    return docs, y
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier

vect = HashingVectorizer(decode_error='ignore', 
                         n_features=2**21,
                         preprocessor=None, 
                         tokenizer=tokenizer)

clf = SGDClassifier(loss='log', random_state=1, n_iter=1)
doc_stream = stream_docs(path='./movie_data.csv')
import pyprind
pbar = pyprind.ProgBar(45)

classes = np.array([0, 1])
for _ in range(45):
    X_train, y_train = get_minibatch(doc_stream, size=1000)
    if not X_train:
        break
    X_train = vect.transform(X_train)
    clf.partial_fit(X_train, y_train, classes=classes)
    pbar.update()
X_test, y_test = get_minibatch(doc_stream, size=5000)
X_test = vect.transform(X_test)
print('Accuracy: %.3f' % clf.score(X_test, y_test))
clf = clf.partial_fit(X_test, y_test)
<br>
<br>
# Serializing fitted scikit-learn estimators
After we trained the logistic regression model as shown above, we know save the classifier along woth the stop words, Porter Stemmer, and `HashingVectorizer` as serialized objects to our local disk so that we can use the fitted classifier in our web application later.
import pickle
import os

dest = os.path.join('movieclassifier', 'pkl_objects')
if not os.path.exists(dest):
    os.makedirs(dest)

pickle.dump(stop, open(os.path.join(dest, 'stopwords.pkl'), 'wb'), protocol=4)   
pickle.dump(clf, open(os.path.join(dest, 'classifier.pkl'), 'wb'), protocol=4)
Next, we save the `HashingVectorizer` as in a separate file so that we can import it later.
%%writefile movieclassifier/vectorizer.py
from sklearn.feature_extraction.text import HashingVectorizer
import re
import os
import pickle

cur_dir = os.path.dirname(__file__)
stop = pickle.load(open(
                os.path.join(cur_dir, 
                'pkl_objects', 
                'stopwords.pkl'), 'rb'))

def tokenizer(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)',
                           text.lower())
    text = re.sub('[\W]+', ' ', text.lower()) \
                   + ' '.join(emoticons).replace('-', '')
    tokenized = [w for w in text.split() if w not in stop]
    return tokenized

vect = HashingVectorizer(decode_error='ignore',
                         n_features=2**21,
                         preprocessor=None,
                         tokenizer=tokenizer)
After executing the preceeding code cells, we can now restart the IPython notebook kernel to check if the objects were serialized correctly.
First, change the current Python directory to `movieclassifer`:
import os
os.chdir('movieclassifier')
import pickle
import re
import os
from vectorizer import vect

clf = pickle.load(open(os.path.join('pkl_objects', 'classifier.pkl'), 'rb'))
import numpy as np
label = {0:'negative', 1:'positive'}

example = ['I love this movie']
X = vect.transform(example)
print('Prediction: %s\nProbability: %.2f%%' %\
      (label[clf.predict(X)[0]], clf.predict_proba(X).max()*100))
<br>
<br>
# Setting up a SQLite database for data storage 
Before you execute this code, please make sure that you are currently in the `movieclassifier` directory.
import sqlite3
import os

if os.path.exists('reviews.sqlite'):
    os.remove('reviews.sqlite')

conn = sqlite3.connect('reviews.sqlite')
c = conn.cursor()
c.execute('CREATE TABLE review_db (review TEXT, sentiment INTEGER, date TEXT)')

example1 = 'I love this movie'
c.execute("INSERT INTO review_db (review, sentiment, date) VALUES (?, ?, DATETIME('now'))", (example1, 1))

example2 = 'I disliked this movie'
c.execute("INSERT INTO review_db (review, sentiment, date) VALUES (?, ?, DATETIME('now'))", (example2, 0))

conn.commit()
conn.close()
conn = sqlite3.connect('reviews.sqlite')
c = conn.cursor()

c.execute("SELECT * FROM review_db WHERE date BETWEEN '2015-01-01 10:10:10' AND DATETIME('now')")
results = c.fetchall()

conn.close()
print(results)
Image(filename='../images/09_01.png', width=700) 
<br>
# Developing a web application with Flask
...
## Our first Flask web application
...
## Form validation and rendering
Image(filename='../images/09_02.png', width=400) 
Image(filename='../images/09_03.png', width=400) 
<br>
<br>
# Turning the movie classifier into a web application
Image(filename='../images/09_04.png', width=400) 
Image(filename='../images/09_05.png', width=400) 
Image(filename='../images/09_06.png', width=400) 
Image(filename='../images/09_07.png', width=200) 
<br>
<br>
# Deploying the web application to a public server
Image(filename='../images/09_08.png', width=600) 
<br>
<br>
## Updating the movie review classifier
Change current directory to `movieclassifier`:
Define a function to update the classifier with the data stored in the local SQLite database:
import pickle
import sqlite3
import numpy as np

# import HashingVectorizer from local dir
from vectorizer import vect

def update_model(db_path, model, batch_size=10000):

    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute('SELECT * from review_db')

    results = c.fetchmany(batch_size)
    while results:
        data = np.array(results)
        X = data[:, 0]
        y = data[:, 1].astype(int)

        classes = np.array([0, 1])
        X_train = vect.transform(X)
        clf.partial_fit(X_train, y, classes=classes)
        results = c.fetchmany(batch_size)

    conn.close()
    return None
Update the model:
cur_dir = '.'

# Use the following path instead if you embed this code into
# the app.py file

# import os
# cur_dir = os.path.dirname(__file__)

clf = pickle.load(open(os.path.join(cur_dir,
                 'pkl_objects',
                 'classifier.pkl'), 'rb'))
db = os.path.join(cur_dir, 'reviews.sqlite')

update_model(db_path=db, model=clf, batch_size=10000)

# Uncomment the following lines to update your classifier.pkl file

# pickle.dump(clf, open(os.path.join(cur_dir, 
#             'pkl_objects', 'classifier.pkl'), 'wb')
#             , protocol=4)
<br>
<br>
# Summary
<br>
...
<br>
