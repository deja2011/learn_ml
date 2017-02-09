Copyright (c) 2015, 2016 [Sebastian Raschka](sebastianraschka.com)

https://github.com/rasbt/python-machine-learning-book

[MIT License](https://github.com/rasbt/python-machine-learning-book/blob/master/LICENSE.txt)# Python Machine Learning - Code Examples# Chapter 8 - Applying Machine Learning To Sentiment AnalysisNote that the optional watermark extension is a small IPython notebook plugin that I developed to make the code reproducible. You can just skip the following line(s).%load_ext watermark
%watermark -a 'Sebastian Raschka' -u -d -v -p numpy,pandas,matplotlib,sklearn,nltk*The use of `watermark` is optional. You can install this IPython extension via "`pip install watermark`". For more information, please see: https://github.com/rasbt/watermark.*<br>
<br>### Overview- [Obtaining the IMDb movie review dataset](#Obtaining-the-IMDb-movie-review-dataset)
- [Introducing the bag-of-words model](#Introducing-the-bag-of-words-model)
  - [Transforming words into feature vectors](#Transforming-words-into-feature-vectors)
  - [Assessing word relevancy via term frequency-inverse document frequency](#Assessing-word-relevancy-via-term-frequency-inverse-document-frequency)
  - [Cleaning text data](#Cleaning-text-data)
  - [Processing documents into tokens](#Processing-documents-into-tokens)
- [Training a logistic regression model for document classification](#Training-a-logistic-regression-model-for-document-classification)
- [Working with bigger data – online algorithms and out-of-core learning](#Working-with-bigger-data-–-online-algorithms-and-out-of-core-learning)
- [Summary](#Summary)<br>
<br># Added version check for recent scikit-learn 0.18 checks
from distutils.version import LooseVersion as Version
from sklearn import __version__ as sklearn_version# Obtaining the IMDb movie review datasetThe IMDB movie review set can be downloaded from [http://ai.stanford.edu/~amaas/data/sentiment/](http://ai.stanford.edu/~amaas/data/sentiment/).
After downloading the dataset, decompress the files.

A) If you are working with Linux or MacOS X, open a new terminal windowm `cd` into the download directory and execute 

`tar -zxf aclImdb_v1.tar.gz`

B) If you are working with Windows, download an archiver such as [7Zip](http://www.7-zip.org) to extract the files from the download archive.### Compatibility Note:

I received an email from a reader who was having troubles with reading the movie review texts due to encoding issues. Typically, Python's default encoding is set to `'utf-8'`, which shouldn't cause troubles when running this IPython notebook. You can simply check the encoding on your machine by firing up a new Python interpreter from the command line terminal and execute

    >>> import sys
    >>> sys.getdefaultencoding()

If the returned result is **not** `'utf-8'`, you probably need to change your Python's encoding to `'utf-8'`, for example by typing `export PYTHONIOENCODING=utf8` in your terminal shell prior to running this IPython notebook. (Note that this is a temporary change, and it needs to be executed in the same shell that you'll use to launch `ipython notebook`.

Alternatively, you can replace the lines 

    with open(os.path.join(path, file), 'r') as infile:
    ...
    pd.read_csv('./movie_data.csv')
    ...
    df.to_csv('./movie_data.csv', index=False)

by 

    with open(os.path.join(path, file), 'r', encoding='utf-8') as infile:
    ...
    pd.read_csv('./movie_data.csv', encoding='utf-8')
    ...
    df.to_csv('./movie_data.csv', index=False, encoding='utf-8')

in the following cells to achieve the desired effect.import pyprind
import pandas as pd
import os

# change the `basepath` to the directory of the
# unzipped movie dataset

#basepath = '/Users/Sebastian/Desktop/aclImdb/'
basepath = './aclImdb'

labels = {'pos': 1, 'neg': 0}
pbar = pyprind.ProgBar(50000)
df = pd.DataFrame()
for s in ('test', 'train'):
    for l in ('pos', 'neg'):
        path = os.path.join(basepath, s, l)
        for file in os.listdir(path):
            with open(os.path.join(path, file), 'r', encoding='utf-8') as infile:
                txt = infile.read()
            df = df.append([[txt, labels[l]]], ignore_index=True)
            pbar.update()
df.columns = ['review', 'sentiment']Shuffling the DataFrame:import numpy as np

np.random.seed(0)
df = df.reindex(np.random.permutation(df.index))Optional: Saving the assembled data as CSV file:df.to_csv('./movie_data.csv', index=False)import pandas as pd

df = pd.read_csv('./movie_data.csv')
df.head(3)<hr>
### Note

If you have problems with creating the `movie_data.csv` file in the previous chapter, you can find a download a zip archive at 
https://github.com/rasbt/python-machine-learning-book/tree/master/code/datasets/movie
<hr><br>
<br># Introducing the bag-of-words model...## Transforming documents into feature vectorsBy calling the fit_transform method on CountVectorizer, we just constructed the vocabulary of the bag-of-words model and transformed the following three sentences into sparse feature vectors:
1. The sun is shining
2. The weather is sweet
3. The sun is shining, the weather is sweet, and one and one is two
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

count = CountVectorizer()
docs = np.array([
        'The sun is shining',
        'The weather is sweet',
        'The sun is shining, the weather is sweet, and one and one is two'])
bag = count.fit_transform(docs)Now let us print the contents of the vocabulary to get a better understanding of the underlying concepts:print(count.vocabulary_)As we can see from executing the preceding command, the vocabulary is stored in a Python dictionary, which maps the unique words that are mapped to integer indices. Next let us print the feature vectors that we just created:Each index position in the feature vectors shown here corresponds to the integer values that are stored as dictionary items in the CountVectorizer vocabulary. For example, the  rst feature at index position 0 resembles the count of the word and, which only occurs in the last document, and the word is at index position 1 (the 2nd feature in the document vectors) occurs in all three sentences. Those values in the feature vectors are also called the raw term frequencies: *tf (t,d)*—the number of times a term t occurs in a document *d*.print(bag.toarray())<br>## Assessing word relevancy via term frequency-inverse document frequencynp.set_printoptions(precision=2)When we are analyzing text data, we often encounter words that occur across multiple documents from both classes. Those frequently occurring words typically don't contain useful or discriminatory information. In this subsection, we will learn about a useful technique called term frequency-inverse document frequency (tf-idf) that can be used to downweight those frequently occurring words in the feature vectors. The tf-idf can be de ned as the product of the term frequency and the inverse document frequency:

$$\text{tf-idf}(t,d)=\text{tf (t,d)}\times \text{idf}(t,d)$$

Here the tf(t, d) is the term frequency that we introduced in the previous section,
and the inverse document frequency *idf(t, d)* can be calculated as:

$$\text{idf}(t,d) = \text{log}\frac{n_d}{1+\text{df}(d, t)},$$

where $n_d$ is the total number of documents, and *df(d, t)* is the number of documents *d* that contain the term *t*. Note that adding the constant 1 to the denominator is optional and serves the purpose of assigning a non-zero value to terms that occur in all training samples; the log is used to ensure that low document frequencies are not given too much weight.

Scikit-learn implements yet another transformer, the `TfidfTransformer`, that takes the raw term frequencies from `CountVectorizer` as input and transforms them into tf-idfs:from sklearn.feature_extraction.text import TfidfTransformer

tfidf = TfidfTransformer(use_idf=True, norm='l2', smooth_idf=True)
print(tfidf.fit_transform(count.fit_transform(docs)).toarray())As we saw in the previous subsection, the word is had the largest term frequency in the 3rd document, being the most frequently occurring word. However, after transforming the same feature vector into tf-idfs, we see that the word is is
now associated with a relatively small tf-idf (0.45) in document 3 since it is
also contained in documents 1 and 2 and thus is unlikely to contain any useful, discriminatory information.
However, if we'd manually calculated the tf-idfs of the individual terms in our feature vectors, we'd have noticed that the `TfidfTransformer` calculates the tf-idfs slightly differently compared to the standard textbook equations that we de ned earlier. The equations for the idf and tf-idf that were implemented in scikit-learn are:$$\text{idf} (t,d) = log\frac{1 + n_d}{1 + \text{df}(d, t)}$$

The tf-idf equation that was implemented in scikit-learn is as follows:

$$\text{tf-idf}(t,d) = \text{tf}(t,d) \times (\text{idf}(t,d)+1)$$

While it is also more typical to normalize the raw term frequencies before calculating the tf-idfs, the `TfidfTransformer` normalizes the tf-idfs directly.

By default (`norm='l2'`), scikit-learn's TfidfTransformer applies the L2-normalization, which returns a vector of length 1 by dividing an un-normalized feature vector *v* by its L2-norm:

$$v_{\text{norm}} = \frac{v}{||v||_2} = \frac{v}{\sqrt{v_{1}^{2} + v_{2}^{2} + \dots + v_{n}^{2}}} = \frac{v}{\big (\sum_{i=1}^{n} v_{i}^{2}\big)^\frac{1}{2}}$$

To make sure that we understand how TfidfTransformer works, let us walk
through an example and calculate the tf-idf of the word is in the 3rd document.

The word is has a term frequency of 3 (tf = 3) in document 3, and the document frequency of this term is 3 since the term is occurs in all three documents (df = 3). Thus, we can calculate the idf as follows:

$$\text{idf}("is", d3) = log \frac{1+3}{1+3} = 0$$

Now in order to calculate the tf-idf, we simply need to add 1 to the inverse document frequency and multiply it by the term frequency:

$$\text{tf-idf}("is",d3)= 3 \times (0+1) = 3$$tf_is = 3
n_docs = 3
idf_is = np.log((n_docs+1) / (3+1))
tfidf_is = tf_is * (idf_is + 1)
print('tf-idf of term "is" = %.2f' % tfidf_is)If we repeated these calculations for all terms in the 3rd document, we'd obtain the following tf-idf vectors: [3.39, 3.0, 3.39, 1.29, 1.29, 1.29, 2.0 , 1.69, 1.29]. However, we notice that the values in this feature vector are different from the values that we obtained from the TfidfTransformer that we used previously. The  nal step that we are missing in this tf-idf calculation is the L2-normalization, which can be applied as follows:$$\text{tfi-df}_{norm} = \frac{[3.39, 3.0, 3.39, 1.29, 1.29, 1.29, 2.0 , 1.69, 1.29]}{\sqrt{[3.39^2, 3.0^2, 3.39^2, 1.29^2, 1.29^2, 1.29^2, 2.0^2 , 1.69^2, 1.29^2]}}$$

$$=[0.5, 0.45, 0.5, 0.19, 0.19, 0.19, 0.3, 0.25, 0.19]$$

$$\Rightarrow \text{tfi-df}_{norm}("is", d3) = 0.45$$As we can see, the results match the results returned by scikit-learn's `TfidfTransformer` (below). Since we now understand how tf-idfs are calculated, let us proceed to the next sections and apply those concepts to the movie review dataset.tfidf = TfidfTransformer(use_idf=True, norm=None, smooth_idf=True)
raw_tfidf = tfidf.fit_transform(count.fit_transform(docs)).toarray()[-1]
raw_tfidf l2_tfidf = raw_tfidf / np.sqrt(np.sum(raw_tfidf**2))
l2_tfidf<br>## Cleaning text datadf.loc[0, 'review'][-50:]import re
def preprocessor(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
    text = re.sub('[\W]+', ' ', text.lower()) +\
        ' '.join(emoticons).replace('-', '')
    return textpreprocessor(df.loc[0, 'review'][-50:])preprocessor("</a>This :) is :( a test :-)!")df['review'] = df['review'].apply(preprocessor)<br>## Processing documents into tokensfrom nltk.stem.porter import PorterStemmer

porter = PorterStemmer()

def tokenizer(text):
    return text.split()


def tokenizer_porter(text):
    return [porter.stem(word) for word in text.split()]tokenizer('runners like running and thus they run')tokenizer_porter('runners like running and thus they run')import nltk

nltk.download('stopwords')from nltk.corpus import stopwords

stop = stopwords.words('english')
[w for w in tokenizer_porter('a runner likes running and runs a lot')[-10:]
if w not in stop]<br>
<br># Training a logistic regression model for document classificationStrip HTML and punctuation to speed up the GridSearch later:X_train = df.loc[:25000, 'review'].values
y_train = df.loc[:25000, 'sentiment'].values
X_test = df.loc[25000:, 'review'].values
y_test = df.loc[25000:, 'sentiment'].valuesfrom sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
if Version(sklearn_version) < '0.18':
    from sklearn.grid_search import GridSearchCV
else:
    from sklearn.model_selection import GridSearchCV

tfidf = TfidfVectorizer(strip_accents=None,
                        lowercase=False,
                        preprocessor=None)

param_grid = [{'vect__ngram_range': [(1, 1)],
               'vect__stop_words': [stop, None],
               'vect__tokenizer': [tokenizer, tokenizer_porter],
               'clf__penalty': ['l1', 'l2'],
               'clf__C': [1.0, 10.0, 100.0]},
              {'vect__ngram_range': [(1, 1)],
               'vect__stop_words': [stop, None],
               'vect__tokenizer': [tokenizer, tokenizer_porter],
               'vect__use_idf':[False],
               'vect__norm':[None],
               'clf__penalty': ['l1', 'l2'],
               'clf__C': [1.0, 10.0, 100.0]},
              ]

lr_tfidf = Pipeline([('vect', tfidf),
                     ('clf', LogisticRegression(random_state=0))])

gs_lr_tfidf = GridSearchCV(lr_tfidf, param_grid,
                           scoring='accuracy',
                           cv=5,
                           verbose=1,
                           n_jobs=-1)gs_lr_tfidf.fit(X_train, y_train)print('Best parameter set: %s ' % gs_lr_tfidf.best_params_)
print('CV Accuracy: %.3f' % gs_lr_tfidf.best_score_)clf = gs_lr_tfidf.best_estimator_
print('Test Accuracy: %.3f' % clf.score(X_test, y_test))<hr>
<hr>####  Start comment:

Please note that `gs_lr_tfidf.best_score_` is the average k-fold cross-validation score. I.e., if we have a `GridSearchCV` object with 5-fold cross-validation (like the one above), the `best_score_` attribute returns the average score over the 5-folds of the best model. To illustrate this with an example:from sklearn.linear_model import LogisticRegression
import numpy as np
if Version(sklearn_version) < '0.18':
    from sklearn.cross_validation import StratifiedKFold
    from sklearn.cross_validation import cross_val_score
else:
    from sklearn.model_selection import StratifiedKFold
    from sklearn.model_selection import cross_val_score

np.random.seed(0)
np.set_printoptions(precision=6)
y = [np.random.randint(3) for i in range(25)]
X = (y + np.random.randn(25)).reshape(-1, 1)

if Version(sklearn_version) < '0.18':
    cv5_idx = list(StratifiedKFold(y, n_folds=5, shuffle=False, random_state=0))

else:
    cv5_idx = list(StratifiedKFold(n_splits=5, shuffle=False, random_state=0).split(X, y))

cross_val_score(LogisticRegression(random_state=123), X, y, cv=cv5_idx)
By executing the code above, we created a simple data set of random integers that shall represent our class labels. Next, we fed the indices of 5 cross-validation folds (`cv3_idx`) to the `cross_val_score` scorer, which returned 5 accuracy scores -- these are the 5 accuracy values for the 5 test folds.  

Next, let us use the `GridSearchCV` object and feed it the same 5 cross-validation sets (via the pre-generated `cv3_idx` indices):if Version(sklearn_version) < '0.18':
    from sklearn.grid_search import GridSearchCV
else:
    from sklearn.model_selection import GridSearchCV

gs = GridSearchCV(LogisticRegression(), {}, cv=cv5_idx, verbose=3).fit(X, y) 
As we can see, the scores for the 5 folds are exactly the same as the ones from `cross_val_score` earlier.Now, the best_score_ attribute of the `GridSearchCV` object, which becomes available after `fit`ting, returns the average accuracy score of the best model:gs.best_score_As we can see, the result above is consistent with the average score computed the `cross_val_score`.cross_val_score(LogisticRegression(), X, y, cv=cv5_idx).mean()#### End comment.

<hr>
<hr><br>
<br># Working with bigger data - online algorithms and out-of-core learningimport numpy as np
import re
from nltk.corpus import stopwords

def tokenizer(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text.lower())
    text = re.sub('[\W]+', ' ', text.lower()) +\
        ' '.join(emoticons).replace('-', '')
    tokenized = [w for w in text.split() if w not in stop]
    return tokenized


def stream_docs(path):
    with open(path, 'r', encoding='utf-8') as csv:
        next(csv)  # skip header
        for line in csv:
            text, label = line[:-3], int(line[-2])
            yield text, labelnext(stream_docs(path='./movie_data.csv'))def get_minibatch(doc_stream, size):
    docs, y = [], []
    try:
        for _ in range(size):
            text, label = next(doc_stream)
            docs.append(text)
            y.append(label)
    except StopIteration:
        return None, None
    return docs, yfrom sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier

vect = HashingVectorizer(decode_error='ignore', 
                         n_features=2**21,
                         preprocessor=None, 
                         tokenizer=tokenizer)

clf = SGDClassifier(loss='log', random_state=1, n_iter=1)
doc_stream = stream_docs(path='./movie_data.csv')import pyprind
pbar = pyprind.ProgBar(45)

classes = np.array([0, 1])
for _ in range(45):
    X_train, y_train = get_minibatch(doc_stream, size=1000)
    if not X_train:
        break
    X_train = vect.transform(X_train)
    clf.partial_fit(X_train, y_train, classes=classes)
    pbar.update()X_test, y_test = get_minibatch(doc_stream, size=5000)
X_test = vect.transform(X_test)
print('Accuracy: %.3f' % clf.score(X_test, y_test))clf = clf.partial_fit(X_test, y_test)<br>
<br># Summary