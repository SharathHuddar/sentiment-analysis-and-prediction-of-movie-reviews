import pandas as pd
import re
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier

df = pd.read_csv('./movie_data.csv')

def preprocessor(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
    text = re.sub('[\W]+', ' ', text.lower()) +\
        ' '.join(emoticons).replace('-', '')
    return text

df['review'] = df['review'].apply(preprocessor)


porter = PorterStemmer()

def tokenizer(text):
    return text.split()


def tokenizer_porter(text):
    return [porter.stem(word) for word in text.split()]


stop = stopwords.words('english')

'''
X_train = df.loc[:25000, 'review'].values
y_train = df.loc[:25000, 'sentiment'].values
X_test = df.loc[25000:, 'review'].values
y_test = df.loc[25000:, 'sentiment'].values
'''

X_train = df.loc[:2500, 'review'].values
y_train = df.loc[:2500, 'sentiment'].values
X_test = df.loc[2500:, 'review'].values
y_test = df.loc[2500:, 'sentiment'].values


tfidf = TfidfVectorizer(strip_accents=None,
                        lowercase=False,
                        preprocessor=None)


#LogisticRegression



print ("LogisticRegression : ")

lr_param_grid = [{'vect__ngram_range': [(1, 1)],
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

lr_pipeline = Pipeline([('vect', tfidf),
                     ('clf', LogisticRegression(random_state=0))])

gs_lr_tfidf = GridSearchCV(lr_pipeline, lr_param_grid,
                           scoring='accuracy',
                           cv=5,
                           verbose=1,
                           n_jobs=-1)


gs_lr_tfidf.fit(X_train, y_train)

#print('LogisticRegression : Best parameter set: %s ' % gs_lr_tfidf.best_params_)
print('LogisticRegression : CV Accuracy: %.3f' % gs_lr_tfidf.best_score_)

lr_clf = gs_lr_tfidf.best_estimator_
print('LogisticRegression : Test Accuracy: %.3f' % lr_clf.score(X_test, y_test))



#LinearSVC

print ("LinearSVC : ")

lsvc_param_grid = [{'vect__ngram_range': [(1, 1)],
               'vect__stop_words': [stop, None],
               'vect__tokenizer': [tokenizer, tokenizer_porter]},
              {'vect__ngram_range': [(1, 1)],
               'vect__stop_words': [stop, None],
               'vect__tokenizer': [tokenizer, tokenizer_porter],
               'vect__use_idf':[False],
               'vect__norm':[None],
               },
              ]

lsvc_pipeline = Pipeline([('vect', tfidf),
                     ('clf', LinearSVC())])

gs_lsvc_tfidf = GridSearchCV(lsvc_pipeline, lsvc_param_grid,
                           scoring='accuracy',
                           cv=5,
                           verbose=1,
                           n_jobs=-1)


gs_lsvc_tfidf.fit(X_train, y_train)

#print('LinearSVC : Best parameter set: %s ' % gs_lsvc_tfidf.best_params_)
print('LinearSVC : CV Accuracy: %.3f' % gs_lsvc_tfidf.best_score_)

lsvc_clf = gs_lsvc_tfidf.best_estimator_
print('LinearSVC : Test Accuracy: %.3f' % lsvc_clf.score(X_test, y_test))




#MultinomialNB

print ("MultinomialNB : ")


mnb_param_grid = [{'vect__ngram_range': [(1, 1)],
               'vect__stop_words': [stop, None],
               'vect__tokenizer': [tokenizer, tokenizer_porter]},
              {'vect__ngram_range': [(1, 1)],
               'vect__stop_words': [stop, None],
               'vect__tokenizer': [tokenizer, tokenizer_porter],
               'vect__use_idf':[False],
               'vect__norm':[None],
               },
              ]

mnb_pipeline = Pipeline([('vect', tfidf),
                     ('clf', MultinomialNB())])

gs_mnb_tfidf = GridSearchCV(mnb_pipeline, mnb_param_grid,
                           scoring='accuracy',
                           cv=5,
                           verbose=1,
                           n_jobs=-1)


gs_mnb_tfidf.fit(X_train, y_train)

#print('MultinomialNB : Best parameter set: %s ' % gs_mnb_tfidf.best_params_)
print('MultinomialNB : CV Accuracy: %.3f' % gs_mnb_tfidf.best_score_)

mnb_clf = gs_mnb_tfidf.best_estimator_
print('MultinomialNB : Test Accuracy: %.3f' % mnb_clf.score(X_test, y_test))





#BernoulliNB

print ("BernoulliNB : ")


bnb_param_grid = [{'vect__ngram_range': [(1, 1)],
               'vect__stop_words': [stop, None],
               'vect__tokenizer': [tokenizer, tokenizer_porter]},
              {'vect__ngram_range': [(1, 1)],
               'vect__stop_words': [stop, None],
               'vect__tokenizer': [tokenizer, tokenizer_porter],
               'vect__use_idf':[False],
               'vect__norm':[None],
               },
              ]

bnb_pipeline = Pipeline([('vect', tfidf),
                     ('clf', BernoulliNB())])

gs_bnb_tfidf = GridSearchCV(bnb_pipeline, bnb_param_grid,
                           scoring='accuracy',
                           cv=5,
                           verbose=1,
                           n_jobs=-1)


gs_bnb_tfidf.fit(X_train, y_train)

#print('BernoulliNB : Best parameter set: %s ' % gs_bnb_tfidf.best_params_)
print('BernoulliNB : CV Accuracy: %.3f' % gs_bnb_tfidf.best_score_)

bnb_clf = gs_bnb_tfidf.best_estimator_
print('BernoulliNB : Test Accuracy: %.3f' % bnb_clf.score(X_test, y_test))

'''


#RandomForestClassifier

print ("RandomForestClassifier : ")


rfc_param_grid = [{'vect__ngram_range': [(1, 1)],
               'vect__stop_words': [stop, None],
               'vect__tokenizer': [tokenizer, tokenizer_porter]},
              {'vect__ngram_range': [(1, 1)],
               'vect__stop_words': [stop, None],
               'vect__tokenizer': [tokenizer, tokenizer_porter],
               'vect__use_idf':[False],
               'vect__norm':[None],
               },
              ]

rfc_pipeline = Pipeline([('vect', tfidf),
                     ('clf', RandomForestClassifier())])

gs_rfc_tfidf = GridSearchCV(rfc_pipeline, rfc_param_grid,
                           scoring='accuracy',
                           cv=5,
                           verbose=1,
                           n_jobs=-1)


gs_rfc_tfidf.fit(X_train, y_train)

#print('RandomForestClassifier : Best parameter set: %s ' % gs_rfc_tfidf.best_params_)
print('RandomForestClassifier : CV Accuracy: %.3f' % gs_rfc_tfidf.best_score_)

rfc_clf = gs_rfc_tfidf.best_estimator_
print('RandomForestClassifier : Test Accuracy: %.3f' % rfc_clf.score(X_test, y_test))

'''

e_clf = VotingClassifier(estimators=[('lr', lr_clf), ('lsvc', lsvc_clf), ('mnb', mnb_clf), ('bnb', bnb_clf)], voting='hard')
e_clf.fit(X_train, y_train)
print('VotingClassifier : Test Accuracy: %.3f' % e_clf.score(X_test, y_test))
