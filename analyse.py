import pandas as pd
import re
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics


from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier

from datetime import datetime
t0 = datetime.now()

df = pd.read_csv('./movie_data.csv')

def preprocessor(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
    text = re.sub('[\W]+', ' ', text.lower()) +\
        ' '.join(emoticons).replace('-', '')
    return text

df['review'] = df['review'].apply(preprocessor)

nltk.download('stopwords')

porter = PorterStemmer()
stemmer = SnowballStemmer("english")

def tokenizer(text):
    return text.split()


def tokenizer_porter(text):
    return [porter.stem(word) for word in text.split()]

def tokenizer_stemmer(text):
    return [stemmer.stem(word) for word in text.split()]


stop = stopwords.words('english')

'''
X_train = df.loc[:25000, 'review'].values
y_train = df.loc[:25000, 'sentiment'].values
X_test = df.loc[25000:, 'review'].values
y_test = df.loc[25000:, 'sentiment'].values
'''

X_train = df.loc[:250, 'review'].values
y_train = df.loc[:250, 'sentiment'].values
X_test = df.loc[250:, 'review'].values
y_test = df.loc[250:, 'sentiment'].values


tfidf = TfidfVectorizer(strip_accents=None,
                        lowercase=False,
                        preprocessor=None)



#LogisticRegression



print ("--------------------LogisticRegression--------------------")

lr_param_grid = [{'vect__ngram_range': [(1, 1)],
               'vect__stop_words': [stop],
               'vect__tokenizer': [tokenizer_porter],
               'clf__penalty': ['l2'],
               'clf__C': [1.0]}
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
#print('LogisticRegression : CV Accuracy: %.3f' % gs_lr_tfidf.best_score_)

lr_clf = gs_lr_tfidf.best_estimator_
#print('LogisticRegression : Test Accuracy: %.3f' % lr_clf.score(X_test, y_test))

y_predicted = lr_clf.predict(X_test)
cm = metrics.confusion_matrix(y_test, y_predicted)
print (cm)
# Print the classification report
print(metrics.classification_report(y_test, y_predicted,target_names=['neg','pos']))


#LinearSVC

print ("--------------------LinearSVC--------------------")

lsvc_param_grid = [{'vect__ngram_range': [(1, 1)],
               'vect__stop_words': [stop],
               'vect__tokenizer': [tokenizer]}
              ]

lsvc_pipeline = Pipeline([('vect', tfidf),
                     ('clf', LinearSVC(C=1000))])

gs_lsvc_tfidf = GridSearchCV(lsvc_pipeline, lsvc_param_grid,
                           scoring='accuracy',
                           cv=5,
                           verbose=1,
                           n_jobs=-1)


gs_lsvc_tfidf.fit(X_train, y_train)

#print('LinearSVC : Best parameter set: %s ' % gs_lsvc_tfidf.best_params_)
#print('LinearSVC : CV Accuracy: %.3f' % gs_lsvc_tfidf.best_score_)

lsvc_clf = gs_lsvc_tfidf.best_estimator_
#print('LinearSVC : Test Accuracy: %.3f' % lsvc_clf.score(X_test, y_test))


y_predicted = lsvc_clf.predict(X_test)
cm = metrics.confusion_matrix(y_test, y_predicted)
print (cm)
# Print the classification report
print(metrics.classification_report(y_test, y_predicted,target_names=['neg','pos']))



#MultinomialNB

print ("--------------------MultinomialNB--------------------")


mnb_param_grid = [{'vect__ngram_range': [(1, 1)],
               'vect__stop_words': [stop],
               'vect__tokenizer': [tokenizer]}
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
#print('MultinomialNB : CV Accuracy: %.3f' % gs_mnb_tfidf.best_score_)

mnb_clf = gs_mnb_tfidf.best_estimator_
#print('MultinomialNB : Test Accuracy: %.3f' % mnb_clf.score(X_test, y_test))


y_predicted = mnb_clf.predict(X_test)
cm = metrics.confusion_matrix(y_test, y_predicted)
print (cm)
# Print the classification report
print(metrics.classification_report(y_test, y_predicted,target_names=['neg','pos']))




#BernoulliNB

print ("--------------------BernoulliNB--------------------")


bnb_param_grid = [{'vect__ngram_range': [(1, 1)],
               'vect__stop_words': [stop],
               'vect__tokenizer': [tokenizer_stemmer]}
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
#print('BernoulliNB : CV Accuracy: %.3f' % gs_bnb_tfidf.best_score_)

bnb_clf = gs_bnb_tfidf.best_estimator_
#print('BernoulliNB : Test Accuracy: %.3f' % bnb_clf.score(X_test, y_test))

y_predicted = bnb_clf.predict(X_test)
cm = metrics.confusion_matrix(y_test, y_predicted)
print (cm)
# Print the classification report
print(metrics.classification_report(y_test, y_predicted,target_names=['neg','pos']))



#RandomForestClassifier

print ("--------------------RandomForestClassifier--------------------")


rfc_param_grid = [{'vect__ngram_range': [(1, 1)],
               'vect__stop_words': [stop],
               'vect__tokenizer': [tokenizer_porter]}
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
#print('RandomForestClassifier : CV Accuracy: %.3f' % gs_rfc_tfidf.best_score_)

rfc_clf = gs_rfc_tfidf.best_estimator_
#print('RandomForestClassifier : Test Accuracy: %.3f' % rfc_clf.score(X_test, y_test))

y_predicted = rfc_clf.predict(X_test)
cm = metrics.confusion_matrix(y_test, y_predicted)
print (cm)
# Print the classification report
print(metrics.classification_report(y_test, y_predicted,target_names=['neg','pos']))




#SGDClassifier

print ("--------------------SGDClassifier--------------------")


sgd_param_grid = [{'vect__ngram_range': [(1, 1)],
               'vect__stop_words': [stop, None],
               'vect__tokenizer': [tokenizer_stemmer]}
              ]

sgd_pipeline = Pipeline([('vect', tfidf),
                     ('clf', SGDClassifier())])

sgd_rfc_tfidf = GridSearchCV(sgd_pipeline, sgd_param_grid,
                           scoring='accuracy',
                           cv=5,
                           verbose=1,
                           n_jobs=-1)


sgd_rfc_tfidf.fit(X_train, y_train)

#print('SGDClassifier : Best parameter set: %s ' % sgd_rfc_tfidf.best_params_)
#print('SGDClassifier : CV Accuracy: %.3f' % sgd_rfc_tfidf.best_score_)

sgd_clf = sgd_rfc_tfidf.best_estimator_
#print('SGDClassifier : Test Accuracy: %.3f' % sgd_clf.score(X_test, y_test))

y_predicted = sgd_clf.predict(X_test)
cm = metrics.confusion_matrix(y_test, y_predicted)
print (cm)
# Print the classification report
print(metrics.classification_report(y_test, y_predicted,target_names=['neg','pos']))


#KNeighborsClassifier

print ("--------------------KNeighborsClassifier--------------------")

kn_param_grid = [{'vect__ngram_range': [(1, 1)],
               'vect__stop_words': [stop],
               'vect__tokenizer': [tokenizer_stemmer]}
              ]

kn_pipeline = Pipeline([('vect', tfidf),
                     ('clf', KNeighborsClassifier())])

gs_kn_tfidf = GridSearchCV(kn_pipeline, kn_param_grid,
                           scoring='accuracy',
                           cv=5,
                           verbose=1,
                           n_jobs=-1)


gs_kn_tfidf.fit(X_train, y_train)

#print('KNeighborsClassifier : Best parameter set: %s ' % gs_kn_tfidf.best_params_)
#print('KNeighborsClassifier : CV Accuracy: %.3f' % gs_kn_tfidf.best_score_)

kn_clf = gs_kn_tfidf.best_estimator_
#print('KNeighborsClassifier : Test Accuracy: %.3f' % kn_clf.score(X_test, y_test))

y_predicted = kn_clf.predict(X_test)
cm = metrics.confusion_matrix(y_test, y_predicted)
print (cm)
# Print the classification report
print(metrics.classification_report(y_test, y_predicted,target_names=['neg','pos']))



print ("--------------------VotingClassifier--------------------")

e_clf = VotingClassifier(estimators=[('lr', lr_clf), ('lsvc', lsvc_clf), ('mnb', mnb_clf), ('bnb', bnb_clf), ('rfc', rfc_clf), ('sgd', sgd_clf), ('kn', kn_clf)], voting='hard')
e_clf.fit(X_train, y_train)
#print('VotingClassifier : Test Accuracy: %.3f' % e_clf.score(X_test, y_test))

y_predicted = e_clf.predict(X_test)
cm = metrics.confusion_matrix(y_test, y_predicted)
print (cm)
# Print the classification report
print(metrics.classification_report(y_test, y_predicted,target_names=['neg','pos']))


print ("------------------------------------------------------------")

print ("Total time : ")
print (datetime.now() - t0)
