import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import warnings
import altair as alt
import pickle
import string
import spacy
import nltk
import re

from nltk import FreqDist
from sklearn.naive_bayes import *
from sklearn.ensemble import *
from sklearn.neighbors import *
from sklearn.tree import *
from sklearn.calibration import *
from sklearn.linear_model import *
from sklearn.multiclass import *
from sklearn.svm import *
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from nltk.stem import WordNetLemmatizer
from collections import Counter
from spacy.lang.en.stop_words import STOP_WORDS
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, auc, roc_curve
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer, HashingVectorizer
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV
from sklearn.pipeline import Pipeline, make_pipeline
import joblib

#nltk.download('stopwords')
sns.set(style='whitegrid')
#matplotlib_inline
warnings.filterwarnings('ignore')

# Load data
df = pd.read_csv('overview-of-recordings.csv')
print(df.head())
#savoir le contenu de notre data
#Analyze Data
def explore_data(df):
    print(f"The data contains {df.shape[0]} rows and {df.shape[1]} columns.")
    print('\n')
    print('Dataset columns:',df.columns)
    print('\n')
    print(df.info())

explore_data(df)
#analyse des valeurs nulles
df.isna().sum()
#analyse de valeurs dupliquées
def checking_removing_duplicates(df):
    count_dups = df.duplicated().sum()
    print("Number of Duplicates: ", count_dups)
    if count_dups >= 1:
        df.drop_duplicates(inplace=True)
        print('Duplicate values removed!')
    else:
        print('No Duplicate values')
checking_removing_duplicates(df)
#corpus
df_text = df[['phrase', 'prompt']]
df_text
#doc terme matrix
cv = CountVectorizer()
df_cv = cv.fit_transform(df_text.phrase)
data_dtm = pd.DataFrame(df_cv.toarray(), columns=cv.get_feature_names_out())
data_dtm.index = df_text.index
data_dtm
# Add features
# Number of characters in the text
df_text['phrase_length'] = df_text['phrase'].apply(len)
# Number of words in the text
df_text['phrase_num_words'] = df_text['phrase'].apply(lambda x: len(x.split()))
# Average length of the words in the text
df_text["mean_word_len"] = df_text["phrase"].apply(lambda x: np.mean([len(w) for w in str(x).split()]))
# Number of non-stopwords in the text
df_text['phrase_non_stopwords'] = df_text['phrase'].apply(lambda x: len([t for t in x.split() if t not in STOP_WORDS]))
df_text.describe().T
#text cleaning
def clean_txt(docs):
    lemmatizer = WordNetLemmatizer()
    # split into words
    speech_words = nltk.word_tokenize(docs)
    # convert to lower case
    lower_text = [w.lower() for w in speech_words]
    # prepare regex for char filtering
    re_punc = re.compile('[%s]' % re.escape(string.punctuation))
    # remove punctuation from each word
    stripped = [re_punc.sub('', w) for w in lower_text]
    # remove remaining tokens that are not alphabetic
    words = [word for word in stripped if word.isalpha()]
    # filter out stop words
    words = [w for w in words if not w in  list(STOP_WORDS)]
    # filter out short tokens
    words = [word for word in words if len(word) > 2]
    #Stemm all the words in the sentence
    lem_words = [lemmatizer.lemmatize(word) for word in words]
    combined_text = ' '.join(lem_words)
    return combined_text
    # Cleaning the text data
df_text['cleaned_phrase'] = df_text['phrase'].apply(clean_txt)
print (df_text)
freq_splits = FreqDist(df_text['phrase'])
print(f"***** 10 most common strings ***** \n{freq_splits.most_common(10)}", "\n")
#
from wordcloud import WordCloud

plt.rcParams['figure.figsize'] = [10, 6]
# generate word cloud and show it

for x in df_text.prompt.unique():
    wc = WordCloud(background_color="black", colormap="Dark2",
               max_font_size=150, random_state=42)
    wc.generate(df_text.cleaned_phrase[(df_text.cleaned_phrase.notnull()) & (df_text.prompt == x)].to_string())
    plt.imshow(wc, interpolation="bilinear")
    plt.title(x, fontdict={'size':16,'weight':'bold'})
    plt.axis("off")
    plt.show()
#préparation de texte et entraînement de modèle
# Spot-Check Normalized Text Models
def NormalizedTextModel(nameOfvect):
    if nameOfvect == 'countvect':
        vectorizer = CountVectorizer()
    elif nameOfvect =='tfvect':
        vectorizer = TfidfVectorizer()
    elif nameOfvect == 'hashvect':
        vectorizer = HashingVectorizer()

    pipelines = []
    pipelines.append((nameOfvect+'MultinomialNB'  , Pipeline([('Vectorizer', vectorizer),('NB'  , MultinomialNB())])))
    pipelines.append((nameOfvect+'CCCV' , Pipeline([('Vectorizer', vectorizer),('CCCV' , CalibratedClassifierCV())])))
    pipelines.append((nameOfvect+'KNN' , Pipeline([('Vectorizer', vectorizer),('KNN' , KNeighborsClassifier())])))
    pipelines.append((nameOfvect+'CART', Pipeline([('Vectorizer', vectorizer),('CART', DecisionTreeClassifier())])))
    pipelines.append((nameOfvect+'PAC'  , Pipeline([('Vectorizer', vectorizer),('PAC'  , PassiveAggressiveClassifier())])))
    pipelines.append((nameOfvect+'SVM' , Pipeline([('Vectorizer', vectorizer),('RC' , RidgeClassifier())])))
    pipelines.append((nameOfvect + 'AB', Pipeline([('Vectorizer', vectorizer), ('AB', AdaBoostClassifier())])))
    pipelines.append(
        (nameOfvect + 'GBM', Pipeline([('Vectorizer', vectorizer), ('GMB', GradientBoostingClassifier())])))
    pipelines.append((nameOfvect + 'RF', Pipeline([('Vectorizer', vectorizer), ('RF', RandomForestClassifier())])))
    pipelines.append((nameOfvect + 'ET', Pipeline([('Vectorizer', vectorizer), ('ET', ExtraTreesClassifier())])))
    pipelines.append((nameOfvect + 'SGD', Pipeline([('Vectorizer', vectorizer), ('SGD', SGDClassifier())])))
    pipelines.append((nameOfvect + 'OVRC',
                      Pipeline([('Vectorizer', vectorizer), ('OVRC', OneVsRestClassifier(LogisticRegression()))])))
    pipelines.append((nameOfvect + 'Bagging', Pipeline([('Vectorizer', vectorizer), ('Bagging', BaggingClassifier())])))
    pipelines.append((nameOfvect + 'NN', Pipeline([('Vectorizer', vectorizer), ('NN', MLPClassifier())])))
    # pipelines.append((nameOfvect+'xgboost', Pipeline([('Vectorizer', vectorizer), ('xgboost', XGBClassifier())])))
    return pipelines
    # Traing model


def fit_model(X_train, y_train, models):
    # Test options and evaluation metric
    num_folds = 10
    scoring = 'accuracy'

    results = []
    names = []
    for name, model in models:
        kfold = KFold(n_splits=num_folds)
        cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)
        results.append(cv_results)
        names.append(name)
        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        print(msg)

    # Split data to training and validation set


def read_in_and_split_data(data, features, target):
    X = data[features]
    y = data[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    return X_train, X_test, y_train, y_test


X = 'cleaned_phrase'
target_class = 'prompt'
X_train, X_test, y_train, y_test = read_in_and_split_data(df_text, X, target_class)
# sample text
sample_text_count = X_train[:10]
# create the transform
vectorizer = CountVectorizer()
# tokenize and build vocab
vectorizer.fit(sample_text_count)
# summarize
print(vectorizer.vocabulary_)
# encode document
vector = vectorizer.transform(sample_text_count)
# summarize encoded vector
print(vector.shape)
print(type(vector))
print(vector.toarray())

# Contvectorizer
models = NormalizedTextModel('countvect')
fit_model(X_train, y_train, models)
# sample text
sample_text_Tfid = X_train[:10]
# create the transform
vectorizer = TfidfVectorizer()
# tokenize and build vocab
vectorizer.fit(sample_text_Tfid)
# summarize
print(vectorizer.vocabulary_)
print(vectorizer.idf_)
# encode document
vector = vectorizer.transform(sample_text_Tfid)
# summarize encoded vector
print(vector.shape)
print(vector.toarray())
# TfidfVectorizer
models = NormalizedTextModel('tfvect')
fit_model(X_train, y_train, models)
# sample text
sample_text_hash = X_train[:10]
# create the transform
vectorizer = HashingVectorizer(n_features=20)
# encode document
vector = vectorizer.transform(sample_text_hash)
# summarize encoded vector
print(vector.shape)
print(vector.toarray())

vectorizer = TfidfVectorizer()
X_train_1 = vectorizer.fit_transform(X_train)
model = BaggingClassifier()
n_estimators = [10, 100, 1000]
#learning_rate= [0.1, 0.001, 0.0001]
#max_depth = [4,5,6]
#min_child_weight=[4,5,6]

#define grid search
grid = dict(n_estimators=n_estimators)
cv = KFold(n_splits=10)
grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy')
grid_result = grid_search.fit(X_train_1, y_train)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))


def classification_metrics(model, y_test, y_pred):
    print(f"Training Accuracy Score: {model.score(X_train, y_train) * 100:.1f}%")
    print(f"Validation Accuracy Score: {model.score(X_test, y_test) * 100:.1f}%")

    conf_matrix = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(pd.DataFrame(conf_matrix), annot=True, cmap='YlGnBu', fmt='g')
    ax.xaxis.set_label_position('top')
    plt.tight_layout()
    plt.title('Confusion matrix for Logisitic Regression Model', fontsize=20, y=1.1)
    plt.ylabel('Actual label', fontsize=15)
    plt.xlabel('Predicted label', fontsize=15)
    plt.show()
    print(classification_report(y_test, y_pred))


text_clf = Pipeline([('vect', TfidfVectorizer()), ('bagging', BaggingClassifier(n_estimators=10))])
model = text_clf.fit(X_train, y_train)
y_pred = model.predict(X_test)
classification_metrics(model, y_test, y_pred)

# Enregistrer le modèle dans une variable
model_filename = 'model.pkl'
joblib.dump(model, model_filename)