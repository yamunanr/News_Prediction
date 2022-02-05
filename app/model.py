import warnings

warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", FutureWarning)
warnings.simplefilter("ignore", DeprecationWarning)
import pandas as pd
import numpy as np
import pickle
import xgboost as xgb
from xgboost import XGBClassifier
import joblib
from sklearn.naive_bayes import GaussianNB
from pandas import MultiIndex, Int64Index
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from sklearn.metrics import f1_score
import seaborn as sns

df = pd.read_csv("../../ML_Flask/Data/data.csv")

text = df["clean_text"]
tfidf_v = TfidfVectorizer(max_features=5000, ngram_range=(1, 3))
X = tfidf_v.fit_transform(text).toarray()
y = df["label"]

# Splitting Dataset into train and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

params = {
    'objective': 'binary:logistic',
    'max_depth': 4,
    'alpha': 10,
    'learning_rate': 1.0,
    'n_estimators': 100,
}

clf = XGBClassifier(**params).fit(X_train, y_train)
pickle.dump(clf, open('../../ML_Flask/model2.pkl', 'wb'))
pickle.dump(tfidf_v, open('../../ML_Flask/tfidfvect2.pkl', 'wb'))




