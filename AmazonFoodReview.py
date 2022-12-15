import re

import numpy as np
import pandas as pd
from nltk.stem import SnowballStemmer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

df = pd.read_csv('Reviews.csv')

df["Sentiment"] = df["Score"].apply(lambda score: "positive" if score > 3 else "negative")
df = df[["Score", "Sentiment", "Summary", "Text"]]



snow = SnowballStemmer('english') 

def cleanup(sentence):
    sentence = str(sentence)
    sentence = sentence.lower() # lower case
    sentence = re.sub(r'[?|!|.|,|)|(|\|/]',r' ',sentence) # replace these punctuation with space
    tokens = sentence.split()
    out = []
    for t in tokens:
        out.append(snow.stem(t))
    out = " ".join(out)
    out = re.sub(r'[\'|"|#]', r'', out) # remove these punctuation
    return out    

df["Summary_Clean"] = df["Summary"].apply(cleanup)



train, test = train_test_split(df, test_size=0.2, random_state = 42)
train = train.reset_index(drop=True)
test = test.reset_index(drop=True)

uni_gram = CountVectorizer(min_df = 5, binary = True)
uni_gram_vectors_train = uni_gram.fit_transform(train['Summary_Clean'].values)
uni_gram_vectors_test = uni_gram.transform(test['Summary_Clean'].values)

bi_gram = CountVectorizer(ngram_range=(1,2), min_df = 5, binary = True)
bi_gram_vectors_train = bi_gram.fit_transform(train['Summary_Clean'].values)
bi_gram_vectors_test = bi_gram.transform(test['Summary_Clean'].values)

tfidf = TfidfVectorizer(ngram_range=(1,2), min_df = 5)
tfidf_vectors_train = tfidf.fit_transform(train['Summary_Clean'].values)
tfidf_vectors_test = tfidf.transform(test['Summary_Clean'].values)



prediction = dict()
prob = dict()

logreg_uni_gram = LogisticRegression(C = 1e5, class_weight = 'balanced')
logreg_uni_gram_result = logreg_uni_gram.fit(uni_gram_vectors_train, train['Sentiment'])
prediction['logistic_uni_gram'] = logreg_uni_gram.predict(uni_gram_vectors_test)
prob['logistic_uni_gram'] = logreg_uni_gram.predict_proba(uni_gram_vectors_test) 

logreg_bi_gram = LogisticRegression(C = 1e5, class_weight = 'balanced')
logreg_bi_gram_result = logreg_bi_gram.fit(bi_gram_vectors_train, train['Sentiment'])
prediction['logistic_bi_gram'] = logreg_bi_gram.predict(bi_gram_vectors_test)
prob['logistic_bi_gram'] = logreg_bi_gram.predict_proba(bi_gram_vectors_test)

logreg_tfidf = LogisticRegression(C = 1e5, class_weight = 'balanced')
logreg_tfidf_result = logreg_tfidf.fit(tfidf_vectors_train, train['Sentiment'])
prediction['logistic_tfidf'] = logreg_tfidf.predict(tfidf_vectors_test)
prob['logistic_tfidf'] = logreg_tfidf.predict_proba(tfidf_vectors_test)


rf_bi_gram = RandomForestClassifier(n_estimators = 100, class_weight = 'balanced', n_jobs = -1)
rf_bi_gram_result = rf_bi_gram.fit(bi_gram_vectors_train, train['Sentiment'])
prediction['rf_bi_gram'] = rf_bi_gram.predict(bi_gram_vectors_test)
prob['rf_bi_gram'] = rf_bi_gram.predict_proba(bi_gram_vectors_test)

