import pandas as pd
import spacy
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_excel("./soc_2018_definitions.xlsx")
df = df.dropna()
df = df.reset_index()
df = df[['Unnamed: 2', 'Unnamed: 3']]
df.columns = ['title', 'desc']
df = df.drop(df.index[0])

nlp = spacy.load('en_core_web_lg')
topic_labels = df['title'].values
stopset = set(stopwords.words('english'))


def removeStopwords(t):
    print(t)
    t = "".join((char.lower() for char in t if char not in string.punctuation))
    tokens = word_tokenize(t)
    tokens = [w for w in tokens if not w in stopset]
    return " ".join(tokens)

df['desc'] = df['desc'].apply(removeStopwords)

topic_keywords = df['desc'].values
topic_docs = list(nlp.pipe(topic_keywords,
                           batch_size=10000,
                           n_threads=3))

topic_vectors = np.array([doc.vector
                          if doc.has_vector else spacy.vocab[0].vector
                          for doc in topic_docs])



keywords = [
    'data',
    'rubbish',
    'teenagers',
    'law',
    'world of warcraft',
    'pedophile',
    'bachelor of science',
    'java',
    'games developer'
]
keyword_docs = list(nlp.pipe(keywords,
                             batch_size=10000,
                             n_threads=3))
keyword_vectors = np.array([doc.vector
                            if doc.has_vector else spacy.vocab[0].vector
                            for doc in keyword_docs])

simple_sim = cosine_similarity(keyword_vectors, topic_vectors)
test = np.argsort(-simple_sim, axis=1)
topic_idx = test
count = 0
for k, i in zip(keywords, topic_idx):
    vectors = simple_sim[count]
    for j in i[:10]:
        print('“%s” is about %s' % (k, topic_labels[j]))
        print(vectors[j])
    count = count + 1
