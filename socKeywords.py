import pandas as pd
from spacy.tokenizer import Tokenizer
import spacy

nlp = spacy.load('en_core_web_lg')
tokenizer = Tokenizer(nlp.vocab)
from spacy.lang.en.stop_words import STOP_WORDS
import numpy as np

df = pd.read_excel("./soc_2018_definitions.xlsx")

df = df.dropna()
df = df.reset_index()
df = df[['Unnamed: 2', 'Unnamed: 3']]
df.columns = ['title', 'desc']
df = df.drop(df.index[0])

import string


def removeStopwords(t):
    t = "".join(char for char in t if char not in string.punctuation)
    t = [token.orth_ for token in tokenizer(t) if
         token.orth_ not in STOP_WORDS and not token.is_punct and not token.is_space]
    return " ".join(char for char in t if char not in string.punctuation)


df['desc'] = df['desc'].apply(removeStopwords)

topic_keywords = df['desc'].values

topic_docs = list(nlp.pipe(topic_keywords,
                           batch_size=10000,
                           n_threads=3))

topic_vectors = np.array([doc.vector if doc.has_vector else spacy.vocab[0].vector for doc in topic_docs])

keywords = [
    'rectum'
]
keyword_docs = list(nlp.pipe(keywords,
                             batch_size=10000,
                             n_threads=3))
keyword_vectors = np.array([doc.vector
                            if doc.has_vector else spacy.vocab[0].vector
                            for doc in keyword_docs])

from sklearn.metrics.pairwise import cosine_similarity

simple_sim = cosine_similarity(keyword_vectors, topic_vectors)
topic_idx = np.argsort(-simple_sim, axis=1)

count = 0
topic_labels = df['title'].values
for k, i in zip(keywords, topic_idx):
    vectors = simple_sim[count]
    for j in i[:10]:
        print('“%s” is about %s' % (k, topic_labels[j]))
        print(vectors[j])
    count = count + 1
