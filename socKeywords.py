
# coding: utf-8

# In[157]:


import pandas as pd
from spacy.tokenizer import Tokenizer
import spacy
nlp = spacy.load('en_core_web_lg')
tokenizer = Tokenizer(nlp.vocab)
from spacy.lang.en.stop_words import STOP_WORDS
import numpy as np


# In[184]:


df = pd.read_excel("./soc_2018_definitions.xlsx")


# In[185]:


df = df.dropna()
df = df.reset_index()
df = df[['Unnamed: 2','Unnamed: 3']]
df.columns = ['title','desc']
df = df.drop(df.index[0])
# display(df)


# In[186]:


import string
def removeStopwords(t):
    t = "".join(char for char in t if char not in string.punctuation)
    t = [token.orth_ for token in tokenizer(t) if token.orth_ not in STOP_WORDS and not token.is_punct and not token.is_space]
    return " ".join(char for char in t if char not in string.punctuation)


# In[187]:


df['desc'] = df['desc'].apply(removeStopwords)
display(df)


# In[188]:


topic_keywords = df['desc'].values


# In[189]:


topic_docs = list(nlp.pipe(topic_keywords,
  batch_size=10000,
  n_threads=3))


# In[225]:


topic_vectors = np.array([doc.vector if doc.has_vector else spacy.vocab[0].vector for doc in topic_docs])


# In[229]:


keywords = [
    'rectum'
]
keyword_docs = list(nlp.pipe(keywords,
  batch_size=10000,
  n_threads=3))
keyword_vectors = np.array([doc.vector
  if doc.has_vector else spacy.vocab[0].vector
  for doc in keyword_docs])
# print('Vector for keyword “%s”: ' % keywords[0])
# print(keyword_vectors[0])


# In[230]:


from sklearn.metrics.pairwise import cosine_similarity
# use numpy and scikit-learn vectorized implementations for performance
simple_sim = cosine_similarity(keyword_vectors, topic_vectors)
topic_idx = np.argsort(-simple_sim,axis=1)
# print(simple_sim)


# In[233]:


count = 0
for k, i in zip(keywords, topic_idx):
    vectors = simple_sim[count]
    for j in i[:10]:
        print('“%s” is about %s' %(k, topic_labels[j]))
        print(vectors[j])
    count=count+1

