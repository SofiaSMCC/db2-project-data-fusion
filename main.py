import pandas as pd
import nltk
import math
import json
import re
from nltk.stem.snowball import SnowballStemmer
from collections import defaultdict
import os

nltk.download('punkt')
stemmer = SnowballStemmer('spanish')

with open("utils/stoplist.txt", encoding="latin1") as file:
    stoplist = [line.rstrip().lower() for line in file]
stoplist += ['?', '-', '.', ':', ',', '!', ';']

def preProcesamiento(texto, stemming=False):
    texto = texto.lower()
    texto = re.sub(r'[^a-zA-Z0-9_À-ÿ]', ' ', texto) 
    words = nltk.word_tokenize(texto)
    words = [word for word in words if word not in stoplist]
    if stemming:
        words = [stemmer.stem(word) for word in words]
    
    return words

def get_tf(word, doc):
    return doc.count(word) / len(doc) if doc else 0

def get_df(docs):
    df = {}
    for words in docs.values():
        unique_words = set(words)  
        for word in unique_words:
            df[word] = df.get(word, 0) + 1
    return df

# mejor esquema de ponderación
def w_tf_idf(tf, df, N):
    return math.log10(1 + tf) * math.log10(N / df)