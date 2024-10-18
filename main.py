import nltk
import regex as re
from nltk.stem.snowball import SnowballStemmer
nltk.download('punkt')
stemmer = SnowballStemmer('spanish')

with open("utils/stoplist.txt", encoding="latin1") as file:
    stoplist = [line.rstrip().lower() for line in file]
stoplist += ['?', '-', '.', ':', ',', '!', ';'] 

def preProcesamiento(texto, stemming=False):
    words = []

    texto = texto.lower()
    texto = re.sub(r'[^a-zA-Z0-9_À-ÿ]', ' ', texto)
    
    words = nltk.word_tokenize(texto)
    words = [word for word in words if word not in stoplist]
    if stemming:
        words = [stemmer.stem(word) for word in words]
    return words