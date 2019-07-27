# Functions file for personality prediction file

import unicodedata
from nltk.tokenize import word_tokenize
import pandas as pd
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
import string
from bs4 import BeautifulSoup

symbols = string.punctuation
numbers = '0123456789'
stopwords_ = set(stopwords.words('english'))
more = ['like','think','infj','feel','know','one','think','realli','thing','get','entp','intp','intj','would','entj','enfj','isfp','feel','istp','infp','enfp','esfj','estj','esfp','estp','istj','isfj','type','becaus','peopl','time']
for w in more:
    stopwords_.append(w)

def remove_accents(string):
    nfkd_form = unicodedata.normalize('NFKD', string)
    only_ascii = nfkd_form.encode('ASCII', 'ignore')
    return only_ascii.decode()

def filter_tokens(string):
    return ([w for w in string.split() if not w in stopwords_])

def remove_link(string):
    return [s for s in string if 'http' not in s]

def wt(string):
    return [word_tokenize(sent) for sent in string.split()]

def flatten(lst):
    return [item for sublist in lst for item in sublist]

def snow_stem(string):
    snowball = SnowballStemmer('english')
    return [snowball.stem(word) for word in string.split()]

def rm_symbols(string):
    for char in symbols:
        string = string.replace(char," ")
    return string


def rm_numbers(string):
    for char in numbers:
        string = string.replace(char,"")
    return string


def clean_text(text):
    text = text.lower()
    text = BeautifulSoup(text, "lxml").text # HTML decoding
    text = rm_symbols(text) # replace symbols by space in text
    text = rm_numbers(text) # delete numbers from text
    #text = ' '.join(word for word in text.split() if word not in stopwords_)
    #print('clean complete')
    return text


def clean_input_text(text):
    """Prepare data to go into our models
    
    INPUT
    --------------
    text = String
    
    OUTPUT
    --------------
    text = String"""
    snowball = SnowballStemmer('english')
    text = clean_text(text)
    text = ' '.join(snowball.stem(word) for word in text.split())
    return text


def clean_df(df):
    """Normalize the data, change all letters to lower case, split into 
    sentences,remove website links, remove symbols, remove numbers then stemminize
    the words
    
    INPUT
    --------------
    df = Pandas DataFrame

    OUTPUT
    --------------
    newdf = Pandas DataFrames
    ______________
    """
    newdf = df.copy()
    newdf.posts = newdf.posts.apply(lambda x: remove_accents(x).lower())
    print('accents removed......... (1/5)')
    newdf = pd.DataFrame((newdf.type,newdf.posts.apply(lambda x: x.split('|||')))).T
    newdf.posts = newdf.posts.apply(lambda x: remove_link(x))
    print('links removed........... (2/5)')
    newdf.posts = newdf.posts.apply(lambda x: ' '.join(x))
    print('lxml removed............ (3/5)')
    newdf.posts = newdf.posts.apply(lambda x: clean_text(x))
    print('symbols/numbers replaced (4/5)')
    newdf.posts = newdf.posts.apply(lambda x: snow_stem(x))
    print('words stemmed........... (5/5)')
    newdf.posts = newdf.posts.apply(lambda x: ' '.join(x))
    return newdf


def result(string):
    lst = []
    if sgd_i.predict_proba([string])[0][1] >= .5:
        lst.append('I')
    else:
        lst.append('E')
    if sgd_n.predict_proba([string])[0][1] >= .5:
        lst.append('N')
    else:
        lst.append('S')
    if sgd_t.predict_proba([string])[0][1] >= .5:
        lst.append('T')
    else:
        lst.append('F')
    if sgd_p.predict_proba([string])[0][1] >= .5:
        lst.append('P')
    else:
        lst.append('J')
    return ''.join(lst)
