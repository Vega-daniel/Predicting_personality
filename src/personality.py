# Functions file for personality prediction file

import unicodedata
from nltk.tokenize import word_tokenize
import pandas as pd
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
import string
import re
from bs4 import BeautifulSoup

REPLACE_BY_SPACE_RE = re.compile('[/(){}\\[\]\|?@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
punctuation_ = set(string.punctuation)
stopwords_ = set(stopwords.words('english'))

def remove_accents(input_str):
    nfkd_form = unicodedata.normalize('NFKD', input_str)
    only_ascii = nfkd_form.encode('ASCII', 'ignore')
    return only_ascii.decode()

def filter_tokens(sent):
    return ([w for w in sent if not w in stopwords_ and not w in punctuation_])

def remove_link(sent):
    return [s for s in sent if 'http' not in s]

def wt(text):
    return [word_tokenize(sent) for sent in text.split()]

def flatten(lst):
    return [item for sublist in lst for item in sublist]

def snow_stem(text):
    snowball = SnowballStemmer('english')
    return [snowball.stem(word) for word in text]

def rm_punc(sent):
	return [w for w in sent if not w in punctuation_]

def clean_text(text):
    text = BeautifulSoup(text, "lxml").text # HTML decoding
    text = REPLACE_BY_SPACE_RE.sub(' ', text) # replace REPLACE_BY_SPACE_RE symbols by space in text
    text = BAD_SYMBOLS_RE.sub('', text) # delete symbols which are in BAD_SYMBOLS_RE from text
    text = ' '.join(word for word in text.split() if word not in stopwords_)
    return text

def clean_df(df):
	"""Normalize the data, change all letters to lower case, split into 
	sentences,remove website links, tokenize words into a new dataframe then:
	filter punctuation, filter stopwords, stemminize
	INPUT
	--------------
	df = Pandas DataFrame

	OUTPUT
	--------------
	word = Pandas DataFrames
	______________
	"""
	newdf = df.copy()
	newdf.posts = newdf.posts.apply(lambda x: remove_accents(x).lower())
	newdf = pd.DataFrame((newdf.type,newdf.posts.apply(lambda x: x.split('|||')))).T
	newdf.posts = newdf.posts.apply(lambda x: remove_link(x))
	newdf.posts = newdf.posts.apply(lambda x: ' '.join(x))
	newdf.posts = newdf.posts.apply(lambda x: clean_text(x))
	newdf.posts = newdf.posts.apply(lambda x: wt(x))
	newdf.posts = newdf.posts.apply(lambda x: flatten(x))
	newdf.posts = newdf.posts.apply(lambda x: snow_stem(x))
	newdf.posts = newdf.posts.apply(lambda x: ' '.join(x))
	return newdf
