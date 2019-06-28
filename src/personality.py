#Functions file for personality prediction file

import unicodedata
from nltk.tokenize import word_tokenize
import pandas as pd
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
import string
import databricks.koalas as ks

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

def wt(lst):
    return [word_tokenize(sent) for sent in lst]

def flatten(lst):
    return [item for sublist in lst for item in sublist]

def snow_stem(lst):
    snowball = SnowballStemmer('english')
    return [snowball.stem(word) for word in lst]

def pipeline(df):
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
	df.posts = df.posts.apply(lambda x: remove_accents(x).lower())
	df = pd.DataFrame((df.type,df.posts.apply(lambda x: x.split('|||')))).T
	df.posts = df.posts.apply(lambda x: remove_link(x))
	df.posts = df.posts.apply(lambda x: wt(x))
	df.posts = df.posts.apply(lambda x: flatten(x))
	df.posts = df.posts.apply(lambda x: filter_tokens(x))
	df.posts = df.posts.apply(lambda x: snow_stem(x))
	return df

def pipeline_sent(df):
	"""Normalize the data, change all letters to lower case, split into 
	sentences,remove website links then join into one string.

	INPUT
	--------------
	df = Pandas DataFrame

	OUTPUT
	--------------
	df = Pandas DataFrames
	______________
	"""
	df.posts = df.posts.apply(lambda x: remove_accents(x).lower())
	df = pd.DataFrame((df.type,df.posts.apply(lambda x: x.split('|||')))).T
	df.posts = df.posts.apply(lambda x: remove_link(x))
	df.posts = df.posts.apply(lambda x: '. '.join(x))
	return df