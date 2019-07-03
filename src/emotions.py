from nltk import word_tokenize
from nltk.stem.snowball import SnowballStemmer
import pandas as pd


def text_emotion(df, column):
    '''
    Takes a DataFrame and a specified column of text. Returns a DataFrame with 
    each of the 10 emotions in the NRC Emotion Lexicon, with each
    column containing the value of the emotions found in the text.
    ------------------------
    INPUT: 
    df = DataFrame 
    column = string
    ------------------------
    OUTPUT: 
    DataFrame with ten columns
    ------------------------
    '''

    new_df = df.copy()

    filepath = ('Data/'
                'NRC-Sentiment-Emotion-Lexicons/'
                'NRC-Sentiment-Emotion-Lexicons/'
                'NRC-Emotion-Lexicon-v0.92/'
                'NRC-Emotion-Lexicon-Wordlevel-v0.92.txt')
    emolex_df = pd.read_csv(filepath,
                            names=["word", "emotion", "association"],
                            sep='\t')
    emolex_words = emolex_df.pivot(index='word',
                                   columns='emotion',
                                   values='association').reset_index()
    emotions = emolex_words.columns.drop('word')
    emo_df = pd.DataFrame(0, index=df.index, columns=emotions)

    stemmer = SnowballStemmer("english")
    
    for i,row in newdf.iterrows():
        print(i)
        document = word_tokenize(new_df.loc[i][column])
        for word in document:
                word = stemmer.stem(word)
                emo_score = emolex_words[emolex_words.word == word]
                if not emo_score.empty:
                    for emotion in list(emotions):
                        emo_df.at[i, emotion] += emo_score[emotion]

    return emo_df
