# Personality Predictor
by Daniel Vega


# Table of Contents
- [Introduction](#Introduction)
- [Overview of the Data](#Overview-of-the-Data)
- [Exploratory Data Analysis](#Exploratory-Data-Analysis)
- [Data Pipeline](#Data-Pipeline)
- [Model Selection](#Model-Selection)
- [Deep Learning](#Deep-Learning)
- [Emotional Analysis](#Emotional-Analysis)
- [Wordclouds](#WordClouds)
- [Conclusion and Next Steps](#Conclusion-and-Next-Steps)

# Introduction

The Myers–Briggs Type Indicator (MBTI) is an introspective self-report questionnaire with the purpose of indicating differing psychological preferences in how people perceive the world around them and make decisions.
The MBTI was constructed by Katharine Cook Briggs and her daughter Isabel Briggs Myers. It is based on the conceptual theory proposed by Carl Jung, who had speculated that humans experience the world using four principal psychological functions – sensation, intuition, feeling, and thinking – and that one of these four functions is dominant for a person most of the time.



![](img/ptypes.png)


#### Goal:
- Learn more about the correlations and differences between each personality type. 
- Derive visuals and compare the personality types against each other. 
- Given sufficient text, predict the personality type of the individual. 

#### Motivation:
I find psychology very interesting, I believe the more information people have, in this case about the personality type, the easier it will be for people to understand each other. Not to mention once we understand an individual's personality, we can help create an environment where they will succeed.

<a href="#Personality-Predictor">Back to top</a>

# Overview of the Data

#### First Dataset:
This data was collected through the PersonalityCafe forum, as it provides a large selection of people and their MBTI personality type, as well as what they have written.

- There are 8675 observations(rows)
- Each row has 1 individual’s personality type and their last 50 posts
- The personality type shown is selected by the user although the forum has a link to the test for those members who do not know what personality type they belong to.

| - | type | posts |
|:---:|:---:|:---:|
| 0 | INFJ | 'http://www.youtube.com/watch?v=qsXHcwe3krw|||...'|
| 1 | ENTP | 'I'm finding the lack of me in these posts ver..' |
| 2 | INTP | 'Good one _____ https://www.youtube.com/wat...' |
| 3 | INTJ | 'Dear INTP, I enjoyed our conversation the o... '|
| 4 | ENTJ | 'You're fired.|||That's another silly misconce... '|

#### Second Dataset:
This Data set comes from "MBTI Manual" published by CPP

- Shows the frequency of each personality type in the population

| - | Type | Frequency |
|:---:|:---:|:---:|
| 0 | ISFJ | 13.8% |
| 1 | ESFJ | 12.3% |
| 2 | ISTJ | 11.6% |
| 3 | ISFP | 8.8% |
| 4 | ESTJ | 8.7% |

<a href="#Personality-Predictor">Back to top</a>


# Exploratory Data Analysis


First let's take a look at how much each personality type was represented in the data set in comparison to the population. 

![](img/samplebarvpop.png)

The results were interesting, the least common personality types seemed to be most represnted in the dataset. In order to compare apples to apples, let's convert our sample count to a percentage and plot them side by side.

![](img/samplevpop.png)

For further EDA please look at the summary [here](ExploratoryDataAnalysis.md)

<a href="#Personality-Predictor">Back to top</a>


# Data Pipeline

<!-- #region -->
Let's create a data pipeline, it will aim to do the following:
- Standardize the text to ASCII
- Remove weblinks
- Tokenize the words
- Use a stemmer on the words
- Remove HTML decoding
- Remove punctuation
- Remove stopwords

The code to do this can be found [here](src/personality.py)

We make a pickle file that creates a list of words as seen below:
- These have been standardized, tokenized, stemmed and punctuations/stopwords have been removed
        
|index|type|posts|
|:---:|:---:|:---:|
|0|INFJ|[life-chang, experi, life, may, perc, experi, ...|
|1|ENTP|['m, find, lack, post, alarm, sex, bore, 's, p...|
|2|INTP|[cours, say, know, 's, bless, curs, absolut, p...|
|3|INTJ|[dear, intp, enjoy, convers, day, esoter, gab,...|
|4|ENTJ|[you, re, fire, 's, anoth, silli, misconcept, ...|

We also make a pickle file of the strings standardized, and stemmed as seen below:

|index|type|posts|
|:---:|:---:|:---:|
|0|INFJ|what has been the most life-chang experi in yo...|
|1|ENTP|i 'm find the lack of me in these post veri al...|
|2|INTP|of cours to which i say i know that 's my bles...|
|3|INTJ|dear intp i enjoy our convers the other day es...|
|4|ENTJ|you re fire that 's anoth silli misconcept tha...|

Next we create another pickle file where the full process has been applied:

|index|type|posts|
|:---:|:---:|:---:|
|0|INFJ|lifechang experi life may perc experi immers h...|
|1|ENTP|im find lack post alarm sex bore posit often e...|
|2|INTP|cours say know that bless curs absolut posit b...|
|3|INTJ|dear intp enjoy convers day esoter gab natur u...|
|4|ENTJ|your fire that anoth silli misconcept approach...|

Finally, we use the [Emotions Lexicon](http://saifmohammad.com/WebPages/NRC-Emotion-Lexicon.htm), which was created by the National Research Council Canada, to derive emotions from the text and store that in a pickle file. The code for this can be found [here](src/emotions.py):

|emotion|anger|anticipation|disgust|fear|joy|negative|positive|sadness|surprise|trust|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|0|3|22|2|3|11|14|22|4|6|13|
|1|14|23|8|15|21|24|37|12|17|18|
|2|7|26|5|12|26|17|42|9|13|31|
|3|6|14|4|7|6|14|30|7|2|20|
|4|17|30|15|13|23|29|43|14|15|24|


<a href="#Personality-Predictor">Back to top</a>
<!-- #endregion -->

# Model Selection


Went through different machine learning algorithms in order to find a model that can predict the personalities. Random would be 1/16 or 0.0625. That is really low, so for our model let's aim to achiece results higher than 50%. The code for this can be found [here](NLP_Models.ipynb)

We will use the following models:
- Random Forest                 - Accuracy = 0.3614985590778098
- Gradient Boosting Classifier  - Accuracy = 0.650787552823665
- Naive Bayes                   - Accuracy = 0.22051479062620052
- Logistic Regression           - Accuracy = 0.6300422589320015
- Support Vector Machine        - Accuracy = 0.6699961582789089

<a href="#Personality-Predictor">Back to top</a>

# Deep Learning

#### Let's create a Neural Network and see if we can get better results. The code for this can be found [here](Deep_Learning.ipynb)

    Accuracy = 0.9865539761813292
    
#### This is very impressive accuracy, let's look at the summary.
    
                  precision    recall  f1-score   support

               0       1.00      1.00      1.00        54
               1       1.00      1.00      1.00       214
               2       1.00      1.00      1.00        77
               3       1.00      1.00      1.00       186
               4       0.00      0.00      0.00        11
               5       0.00      0.00      0.00        15
               6       0.00      0.00      0.00         9
               7       1.00      1.00      1.00        21
               8       1.00      1.00      1.00       425
               9       0.98      1.00      0.99       538
              10       0.96      1.00      0.98       327
              11       0.97      1.00      0.99       408
              12       1.00      1.00      1.00        53
              13       1.00      1.00      1.00        88
              14       1.00      1.00      1.00        69
              15       1.00      1.00      1.00       108

       micro avg       0.99      0.99      0.99      2603
       macro avg       0.81      0.81      0.81      2603
    weighted avg       0.97      0.99      0.98      2603
    
#### Let's also take a look at the confusion matrix:

    [[ 54   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0]
     [  0 214   0   0   0   0   0   0   0   0   0   0   0   0   0   0]
     [  0   0  77   0   0   0   0   0   0   0   0   0   0   0   0   0]
     [  0   0   0 186   0   0   0   0   0   0   0   0   0   0   0   0]
     [  0   0   0   0   0   0   0   0   0   0   0  11   0   0   0   0]
     [  0   0   0   0   0   0   0   0   0   0  15   0   0   0   0   0]
     [  0   0   0   0   0   0   0   0   0   9   0   0   0   0   0   0]
     [  0   0   0   0   0   0   0  21   0   0   0   0   0   0   0   0]
     [  0   0   0   0   0   0   0   0 425   0   0   0   0   0   0   0]
     [  0   0   0   0   0   0   0   0   0 538   0   0   0   0   0   0]
     [  0   0   0   0   0   0   0   0   0   0 327   0   0   0   0   0]
     [  0   0   0   0   0   0   0   0   0   0   0 408   0   0   0   0]
     [  0   0   0   0   0   0   0   0   0   0   0   0  53   0   0   0]
     [  0   0   0   0   0   0   0   0   0   0   0   0   0  88   0   0]
     [  0   0   0   0   0   0   0   0   0   0   0   0   0   0  69   0]
     [  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0 108]]

<a href="#Personality-Predictor">Back to top</a>


# Emotional Analysis

Next let's dive into the emotions by each personality type. The code for this can be found [here](Emotional_Analysis.ipynb).
![](img/emoByType.png)

<a href="#Personality-Predictor">Back to top</a>


# WordClouds

#### Now let's go back to the data and see what we can derive
- Created another dictionary with high frequency words by Personality Type
 - This can help us make some word clouds but first we need to clean our data
- Created a list of the 30 most common words among all personality types
- Removed the words in that list from our dataset

Let's get a bit fancy, instead of the default wordclouds, we can use a template for them, since we are talking about the mind, let's use a head.

|Extroverted|Introverted|
|:---:|:---:|
|ENTP|INTP|
|![](img/emoENTP.png)|![](img/emoINTP.png)|

After transforming this image, using the pillow library and numpy, we can use it to produce the following wordclouds.

|Extroverted|Introverted|
|:---:|:---:|
|ENTP|INTP|
|![](img/ENTP.png)|![](img/INTP.png)|


<a href="#Personality-Predictor">Back to top</a>


# Conclusion and Next Steps

- Took the datasets and performed Exploratory Data Analysis
- Created a data pipeline
- Built several models and picked support vector machine with stochastic gradient descent due to it's high accuracy and precision
- Built a Neural Network which improved gave great accuracy but was overfit to the over represnted classes
- Performed emotional analysis for each personality type
- Created Word Clouds based on the frequancy of words used by each personality type.
- Next step would be to gather data from another place like twitter or facebook and see if we can predict personalities based on that text

<a href="#Personality-Predictor">Back to top</a>

```python

```
