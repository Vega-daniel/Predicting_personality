# Personality Predictor
by Daniel Vega


# Table of Contents
- [Introduction](#Introduction)
- [Strategy and Process](#Strategy-and-Process)
- [Overview of the Data](#Overview-of-the-Data)
- [Exploratory Data Analysis](#Exploratory-Data-Analysis)
- [Hypothesis Testing](#Hypothesis-Testing)
- [Wordclouds](#WordClouds)
- [Conclusion and Next Steps](#Conclusion-and-Next-Steps)

# Introduction

The Myers–Briggs Type Indicator (MBTI) is an introspective self-report questionnaire with the purpose of indicating differing psychological preferences in how people perceive the world around them and make decisions.
The MBTI was constructed by Katharine Cook Briggs and her daughter Isabel Briggs Myers. It is based on the conceptual theory proposed by Carl Jung, who had speculated that humans experience the world using four principal psychological functions – sensation, intuition, feeling, and thinking – and that one of these four functions is dominant for a person most of the time.



![](img/ptypes.png)


#### Goal:
Learn more about the correlations and differences between each personality type. Derive visuals and compare the personality types against each other.

#### Motivation:
I find psychology very interesting, I believe the more information people have, in this case about the personality type, the easier it will be for people to understand each other.

<a href="#Personality-Predictor">Back to top</a>


# Strategy and Process
- Overview of the Data
- Exploratory Data Analysis
- Hypothesis Testing
- Visual Represantations


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

Now we can make better sense of the data. It looks like IN-- personality types are highly represented in this sample.

The takeaway being IN-- personalities seem to have a stronger presence in this online forum.

<a href="#Personality-Predictor">Back to top</a>


# Hypothesis Testing




Looking at the data above, we can see that having “IN--” in the personality increases the chances of being active on this forum. Let's test this:

> $H_0$:"IN--" personalities have an equal chance of being in this online forum


| IN-- | PopulationFreq | SampleFreq | Count |
|:---:|:---:|:---:|:---:|
| 0 | 89 | 34.33 | 2978 |
| 1 | 11 | 65.67 | 5697 |


We have a total of 8675 users, and 2978 of them are "IN--" we will test our hypothesis below, we will reject our hypothesis if we get a p-value greater than 0.05

$$ \text{# of "IN--"} \approx Binomial(8675, 0.11) $$

The central limit theorem tells us that a binomial with large $N$ is well approximated by a Normal distribution with the appropriate mean and varaince. Let's take a look at both plots belows.

$$ Binomial(8675, 0.11) \approx N(8675 \times 0.11, \sqrt{8675 \times 0.11 \times 0.89}) $$


![](img/distributions.png)


Let's continue with the Normal Distribution

The p-value for this is:

$$ P(\geq \text{ 2978 'IN--' observations} \mid \text{Null Hypothesis} ) $$

![](img/pvalue.png)


#### Based on the data (p-value was below 0), we reject the Null Hypothesis

Let's take a look at all "IN--" personalities

| Type | PopulationFreq | SampleFreq | Count | Questions/Post |
|:---:|:---:|:---:|:---:|:---:|
| INFP | 4.4 | 21.12 | 1832 | 0.20 |
| INTP | 3.3 | 15.03 | 1304 | 0.22 |
| INTJ | 2.1 | 12.58 | 1091 | 0.21 |
| INFJ | 1.5 | 16.95 | 1470 | 0.21 |


INTPs seem to ask more questions per post. Can we confidently say that INTPs ask more questions than the rest?

![](img/ratios.png)


#### Let's take a skeptical stance, and clearly state this Hypothesis.

> $H_0$: there is no difference in the average amount of questions asked between INTP and INTJ.

> $H_0$: there is no difference in the average amount of questions asked between INTP and INFJ.

> $H_0$: there is no difference in the average amount of questions asked between INTP and INFP.

Our question concerns population averages (is INTP's question/post average different than INTJ, INFJ and INFP).  Our measurements are sample averages, which, from the central limit theorem, we know are approximately normally distributed given the population average

$$ \text{Sample average of INTP's questions} \sim Normal \left( \mu_T, \sqrt{\frac{\sigma^2_T}{1304}} \right) $$
$$ \text{Sample average of INTJ's questions} \sim Normal \left( \mu_J, \sqrt{\frac{\sigma^2_J}{1091}} \right) $$
$$ \text{Sample average of INFJ's questions} \sim Normal \left( \mu_F, \sqrt{\frac{\sigma^2_F}{1470}} \right) $$
$$ \text{Sample average of INFP's questions} \sim Normal \left( \mu_P, \sqrt{\frac{\sigma^2_P}{1832}} \right) $$

If we are willing to assume that the Questions posted by INTP are independent from the other personalities, then we can compress the important information into one normal distribution

$$ \text{Difference in sample averages} \sim Normal \left( \mu_T - \mu_J, \sqrt{\frac{\sigma^2_T}{1304} + \frac{\sigma^2_J}{1091}} \right) $$
$$ \text{Difference in sample averages} \sim Normal \left( \mu_T - \mu_F, \sqrt{\frac{\sigma^2_T}{1304} + \frac{\sigma^2_F}{1470}} \right) $$
$$ \text{Difference in sample averages} \sim Normal \left( \mu_T - \mu_P, \sqrt{\frac{\sigma^2_T}{1304} + \frac{\sigma^2_P}{1832}} \right) $$

Under the assumption of the null hypothesis

$$ \text{Difference in sample averages} \sim Normal \left( 0, \sqrt{\frac{\sigma^2_T}{1304} + \frac{\sigma^2_J}{1091}} \right) $$
$$ \text{Difference in sample averages} \sim Normal \left( 0, \sqrt{\frac{\sigma^2_T}{1304} + \frac{\sigma^2_F}{1470}} \right) $$
$$ \text{Difference in sample averages} \sim Normal \left( 0, \sqrt{\frac{\sigma^2_T}{1304} + \frac{\sigma^2_P}{1832}} \right) $$


In cases where we have to independently estiamte the variance of a normal distribution from the same samples we are testing, this estimation of the variance contributes to uncertenty in our test.  This means that the Normal distribution is then **too precise** to use as a conservative estimate of the p-value.


### Welch's t-test

To recify the problem, we first convert to a sample statistic whose variance is expected to be $1$

$$ \frac{\text{Difference in sample averages}}{\sqrt{\frac{\sigma^2_T}{1304} + \frac{\sigma^2_J}{1091}}} $$
$$ \frac{\text{Difference in sample averages}}{\sqrt{\frac{\sigma^2_T}{1304} + \frac{\sigma^2_F}{1470}}} $$
$$ \frac{\text{Difference in sample averages}}{\sqrt{\frac{\sigma^2_T}{1304} + \frac{\sigma^2_P}{1832}}} $$


Now we still have a similar issue to the two sample test of population proportions, we do not know the population varainces in the denominator of the formula, so our only recourse is to substitute in the sample variances

>Welch Test Statistic(INTP v. INTJ): 1.17

>Welch Test Statistic(INTP v. INFJ): 2.35

>Welch Test Statistic(INTP v. INFP): 3.89


Unfortuantely, this changes the distribution of the test statistic.  Instead of using a normal distribution, we must now use a **Student's t-distribution**, which accounts for the extra uncertainty in estimating the two new parameters.

The t-distribution always has mean $0$ and varaince $1$, and has one parameter, the **degrees of freedom**.  Smaller degrees of freedom have heavyer tails, with the distribution becoming more normal as the degrees of freedom gets larger.

The resulting application to our situation results in [Welch's t-test](https://en.wikipedia.org/wiki/Welch's_t-test).

> Degrees of Freedom for Welch's Test: 2312.08

>Degrees of Freedom for Welch's Test: 2751.33

>Degrees of Freedom for Welch's Test: 2678.67


![](img/welchttest.png)

![](img/pvalueregion.png)

Based on the result we can see that our datasets are not normally distributed, this is a good lesson for next time. Plot the distribution in the beginning before moving forward.


#### This means we must use: *Mann-Whitney Signed Rank Test*
Let us rephrase our null hypothesis to what we started with:

> $H_0$: INTPs ratio of questions to posts are equally likely to INTJs. i.e
  
  $$P(\text{INTPs questions/post} > \text{INTJs questions/post}) = 0.5$$

> $H_0$: INTPs ratio of questions to posts are equally likely to INFJs. i.e
  
  $$P(\text{INTPs questions/post} > \text{INFJs questions/post}) = 0.5$$

> $H_0$: INTPs ratio of questions to posts are equally likely to INFPs. i.e  
  
  $$P(\text{INTPs questions/post} > \text{INFPs questions/post}) = 0.5$$

We will set a rejection threshold of **0.01**

>p-value for INTP > INTJ: 0.07070

>p-value for INTP > INFJ: 0.00052

>p-value for INTP > INFP: 0.00002


Based on our results:

> we fail to reject the first Null Hypothesis

> we reject the second Null Hypothesis

> we reject the third Null Hypothesis

<a href="#Personality-Predictor">Back to top</a>


# WordClouds

#### Now let's go back to the data and see what we can derive
- Created a dictionary with all the observations of each Personality Type
 - After doing so it did not prove to be very useful
- Created another dictionary with high frequency words by Personality Type
 - This can help us make some word clouds but first we need to clean our data
- Created a list of the 30 most common words among all personality types
- Removed the words in that list from our dataset

Let's get a bit fancy, instead of the default wordclouds, we can use a template for them, since we are talking about the mind, let's use a head.

<img src="img/head2.png" alt="drawing" width="250"/>

After transforming this image, using the pillow library and numpy, we can use it to produce the following wordclouds.

|Extroverted|Introverted|
|:---:|:---:|
|ENFJ|INFJ|
|![](img/ENFJ.png)|![](img/INFJ.png)|
|ENFP|INFP|
|![](img/ENFP.png)|![](img/INFP.png)|
|ENTJ|INTJ|
|![](img/ENTJ.png)|![](img/INTJ.png)|
|ENTP||INTP|
|![](img/ENTP.png)|![](img/INTP.png)|
|ESFJ|ISFJ|
|![](img/ESFJ.png)|![](img/ISFJ.png)|
|ESFP|ISFP|
|![](img/ESFP.png)|![](img/ISFP.png)|
|ESTJ|ISTJ|
|![](img/ESTJ.png)|![](img/ISTJ.png)|
|ESTP|ISTP|
|![](img/ESTP.png)|![](img/ISTP.png)|

<a href="#Personality-Predictor">Back to top</a>


# Conclusion and Next Steps

- Took the datasets and compared the sample agianst the population
- Found "IN--" personalities are more common in the PersonalityCafe Forum.
- Found that INTPs post more questions per post than INFJ and INFP
- Created Word Clouds based on the frequancy of words used by each personality type.
- Next step would be to run our PersonalityCafe dataset through an Natural Language Processing (NLP) model and see if we can predict each personality type based on posts.

<a href="#Personality-Predictor">Back to top</a>
