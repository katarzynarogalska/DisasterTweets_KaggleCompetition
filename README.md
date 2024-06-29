# DisasterTweets Kaggle competition
Social Media like Tweeter, Facebook and Instagram play a huge role in everyday life. 
It is a big source of information but is it possible for a computer to understand a person's true intentions?
In both of the Tweets: 
* *"Fire near Brooke Street! Stay safe"*
* *"Your photo is absolutely fire!!!"*

appears the word 'fire'. We humans have no trouble understanding the different meaning of the same word. Will a computer also be able to do so?

# Main goal
The main purpose of our project is to construct a machine learning model capable of discerning genuine meaning and intentions behing a given Tweet.

# Dataset
We are using data from [Kaggle competition](https://www.kaggle.com/competitions/nlp-getting-started?fbclid=IwZXh0bgNhZW0CMTAAAR3cCDGk3Lp4ExV1M4CNy-hRDu8fXc8Pqno1sBpDEHzr1JWog2lxoCRI7j8_aem_AYZB3kUeXJIXS8j73e8LUYSfA8oGcbd_-2ir18kaNF1b2ldTSq3Q3nDRB8dj61hMFs9sDyeeXcvkg57fuFbfMfdQ), which is a dataframe created by figure-eight and originally shared on their ‘Data For Everyone’ website.
 In the dataset, the following columns are included:
* id - a unique identifier for each tweet
* keyword -  a particular keyword from the tweet
* location - the location the tweet was sent from
* text - the text of the tweet
* **target** - denotes whether a tweet is about a real disaster (1) or not (0)
  
# Project Structure
```
DisasterTweets_KaggleCompetition/
  ├── data/               # Raw data & submission file
  ├── img/                # Where images are stored
  ├── notebooks/          # Jupyter Notebooks 
  ├── pipelines/          # pipelines for processing data 
  ├── reqirements.txt     # needed tool versions
  ├── Project_report.pdf  # documentation for this project
  └── README.md           # This file
```
# Milestones 
## EDA and Feature engineering
In this part we focused on analysing the data. How can we gather more information from a plain text? 
Using available NLP methods and data visualisations we tried to discover interesting patterns.

**Technical information:** *Code to this part can be found in Tweets_EDA, but all transformations like adding new columns are performed inside transformers for pipelines in /src files*
## Feature importance 
After adding lots of new features we need to take a step back and analyse which of them are really relevant for our models, and which of them are just a noise.
In this part we used common feature importance methods, as well as correlation matrix analysis. 

**Technical information:** Code to this step is can be found in Tweets_Feature_Importance, however as previous all dataframe transformations will be performed in transformer classes in /src

## Model building
...
# Authors
* [Katarzyna Rogalska](https://github.com/katarzynarogalska)
* [Michał Pytel](https://github.com/Michael-Pytel)
* [Jakub Półtorak](https://github.com/JakubPoltorak147)
