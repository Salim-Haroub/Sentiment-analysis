#!/usr/bin/env python
# coding: utf-8

# # Sentiment Analysis on twitter_samples and Movie_reviews Corpora.

# In[3]:


#Importing necessary libraries

import nltk
import matplotlib.pyplot as plt
from nltk.corpus import movie_reviews
from nltk.corpus import twitter_samples
from nltk.sentiment import SentimentIntensityAnalyzer


# In[4]:


# Using nltk built-in sentiment analyzer on twitter_samples
sia = SentimentIntensityAnalyzer()
fileid = twitter_samples.fileids()
fileid


# In[5]:


twitter_samples.strings()


# In[6]:


tweets =[t.replace('://', '//') for t in twitter_samples.strings()]
#cleantweets = [t.replace('://', '//') for t in tweets]


# In[7]:


def is_positive(tweet: str)-> bool:
    return sia.polarity_scores(tweet)['compound'] > 0


# In[8]:


import random
from random import shuffle
shuffle(tweets)
for tweet in tweets[:20]:
    print(is_positive(tweet), tweet)


# In[20]:


def gender_features(word):
    return{'suffix1': word[-1], 'suffix2': word[-2:], 'suffix3':word[-3:]}
labeled_names = ([(name, 'male') for name in nltk.corpus.names.words('male.txt')] +
                [(name, 'female') for name in nltk.corpus.names.words('female.txt')])

shuffle(labeled_names)

len(labeled_names)


# In[21]:


from nltk.classify import apply_features
train_set = apply_features(gender_features, labeled_names[1500:])
devtest_set = apply_features(gender_features, labeled_names[500:1500])
test_set = apply_features(gender_features, labeled_names[:500])

classifier = nltk.classify.NaiveBayesClassifier.train(train_set)


# In[22]:


print(nltk.classify.accuracy(classifier, devtest_set))


# In[23]:


classifier.show_most_informative_features()


# In[37]:


classifier.classify(gender_features('Erin'))


# In[24]:


print(nltk.classify.accuracy(classifier, test_set))


# In[39]:


error = []

for (name, tag) in labeled_names:
    guess = classifier.classify(gender_features(name))
    if guess!= tag:
        error.append((tag, guess, name))
        
for (tag, guess, name) in error:
    print('correct = {:<10} guessed = {:<10} name = {}'.format(tag, guess, name))


# In[43]:


print('\U0001F44E', '\U0001F44D')


# In[44]:


twitter_samples.fileids()


# In[67]:


tweets = [w.replace('://', ' //') for w in twitter_samples.strings()]
shuffle(tweets)


# In[50]:


from nltk.sentiment import SentimentIntensityAnalyzer


# In[65]:


sia = SentimentIntensityAnalyzer()
#unctweets = twitter_samples('tweets.20150430-223406.json')
sia.polarity_scores(str(twitter_samples.strings('tweets.20150430-223406.json')))


# In[110]:


def is_positive(strings):
    return sia.polarity_scores(strings)['compound']>0

for strings in tweets[:30]:
    
    if is_positive(strings) == True:
        mess = '\U0001F44D'
    else:
        
        mess = '\U0001F44E'
    print('{:<5} {} {}'.format(mess, strings, sia.polarity_scores(strings)['compound']))


# In[112]:


movie_reviews.categories()


# In[113]:


posreviews = movie_reviews.words(categories = 'pos')
negreviews = movie_reviews.words(categories = 'neg')

allreviews = posreviews + negreviews


# In[158]:


pos_ids = movie_reviews.fileids(categories = 'pos')
neg_ids = movie_reviews.fileids(categories = 'neg')
all_ids = pos_ids + neg_ids

raw = movie_reviews.raw(all_ids)
type(raw)


# In[115]:


from statistics import mean


# In[163]:


def positive(ids):
    
    score = [sia.polarity_scores(text)['compound'] for text in nltk.sent_tokenize(movie_reviews.raw(ids))]
    
    return mean(score)>0
    
    
    


# In[164]:


positive(pos_ids)


# In[165]:


correct = 0

shuffle(all_ids)
x = all_ids
for reviews in x:
    if positive(reviews):
        
        if reviews in pos_ids:
        
            text = movie_reviews.raw(reviews)[0:80]
       
            print(f'\U0001F44D, {text}')
            
            correct += 1
            
        else:
        
            if reviews in neg_ids:
                
                text2 = movie_reviews.raw(reviews)[0:80]
            
                print(f'\U0001F44E, {text2}')
        
                correct +=1


# In[166]:


print(f'{correct/len(all_ids):.2%} correct')


# In[167]:


shuffle(all_ids)

for reviews in all_ids:
    if positive(reviews):
        if reviews in pos_ids:
            correct+=1
            
        else:
            if reviews in neg_ids:
                correct+=1
                
                
print(f'{correct/len(all_ids):.2%} correct')


# In[168]:


y = movie_reviews.raw(all_ids)
type(y)


# In[169]:


z = movie_reviews.words(all_ids)
type(z)


# In[170]:


z[:10]


# In[171]:


y[:40]


# In[ ]:





# In[ ]:




