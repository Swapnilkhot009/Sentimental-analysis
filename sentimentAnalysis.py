# importing libraries
import tweepy
from textblob import TextBlob
from wordcloud import WordCloud
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')

log = pd.read_csv('Login.csv')

# Havinf Twitter Crenditials
costomersKey = log['Keys'][0]
costomersSecret = log['Keys'][1]
accessToken = log['Keys'][2]
accessTokenSecret = log['Keys'][3]

print(costomersSecret)
print(costomersKey)

# creating authentication object
authenticate = tweepy.OAuthHandler(costomersKey, costomersSecret)
# Set access token and access token secret
authenticate.set_access_token(accessToken, accessTokenSecret)
# Create an APi object while passing in th eauth info
api = tweepy.API(authenticate, wait_on_rate_limit=True)

# Extrating 100 tweets form account
posts = api.user_timeline(screen_name=input(), count=100, land='en', tweet_mode="extended")
# printing last 5 tweets from twitter accout
print("Show 5 recent tweets.")
i = 1
for tweet in posts[0:5]:
    print(str(i) + ') ' + tweet.full_text + '\n')
    i = i + 1

# create dataframe with column called Tweets
df = pd.DataFrame([tweet.full_text for tweet in posts], columns=['Tweets'])


# Clean the text
# create a function to clean the text
def cleanTxt(text):
    text = re.sub(r'@[A-Za-z0-9]+', '', text)  # Removed @mentioned
    text = re.sub(r'#', '', text)  # Removed # mentioned
    text = re.sub(r'RT[\s]+', '', text)  # Removes pics
    text = re.sub(r'https?:\/\/\S+', '', text)  # Removed @websitesMentioned
    text = re.sub(r'_[A-Za-z0-9]+', '', text)
    text = re.sub(r':', '', text)
    return text


# Cleaning the text
df['Tweets'] = df['Tweets'].apply(cleanTxt)


# create function to get Subjectivity
def getSubjectivity(text):
    return TextBlob(text).sentiment.subjectivity


# create function to get Polarity
def getPolarity(text):
    return TextBlob(text).sentiment.polarity


# create teo column
df['Subjectivity'] = df['Tweets'].apply(getSubjectivity)
df['Polarity'] = df['Tweets'].apply(getPolarity)

# Ploting WordCloud
allWords = ' '.join([twts for twts in df['Tweets']])
wordCloud = WordCloud(width=500, height=300, random_state=21, max_font_size=119).generate(allWords)

plt.imshow(wordCloud, interpolation='bilinear')
plt.axis('off')
plt.show()


# create a function to compute negative , nutral or positive sentiment
def getAnalysis(score):
    if score < 0:
        return 'Negative'
    elif score == 0:
        return 'Neutral'
    else:
        return 'Positive'


df['Analysis'] = df['Polarity'].apply(getAnalysis)

# ploting polarity and subjectivity
# plt.figure(figsize=(8, 6))
# for i in range(0, df.shape[0]):
#     plt.scatter(df['Polarity'][i], df['Subjectivity'][i], color="Green")
#
# plt.title("Semtiment Analysis")
# plt.xlabel("Polarity.")
# plt.ylabel("Subjectivity.")

# Get the percetage of positive and negative tweets
ptweets = df[df.Analysis == 'Positive']
ptweets = ptweets['Tweets']

print("Positive tweets percentge")
print(round((ptweets.shape[0] / df.shape[0]) * 100, 1))

ntweets = df[df.Analysis == 'Negative']
ntweets = ntweets['Tweets']

print("Negative tweets percentge")
print(round((ntweets.shape[0] / df.shape[0]) * 100, 1))

nutweets = df[df.Analysis == 'Neutral']
nutweets = nutweets['Tweets']

print("Neutral tweets percentge")
print(round((nutweets.shape[0] / df.shape[0]) * 100, 1))


# showing bar graph
print(df['Analysis'].value_counts())


# Plot and visualize the graph
plt.title("Sentiment Analysis Bar Graph.")
plt.xlabel("Sentiment.")
plt.ylabel("Percentage.")
plt.bar(df['Analysis'].value_counts().keys(), df['Analysis'].value_counts().values)
# df['Analysis'].value_counts().plot(kind='bar')
plt.show()
