import string
from collections import Counter
import matplotlib.pyplot as plt
from nltk.corpus import stopwords  # has all the stop words
from nltk.tokenize import word_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# opening the txt file
my_text = open('SentiText.txt', encoding='utf-8').read()

# converting to lower case
low_case = my_text.lower()

# removing punctuations
cleaned_text = low_case.translate(str.maketrans('', '', string.punctuation))
# 1: chars to replace
# 2: with which they had to replace
# 3: chars that needs to be deleted

# Converting to list
tokenized_word = word_tokenize(cleaned_text, "english")

# creating a list of final words
final_words = []
for word in tokenized_word:
    if word not in stopwords.words('english'):
        final_words.append(word)

# here we get the emotions related with the words in final_words[] list
emotion_list = []
with open('emotions.txt', 'r') as file:  # opening the file
    for line in file:  # looping through file
        clear_line = line.replace('\n', '').replace(",", '').replace("'", "").strip()  # first removed the empty line
        # \n then the , then ' and then extra spaces by strip

        word, emotion = clear_line.split(":")  # unpacking the clearLine

        if word in final_words:
            emotion_list.append(emotion)

print(emotion_list)
#  count each emotion

w = Counter(emotion_list)
print(w)


def sentiment_analyser(sentiment_text):
    score = SentimentIntensityAnalyzer().polarity_scores(sentiment_text)
    print(score)
    neg = score['neg']
    neu = score['neu']
    pos = score['pos']

    if neg > pos:
        print("Negative sentiment")
    elif pos > neg:
        print("Positive Sentiment")


sentiment_analyser(cleaned_text)

fig, axl = plt.subplots()
plt.title("Sentiment Analysis Bar Graph.")
plt.xlabel("Sentiment(Emotions).")
plt.ylabel("Occurrence")
axl.bar(w.keys(), w.values())
fig.autofmt_xdate()
plt.savefig('graph.png')
plt.show()
