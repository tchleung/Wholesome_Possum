import re
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# instantiate dependables
stopwordslist = stopwords.words('english')
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    try:
        # get rid of emojis
        emoji_pattern = re.compile("["
            u"\U0001F600-\U0001F64F"  # emoticons
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # transport & map symbols
            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
            u"\U0001F1F2-\U0001F1F4"  # Macau flag
            u"\U0001F1E6-\U0001F1FF"  # flags
            u"\U0001F600-\U0001F64F"
            u"\U00002702-\U000027B0"
            u"\U000024C2-\U0001F251"
            u"\U0001f926-\U0001f937"
            u"\U0001F1F2"
            u"\U0001F1F4"
            u"\U0001F620"
            u"\u200d"
            u"\u2640-\u2642"
            "]+", flags=re.UNICODE)
        text = emoji_pattern.sub(r'', text)

        # get rid of url
        text = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|'\
                           '(?:%[0-9a-fA-F][0-9a-fA-F]))+','', text)

        # get rid of users mention
        text = re.sub("(u/[A-Za-z0-9-_]+)","", text)

        # get rid of subreddit mention
        text = re.sub("(r/[A-Za-z0-9-_]+)","", text)

        # get rid of Nintendo Friend Code reference
        text = re.sub("([Ss][Ww]-[0-9-_]+)","", text)
        text = re.sub("([Ss][Ww])","", text)

        # remove all special characters and numbers
        text = re.sub('[^A-Za-z]+', ' ', text)
        text = text.lower()
        text = text.strip()

        # remove stopwords
        textlist = [word for word in text.split(' ') if word not in stopwordslist]

        # lemmatize
        textlist = [lemmatizer.lemmatize(word) for word in textlist]

        text = ' '.join(textlist)

        return text
    except:
        print(text)