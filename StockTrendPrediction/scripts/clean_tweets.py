import pandas as pd
import re
import json
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tqdm import tqdm
import emoji

# Download NLTK resources if not already present
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Function to remove URLs
def remove_urls(text):
    return re.sub(r'http\S+|www\S+', '', text)

# Function to remove emojis
def remove_emojis(text):
    return emoji.replace_emoji(text, replace='')

# Function to clean and preprocess text
def clean_text(text, stop_words, lemmatizer):
    text = remove_urls(text)
    text = remove_emojis(text)
    text = text.lower()
    tokens = text.split()  # Use simple whitespace split instead of nltk.word_tokenize
    tokens = [t for t in tokens if t.isalpha() and t not in stop_words]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return ' '.join(tokens)

# Read Excel file
df = pd.read_excel('../../stock_tweets.xlsx')  # Adjust path if needed

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

TWEET_COLUMN = 'Tweet'
if TWEET_COLUMN not in df.columns:
    raise ValueError(f"Expected a column named '{TWEET_COLUMN}' in the Excel file.")

tqdm.pandas()
df['cleaned_text'] = df[TWEET_COLUMN].progress_apply(lambda x: clean_text(str(x), stop_words, lemmatizer))

# Save to JSON (one record per line)
df[['cleaned_text']].to_json('data/cleaned_tweets.json', orient='records', lines=True) 