import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter
import pandas as pd
import os

# Download necessary NLTK resources
nltk.download('stopwords')
nltk.download('punkt')

# Read data from CSV file
samples_add = '../mtsamples.csv'  # Assuming the CSV file is in the parent directory
samples = pd.read_csv(samples_add)
samples.fillna({'transcription': ''}, inplace=True)  # Fill NaN values with empty strings
samples = samples['transcription']  # Select only the 'transcription' column

# Text preprocessing function
def preprocess_text(text):
    # Tokenize the text
    tokens = word_tokenize(text)
    # Convert tokens to lowercase and remove non-alphanumeric tokens
    tokens = [word.lower() for word in tokens if word.isalnum()]
    return tokens

# Collect all words from the transcriptions
all_words = []
for record in samples:
    tokens = preprocess_text(record)
    all_words.extend(tokens)

# Calculate word frequencies
word_counts = Counter(all_words)

# Identify common words (assuming words with frequency greater than 20 are common)
common_words = [word for word, count in word_counts.items() if count > 60]

# Get NLTK's English stopwords list
nltk_stopwords = set(stopwords.words('english'))

# Create a custom stopwords list by combining NLTK stopwords and common words
custom_stopwords = set(common_words).union(nltk_stopwords)

# Write custom stopwords list to a file
stopwords_dir = "../stopwords"  # Directory where the stopwords file will be saved
custom_stopwords_file = "custom_stopwords.txt"  # Filename for the custom stopwords file
custom_stopwords_path = os.path.join(stopwords_dir, custom_stopwords_file)

# Create the directory 'stopwords' if it doesn't exist
if not os.path.exists(stopwords_dir):
    os.makedirs(stopwords_dir)

# Write custom stopwords to the file
with open(custom_stopwords_path, "w") as file:
    for word in custom_stopwords:
        file.write(word + "\n")

print("Custom Stopwords saved to:", custom_stopwords_path)
