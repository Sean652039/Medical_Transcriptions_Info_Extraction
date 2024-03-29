import re
import os
import torch
import numpy as np
from tqdm import tqdm
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer

from TreatmentExtractionModel import TreatmentLDA

nltk.download('punkt')

class ExtractTreatment:
    def __init__(self, df_samples, num_topics):
        self.samples = df_samples['transcription']  # Initialize with the samples dataframe
        self.num_topics = num_topics  # Number of topics to extract

        # Check and set the device to GPU if available
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cuda")

    # Function for text cleaning and preprocessing
    def preprocess_text(self, text):
        # Remove special characters and punctuation
        text = re.sub(r'[^\w\s]', '', text)
        # Convert text to lowercase
        text = text.lower()
        # Tokenize the text
        tokens = word_tokenize(text)
        # Remove stopwords
        # Load custom stopwords
        with open('stopwords/custom_stopwords.txt', 'r') as file:
            custom_stopwords = set(file.read().splitlines())
        tokens = [word for word in tokens if word not in custom_stopwords]
        # Stemming
        stemmer = SnowballStemmer('english')  # Stemmer
        tokens = [stemmer.stem(word) for word in tokens]
        # Join tokens back into a string
        preprocessed_text = ' '.join(tokens)

        return preprocessed_text

    # Function to train the LDA model
    def train_model(self):
        # Preprocess the samples
        """
        For each patient, use their transcription to train LDA.
        """
        treatments_info = {}
        for idx, sample in tqdm(enumerate(self.samples), desc="Processing samples"):
            treatment = []
            if sample == '':
                treatments_info[f"patient {idx}"] = treatment
                continue

            sample = [self.preprocess_text(sample)]

            # deal with the problems that all content are in stop words
            if sample[0] == '':
                treatments_info[f"patient {idx}"] = treatment
                continue

            # Build the document-term matrix
            vectorizer = CountVectorizer()
            X = vectorizer.fit_transform(sample)
            X = torch.from_numpy(X.toarray()).float()  # Convert to PyTorch tensor
            X = X.to(self.device)  # Move to GPU if available

            # Create and move the LDA model to GPU (if available)
            model = TreatmentLDA(X.shape[1], self.num_topics).to(self.device)

            # Define optimizer
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

            num_epochs = 1000
            for epoch in range(num_epochs):
                optimizer.zero_grad()
                x_hat = model(X)
                loss = torch.sum(-X * torch.log(x_hat + 1e-10))  # Cross-entropy loss
                loss.backward()
                optimizer.step()

            # Generate document-topic distribution using the trained LDA model
            topic_word_dist = torch.softmax(model.phi, dim=0).cpu().detach().numpy()

            feature_names = np.array(vectorizer.get_feature_names_out())
            top_words_all_topics = []
            from collections import Counter
            for topic_idx, topic in enumerate(topic_word_dist):
                num_words = len(topic) // 2
                top_words_idx = topic.argsort()
                top_words = feature_names[top_words_idx][:num_words]
                top_words_all_topics.extend(top_words)

            # Count frequencies of all words
            word_counts_all_topics = Counter(top_words_all_topics)
            # choose top 5 words as treatment, since 1 patience may get multiple treatments
            top_5 = word_counts_all_topics.most_common(5)
            treatment = [word for word, _ in top_5]
            treatments_info[f"patient {idx}"] = treatment

        # Create the 'treatments' directory if it doesn't exist
        treatments_dir = 'treatments'
        if not os.path.exists(treatments_dir):
            os.makedirs(treatments_dir)
        filename = os.path.join(treatments_dir, f'treatments.txt')
        with open(filename, 'w') as file:
            for key, value in treatments_info.items():
                file.write(f'{key}: {value}\n')
