import re
import os
import torch
import torch.nn as nn
from transformers import BertTokenizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from AgeDataset import AgeDataset
from AgeExtractionModel import AgeExtraction


class ExtractAge:
    def __init__(self, df_samples):
        self.df_samples = df_samples

        # Load pre-trained BERT model and tokenizer
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = AgeExtraction()

        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cuda")
        self.model.to(self.device)

    def pseudo_labels(self, data):

        # define it based on the results from Context Analysis
        age_pattern = re.compile(r'(\d{1,3})[-\s]?year[-\s]?old')
        pseudo_labels = []
        for text in data:
            if text == '':
                pseudo_labels.append(None)
                continue
            matches = age_pattern.findall(text)
            age = None
            if matches:
                age = int(matches[0])
            pseudo_labels.append(age)

        return pseudo_labels

    def train_model(self):

        transcriptions = self.df_samples['transcription'].tolist()

        dataset = AgeDataset(transcriptions, self.pseudo_labels(transcriptions), self.tokenizer)
        dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

        # Define loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=2e-5)

        num_epochs = 3
        for epoch in range(num_epochs):  # Example: 10 epochs
            self.model.train()
            progress_bar = tqdm(dataloader, desc=f'Epoch {epoch + 1}', leave=False)
            for batch in progress_bar:
                input_ids = batch['input_ids']
                attention_mask = batch['attention_mask']
                token_type_ids = batch['token_type_ids']
                labels = batch['label'].to("mps")

                optimizer.zero_grad()
                logits = self.model(input_ids, attention_mask, token_type_ids).squeeze(-1)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()

            print(f'Epoch {epoch + 1}, Loss: {loss.item()}')

        # Check if directory exists, create if it doesn't
        save_dir = 'model'
        os.makedirs(save_dir, exist_ok=True)

        # Save the model
        torch.save(self.model.state_dict(), os.path.join(save_dir, 'trained_model_age.pth'))