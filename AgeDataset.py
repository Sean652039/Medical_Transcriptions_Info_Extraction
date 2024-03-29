from torch.utils.data import Dataset
import torch


class AgeDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cuda")

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            return_token_type_ids=True,
            truncation=True
        )

        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]

        # Replace None label with a specific value, e.g., -1
        if label is None:
            label = -1

        # Move tensors to the specified device
        label = torch.tensor(label).clone().detach().to(self.device)

        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long).clone().detach().to(self.device),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long).clone().detach().to(self.device),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long).clone().detach().to(self.device),
            'label': torch.tensor(label, dtype=torch.float).clone().detach().to(self.device)
        }
