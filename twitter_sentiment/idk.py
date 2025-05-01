import os
from functools import partial
import torch.utils.data as data

from pytorch_lightning.callbacks import ModelCheckpoint

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from transformer_implementation import *
import pandas as pd

# NOTE: Setting up a tokenizer to transform the text data into a format that can be fed into the model.
# pip install transformers torch (OR pip install transformers tensorflow, if you want to use tensorflow instead)
# from transformers import AutoTokenizer
# tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
# input_dim = tokenizer.vocab_size # 30522

import string
class Tokenizer():
    def __init__(self):
        self.vocab = {c: i for i, c in enumerate(string.printable)}
        self.vocab_size = len(self.vocab)
    def encode(self, text, add_special_tokens=False, truncation=False, padding=False, max_length=None, return_tensors=None):
        tokens = [self.vocab.get(c, -1) for c in text if c in self.vocab]
        if truncation and max_length is not None:
            tokens = tokens[:max_length]
        if padding and max_length is not None:
            tokens += [0] * (max_length - len(tokens))
        # if add_special_tokens:
        #     tokens = [self.vocab['[CLS]']] + tokens + [self.vocab['[SEP]']]
        if return_tensors == 'pt':
            return torch.tensor(tokens, dtype=torch.long)
        return tokens
tokenizer = Tokenizer()
input_dim = tokenizer.vocab_size # 62
root_dir  = 'C:/Files/Development/AI/Transformers/transformer_practice/integrations'

# Path to the folder where the datasets are/will be downloaded
DATASET_PATH = root_dir+ '/twitter_sentiment/datasets'
# Path to the folder where the model checkpoints will be saved
CHECKPOINT_PATH = root_dir+ '/twitter_sentiment/checkpoints'

print(CHECKPOINT_PATH)

pl.seed_everything(42) # Set the random seed for reproducibility

# Ensure that all operations are deterministic on GPU (if used) for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# The below code trains a transformer model to answer the question: "Is this a positive, negative or neutral sentiment?"

sentiment_labels = {
    'Positive': 0,
    'Negative': 1,
    'Neutral': 2,
    'Irrelevant': 3
}

def path_join(path, fname):
    return os.path.abspath(os.path.join(DATASET_PATH, fname)).replace('\\', '/')
class SentimentDataset(data.Dataset):
    def __init__(self, seq_len, num_categories, test_sample_frac, mode, test_rows=None):
        super().__init__()
        self.num_categories = num_categories
        self.seq_len = seq_len
        # self.size = dataset_size

        self.mode = mode.lower().strip()
        if mode == "train":
            self.data = pd.read_csv(path_join(DATASET_PATH, "twitter_training.csv")).drop_duplicates().reset_index()
            self.data = self.data[~self.data.index.isin(test_rows)]
        elif mode == "validation":
            self.data = pd.read_csv(path_join(DATASET_PATH, "twitter_validation.csv")).drop_duplicates().reset_index()
        elif mode == "test":
            self.data = pd.read_csv(path_join(DATASET_PATH, "twitter_training.csv")).drop_duplicates().reset_index()
            self.data = self.data.sample(n=round(len(self.data)*test_sample_frac), random_state=42)
            self.index_vals = self.data.index
        self.size = len(self.data)
        self.labels = torch.tensor(self.data['sentiment'].map(sentiment_labels).tolist())#.unsqueeze(1).unsqueeze(1)
        # self.data = torch.tensor(self.data['message'].apply(lambda x: torch.Tensor(tokenizer.encode(str(x), add_special_tokens=True, truncation=True, padding='max_length', max_length=seq_len)).float()).values)
        self.data = torch.stack(
            self.data['message'].apply(
                lambda x: torch.tensor(
                    tokenizer.encode(
                        str(x),
                        add_special_tokens=True,
                        truncation=True,
                        padding='max_length',
                        max_length=seq_len,
                        return_tensors='pt'
                    ),
                    dtype=torch.float32
                )
            ).tolist()
        )

        # df_validation = pd.read_csv(os.path.join(DATASET_PATH, "twitter_validation.csv")).drop_duplicates()
        # # self.data = torch.randint(num_categories, size=(dataset_size, seq_len)) # [dataset_size, seq_len]
        # csv_path = os.path.join(DATASET_PATH, "financial_sentiment_data.csv")  # Replace with your CSV file path
        # df = pd.read_csv(csv_path)
        # # seq_len = len(max(df['Sentence'], key=lambda x: len(x)))
        # df['Sentence'] = df['Sentence'].apply(lambda x: tokenizer.encode(x, add_special_tokens=True, truncation=True, padding='max_length', max_length=seq_len))  # Tokenize the sentences]
        # self.data = torch.tensor(df['Sentence'].values, dtype=torch.long)  # Convert DataFrame to torch tensor
        # df['Sentiment'] = df['Sentiment'].map(sentiment_labels)  # Map sentiment labels to integers
        # self.labels = torch.tensor(df['Sentiment'].values, dtype=torch.long)  # Convert labels to torch tensor

    def get_index(self):
        return None if self.mode != 'test' else self.index_vals

    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        # try:
        #     labels = self.labels[idx]
        # except KeyError:
        #     print(f'KeyError: {idx}\t|\t{self.labels.shape}, {self.data.shape}')
        #     raise KeyError
        return self.data[idx], self.labels[idx]

dataset = partial(SentimentDataset, 280, 4, 0.1)
test_dataset = dataset("test")
test_loader  = data.DataLoader(test_dataset, batch_size=128)
train_loader = data.DataLoader(dataset("train", test_rows=test_dataset.get_index()), batch_size=128, shuffle=True, drop_last=True, pin_memory=True)
val_loader   = data.DataLoader(dataset("validation"), batch_size=128)
# inp_data, labels = train_loader.dataset[0]
# print(f"Input data: {inp_data}\nLabels:     {labels}")

class SentimentPredictor(TransformerPredictor):
    def _calculate_loss(self, batch, mode="train"):
        inp_data, labels = batch

        # inp_data = F.one_hot(inp_data, num_classes=self.hparams.num_classes).float()
        preds = self.forward(inp_data, add_positional_encoding=True)
        # NOTE: by transforming the predictions and labels as shown below, we make the loss function treat each token prediction independently when
        # calculating the loss. 
        # preds.view(-1, preds.size(-1)): [batch_size, seq_len, num_classes] => [batch_size * seq_len, num_classes]
        # labels.view(-1): [batch_size, seq_len] => [batch_size * seq_len]
        loss = F.cross_entropy(preds.view(-1, preds.size(-1)), labels.view(-1))
        acc = (preds.argmax(dim=-1) == labels).float().mean()

        self.log(f"{mode}_loss", loss)
        self.log(f"{mode}_acc", acc)
        return loss, acc
    
    def training_step(self, batch, batch_idx):
        loss, _ = self._calculate_loss(batch, mode="train")
        return loss
    
    def validation_step(self, batch, batch_idx):
        _ = self._calculate_loss(batch, mode="val")

    def test_step(self, batch, batch_idx):
        _ = self._calculate_loss(batch, mode="test")


def train_sentiment(**kwargs):
    # Create a PyTorch Lightning trainer with the generation callback
    root_dir = os.path.join(CHECKPOINT_PATH, "twitter_sentiment")
    os.makedirs(root_dir, exist_ok=True)
    # NOTE: The gradient_clip_val argument prevents exploding gradients during backpropagation
    trainer = pl.Trainer(default_root_dir=root_dir,
                         callbacks=[ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_acc")],
                         accelerator="gpu" if str(device).startswith("cuda") else "cpu",
                         devices=1,
                         max_epochs=10,
                         gradient_clip_val=1
                         )
    trainer.logger._default_hp_metric = None # Optional logging argument that we don't need
    pretrained_filename = os.path.join(CHECKPOINT_PATH, "SentimentTask.ckpt")
    if os.path.isfile(pretrained_filename):
        print("Found pretrained model. Loading...")
        model = SentimentPredictor.load_from_checkpoint(pretrained_filename)
    else:
        model = SentimentPredictor(max_iters=trainer.max_epochs*len(train_loader), **kwargs)
        trainer.fit(model, train_loader, val_loader)
    # Test best model on validation and test set
    val_result = trainer.test(model, val_loader, verbose=False)
    test_result = trainer.test(model, test_loader, verbose=False)
    result = {"test_acc": test_result[0]["test_acc"], "val_acc": val_result[0]["test_acc"]}
    model = model.to(device)
    return model, result

# print(input_dim) 
# 5087
model, result = train_sentiment(
    input_dim = 280,#tokenizer.vocab_size,
    model_dim = 600,
    num_heads =  5,
    num_classes = train_loader.dataset.num_categories,
    num_layers = 10,
    dropout = 0.2,
    lr = 5e-5,
    warmup = 50,
    max_seq_len=1
)


print(f"\nVal accuracy:  {(100.0 * result['val_acc']):4.2f}%")
print(f"Test accuracy: {(100.0 * result['test_acc']):4.2f}%")
