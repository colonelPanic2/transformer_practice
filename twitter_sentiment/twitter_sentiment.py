import os
from functools import partial
import torch.utils.data as data

from pytorch_lightning.callbacks import ModelCheckpoint

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from transformer_implementation import *
import pandas as pd
import re
# NOTE: Setting up a tokenizer to transform the text data into a format that can be fed into the model.
# pip install transformers torch (OR pip install transformers tensorflow, if you want to use tensorflow instead)
from transformers import AutoTokenizer, AutoModel
TOKENIZER = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")
EMBEDDING_LAYER = model.embeddings.word_embeddings
input_dim = TOKENIZER.vocab_size # 30522

# import string
# class Tokenizer():
#     def __init__(self):
#         self.vocab = {c: i for i, c in enumerate(string.printable)}
#         self.vocab_size = len(self.vocab)
#     def encode(self, text, add_special_tokens=False, truncation=False, padding=False, max_length=None, return_tensors=None):
#         tokens = [self.vocab.get(c, -1) for c in text if c in self.vocab]
#         if truncation and max_length is not None:
#             tokens = tokens[:max_length]
#         if padding and max_length is not None:
#             tokens += [0] * (max_length - len(tokens))
#         # if add_special_tokens:
#         #     tokens = [self.vocab['[CLS]']] + tokens + [self.vocab['[SEP]']]
#         if return_tensors == 'pt':
#             return torch.tensor(tokens, dtype=torch.long)
#         return tokens
# tokenizer = Tokenizer()
# input_dim = tokenizer.vocab_size # 62
root_dir  = 'C:/Files/Development/AI/Transformers/transformer_practice'



# Path to the folder where the datasets are/will be downloaded
DATASET_PATH = root_dir+ '/twitter_sentiment/datasets'
# Path to the folder where the model checkpoints will be saved
CHECKPOINT_PATH = root_dir+ '/twitter_sentiment/checkpoints'

# print(CHECKPOINT_PATH)

pl.seed_everything(42) # Set the random seed for reproducibility

# Ensure that all operations are deterministic on GPU (if used) for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Prioritize performance over precision for float32 matrix multiplication
torch.set_float32_matmul_precision('medium')

# The below code trains a transformer model to answer the question: "Is this a positive, negative or neutral sentiment?"

def save_datasets_to_latest_version(train_dataset, val_dataset, test_dataset):
    logs_dir = Path(f"{CHECKPOINT_PATH}/twitter_sentiment/lightning_logs")
    if not logs_dir.exists():
        print("No lightning_logs directory found.")
        return

    # Find all version directories
    version_dirs = [d for d in logs_dir.iterdir() if d.is_dir() and re.match(r'version_\d+', d.name)]
    if not version_dirs:
        print("No version directories found.")
        return

    # Get the most recent version by highest number
    latest_version = max(version_dirs, key=lambda d: int(re.search(r'\d+', d.name).group()))
    datasets_path = latest_version / "datasets.xlsx"

    # Convert datasets to DataFrames
    def dataset_to_df(dataset):
        if hasattr(dataset, 'data') and hasattr(dataset, 'labels'):
            return pd.DataFrame({'message': dataset.data, 'sentiment': dataset.labels})
        else:
            # fallback: try to iterate
            messages, sentiments = [], []
            for x, y in dataset:
                messages.append(x)
                sentiments.append(y.item() if hasattr(y, 'item') else y)
            return pd.DataFrame({'message': messages, 'sentiment': sentiments})

    with pd.ExcelWriter(datasets_path) as writer:
        dataset_to_df(train_dataset).to_excel(writer, sheet_name='train', index=False)
        dataset_to_df(val_dataset).to_excel(writer, sheet_name='validation', index=False)
        dataset_to_df(test_dataset).to_excel(writer, sheet_name='test', index=False)

    print(f"Datasets saved to {datasets_path}")

# Example usage (call after datasets are created):
# save_datasets_to_latest_version(train_dataset, val_dataset, test_dataset)

def permute_df(df):
    return df.sample(frac=1, random_state=42)#.reset_index(drop=True)

from copy import deepcopy
from sklearn.model_selection import train_test_split
def path_join(path, fname):
    return os.path.abspath(os.path.join(DATASET_PATH, fname)).replace('\\', '/')

class SentimentDataset(data.Dataset):
    def __init__(self, seq_len, num_categories, test_sample_frac, val_sample_frac, mode, test_rows=None):
        super().__init__()
        self.num_categories = num_categories
        self.seq_len = seq_len            
        self.mode = mode.lower().strip()
        # self.sentiment_labels = {
        #             'Positive': 0,
        #             'Negative': 1,
        #             'Neutral': 2,
        #             'Irrelevant': 2
        #         }
        self.sentiment_labels = {
                    0: 0,
                    1: 0, # I'm pretty sure 1 and 3 don't exist, and 2 doesn't exist in this particular file
                    2: 2,
                    3: 1,
                    4: 1
                }
        if mode == "train":
            self.data = pd.read_csv(path_join(DATASET_PATH, "training.1600000.processed.noemoticon.csv")).drop_duplicates().reset_index(drop=True)
            # self.data = pd.read_csv(path_join(DATASET_PATH, "twitter_training.csv")).drop_duplicates().reset_index(drop=True)
            self.data = self.data[~self.data.index.isin(test_rows)]
            self.labels = [c for c in self.data['sentiment'].map(self.sentiment_labels).tolist()]
            # # NOTE: Trying to add more entries to the training dataset by splitting the text into subsets of words
            # data = []
            # labels = []
            # for i,msg in enumerate(self.data['message'].tolist()):
            #     msg = re.sub(r'[-_ ]+', ' ', str(msg))  # Remove underscores and hyphens
            #     msg_words = msg.split(' ')
            #     for j in range(0,len(msg_words)):
            #         data.append(" ".join(msg_words[0:j+1]))
            #         labels.append(self.labels[i])
            # self.data = data
            # self.labels = labels
            self.data = self.data['message'].tolist()
        elif mode == "validation":
            self.data = permute_df(pd.read_csv(path_join(DATASET_PATH, "training.1600000.processed.noemoticon.csv")).drop_duplicates().reset_index(drop=True))#"twitter_validation.csv")).drop_duplicates().reset_index(drop=True)

            self.data = self.data[~self.data.index.isin(test_rows)]
            df = pd.DataFrame(columns=self.data.columns)
            self.data['sentiment'] = self.data['sentiment'].map(self.sentiment_labels)
            min_category = self.data['sentiment'].value_counts().min()
            N = math.floor(min_category * val_sample_frac)
            N_train = min_category - N
            for i in range(num_categories):
                df_sentiment = self.data[self.data['sentiment'] == i]
                N_ = math.floor(len(df_sentiment) - N_train)
                if len(df_sentiment) > 1 and N_ == 0:
                    N_ = 1
                elif len(df_sentiment) > min_category:
                    N_ -= math.floor(9.85*N_*test_sample_frac)
                df = pd.concat([df, df_sentiment.sample(n=N_, random_state=42)])
            self.data = df['message'].tolist()
            self.labels = df['sentiment'].tolist()
            self.index_vals = df.index

            # self.labels = self.data['sentiment'].tolist()
            # self.data = self.data['message'].tolist()
        elif mode == "test":
            self.data = permute_df(pd.read_csv(path_join(DATASET_PATH, "training.1600000.processed.noemoticon.csv")).drop_duplicates().reset_index(drop=True))
            # self.data = pd.read_csv(path_join(DATASET_PATH, "twitter_training.csv")).drop_duplicates().reset_index(drop=True)
            # self.data = self.data[self.data.duplicated('id', keep='first')]
            # self.labels = self.data['sentiment'].map(self.sentiment_labels).tolist()
            # self.index_vals = self.data.index
            # self.data = self.data['message'].tolist()

            df = pd.DataFrame(columns=self.data.columns)
            self.data['sentiment'] = self.data['sentiment'].map(self.sentiment_labels)
            min_category = self.data['sentiment'].value_counts().min()
            N = math.floor(min_category * test_sample_frac)
            N_train = min_category - N
            for i in range(num_categories):
                df_sentiment = self.data[self.data['sentiment'] == i]
                N_ = math.floor(len(df_sentiment) - N_train)
                # N = math.floor(len(df_sentiment)*test_sample_frac)
                # print(f'(TEST) {i}: {N}')
                if len(df_sentiment) > 1 and N_ == 0:
                    N_ = 1
                elif len(df_sentiment) > min_category:
                    # print(N_, math.floor(20*N_*test_sample_frac))
                    N_ -= math.floor(9.75*N_*test_sample_frac)
                    
                df = pd.concat([df, df_sentiment.sample(n=N_, random_state=42)])
            self.data = df['message'].tolist()
            self.labels = df['sentiment'].tolist()
            self.index_vals = df.index
            # self.data = self.data.sample(n=round(len(self.data)*test_sample_frac), random_state=42)

        self.size = len(self.data)
        print(f"({mode}) Dataset size: {self.size}")
        df_distribution = pd.DataFrame({'sentiment': self.labels}).value_counts().reset_index(name='row_count')
        df_distribution['% of total'] = df_distribution.apply(lambda x: f"{round(x['row_count'] / self.size * 100, 3)}%", axis=1)
        df_distribution = df_distribution.sort_values(by=['sentiment']).reset_index(drop=True)
        print(f"({mode}) Category distribution:\n{df_distribution}")

        # Load tokenizer and embedding model
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    def get_index(self):
        return None if self.mode == 'train' else list(self.index_vals)

    def __len__(self):
        return self.size
    
    def __getitem__(self, idx, mask_padding=True):
        text = str(self.data[idx])
        tokenized_text = self.tokenizer.encode(
            text,
            add_special_tokens=True,
            truncation=True,
            padding='max_length',
            max_length=self.seq_len,
            return_tensors='pt'
        ).squeeze(0)  # Remove the batch dimension
        # mask = None if not mask_padding else (tokenized_text != PADDING_TOKEN_ID).long()
        # with torch.no_grad():
        #     embedded_text = self.embedding_layer(tokenized_text).squeeze(0).float()  # Remove the batch dimension
        labels = torch.tensor([self.labels[idx]], dtype=torch.float32)
        return tokenized_text, labels#, mask


from torchmetrics.classification import MulticlassPrecision, MulticlassRecall, MulticlassF1Score
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import re
from pathlib import Path
# def main():
if __name__ == "__main__":
    MAX_SEQ_LEN = 280 + 2 # 280 characters + 2 special tokens (CLS and SEP)
    BATCH_SIZE = 32#int(input("Enter batch size: "))
    NUM_CLASSES = 2

    ### TODO: Figure out how to make more efficient use of the dataset. At the moment, the model appears to be overfitting to the training data
    ### and just guessing the most common class every time.
    dataset = partial(SentimentDataset, MAX_SEQ_LEN, NUM_CLASSES, 0.1, 0.05)
    test_dataset = dataset("test")
    val_dataset = dataset("validation", test_rows=test_dataset.get_index())
    train_dataset = dataset("train", test_rows=test_dataset.get_index() + val_dataset.get_index())
    test_loader  = data.DataLoader(test_dataset, batch_size=BATCH_SIZE)
    val_loader   = data.DataLoader(val_dataset, batch_size=BATCH_SIZE)
    train_loader = data.DataLoader(train_dataset, num_workers = 8, persistent_workers=True, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, pin_memory=True)
    inp_data, labels = test_loader.dataset[0]
    # print(f"Input data: {inp_data.shape}\nLabels:     {labels.shape}")

    train_class_weights = compute_class_weight('balanced', classes=np.unique(train_loader.dataset.labels), y=train_loader.dataset.labels)
    # val_class_weights = compute_class_weight('balanced', classes=np.unique(val_loader.dataset.labels), y=val_loader.dataset.labels)
    val_class_weights = compute_class_weight(class_weight=None, classes=np.unique(val_loader.dataset.labels), y=val_loader.dataset.labels)
    # val_class_weights = train_class_weights
    class SentimentPredictor(TransformerPredictor):
        def _calculate_f1(self, preds, labels):
            f1 = MulticlassF1Score(num_classes=self.hparams.num_classes, average='macro').to(device)(preds, labels)
            return f1
        
        def _calculate_loss(self, batch, mode="train"):
            inp_data, labels = batch

            preds = self.forward(inp_data, add_positional_encoding=True)#.sum(dim=1)
            class_weights = torch.tensor(train_class_weights if mode == "train" else val_class_weights, dtype=torch.float32).to(device)
            
            # NOTE: by transforming the predictions and labels as shown below, we make the loss function take the mean of the class predictions of each token
            # and compare it to the labels.
            # labels_onehot = F.one_hot(labels.view(-1).long(), num_classes=self.hparams.num_classes).float()
            loss = F.cross_entropy(
                preds, 
                labels.view(-1).long(),
                weight = class_weights
            )
            eval_preds = preds.argmax(dim=-1).float()
            acc = (eval_preds == labels).float().mean()
            # probs = torch.sigmoid(preds)
            # preds = (probs > 0.5).float()
            # preds = torch.argmax(probs, dim=-1)
            f1 = self._calculate_f1(eval_preds, labels.squeeze(-1))

            self.log(f"{mode}_loss", loss)
            self.log(f"{mode}_acc", acc)
            self.log(f"{mode}_f1", f1)
            return loss, acc
        
        def training_step(self, batch, batch_idx):
            loss, _ = self._calculate_loss(batch, mode="train")
            return loss
        
        def validation_step(self, batch, batch_idx):
            _ = self._calculate_loss(batch, mode="val")

        def test_step(self, batch, batch_idx):
            _ = self._calculate_loss(batch, mode="test")

    ## TODO: Try to figure out a way to use masking in a way that allows the same input to be used multiple times, 
    ## but with different masks.
    def train_sentiment(**kwargs):
        # Create a PyTorch Lightning trainer with the generation callback
        root_dir = os.path.join(CHECKPOINT_PATH, "twitter_sentiment")
        os.makedirs(root_dir, exist_ok=True)
        # NOTE: The gradient_clip_val argument prevents exploding gradients during backpropagation
        callbacks = [
            ModelCheckpoint(mode="min", monitor="val_loss", filename="best_loss", save_top_k=1),
            ModelCheckpoint(mode="max", monitor="val_acc", filename="best_acc", save_top_k=1),
            ModelCheckpoint(mode="max", monitor="val_f1" , filename="best_f1" , save_top_k=1),
        ]
        trainer = pl.Trainer(default_root_dir=root_dir,
                            callbacks=callbacks,
                            accelerator="gpu" if str(device).startswith("cuda") else "cpu",
                            devices=1,
                            max_epochs=2,
                            gradient_clip_val=1
                            )
        trainer.logger._default_hp_metric = None # Optional logging argument that we don't need
        pretrained_filename = os.path.join(CHECKPOINT_PATH, "hi.ckpt")#"twitter_sentiment/lightning_logs/version_123/checkpoints/best_acc.ckpt")#"SentimentTask.ckpt")
        if os.path.isfile(pretrained_filename):
            print("Found pretrained model. Loading...")
            model = SentimentPredictor.load_from_checkpoint(pretrained_filename)
        else:
            model = SentimentPredictor(max_iters=trainer.max_epochs*len(train_loader), **kwargs)
            trainer.fit(model, train_loader, val_loader)
            save_datasets_to_latest_version(train_dataset, val_dataset, test_dataset)

        # Test best model on validation and test set
        val_result = trainer.test(model, val_loader, verbose=False)
        test_result = trainer.test(model, test_loader, verbose=False)
        result = {"test_acc": test_result[0]["test_acc"], "val_acc": val_result[0]["test_acc"]}
        model = model.to(device)
        return model, result

    # print(input_dim) 
    # 5087
    # TODO: Find more ways to improve accuracy. We're getting about 33% accuracy on 3 categories, which is about the same as random guessing.
    try:
        model, result = train_sentiment(
            input_dim = 768,#tokenizer.vocab_size,
            model_dim = 768,
            num_heads =  6,
            num_classes = NUM_CLASSES,#train_loader.dataset.num_categories,
            num_layers = 3,
            dropout = 0.4,
            lr = 7.5e-6,
            warmup = 100,
            max_seq_len=MAX_SEQ_LEN,
            input_dropout = 0.4
        )
        save_datasets_to_latest_version(train_dataset, val_dataset, test_dataset)
    except KeyboardInterrupt:
        print("Training interrupted. Saving datasets...")
        save_datasets_to_latest_version(train_dataset, val_dataset, test_dataset)
        raise

    print(f"\nVal accuracy:  {(100.0 * result['val_acc']):4.2f}%")
    print(f"Test accuracy: {(100.0 * result['test_acc']):4.2f}%")
    # return model,result

    ### NOTE: So far, BATCH_SIZE = 8 with the custom Tokenizer, gradient_clip_val = 0.8, and the inputs shown below has 
    ## In general, decreasing the batch size and learning rate has been improving the results, albeit not by much.
    ## In this case, thie behaviour appears to indicate that the the success rate may be converging towards the 
    ## probability of randomly guessing the correct answer.
    ### given the best results:
    #### Val accuracy:  35.40%
    #### Test accuracy: 33.73%
    # model, result = train_sentiment(
    #     input_dim = 280,#tokenizer.vocab_size,
    #     model_dim = 600,
    #     num_heads =  6,
    #     num_classes = train_loader.dataset.num_categories,
    #     num_layers = 10,
    #     dropout = 0.1,
    #     lr = 5e-6,
    #     warmup = 50,
    #     max_seq_len=1
    # )