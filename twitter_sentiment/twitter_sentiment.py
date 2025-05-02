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
root_dir  = 'C:/Files/Development/AI/Transformers/integrations'



# Path to the folder where the datasets are/will be downloaded
DATASET_PATH = root_dir+ '/twitter_sentiment/datasets'
# Path to the folder where the model checkpoints will be saved
CHECKPOINT_PATH = root_dir+ '/twitter_sentiment/checkpoints'

print(CHECKPOINT_PATH)

pl.seed_everything(42) # Set the random seed for reproducibility

# Ensure that all operations are deterministic on GPU (if used) for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Prioritize performance over precision for float32 matrix multiplication
torch.set_float32_matmul_precision('medium')

# The below code trains a transformer model to answer the question: "Is this a positive, negative or neutral sentiment?"


from copy import deepcopy
from sklearn.model_selection import train_test_split
def path_join(path, fname):
    return os.path.abspath(os.path.join(DATASET_PATH, fname)).replace('\\', '/')

class SentimentDataset(data.Dataset):
    def __init__(self, seq_len, num_categories, test_sample_frac, mode, test_rows=None):
        super().__init__()
        self.num_categories = num_categories
        self.seq_len = seq_len            
        self.mode = mode.lower().strip()
        self.sentiment_labels = {
                    'Positive': 0,
                    'Negative': 1,
                    'Neutral': 2,
                    'Irrelevant': 2#3
                }
        # if mode == "train":
        #     self.data = pd.read_csv(path_join(DATASET_PATH, "twitter_training.csv")).drop_duplicates().reset_index()
        #     self.data = self.data[~self.data.index.isin(test_rows)]
        #     self.labels = torch.tensor([[c]*num_categories for c in self.data['sentiment'].map(sentiment_labels).tolist()])
        # elif mode == "validation":
        #     self.data = pd.read_csv(path_join(DATASET_PATH, "twitter_validation.csv")).drop_duplicates().reset_index()
        #     self.labels = torch.tensor([[c]*num_categories for c in self.data['sentiment'].map(sentiment_labels).tolist()])
        # elif mode == "test":
        #     self.data = pd.read_csv(path_join(DATASET_PATH, "twitter_training.csv")).drop_duplicates().reset_index()
        #     df = pd.DataFrame(columns=self.data.columns)
        #     self.data['sentiment'] = self.data['sentiment'].map(sentiment_labels)
        #     for i in range(num_categories):
        #         df_sentiment = self.data[self.data['sentiment'] == i]
        #         N = math.floor(len(df_sentiment)*test_sample_frac)
        #         # print(f'(TEST) {i}: {N}')
        #         if len(df_sentiment) > 1 and N == 0:
        #             N = 1
        #         df = pd.concat([df, df_sentiment.sample(n=N, random_state=42)])
        if mode == "train":
            self.data = pd.read_csv(path_join(DATASET_PATH, "twitter_training.csv")).drop_duplicates().reset_index(drop=True)
            # self.data, _ = train_test_split(self.data, test_size=test_sample_frac, random_state=42, stratify=self.data['sentiment'])
            self.data = self.data[~self.data.index.isin(test_rows)]
            self.labels = [c for c in self.data['sentiment'].map(self.sentiment_labels).tolist()]
        elif mode == "validation":
            self.data = pd.read_csv(path_join(DATASET_PATH, "twitter_validation.csv")).drop_duplicates().reset_index(drop=True)
            self.labels = torch.tensor([c for c in self.data['sentiment'].map(self.sentiment_labels).tolist()])
        elif mode == "test":
            # NOTE: This approach to evenly distributing the sentiment labels in the training set only works because the dataset is already
            # close to being evenly distributed, so we can still get a decent sample size for each class.
            self.data = pd.read_csv(path_join(DATASET_PATH, "twitter_training.csv")).drop_duplicates().reset_index(drop=True)
            # _, self.data = train_test_split(self.data, test_size=test_sample_frac, random_state=42, stratify=self.data['sentiment'])
            # self.labels = torch.tensor([[c]*num_categories for c in self.data['sentiment'].map(self.sentiment_labels).tolist()])
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
                df = pd.concat([df, df_sentiment.sample(n=N_, random_state=42)])
            self.data = df
            self.labels = [c for c in self.data['sentiment'].tolist()]
            self.index_vals = self.data.index
            self.data = self.data.sample(n=round(len(self.data)*test_sample_frac), random_state=42)
        # self.data.reset_index(drop=True, inplace=True)
        self.size = len(self.data)
        print(f"({mode}) Dataset size: {self.size}")
        print(f"({mode}) Category distribution: {self.data['sentiment'].value_counts()}")
        # self.labels = [c for c in self.data['sentiment'].map(self.sentiment_labels).tolist()]
        self.data = self.data['message'].tolist()
        # self.data = torch.stack(
        #     self.data['message'].apply(
        #         lambda x: torch.tensor(
        #             TOKENIZER.encode(
        #                 str(x),
        #                 add_special_tokens=True,
        #                 truncation=True,
        #                 padding='max_length',
        #                 max_length=seq_len,
        #                 return_tensors='pt'
        #             ),
        #             dtype=torch.long
        #         )
        #     ).tolist()
        # )

        # Load tokenizer and embedding model
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.embedding_layer = AutoModel.from_pretrained("bert-base-uncased").embeddings.word_embeddings
        self.embedding_layer.eval()  # disable dropout, etc.
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
        # data = embed_tokens(self.data[idx])
        # return data, self.labels[idx]
        text = str(self.data[idx])
        tokenized_text = self.tokenizer.encode(
            text,
            add_special_tokens=True,
            truncation=True,
            padding='max_length',
            max_length=self.seq_len,
            return_tensors='pt'
        )
        with torch.no_grad():
            embedded_text = self.embedding_layer(tokenized_text).squeeze(0).float()  # Remove the batch dimension
        labels = torch.tensor([self.labels[idx]], dtype=torch.float32)
        return embedded_text, labels
        # return EMBEDDING_LAYER(self.data[idx]).squeeze(0).float(), self.labels[idx].float()





if __name__ == "__main__":
    BATCH_SIZE = 32#int(input("Enter batch size: "))


    ### TODO: Figure out how to make more efficient use of the dataset. At the moment, the model appears to be overfitting to the training data
    ### and just guessing the most common class every time.
    dataset = partial(SentimentDataset, 280, 3, 0.1)
    test_dataset = dataset("test")
    test_loader  = data.DataLoader(test_dataset, batch_size=BATCH_SIZE)
    train_loader = data.DataLoader(dataset("train", test_rows=test_dataset.get_index()), num_workers = 10, persistent_workers=True, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, pin_memory=True)
    # train_loader = data.DataLoader(dataset("train"), num_workers = 10, persistent_workers=True, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, pin_memory=True)
    val_loader   = data.DataLoader(dataset("validation"), batch_size=BATCH_SIZE)
    inp_data, labels = train_loader.dataset[0]
    print(f"Input data: {inp_data.shape}\nLabels:     {labels.shape}")

    class SentimentPredictor(TransformerPredictor):
        def _calculate_loss(self, batch, mode="train"):
            inp_data, labels = batch

            # inp_data = F.one_hot(inp_data, num_classes=self.hparams.num_classes).float()
            preds = self.forward(inp_data, add_positional_encoding=True)
            # NOTE: by transforming the predictions and labels as shown below, we make the loss function treat each token prediction independently when
            # calculating the loss. 
            # preds.view(-1, preds.size(-1)): [batch_size, seq_len, num_classes] => [batch_size * seq_len, num_classes]
            # labels.view(-1): [batch_size, seq_len] => [batch_size * seq_len]
            # print(f"Preds: {preds.shape}, Labels: {labels.shape}")
            # print(f"Preds: {preds.view(-1, preds.size(-1)).shape}, Labels: {labels.view(-1).shape}")
            # print(f"Preds: {preds.permute(1,0,2).squeeze(0).shape}, Labels: {labels.view(-1).shape}")
            # print(f"Preds : {preds.mean(dim=1).shape}\nLabels: {labels.view(-1).shape}\n\n")
            # print(f"Preds : {preds.mean(dim=1)}\nLabels: {labels}\n\n")
            # loss = F.cross_entropy(preds.view(-1, preds.size(-1)), labels.view(-1))
            # loss = F.cross_entropy(preds.permute(1,0,2).squeeze(0).view(-1), labels.view(-1))
            # if mode == 'val':
            #     print(preds.shape)
            loss = F.cross_entropy(preds.mean(dim=1), F.one_hot(labels.view(-1).long(), num_classes=self.hparams.num_classes).float())
            acc = (preds.mean(dim=1).argmax(dim=-1) == labels).float().mean()

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
                            callbacks=[ModelCheckpoint(save_weights_only=True, mode="min", monitor="val_acc")],
                            accelerator="gpu" if str(device).startswith("cuda") else "cpu",
                            devices=1,
                            max_epochs=1,
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
    # TODO: Find more ways to improve accuracy. We're getting about 33% accuracy on 3 categories, which is about the same as random guessing.
    model, result = train_sentiment(
        input_dim = 768,#tokenizer.vocab_size,
        model_dim = 1000,
        num_heads =  10,
        num_classes = train_loader.dataset.num_categories,
        num_layers = 10,
        dropout = 0.1,
        lr = 5e-6,
        warmup = 100,
        max_seq_len=280
    )


    print(f"\nVal accuracy:  {(100.0 * result['val_acc']):4.2f}%")
    print(f"Test accuracy: {(100.0 * result['test_acc']):4.2f}%")


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