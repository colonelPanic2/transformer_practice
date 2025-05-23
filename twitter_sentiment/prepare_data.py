from copy import deepcopy
from sklearn.model_selection import train_test_split
import os
import math
import re
import pandas as pd
from pathlib import Path
import torch.utils.data as data
from math import floor
import torch
from transformers import AutoTokenizer, AutoModel
TOKENIZER = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")
EMBEDDING_LAYER = model.embeddings.word_embeddings
MAX_SEQ_LEN = 280 + 2 # 280 characters + 2 special tokens (CLS and SEP)
NUM_CLASSES = 2
def path_join(path, fname):
    return os.path.abspath(os.path.join(path, fname)).replace('\\', '/')
root_dir  = 'C:/Files/Development/AI/Transformers/transformer_practice'

# Path to the folder where the datasets are/will be downloaded
DATASET_PATH = root_dir+ '/twitter_sentiment/datasets'
# Path to the folder where the model checkpoints will be saved
CHECKPOINT_PATH = root_dir+ '/twitter_sentiment/checkpoints'

train_diff = -0.00#22
dataset_distributions = {
    'train': {
        0: 0.50251-train_diff,
        1: 0.49749+train_diff
    },
    'validation': {
        0: 0.5,
        1: 0.5
    }
}

redistribute_allocs = {
    'train': 0.7,
    'validation': 0.13,
    'test': 0.17
}

# Path to the folder where the datasets are/will be downloaded
DATASET_PATH = root_dir+ '/twitter_sentiment/datasets'
# Positive -> 0, Negative -> 1, Neutral -> 2 (neutral is being excluded)
datasets_map = {
    'Sentiment Analysis Dataset': {
        'author': 'Abhishek Shrivastava',
        'file_path': 'training.1600000.processed.noemoticon.csv',
        'colname_map': {
            'sentiment': 'sentiment',
            'message': 'message'
        },
        'category_map': {
            0: 0,
            1: 0, # I'm pretty sure 1 and 3 don't exist
            2: 2, # 2 doesn't exist in this particular file
            3: 1,
            4: 1
        },
        'primary_dataset': True
    },
    'Twitter and Reddit Sentimental analysis Dataset': {
        'author': '',
        'file_path': 'twitter-and-reddit-sentimental-analysis-dataset/Twitter_Data.csv',
        'colname_map': {
            'category': 'sentiment',
            'clean_text': 'message'
        },
        'category_map': {
            1.0: 0,
            0: 2,
            -1.0: 1
        },
        'primary_dataset': False
    },
    'twitter_training': {
        'author': '<UNKNOWN>',
        'file_path': 'twitter_training.csv',
        'colname_map': {
            'sentiment': 'sentiment',
            'message': 'message'
        },
        'category_map': {
            'Positive': 0,
            'Negative': 1,
            'Neutral': 2,
            'Irrelevant': 2
        },
        'primary_dataset': False
    },

}
def permute_df(df):
    return df.sample(frac=1, random_state=42)#.reset_index(drop=True)
def prepare_dataset(dataset_info):
    file_path = path_join(DATASET_PATH, dataset_info['file_path'])
    df_data = pd.read_csv(file_path)
    df_data.rename(columns=dataset_info['colname_map'], inplace=True)
    df_data['sentiment'] = df_data['sentiment'].map(dataset_info['category_map'])
    df_data['file'] = dataset_info['file_path']
    df_data = df_data[['file', 'message', 'sentiment']].drop_duplicates()
    return df_data
def combine_sentiment_datasets(target_classes=[0,1],supplement_minority_classes={}):
    """
    Combine multiple sentiment datasets into a single dataset.
    """
    df = pd.DataFrame()
    for dataset_name, dataset_info in datasets_map.items():
        df_data = prepare_dataset(dataset_info)
        if dataset_info['primary_dataset']:
            df = pd.concat([df, df_data[df_data['sentiment'].isin(target_classes)]])
            continue
        if len(supplement_minority_classes) > 0:
            for i,portion in supplement_minority_classes.items():
                if i not in target_classes:
                    continue
                df = pd.concat([df, df_data[df_data['sentiment'] == i].sample(frac=portion, random_state=42)])
        else:
            df = pd.concat([df, df_data[df_data['sentiment'].isin(target_classes)]])
    return df.reset_index(drop=True)
def show_distributions(df, colname='sentiment'):
    """
    Get the distribution of a column in a dataframe.
    """
    df_counts = df[colname].value_counts().reset_index(name='row_count')
    df_counts['% of total'] = df_counts.apply(lambda x: f"{round(x['row_count'] / len(df) * 100, 3)}%", axis=1)
    return df_counts.sort_values(by=[colname]).reset_index(drop=True)
def get_N(df, category, portion):
    size = len(df)
    m = size * portion
    n = len(df[df['sentiment'] == category])
    # Calculate the number of samples needed from the specified category so that it makes up 'portion' of the total dataset size after sampling.
    if n == 0 or portion == 0:
        return 0
    N = int((m * portion) / (n / size))
    N = min(N, n)
    return N
def get_distributions(df_input, dataset_distributions, df_redistribute):
    """
    Get the distribution of the dataset based on the specified proportions.
    """
    df_train = pd.DataFrame()
    df_val = pd.DataFrame()
    for category, portion in dataset_distributions['train'].items():
        N = get_N(df_input, category, portion)
        df_train = pd.concat([df_train, df_input[df_input['sentiment'] == category].sample(N)])
    df_remaining = df_input[~df_input.index.isin(list(df_train.index)+list(df_redistribute.index))]
    # print(len(df_remaining))
    df_remaining = pd.concat([df_remaining,df_redistribute])#.reset_index(drop=True)
    # print(show_distributions(df_remaining))
    for category, portion in dataset_distributions['validation'].items():
        N = get_N(df_remaining, category, portion)
        df_val = pd.concat([df_val, df_remaining[df_remaining['sentiment'] == category].sample(N)])
    df_test = df_remaining[~df_remaining.index.isin(list(df_train.index)+list(df_val.index))]
    return df_train, df_val, df_test

def redistribute_classes(df_train, df_val, df_test, class_col='sentiment', redistribute_allocs = {'train': 0.34, 'validation': 0.33, 'test': 0.33}):
    """
    Redistribute the classes in df_test evenly among df_train, df_val, and df_test.
    Returns new dataframes: df_train_new, df_val_new, df_test_new.
    """

    # Concatenate all dataframes
    df_all = pd.concat([df_train, df_val, df_test])
    # Get unique classes
    classes = df_all[class_col].unique()
    # Prepare new splits
    df_train_new = pd.DataFrame(columns=df_all.columns)
    df_val_new = pd.DataFrame(columns=df_all.columns)
    df_test_new = pd.DataFrame(columns=df_all.columns)

    for cls in classes:
        df_cls = df_test[df_test[class_col] == cls]
        n = len(df_cls)
        n_train = floor(n * redistribute_allocs['train'])
        n_val = floor(n * redistribute_allocs['validation'])
        n_test = n - n_train - n_val  # remainder goes to test

        df_cls_shuffled = df_cls.sample(frac=1, random_state=42)
        df_train_new = pd.concat([df_train[df_train[class_col] == cls],df_train_new, df_cls_shuffled.iloc[:n_train]])
        df_val_new = pd.concat([df_val[df_val[class_col] == cls], df_val_new, df_cls_shuffled.iloc[n_train:n_train + n_val]])
        df_test_new = pd.concat([df_test_new, df_cls_shuffled.iloc[n_train + n_val:]])

    # Reset indices
    return df_train_new, df_val_new, df_test_new

def get_datasets(dataset_distributions,redistribute_allocs):
    df_original = combine_sentiment_datasets(target_classes=[0,1], supplement_minority_classes={1:1})
    print(f'Before balancing ({len(df_original)}):')
    print(show_distributions(df_original))

    min_category = df_original['sentiment'].value_counts().min()
    max_category = df_original['sentiment'].value_counts().max()
    df_redistribute = pd.DataFrame()
    for i in list(set(df_original['sentiment'])):
        cur_category = len(df_original[df_original['sentiment'] == i])
        if cur_category == min_category:
            continue
        df_redistribute = pd.concat([df_redistribute, df_original[df_original['sentiment']==i].sample(n=cur_category-min_category,random_state=42)])
    redistribute_index = list(df_redistribute.index)
    # df_redistribute = df_redistribute.reset_index(drop=True)
    df = df_original[~df_original.index.isin(redistribute_index)]#.reset_index(drop=True)
    print(f'\nAfter balancing ({len(df)}):')
    print(show_distributions(df))
    print(f'\ndf_redistribute ({len(df_redistribute)}):')
    print(show_distributions(df_redistribute))
    assert len(set(list(df_redistribute.index)+list(df.index))) == len(set(df_original.index))

    df_train, df_val, df_test = get_distributions(df, dataset_distributions, df_redistribute)
    df_train, df_test, df_val = redistribute_classes(df_train, df_val, df_test,redistribute_allocs=redistribute_allocs)
    df_train, df_test, df_val = redistribute_classes(df_train, df_val, df_test)

    print(f'Train ({len(df_train)}):\n{show_distributions(df_train)}')
    print(f'Val   ({len(df_val)}):\n{show_distributions(df_val)}')
    print(f'Test  ({len(df_test)}):\n{show_distributions(df_test)}')
    assert len(list(set(list(df_train.index)+list(df_val.index)+list(df_test.index))))-len(df_original) == 0

    return df_train, df_val, df_test

# def get_sentiment_datasets(seq_len, num_categories, test_sample_frac, val_sample_frac, mode, test_rows=None):
#         base_df = combine_sentiment_datasets(target_classes=[0,1], supplement_minority_classes={1: 1})
#         if mode == "train":
#             data = base_df
#             data = data[~data.index.isin(test_rows)]
#             labels = [c for c in data['sentiment'].tolist()]
#             data = data['message'].tolist()
#             index_vals=None
#         elif mode == "validation":
#             data = permute_df(base_df)
#             data = data[~data.index.isin(test_rows)]
#             df = pd.DataFrame(columns=data.columns)
#             min_category = data['sentiment'].value_counts().min()
#             N = math.floor(min_category * val_sample_frac)
#             N_train = min_category - N
#             for i in range(num_categories):
#                 df_sentiment = data[data['sentiment'] == i]
#                 N_ = math.floor(len(df_sentiment) - N_train)
#                 if len(df_sentiment) > 1 and N_ == 0:
#                     N_ = 1
#                 elif len(df_sentiment) > min_category:
#                     N_ -= math.floor(9.85*N_*test_sample_frac)
#                 df = pd.concat([df, df_sentiment.sample(n=N_, random_state=42)])
#             data = df['message'].tolist()
#             labels = df['sentiment'].tolist()
#             index_vals = df.index
#         elif mode == "test":
#             data = permute_df(base_df)
#             df = pd.DataFrame(columns=data.columns)
#             min_category = data['sentiment'].value_counts().min()
#             N = math.floor(min_category * test_sample_frac)
#             N_train = min_category - N
#             for i in range(num_categories):
#                 df_sentiment = data[data['sentiment'] == i]
#                 N_ = math.floor(len(df_sentiment) - N_train)
#                 if len(df_sentiment) > 1 and N_ == 0:
#                     N_ = 1
#                 elif len(df_sentiment) > min_category:
#                     N_ -= math.floor(9.75*N_*test_sample_frac)
#                 df = pd.concat([df, df_sentiment.sample(n=N_, random_state=42)])
#             data = df['message'].tolist()
#             labels = df['sentiment'].tolist()
#             index_vals = df.index
#         return data, labels, index_vals, len(data)
def get_sentiment_datasets(dataset,mode,num_categories):
        data = dataset['message'].tolist()
        labels = dataset['sentiment'].tolist()
        index_vals = dataset.index
        size = len(data)
        return data, labels, index_vals, size
class SentimentDataset(data.Dataset):
    def __init__(self, dataset, mode, seq_len, num_categories):
    # def __init__(self, seq_len, num_categories, test_sample_frac, val_sample_frac, mode, test_rows=None):
        super().__init__()
        self.num_categories = num_categories
        self.seq_len = seq_len            
        self.mode = mode.lower().strip()
        
        self.data, self.labels, self.index_vals, self.size = get_sentiment_datasets(dataset, mode, num_categories)
        # print(f"({mode}) Dataset size: {self.size}")
        # df_distribution = pd.DataFrame({'sentiment': self.labels}).value_counts().reset_index(name='row_count')
        # df_distribution['% of total'] = df_distribution.apply(lambda x: f"{round(x['row_count'] / self.size * 100, 3)}%", axis=1)
        # df_distribution = df_distribution.sort_values(by=['sentiment']).reset_index(drop=True)
        # print(f"({mode}) Category distribution:\n{df_distribution}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

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

def get_dataset_objects(dataset_distributions,redistribute_allocs):
    df_train, df_val, df_test = get_datasets(dataset_distributions,redistribute_allocs)
    train_dataset = SentimentDataset(df_train, 'train', MAX_SEQ_LEN, NUM_CLASSES)
    val_dataset = SentimentDataset(df_val, 'validation', MAX_SEQ_LEN, NUM_CLASSES)
    test_dataset = SentimentDataset(df_test, 'test', MAX_SEQ_LEN, NUM_CLASSES)
    return train_dataset, val_dataset, test_dataset

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


if __name__ == '__main__':
    df_train, df_val, df_test = get_datasets(dataset_distributions,redistribute_allocs)