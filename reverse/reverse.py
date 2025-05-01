import os
from functools import partial
import torch.utils.data as data

from pytorch_lightning.callbacks import ModelCheckpoint

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from transformer_implementation import *

root_dir  = 'C:/Files/Development/AI/Transformers/integrations/reverse'

# Path to the folder where the datasets are/will be downloaded
DATASET_PATH = root_dir+ '/reverse/datasets'
# Path to the folder where the model checkpoints will be saved
CHECKPOINT_PATH = root_dir+ '/reverse/checkpoints'

print(CHECKPOINT_PATH)

pl.seed_everything(42) # Set the random seed for reproducibility

# Ensure that all operations are deterministic on GPU (if used) for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# The below code trains a transformer model to answer the question: "What is the reverse of this sequence of digits?"
# The model is trained on sequences of digits from 0-9, where each digit is represented by a one-hot vector of length 10.

# Since we know that the expected result is just input_seq[::-1], it is a simple way to introduce us to transformer models.

class ReverseDataset(data.Dataset):
    def __init__(self, num_categories, seq_len, dataset_size):
        super().__init__()
        self.num_categories = num_categories
        self.seq_len = seq_len
        self.size = dataset_size

        self.data = torch.randint(num_categories, size=(dataset_size, seq_len)) # [dataset_size, seq_len]

    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        inp_data = self.data[idx]
        labels = torch.flip(inp_data, dims=(0,)) # Reverse the sequence of tokens in the input data to get the expected/correct output for training
        return inp_data, labels

dataset = partial(ReverseDataset, 10, 16)
train_loader = data.DataLoader(dataset(50000), batch_size=128, shuffle=True, drop_last=True, pin_memory=True)
val_loader   = data.DataLoader(dataset(1000), batch_size=128)
test_loader  = data.DataLoader(dataset(10000), batch_size=128)
inp_data, labels = train_loader.dataset[0]
print(f"Input data: {inp_data}\nLabels:     {labels}")


class ReversePredictor(TransformerPredictor):
    def _calculate_loss(self, batch, mode="train"):
        inp_data, labels = batch
        # In this case, each class is a digit from 0-9. So each digit is represented by a one-hot vector of length 10.
        # For example, the digit 3 is represented as [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]. If it was at the first position in the
        # input sequence, then the first vector in the sequence would be [0, 0, 0, 1, 0, 0, 0, 0, 0, 0].
        inp_data = F.one_hot(inp_data, num_classes=self.hparams.num_classes).float()
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


def train_reverse(**kwargs):
    # Create a PyTorch Lightning trainer with the generation callback
    root_dir = os.path.join(CHECKPOINT_PATH, "reverse")
    os.makedirs(root_dir, exist_ok=True)
    trainer = pl.Trainer(default_root_dir=root_dir,
                         callbacks=[ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_acc")],
                         accelerator="gpu" if str(device).startswith("cuda") else "cpu",
                         devices=1,
                         max_epochs=10,
                         gradient_clip_val=5
                         )
    trainer.logger._default_hp_metric = None # Optional logging argument that we don't need
    pretrained_filename = os.path.join(CHECKPOINT_PATH, "ReverseTask.ckpt")
    if os.path.isfile(pretrained_filename):
        print("Found pretrained model. Loading...")
        model = ReversePredictor.load_from_checkpoint(pretrained_filename)
    else:
        model = ReversePredictor(max_iters=trainer.max_epochs*len(train_loader), **kwargs)
        trainer.fit(model, train_loader, val_loader)
    # Test best model on validation and test set
    val_result = trainer.test(model, val_loader, verbose=False)
    test_result = trainer.test(model, test_loader, verbose=False)
    result = {"test_acc": test_result[0]["test_acc"], "val_acc": val_result[0]["test_acc"]}
    model = model.to(device)
    return model, result


model, result = train_reverse(
    input_dim = train_loader.dataset.num_categories,
    model_dim = 32,
    num_heads =  2,
    num_classes = train_loader.dataset.num_categories,
    num_layers = 2,
    dropout = 0.0,
    lr = 5e-4,
    warmup = 50
)


print(f"\nVal accuracy:  {(100.0 * result['val_acc']):4.2f}%")
print(f"Test accuracy: {(100.0 * result['test_acc']):4.2f}%")
