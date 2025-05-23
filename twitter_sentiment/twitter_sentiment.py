
from functools import partial
import torch.nn.functional as F
from pytorch_lightning.callbacks import ModelCheckpoint
from torchmetrics.classification import MulticlassPrecision, MulticlassRecall, MulticlassF1Score
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from prepare_data import *
from transformer_implementation import *
# Set the random seed for reproducibility
pl.seed_everything(42) 
# Ensure that all operations are deterministic on GPU (if used) for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# Prioritize performance over precision for float32 matrix multiplication
torch.set_float32_matmul_precision('medium')

# The below code trains a transformer model to answer the question: "Is this a positive, negative or neutral sentiment?"
if __name__ == "__main__":
    BATCH_SIZE = 48#int(input("Enter batch size: "))
    GAMMA = 0.5#1.0

    train_dataset, val_dataset, test_dataset = get_dataset_objects(dataset_distributions,redistribute_allocs)
    test_loader  = data.DataLoader(test_dataset, batch_size=BATCH_SIZE)
    val_loader   = data.DataLoader(val_dataset, batch_size=BATCH_SIZE)
    train_loader = data.DataLoader(train_dataset, num_workers = 8, persistent_workers=True, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, pin_memory=True)
    inp_data, labels = test_loader.dataset[0]

    train_class_weights = compute_class_weight('balanced', classes=np.unique(train_loader.dataset.labels), y=train_loader.dataset.labels)
    # smoothed_weights = np.sqrt(train_class_weights)
    # train_class_weights = torch.tensor(smoothed_weights / smoothed_weights.sum(), dtype=torch.float32).to(device)
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
            # loss = FocalLoss(class_weights, gamma=self.hparams.gamma).forward(preds,labels.view(-1).long())
            loss = F.cross_entropy(
                preds, 
                labels.view(-1).long(),
                weight = class_weights
            )
            eval_preds = preds.argmax(dim=-1).float()
            acc = (eval_preds == labels).float().mean()
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
                            max_epochs=1#,
                            # gradient_clip_val=1
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

    try:
        model, result = train_sentiment(
            input_dim = 768,
            model_dim = 768,
            num_heads =  8,
            num_classes = 2,
            num_layers = 6,
            dropout = 0.4,
            lr = 5e-5,
            warmup = 100,
            max_seq_len=282,
            input_dropout = 0.0#0.2
        )
        save_datasets_to_latest_version(train_dataset, val_dataset, test_dataset)
    except KeyboardInterrupt:
        print("Training interrupted. Saving datasets...")
        save_datasets_to_latest_version(train_dataset, val_dataset, test_dataset)
        raise

    print(f'\nTrain weights: {train_class_weights}')
    print(f'Gamme: {GAMMA}')
    print(f"\nVal accuracy:  {(100.0 * result['val_acc']):4.2f}%")
    print(f"Test accuracy: {(100.0 * result['test_acc']):4.2f}%")