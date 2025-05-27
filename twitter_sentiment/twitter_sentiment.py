
from functools import partial
import torch.nn.functional as F
from pytorch_lightning.callbacks import ModelCheckpoint
from torchmetrics.classification import MulticlassPrecision, MulticlassRecall, MulticlassF1Score
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import sys, os
import re
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import requests, json
import traceback
url = 'https://ntfy.sh/desktop-notifs'
NTFY = lambda msg, new_version=-1, priority="Low": requests.post(url, data=msg, headers={'Title': f'({new_version}) Transformer Practice', 'Priority': str(priority)})
EPOCH = 0
MAX_EPOCHS = 1

try:
    from prepare_data import *
    from transformer_implementation import *
    new_version = int(re.findall(r'version_(\d+)', str(get_latest_version()).split('\\')[-1])[0])+1

    # Set the random seed for reproducibility
    pl.seed_everything(42) 
    # Ensure that all operations are deterministic on GPU (if used) for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Prioritize performance over precision for float32 matrix multiplication
    torch.set_float32_matmul_precision('medium')
    class SentimentPredictor(TransformerPredictor):
        def _calculate_f1(self, preds, labels):
            f1 = MulticlassF1Score(num_classes=self.hparams.num_classes, average='macro').to(device)(preds, labels)
            return f1
        
        def _calculate_loss(self, batch, mode="train"):
            # inp_data, labels = batch
            inp_data = batch['x']
            labels = batch['y']
            mask = batch['mask']
            preds = self.forward(inp_data, mask=mask, add_positional_encoding=True)#.sum(dim=1)
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
            if mode == "val" and globals()['EPOCH'] < MAX_EPOCHS-1:
                globals()['EPOCH'] += 1
                NTFY(f"EPOCH_COMPLETED: {EPOCH}/{MAX_EPOCHS}\n\tValidation accuracy: {100.0 * acc:.2f}%\n\tValidation loss: {100*loss:.2f}\n\tValidation F1: {100*f1:.2f}", new_version=new_version, priority="Low")
            return loss, acc
        
        def training_step(self, batch, batch_idx):
            loss, _ = self._calculate_loss(batch, mode="train")
            return loss
        
        def validation_step(self, batch, batch_idx):
            _ = self._calculate_loss(batch, mode="val")

        def test_step(self, batch, batch_idx):
            _ = self._calculate_loss(batch, mode="test")
    # The below code trains a transformer model to answer the question: "Is this a positive, negative or neutral sentiment?"
    if __name__ == "__main__":
        BATCH_SIZE = 48#int(input("Enter batch size: "))
        GAMMA = 0.5#1.0
        NTFY("INITIALIZING: Preparing to train the transformer model for sentiment analysis...", new_version=new_version, priority="High")

        train_dataset, val_dataset, test_dataset = get_dataset_objects(dataset_distributions,redistribute_allocs)
        test_loader  = data.DataLoader(test_dataset, batch_size=BATCH_SIZE,collate_fn=collate_fn)
        val_loader   = data.DataLoader(val_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn)
        train_loader = data.DataLoader(train_dataset, num_workers = 8, collate_fn=collate_fn, persistent_workers=True, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, pin_memory=True)
        # train_loader = data.DataLoader(train_dataset, num_workers = 8, collate_fn=collate_fn, persistent_workers=True, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, pin_memory=True)
        # inp_data, labels = test_loader.dataset[0]
        batch = test_loader.dataset[0]

        train_class_weights = compute_class_weight('balanced', classes=np.unique(train_loader.dataset.labels), y=train_loader.dataset.labels)
        # smoothed_weights = np.sqrt(train_class_weights)
        # train_class_weights = torch.tensor(smoothed_weights / smoothed_weights.sum(), dtype=torch.float32).to(device)
        # val_class_weights = compute_class_weight('balanced', classes=np.unique(val_loader.dataset.labels), y=val_loader.dataset.labels)
        val_class_weights = compute_class_weight(class_weight=None, classes=np.unique(val_loader.dataset.labels), y=val_loader.dataset.labels)
        # val_class_weights = train_class_weights


        def train_sentiment(**kwargs):
            globals()['new_version'] = int(re.findall(r'version_(\d+)', str(get_latest_version()).split('\\')[-1])[0])+1
            new_version = globals()['new_version']
            # Create a PyTorch Lightning trainer with the generation callback
            NTFY(f"Starting training with parameters:\nBATCH_SIZE = {BATCH_SIZE}\nhparams = {json.dumps(kwargs, indent=2)}\n",new_version=new_version)
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
            INPUT_DIM = 768
            heads = [i for i in list(range(6,16)) if INPUT_DIM%i==0]
            heads = [6]
            model_dim_scales = [1, 2,  3][:1]
            layers           = [6, 8, 10][2::2]
            dropouts         = [0.4, 0.2][1:]
            learning_rates   = [5e-5, 5e-6][:1]
            warmups = [100, 1000][:1]
            skip_model_dim_scale = False
            for model_dim_scale in model_dim_scales:
                MODEL_DIM = INPUT_DIM * model_dim_scale
                for num_heads in heads:
                    for num_layers in layers:
                        for dropout in dropouts:
                            for learning_rate in learning_rates:
                                for warmup in warmups:
                                    model, result = train_sentiment(
                                        input_dim = INPUT_DIM,
                                        model_dim = MODEL_DIM,
                                        num_heads =  num_heads,
                                        num_classes = 2,
                                        num_layers = num_layers,
                                        dropout = dropout,
                                        lr = learning_rate,
                                        warmup = warmup,
                                        max_seq_len=282,
                                        input_dropout = 0.0#0.2
                                    )
                                    save_datasets_to_latest_version(train_dataset, val_dataset, test_dataset)
                                    val_acc_msg = f"Validation accuracy: {100.0 * result['val_acc']:.2f}%"
                                    test_acc_msg = f"Test accuracy: {100.0 * result['test_acc']:.2f}%"
                                    display_msg = f"COMPLETE: Training completed\n\t{val_acc_msg}\n\t{test_acc_msg}\n"
                                    print(f'\nTrain weights: {train_class_weights}')
                                    print(f'\n{val_acc_msg}\n{test_acc_msg}')
                                    globals()['EPOCH'] = 0
                                    NTFY(display_msg, new_version=globals()['new_version'], priority="High")
                                    if model_dim_scale != 1:
                                        skip_model_dim_scale = True
                                        break
                                if skip_model_dim_scale:
                                    break
                            if skip_model_dim_scale:
                                break
                        if skip_model_dim_scale:
                            break
                    if skip_model_dim_scale:
                        break
                skip_model_dim_scale = False
        except KeyboardInterrupt:
            print("Training interrupted. Saving datasets...")
            save_datasets_to_latest_version(train_dataset, val_dataset, test_dataset)
            raise Exception("Training interrupted by user.")
except Exception as e:
    e = traceback.format_exc()
    display_msg = f"ERROR: encountered an unexpected error\n{str(e)}\n"
    try:
        from prepare_data import get_latest_version
        new_version_tmp = int(re.findall(r'version_(\d+)', str(get_latest_version()).split('\\')[-1])[0])+1
        if new_version_tmp != globals().get('new_version', -999):
            new_version = new_version_tmp
    except Exception as e:
        new_version = -1
        e = traceback.format_exc()
        display_msg += f"Failed to get new version number:\n{str(e)}\n"
    NTFY(display_msg, new_version=new_version, priority="High")
    print(e)