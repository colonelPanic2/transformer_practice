import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

import torch.optim as optim

try:
    import pytorch_lightning as pl
except ModuleNotFoundError:
    os.system("pip install --quiet pytorch-lightning>=1.4")
    import pytorch_lightning as pl

# Globals, Helper functions

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print(f"Device: {device}")

def expand_mask(mask):
    assert mask.ndim >= 2, "Mask should have at least 2 dimensions"
    if mask.ndim == 3:
        mask = mask.unsqueeze(1)
    while mask.ndim < 4:
        mask = mask.unsqueeze(0)
    return mask

def scaled_dot_product(q, k, v, mask=None):
    d_k = q.size(-1)
    dot_product = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        # Use masking to prevent "future" tokens (any tokens below the diagonal of the seq_lenxseq_len matrices) from giving away information during training
        # NOTE: For some reason, using -torch.inf doesn't return zeros from the softmax function here
        #       This is because the softmax function is not numerically stable when using -torch.inf
        #       Instead, we use a large negative number (-1e9) to mask out the future tokens
        dot_product = dot_product.masked_fill(mask == 0, -1e9)#-torch.inf)
    attention = F.softmax(dot_product, dim=-1) # Softmax over the columns in the dot product
    values = torch.matmul(attention, v)
    return values, attention



# Implementation

class MultiHeadAttention(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads, bias=True):        
        super().__init__()
        assert embed_dim % num_heads == 0, str(f"Embedding dimension, '{embed_dim}' must be divisible by number of heads, '{num_heads}'")
        self.bias = bias
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # self.causal_mask = torch.tril(torch.ones(input_dim, input_dim)).unsqueeze(0).unsqueeze(1).transpose(-1,-2).to('cuda:0')
        self.qkv_proj = nn.Linear(input_dim, 3*embed_dim)
        self.o_proj = nn.Linear(embed_dim, input_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        nn.init.xavier_uniform_(self.o_proj.weight)
        if not self.bias:
            self.qkv_proj.bias.data.fill_(0)
            self.o_proj.bias.data.fill_(0)
    
    def forward(self, x, mask=None, return_attention=False):
        batch_size, seq_len, _ = x.size()
        if mask is not None:
            mask = expand_mask(mask)# * self.causal_mask
        qkv = self.qkv_proj(x) # [batch_size, seq_len, 3 * embed_dim]
        qkv = qkv.reshape(batch_size, seq_len, self.num_heads, 3 * self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3) # [batch_size, num_heads, seq_len, 3 * head_dim]
        q, k, v = qkv.chunk(3, dim=-1)

        values, attention = scaled_dot_product(q, k, v, mask=mask) # [batch_size, num_heads, seq_len, head_dim]
        values = values.permute(0, 2, 1, 3) # [batch_size, seq_len, num_heads, head_dim]
        values = values.reshape(batch_size, seq_len, self.embed_dim)
        o = self.o_proj(values)

        if return_attention:
            return o, attention
        return o


class EncoderBlock(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads, dim_feedforward, dropout=0.0):
        super().__init__()

        self.MultiHeadAttention = MultiHeadAttention(input_dim, embed_dim, num_heads)
        # self.MultiHeadAttention = MultiHeadAttention(input_dim, input_dim, num_heads)

        self.MultiLayerPerceptron = nn.Sequential(
            nn.Linear(input_dim, dim_feedforward),
            nn.Dropout(dropout),
            nn.ReLU(inplace=True),
            nn.Linear(dim_feedforward, input_dim)
        )

        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        mha_out = self.MultiHeadAttention(x, mask=mask)
        x = x + self.dropout(mha_out)
        x = self.norm1(x)

        mlp_out = self.MultiLayerPerceptron(x)
        x = x + self.dropout(mlp_out)
        x = self.norm2(x)

        return x


class TransformerEncoder(nn.Module):
    def __init__(self, n_layers, **block_args):
        super().__init__()
        # NOTE: It is very important to make this an nn.ModuleList object. Otherwise, there will 
        # likely be issues caused by tensors being sent to different devices
        self.layers = nn.ModuleList([EncoderBlock(**block_args) for _ in range(n_layers)])
    
    def forward(self, x, mask=None):
        for l in self.layers:
            x = l(x, mask=mask) # For some reason, we're getting all NANs when x is passed through the second layer
        return x
    
    def get_attention_maps(self, x, mask=None):
        attention_maps = []
        for l in self.layers:
            _, attention = l.self_attn(x, mask=mask, return_attention=True)
            attention_maps.append(attention)
            x = l(x)
        return attention_maps


class CosineWarmupScheduler(optim.lr_scheduler._LRScheduler):
    def __init__(self, max_iters, warmup, optimizer):
        self.max_num_iters = max_iters # The number of training epochs
        self.warmup = warmup  # The number of epochs for which the learning rate is to be reduced by a factor of epoch_idx / warmup
        super().__init__(optimizer)
        
    def get_lr(self):
        lr_factor = self.get_lr_factor(last_epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]
    
    def get_lr_factor(self, last_epoch):
        return 0.5 * ( 1 + np.cos(np.pi * last_epoch / self.max_num_iters) ) * min(last_epoch, self.warmup) / self.warmup


class PositionalEncoding(nn.Module):
    def __init__(self, input_dim, max_seq_len=1):#=5000):
        super().__init__()
        self.input_dim = input_dim # The length of the vector representing each token in the input sequence
        self.max_len = max_seq_len # The maximum number of tokens in the input sequence
        pe = torch.zeros(max_seq_len, input_dim)
        position = torch.arange(0, max_seq_len, dtype = torch.float).unsqueeze(1) # [max_len, 1]
        # This is a more efficient way to get the denominator from the notes
        denominator = torch.exp(torch.arange(0, input_dim, 2).float() * -math.log(10000.0) / input_dim) # [input_dim // 2]

        # NOTE: [max_len, 1] x [input_dim // 2] == [max_len, input_dim // 2] because pytorch automatically broadcasts [input_dim // 2] to [1, input_dim // 2]
        #       when performing this operation
        # Broadcasting summary: Basically, whenever an operation uses data from 2+ different tensors, any dimensions that are 1 in one tensor but not 1 in the 
        # second tensor will be automatically transformed so that the "1" dimension from the first tensor matches the number in the same dimension of the 
        # second tensor
        pe[:, 0::2] = torch.sin(position * denominator)
        pe[:, 1::2] = torch.cos(position * denominator)

        pe = pe.unsqueeze(0) # transform pe to [1, max_seq_len, input_dim] so that it can be broadcasted over the batch_size dimension of the input, x
        
        # register_buffer => Tensor which is not a parameter, but should be part of the module's state.
        # Used for tensors that need to be on the same device as the module.
        # persistent=False tells PyTorch to not add the buffer to the state dict (e.g. when we save the model)
        self.register_buffer('pe', pe, persistent=False)
    
    def forward(self, x):
        # Broadcast pe across the batch for the input, and allow the seq_len of the input to be any size, so long as it is <= max_seq_len
        # print('\n\n\n\n')
        # print(f'x shape: {x.shape}')
        # # print(x)
        # print(f'pe shape: {self.pe.shape}')
        x = x + self.pe[:, :x.size(1)]
        return x

from transformers import AutoModel
PADDING_TOKEN_ID = 0

class TransformerPredictor(pl.LightningModule):
    def __init__(self, input_dim, model_dim, num_classes, num_heads, num_layers, lr, warmup, max_iters, dropout=0.0, input_dropout=0.0, max_seq_len=5000):
        super().__init__()
        self.save_hyperparameters()
        self._create_model()
    def _create_model(self):
        # self.causal_mask = torch.tril(torch.ones(self.hparams.max_seq_len, self.hparams.max_seq_len)).unsqueeze(0).unsqueeze(1).transpose(-1,-2).to('cuda:0')
        self.upper_mask = torch.triu(torch.ones(self.hparams.max_seq_len, self.hparams.max_seq_len), diagonal=1).bool().to('cuda:0')#.unsqueeze(0).unsqueeze(1).transpose(-1,-2).to('cuda:0')
        
        self.embedding_layer = AutoModel.from_pretrained("bert-base-uncased").embeddings.word_embeddings
        self.embedding_layer.eval()  # disable dropout, etc.

        # Input dim -> Model dim (length of the vector representing each token)
        self.input_net = nn.Sequential(
            nn.Dropout(self.hparams.input_dropout),
            nn.Linear(self.hparams.input_dim, self.hparams.model_dim)
        )
        self.PositionalEncoding = PositionalEncoding(self.hparams.model_dim, max_seq_len=self.hparams.max_seq_len)
        self.TransformerEncoder = TransformerEncoder(self.hparams.num_layers, 
                                                input_dim=self.hparams.model_dim, 
                                                embed_dim = self.hparams.model_dim,
                                                num_heads=self.hparams.num_heads,
                                                dim_feedforward=2*self.hparams.model_dim,
                                                dropout=self.hparams.dropout
                                            )
        self.output_net = nn.Sequential(
            nn.Linear(self.hparams.model_dim, self.hparams.model_dim),
            nn.LayerNorm(self.hparams.model_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(self.hparams.dropout),
            nn.Linear(self.hparams.model_dim, self.hparams.num_classes)
        )
        self.final_classifier = nn.Sequential(
            nn.Linear(self.hparams.max_seq_len, 1)
        )
    def get_input_mask(self, x):
        mask = (x != PADDING_TOKEN_ID).long()
        mask_diag = torch.diag_embed(mask)
        nonzero_diag_mask = (mask != PADDING_TOKEN_ID).unsqueeze(-2)
        # conditional_mask = self.upper_mask & nonzero_diag_mask
        conditional_mask = nonzero_diag_mask & torch.ones_like(self.upper_mask).bool()
        # This only masks the padding tokens. Non-padding tokens are not masked
        mask_diag[conditional_mask] = 1
        return mask_diag
    def forward(self, x, mask=None, add_positional_encoding=True):
        # with torch.no_grad():
        #     x = self.embedding_layer(x).squeeze(0).float()  # Remove the batch dimension
        mask = self.get_input_mask(x)
        # mask = (x != PADDING_TOKEN_ID).long()
        # mask = (mask * self.causal_mask).long().squeeze(0)
        x = self.embedding_layer(x).squeeze(0).float()  # Remove the batch dimension
        x = self.input_net(x)
        # mask2 = torch.tril(torch.ones(self.hparams.max_seq_len, self.hparams.max_seq_len)).unsqueeze(0).unsqueeze(1).transpose(-1,-2).to('cuda:0')
        if add_positional_encoding:
            x = self.PositionalEncoding(x)
        x = self.TransformerEncoder(x, mask=mask)
        cls_output = x[:, 0, :].squeeze(1) # [batch_size, model_dim]
        x = self.output_net(cls_output)
        # x = x.permute(0, 2, 1) # [batch_size, num_classes, seq_len]
        # x = self.final_classifier(x) # [batch_size, num_classes, 1]
        # return x.squeeze(-1) # [batch_size, num_classes]
        # cls_output = x[:, 0, :].squeeze(1) # [batch_size, num_classes]
        # return cls_output # [batch_size, num_classes]
        return x # [batch_size, num_classes]
    @torch.no_grad() # This function is for evaluation of the model, so we don't need to track gradients for tensor operations within it
    def get_attention_maps(self, x, mask=None, add_positional_encoding=True):
        x = self.input_net(x)
        if add_positional_encoding:
            x = self.PositionalEncoding(x)
        attention_maps = self.TransformerEncoder.get_attention_maps(x, mask=mask)
        return attention_maps
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr)
        # Apply lr scheduler per step
        lr_scheduler = CosineWarmupScheduler(warmup=self.hparams.warmup,
                                             max_iters=self.hparams.max_iters,
                                             optimizer=optimizer)
        return [optimizer], [{'scheduler': lr_scheduler, 'interval': 'step'}]
    # Create placeholders for the training/validation/test steps. These functions are to be implemented by downstream classes
    # that will configure an input dataset for the transformer to use
    def training_step(self, batch, batch_idx):
        raise NotImplementedError
    def validation_step(self, batch, batch_idx):
        raise NotImplementedError
    def test_step(self, batch, batch_idx):
        raise NotImplementedError
    





class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        """
        Args:
            alpha: scalar or tensor of shape (num_classes,)
            gamma: focusing parameter
            reduction: 'mean', 'sum', or 'none'
        """
        super(FocalLoss, self).__init__()
        if isinstance(alpha, (list, tuple)):
            self.alpha = torch.tensor(alpha)
        else:
            self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        """
        logits: (batch_size, num_classes)
        targets: (batch_size,) with values in [0, num_classes)
        """
        num_classes = logits.size(1)
        device = logits.device

        # Convert alpha to tensor and move to device
        if isinstance(self.alpha, torch.Tensor):
            alpha = self.alpha.to(device)
            alpha_t = alpha[targets]  # Shape: (batch_size,)
        else:
            alpha_t = torch.full_like(targets, fill_value=self.alpha, dtype=torch.float)

        # Compute probabilities
        ce_loss = F.cross_entropy(logits, targets, reduction='none')  # Shape: (batch_size,)
        probs = F.softmax(logits, dim=1)
        targets_one_hot = F.one_hot(targets, num_classes=num_classes).float()
        pt = (probs * targets_one_hot).sum(dim=1)  # Shape: (batch_size,)

        # Compute focal weight
        focal_weight = alpha_t * (1 - pt) ** self.gamma  # Shape: (batch_size,)
        loss = focal_weight * ce_loss  # Shape: (batch_size,)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss