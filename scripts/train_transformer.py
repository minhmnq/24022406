import sys
import os
import site
site.addsitedir(r'C:\Users\minhn\AppData\Roaming\Python\Python313\site-packages')

import math
import copy
import json
import time
import re
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

# Set seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Enable fast GPU math paths: cuDNN autotuner + TF32 matmuls.
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# --- 1. Tokenizer & Vocabulary ---
class SimpleVocab:
    def __init__(self, pad_token='<pad>', unk_token='<unk>', sos_token='<sos>', eos_token='<eos>'):
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.sos_token = sos_token
        self.eos_token = eos_token
        
        self.stoi = {pad_token: 0, unk_token: 1, sos_token: 2, eos_token: 3}
        self.itos = {0: pad_token, 1: unk_token, 2: sos_token, 3: eos_token}
        self.freqs = {}

    def tokenize(self, text):
        text = re.sub(r"[\*\"“”\n\\…\+\-\/\=\(\)‘•:\[\]\|’\!;]", " ", str(text))
        text = re.sub(r"\!+", "!", text)
        text = re.sub(r"\,+", ",", text)
        text = re.sub(r"\?+", "?", text)
        text = re.sub(r"[ ]+", " ", text).strip().lower()
        return text.split()

    def build_vocab(self, sentences, max_size=30000, min_freq=2):
        for sent in sentences:
            for word in self.tokenize(sent):
                self.freqs[word] = self.freqs.get(word, 0) + 1
        
        sorted_words = sorted(self.freqs.items(), key=lambda x: x[1], reverse=True)
        for word, count in sorted_words:
            if count >= min_freq and len(self.stoi) < max_size:
                idx = len(self.stoi)
                self.stoi[word] = idx
                self.itos[idx] = word

    def encode(self, sentence, add_sos=False, add_eos=False):
        tokens = self.tokenize(sentence)
        ids = [self.stoi.get(tok, self.stoi[self.unk_token]) for tok in tokens]
        if add_sos:
            ids = [self.stoi[self.sos_token]] + ids
        if add_eos:
            ids = ids + [self.stoi[self.eos_token]]
        return ids

    def decode(self, ids):
        tokens = []
        for idx in ids:
            if idx == self.stoi[self.eos_token]:
                break
            if idx in (self.stoi[self.pad_token], self.stoi[self.sos_token]):
                continue
            tokens.append(self.itos.get(idx, self.unk_token))
        return " ".join(tokens)

    def __len__(self):
        return len(self.stoi)

# --- 2. PyTorch Dataset ---
class ParallelTextDataset(Dataset):
    def __init__(self, src_lines, trg_lines, src_vocab, trg_vocab, max_len=100):
        self.data = []
        for s, t in zip(src_lines, trg_lines):
            s_ids = src_vocab.encode(s)
            t_ids = trg_vocab.encode(t, add_sos=True, add_eos=True)
            if 0 < len(s_ids) <= max_len and 0 < len(t_ids) <= max_len:
                self.data.append((s_ids, t_ids))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def pad_collate_fn(batch, src_pad_idx, trg_pad_idx):
    src_list, trg_list = zip(*batch)
    max_src_len = max(len(s) for s in src_list)
    max_trg_len = max(len(t) for t in trg_list)
    
    padded_src = [s + [src_pad_idx] * (max_src_len - len(s)) for s in src_list]
    padded_trg = [t + [trg_pad_idx] * (max_trg_len - len(t)) for t in trg_list]
    
    return torch.tensor(padded_src, dtype=torch.long), torch.tensor(padded_trg, dtype=torch.long)

class LengthGroupedBatchSampler(torch.utils.data.Sampler):
    """Groups examples of similar length into the same batch.

    Sorting by length before batching drastically cuts the amount of padding,
    so each batch has fewer wasted tokens and the epoch runs much faster
    without changing the model. Batch order is reshuffled every epoch to keep
    training stochastic.
    """
    def __init__(self, lengths, batch_size, shuffle=True):
        self.lengths = lengths
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        indices = list(range(len(self.lengths)))
        if self.shuffle:
            random.shuffle(indices)
        # Sort within large pools so batches share similar lengths but still vary.
        pool_size = self.batch_size * 50
        batches = []
        for i in range(0, len(indices), pool_size):
            pool = indices[i:i + pool_size]
            pool.sort(key=lambda idx: self.lengths[idx])
            for j in range(0, len(pool), self.batch_size):
                batches.append(pool[j:j + self.batch_size])
        if self.shuffle:
            random.shuffle(batches)
        for batch in batches:
            yield batch

    def __len__(self):
        return (len(self.lengths) + self.batch_size - 1) // self.batch_size

# --- 3. Transformer Model Architecture (from demo_transformer.ipynb) ---
class Embedder(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
    def forward(self, x):
        return self.embed(x)

class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_length=300, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_seq_length, d_model)
        for pos in range(max_seq_length):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (10000 ** (2 * i / d_model)))
                if i + 1 < d_model:
                    pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * i + 1) / d_model)))
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x * math.sqrt(self.d_model)
        seq_len = x.size(1)
        pe = self.pe[:, :seq_len]
        x = x + pe
        return self.dropout(x)

def attention(q, k, v, mask=None, dropout=None):
    d_k = q.size(-1)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        mask = mask.unsqueeze(1)
        # Use the dtype minimum so this also works under fp16 (AMP).
        scores = scores.masked_fill(mask == 0, torch.finfo(scores.dtype).min)
    scores = F.softmax(scores, dim=-1)
    if dropout is not None:
        scores = dropout(scores)
    output = torch.matmul(scores, v)
    return output, scores

class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout=0.1):
        super().__init__()
        assert d_model % heads == 0
        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads
        self.dropout_p = dropout
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        bs = q.size(0)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k).transpose(1, 2)
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k).transpose(1, 2)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k).transpose(1, 2)
        # Use PyTorch's fused scaled-dot-product attention (Flash / mem-efficient
        # kernels). It is far faster and lighter than the manual softmax path.
        attn_mask = mask.unsqueeze(1).bool() if mask is not None else None
        concat = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout_p if self.training else 0.0,
        )
        concat = concat.transpose(1, 2).contiguous().view(bs, -1, self.d_model)
        return self.out(concat)

class Norm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        return self.alpha * (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=2048, dropout=0.1):
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.linear_2(self.dropout(F.relu(self.linear_1(x))))

class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.attn = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.ff = FeedForward(d_model, dropout=dropout)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x, mask):
        x2 = self.norm_1(x)
        x = x + self.dropout_1(self.attn(x2, x2, x2, mask))
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.ff(x2))
        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.norm_3 = Norm(d_model)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.dropout_3 = nn.Dropout(dropout)
        self.attn_1 = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.attn_2 = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.ff = FeedForward(d_model, dropout=dropout)

    def forward(self, x, e_outputs, src_mask, trg_mask):
        x2 = self.norm_1(x)
        x = x + self.dropout_1(self.attn_1(x2, x2, x2, trg_mask))
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.attn_2(x2, e_outputs, e_outputs, src_mask))
        x2 = self.norm_3(x)
        x = x + self.dropout_3(self.ff(x2))
        return x

def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, N, heads, dropout):
        super().__init__()
        self.N = N
        self.embed = Embedder(vocab_size, d_model)
        self.pe = PositionalEncoder(d_model, dropout=dropout)
        self.layers = get_clones(EncoderLayer(d_model, heads, dropout), N)
        self.norm = Norm(d_model)

    def forward(self, src, mask):
        x = self.embed(src)
        x = self.pe(x)
        for i in range(self.N):
            x = self.layers[i](x, mask)
        return self.norm(x)

class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, N, heads, dropout):
        super().__init__()
        self.N = N
        self.embed = Embedder(vocab_size, d_model)
        self.pe = PositionalEncoder(d_model, dropout=dropout)
        self.layers = get_clones(DecoderLayer(d_model, heads, dropout), N)
        self.norm = Norm(d_model)

    def forward(self, trg, e_outputs, src_mask, trg_mask):
        x = self.embed(trg)
        x = self.pe(x)
        for i in range(self.N):
            x = self.layers[i](x, e_outputs, src_mask, trg_mask)
        return self.norm(x)

class Transformer(nn.Module):
    def __init__(self, src_vocab, trg_vocab, d_model=256, N=4, heads=8, dropout=0.1, tie_weights=True):
        super().__init__()
        self.encoder = Encoder(src_vocab, d_model, N, heads, dropout)
        self.decoder = Decoder(trg_vocab, d_model, N, heads, dropout)
        self.out = nn.Linear(d_model, trg_vocab)
        # Tie the decoder input embedding with the output projection.
        # This shares parameters, regularises the model and usually lifts BLEU.
        if tie_weights:
            self.out.weight = self.decoder.embed.embed.weight

    def forward(self, src, trg, src_mask, trg_mask):
        e_outputs = self.encoder(src, src_mask)
        d_output = self.decoder(trg, e_outputs, src_mask, trg_mask)
        return self.out(d_output)

# --- 4. Mask Helper Functions ---
def nopeak_mask(size, device):
    np_mask = np.triu(np.ones((1, size, size)), k=1).astype('uint8')
    np_mask = torch.from_numpy(np_mask) == 0
    return np_mask.to(device)

def create_masks(src, trg, src_pad, trg_pad, device):
    src_mask = (src != src_pad).unsqueeze(-2)
    if trg is not None:
        trg_mask = (trg != trg_pad).unsqueeze(-2)
        size = trg.size(1)
        np_mask = nopeak_mask(size, device)
        trg_mask = trg_mask & np_mask
    else:
        trg_mask = None
    return src_mask, trg_mask

# --- 5. Scheduled Optimizer & Label Smoothing Loss ---
class ScheduledOptim:
    def __init__(self, optimizer, init_lr, d_model, n_warmup_steps=4000):
        self._optimizer = optimizer
        self.init_lr = init_lr
        self.d_model = d_model
        self.n_warmup_steps = n_warmup_steps
        self.n_steps = 0

    def step_and_update_lr(self):
        self._update_learning_rate()
        self._optimizer.step()

    def zero_grad(self):
        self._optimizer.zero_grad()

    def _get_lr_scale(self):
        return (self.d_model ** -0.5) * min(self.n_steps ** (-0.5), self.n_steps * (self.n_warmup_steps ** -1.5))

    def _update_learning_rate(self):
        self.n_steps += 1
        lr = self.init_lr * self._get_lr_scale()
        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr

class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, padding_idx, smoothing=0.1):
        super().__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.padding_idx = padding_idx

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=-1)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 2))
            true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
            true_dist[:, self.padding_idx] = 0
            mask = (target == self.padding_idx)
            if mask.sum() > 0:
                true_dist[mask] = 0.0
        return torch.mean(torch.sum(-true_dist * pred, dim=-1))

# --- 6. Inference Functions ---
def translate_greedy(model, src_sentence, src_vocab, trg_vocab, device, max_len=80):
    model.eval()
    src_ids = src_vocab.encode(src_sentence)
    src_tensor = torch.tensor([src_ids], dtype=torch.long, device=device)
    src_mask = (src_tensor != src_vocab.stoi[src_vocab.pad_token]).unsqueeze(-2)

    with torch.no_grad():
        e_outputs = model.encoder(src_tensor, src_mask)
        trg_ids = [trg_vocab.stoi[trg_vocab.sos_token]]

        for _ in range(max_len):
            trg_tensor = torch.tensor([trg_ids], dtype=torch.long, device=device)
            trg_mask = nopeak_mask(len(trg_ids), device)
            d_out = model.decoder(trg_tensor, e_outputs, src_mask, trg_mask)
            out = model.out(d_out)
            next_word_id = out[0, -1].argmax(dim=-1).item()

            if next_word_id == trg_vocab.stoi[trg_vocab.eos_token]:
                break
            trg_ids.append(next_word_id)

    return trg_vocab.decode(trg_ids)


def translate_beam(model, src_sentence, src_vocab, trg_vocab, device,
                   beam_size=5, max_len=80, length_penalty=0.7):
    """Beam search decoding with length normalisation.

    Encoder output is computed once and reused across beams, so this stays
    fast enough to evaluate the whole test set. Length penalty follows the
    GNMT formulation and stops short translations from dominating.
    """
    model.eval()
    sos = trg_vocab.stoi[trg_vocab.sos_token]
    eos = trg_vocab.stoi[trg_vocab.eos_token]

    with torch.no_grad():
        src_ids = src_vocab.encode(src_sentence)
        src_tensor = torch.tensor([src_ids], dtype=torch.long, device=device)
        src_mask = (src_tensor != src_vocab.stoi[src_vocab.pad_token]).unsqueeze(-2)
        e_outputs = model.encoder(src_tensor, src_mask)

        # Each beam: (token_ids list, cumulative log-prob, finished flag)
        beams = [([sos], 0.0, False)]

        for _ in range(max_len):
            if all(finished for _, _, finished in beams):
                break

            candidates = []
            for tokens, score, finished in beams:
                if finished:
                    candidates.append((tokens, score, True))
                    continue

                trg_tensor = torch.tensor([tokens], dtype=torch.long, device=device)
                trg_mask = nopeak_mask(len(tokens), device)
                d_out = model.decoder(trg_tensor, e_outputs, src_mask, trg_mask)
                logits = model.out(d_out)[0, -1]
                log_probs = F.log_softmax(logits, dim=-1)

                top_lp, top_ix = log_probs.topk(beam_size)
                for lp, ix in zip(top_lp.tolist(), top_ix.tolist()):
                    new_tokens = tokens + [ix]
                    new_score = score + lp
                    candidates.append((new_tokens, new_score, ix == eos))

            # Keep the best `beam_size` candidates by length-normalised score.
            def norm_score(item):
                tokens, score, _ = item
                length = max(len(tokens) - 1, 1)
                return score / (length ** length_penalty)

            candidates.sort(key=norm_score, reverse=True)
            beams = candidates[:beam_size]

        best_tokens = max(
            beams,
            key=lambda it: it[1] / (max(len(it[0]) - 1, 1) ** length_penalty),
        )[0]

    return trg_vocab.decode(best_tokens)

# --- 7. Main Training Script ---
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Training Transformer Model on {device} ===")

    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
    models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')
    logs_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'logs')
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    with open(os.path.join(data_dir, 'train.en'), encoding='utf-8') as f:
        train_en = [line.strip() for line in f]
    with open(os.path.join(data_dir, 'train.vi'), encoding='utf-8') as f:
        train_vi = [line.strip() for line in f]
    with open(os.path.join(data_dir, 'tst2012.en'), encoding='utf-8') as f:
        val_en = [line.strip() for line in f]
    with open(os.path.join(data_dir, 'tst2012.vi'), encoding='utf-8') as f:
        val_vi = [line.strip() for line in f]
    with open(os.path.join(data_dir, 'tst2013.en'), encoding='utf-8') as f:
        test_en = [line.strip() for line in f]
    with open(os.path.join(data_dir, 'tst2013.vi'), encoding='utf-8') as f:
        test_vi = [line.strip() for line in f]

    print(f"Train pairs: {len(train_en)} | Val: {len(val_en)} | Test: {len(test_en)}")

    print("Building vocabularies...")
    src_vocab = SimpleVocab()
    trg_vocab = SimpleVocab()
    # Lower min_freq keeps more word types, which reduces <unk> and helps BLEU.
    src_vocab.build_vocab(train_en, max_size=30000, min_freq=2)
    trg_vocab.build_vocab(train_vi, max_size=30000, min_freq=2)
    print(f"Source (EN) vocab size: {len(src_vocab)}, Target (VI) vocab size: {len(trg_vocab)}")

    # Create Datasets & Loaders
    train_ds = ParallelTextDataset(train_en, train_vi, src_vocab, trg_vocab, max_len=80)
    val_ds = ParallelTextDataset(val_en, val_vi, src_vocab, trg_vocab, max_len=80)
    test_ds = ParallelTextDataset(test_en, test_vi, src_vocab, trg_vocab, max_len=80)
    print(f"Usable train pairs after length filter: {len(train_ds)}")

    src_pad_idx = src_vocab.stoi[src_vocab.pad_token]
    trg_pad_idx = trg_vocab.stoi[trg_vocab.pad_token]

    batch_size = 128
    # Bucket by length to minimise padding -> big speedup, same model.
    train_lengths = [len(s) + len(t) for s, t in train_ds.data]
    train_sampler = LengthGroupedBatchSampler(train_lengths, batch_size, shuffle=True)
    train_loader = DataLoader(
        train_ds,
        batch_sampler=train_sampler,
        collate_fn=lambda b: pad_collate_fn(b, src_pad_idx, trg_pad_idx),
        num_workers=2,
        pin_memory=(device.type == 'cuda'),
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda b: pad_collate_fn(b, src_pad_idx, trg_pad_idx),
        num_workers=2,
        pin_memory=(device.type == 'cuda'),
        persistent_workers=True,
    )

    # Model Hyperparameters
    d_model = 512
    n_layers = 4
    heads = 8
    dropout = 0.2

    model = Transformer(len(src_vocab), len(trg_vocab), d_model=d_model, N=n_layers, heads=heads, dropout=dropout).to(device)

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params/1e6:.1f}M")

    criterion = LabelSmoothingLoss(len(trg_vocab), padding_idx=trg_pad_idx, smoothing=0.1)
    inner_optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, betas=(0.9, 0.98), eps=1e-9)
    optimizer = ScheduledOptim(inner_optimizer, init_lr=1.0, d_model=d_model, n_warmup_steps=4000)

    # Mixed precision speeds training on the RTX 3060 and lets us fit more data.
    use_amp = device.type == 'cuda'
    # bf16 (if supported) avoids gradient-scaling overhead and is more stable.
    use_bf16 = use_amp and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp and not use_bf16)

    epochs = 25
    patience = 3  # Early-stop when val loss has not improved for this many epochs.
    epochs_no_improve = 0
    history = {'train_loss': [], 'val_loss': [], 'bleu_score': []}
    best_val_loss = float('inf')
    best_state = None

    print(f"Starting training for {epochs} epochs on {device}...", flush=True)
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        total_train_loss = 0.0

        for src, trg in train_loader:
            src = src.to(device, non_blocking=True)
            trg = trg.to(device, non_blocking=True)
            trg_input = trg[:, :-1]
            trg_y = trg[:, 1:]

            src_mask, trg_mask = create_masks(src, trg_input, src_pad_idx, trg_pad_idx, device)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=use_amp, dtype=amp_dtype):
                preds = model(src, trg_input, src_mask, trg_mask)
                loss = criterion(preds.contiguous().view(-1, preds.size(-1)), trg_y.contiguous().view(-1))

            optimizer._update_learning_rate()
            if use_bf16:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer._optimizer.step()
            else:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer._optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer._optimizer)
                scaler.update()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)

        # Validation Loss
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for src, trg in val_loader:
                src = src.to(device, non_blocking=True)
                trg = trg.to(device, non_blocking=True)
                trg_input = trg[:, :-1]
                trg_y = trg[:, 1:]
                src_mask, trg_mask = create_masks(src, trg_input, src_pad_idx, trg_pad_idx, device)
                with torch.cuda.amp.autocast(enabled=use_amp, dtype=amp_dtype):
                    preds = model(src, trg_input, src_mask, trg_mask)
                    loss = criterion(preds.contiguous().view(-1, preds.size(-1)), trg_y.contiguous().view(-1))
                total_val_loss += loss.item()
        avg_val_loss = total_val_loss / len(val_loader)

        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)

        # Keep the checkpoint with the lowest validation loss, not the last epoch.
        marker = ""
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            marker = " *best*"
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}{marker}", flush=True)

        # Stop early once validation loss stops improving to save time.
        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch} (no val improvement for {patience} epochs).", flush=True)
            break

    elapsed_time = time.time() - start_time
    print(f"Training completed in {elapsed_time:.2f} seconds.", flush=True)

    # Restore the best checkpoint before evaluating.
    if best_state is not None:
        model.load_state_dict(best_state)

    # Evaluate BLEU on the full tst2013 test set using beam search.
    print("Evaluating BLEU score on the full tst2013 test set (beam search)...")
    hypotheses = []
    references = []
    smooth = SmoothingFunction().method4

    sample_translations = []

    for i, (en_sent, vi_sent) in enumerate(zip(test_en, test_vi)):
        pred_vi = translate_beam(model, en_sent, src_vocab, trg_vocab, device, beam_size=5)
        hypotheses.append(trg_vocab.tokenize(pred_vi))
        references.append([trg_vocab.tokenize(vi_sent)])
        if i < 5:
            sample_translations.append({'src': en_sent, 'ref': vi_sent, 'pred': pred_vi})

    bleu_score = corpus_bleu(references, hypotheses, smoothing_function=smooth) * 100
    print(f"Transformer BLEU Score on tst2013 (full, beam=5): {bleu_score:.2f}")

    history['final_bleu'] = bleu_score
    history['elapsed_seconds'] = elapsed_time
    history['sample_translations'] = sample_translations

    # Save model & metrics
    torch.save({
        'model_state_dict': model.state_dict(),
        'src_vocab': src_vocab,
        'trg_vocab': trg_vocab
    }, os.path.join(models_dir, 'transformer_model.pt'))

    with open(os.path.join(logs_dir, 'transformer_metrics.json'), 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=2, ensure_ascii=False)

    print("Saved Transformer model checkpoint and metrics successfully.")

if __name__ == '__main__':
    main()
