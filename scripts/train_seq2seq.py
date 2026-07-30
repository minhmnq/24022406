import sys
import os
import site
site.addsitedir(r'C:\Users\minhn\AppData\Roaming\Python\Python313\site-packages')

import math
import json
import time
import re
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

# Import vocabulary and dataset loader from train_transformer
from train_transformer import SimpleVocab, ParallelTextDataset, pad_collate_fn

# Set seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# --- 1. Luong Attention Mechanism (Scaled Luong Attention as in TF NMT) ---
class LuongAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.wa = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, decoder_hidden, encoder_outputs, src_mask=None):
        # decoder_hidden: [batch_size, hidden_dim]
        # encoder_outputs: [batch_size, src_seq_len, hidden_dim]
        score = torch.bmm(encoder_outputs, self.wa(decoder_hidden).unsqueeze(2)).squeeze(2) # [batch_size, src_seq_len]
        score = score / math.sqrt(decoder_hidden.size(-1)) # Scaled score
        
        if src_mask is not None:
            score = score.masked_fill(src_mask == 0, -1e9)
            
        attn_weights = F.softmax(score, dim=-1) # [batch_size, src_seq_len]
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs).squeeze(1) # [batch_size, hidden_dim]
        return context, attn_weights

# --- 2. Seq2Seq Model with Luong Attention (matching TF NMT architecture) ---
class Seq2SeqAttention(nn.Module):
    def __init__(self, src_vocab_size, trg_vocab_size, embed_dim=256, hidden_dim=256, num_layers=2, dropout=0.2):
        super().__init__()
        self.src_vocab_size = src_vocab_size
        self.trg_vocab_size = trg_vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Encoder: 2-layer Bidirectional LSTM
        self.src_embed = nn.Embedding(src_vocab_size, embed_dim)
        self.encoder_lstm = nn.LSTM(embed_dim, hidden_dim // 2, num_layers=num_layers,
                                    batch_first=True, bidirectional=True, dropout=dropout if num_layers > 1 else 0)

        # Decoder: 2-layer Unidirectional LSTM
        self.trg_embed = nn.Embedding(trg_vocab_size, embed_dim)
        self.attention = LuongAttention(hidden_dim)
        self.decoder_lstm = nn.LSTM(embed_dim + hidden_dim, hidden_dim, num_layers=num_layers,
                                    batch_first=True, dropout=dropout if num_layers > 1 else 0)

        # Output projection with dropout
        self.dropout = nn.Dropout(dropout)
        self.wc = nn.Linear(hidden_dim + hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, trg_vocab_size)

    def forward(self, src, trg, src_mask=None):
        batch_size = src.size(0)
        src_seq_len = src.size(1)
        trg_seq_len = trg.size(1)

        # --- ENCODER ---
        src_embedded = self.src_embed(src) # [batch_size, src_len, embed_dim]
        encoder_outputs, (enc_h, enc_c) = self.encoder_lstm(src_embedded) 
        # encoder_outputs: [batch_size, src_len, hidden_dim]

        # Combine bidirectional states for decoder initialization
        # enc_h is [2 * num_layers, batch_size, hidden_dim // 2]
        dec_h = enc_h.view(self.num_layers, 2, batch_size, self.hidden_dim // 2)
        dec_h = torch.cat([dec_h[:, 0, :, :], dec_h[:, 1, :, :]], dim=-1) # [num_layers, batch_size, hidden_dim]

        dec_c = enc_c.view(self.num_layers, 2, batch_size, self.hidden_dim // 2)
        dec_c = torch.cat([dec_c[:, 0, :, :], dec_c[:, 1, :, :]], dim=-1) # [num_layers, batch_size, hidden_dim]

        # --- DECODER ---
        trg_embedded = self.trg_embed(trg) # [batch_size, trg_len, embed_dim]

        outputs = []
        context = torch.zeros(batch_size, self.hidden_dim, device=src.device)

        for t in range(trg_seq_len):
            trg_token_embed = trg_embedded[:, t, :] # [batch_size, embed_dim]
            lstm_input = torch.cat([trg_token_embed, context], dim=-1).unsqueeze(1) # [batch_size, 1, embed_dim + hidden_dim]

            decoder_output, (dec_h, dec_c) = self.decoder_lstm(lstm_input, (dec_h, dec_c))
            # decoder_output: [batch_size, 1, hidden_dim]
            dec_hidden_t = decoder_output.squeeze(1) # [batch_size, hidden_dim]

            # Compute Luong Attention Context
            context, _ = self.attention(dec_hidden_t, encoder_outputs, src_mask=src_mask)

            # Combine decoder output state and context vector (Input Feeding Architecture)
            concat_h = torch.tanh(self.wc(torch.cat([dec_hidden_t, context], dim=-1)))
            logits_t = self.out(self.dropout(concat_h)) # [batch_size, trg_vocab_size]
            outputs.append(logits_t.unsqueeze(1))

        return torch.cat(outputs, dim=1) # [batch_size, trg_seq_len, trg_vocab_size]

# --- 3. Greedy Inference Function ---
def translate_greedy_seq2seq(model, src_sentence, src_vocab, trg_vocab, device, max_len=80):
    model.eval()
    src_ids = src_vocab.encode(src_sentence)
    src_tensor = torch.tensor([src_ids], dtype=torch.long, device=device)
    src_mask = (src_tensor != src_vocab.stoi[src_vocab.pad_token])

    with torch.no_grad():
        src_embedded = model.src_embed(src_tensor)
        encoder_outputs, (enc_h, enc_c) = model.encoder_lstm(src_embedded)

        batch_size = 1
        dec_h = enc_h.view(model.num_layers, 2, batch_size, model.hidden_dim // 2)
        dec_h = torch.cat([dec_h[:, 0, :, :], dec_h[:, 1, :, :]], dim=-1)

        dec_c = enc_c.view(model.num_layers, 2, batch_size, model.hidden_dim // 2)
        dec_c = torch.cat([dec_c[:, 0, :, :], dec_c[:, 1, :, :]], dim=-1)

        trg_ids = [trg_vocab.stoi[trg_vocab.sos_token]]
        context = torch.zeros(batch_size, model.hidden_dim, device=device)

        for _ in range(max_len):
            last_id = torch.tensor([[trg_ids[-1]]], dtype=torch.long, device=device)
            trg_token_embed = model.trg_embed(last_id).squeeze(1)
            lstm_input = torch.cat([trg_token_embed, context], dim=-1).unsqueeze(1)

            decoder_output, (dec_h, dec_c) = model.decoder_lstm(lstm_input, (dec_h, dec_c))
            dec_hidden_t = decoder_output.squeeze(1)

            context, _ = model.attention(dec_hidden_t, encoder_outputs, src_mask=src_mask)
            concat_h = torch.tanh(model.wc(torch.cat([dec_hidden_t, context], dim=-1)))
            logits_t = model.out(concat_h)

            next_word_id = logits_t.argmax(dim=-1).item()
            if next_word_id == trg_vocab.stoi[trg_vocab.eos_token]:
                break
            trg_ids.append(next_word_id)

    return trg_vocab.decode(trg_ids)

# --- 4. Main Training Script ---
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Training Seq2Seq Attention Model on {device} ===")

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
    src_vocab.build_vocab(train_en, max_size=30000, min_freq=2)
    trg_vocab.build_vocab(train_vi, max_size=30000, min_freq=2)
    print(f"Source (EN) vocab size: {len(src_vocab)}, Target (VI) vocab size: {len(trg_vocab)}")

    train_ds = ParallelTextDataset(train_en, train_vi, src_vocab, trg_vocab, max_len=80)
    val_ds = ParallelTextDataset(val_en, val_vi, src_vocab, trg_vocab, max_len=80)
    print(f"Usable train pairs after length filter: {len(train_ds)}")

    src_pad_idx = src_vocab.stoi[src_vocab.pad_token]
    trg_pad_idx = trg_vocab.stoi[trg_vocab.pad_token]

    batch_size = 128
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=lambda b: pad_collate_fn(b, src_pad_idx, trg_pad_idx))
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=lambda b: pad_collate_fn(b, src_pad_idx, trg_pad_idx))

    # Model Hyperparameters
    embed_dim = 512
    hidden_dim = 512
    num_layers = 2
    dropout = 0.3

    model = Seq2SeqAttention(len(src_vocab), len(trg_vocab), embed_dim=embed_dim, hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params/1e6:.1f}M")

    # Label smoothing improves generalisation for the RNN model too.
    criterion = nn.CrossEntropyLoss(ignore_index=trg_pad_idx, label_smoothing=0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # Halve the LR when validation loss plateaus.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1)

    epochs = 20
    history = {'train_loss': [], 'val_loss': [], 'bleu_score': []}
    best_val_loss = float('inf')
    best_state = None

    print(f"Starting training for {epochs} epochs on {device}...", flush=True)
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        total_train_loss = 0.0

        for src, trg in train_loader:
            src, trg = src.to(device), trg.to(device)
            trg_input = trg[:, :-1]
            trg_y = trg[:, 1:]

            src_mask = (src != src_pad_idx)
            preds = model(src, trg_input, src_mask=src_mask)

            optimizer.zero_grad()
            loss = criterion(preds.contiguous().view(-1, preds.size(-1)), trg_y.contiguous().view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)

        # Validation Loss
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for src, trg in val_loader:
                src, trg = src.to(device), trg.to(device)
                trg_input = trg[:, :-1]
                trg_y = trg[:, 1:]
                src_mask = (src != src_pad_idx)
                preds = model(src, trg_input, src_mask=src_mask)
                loss = criterion(preds.contiguous().view(-1, preds.size(-1)), trg_y.contiguous().view(-1))
                total_val_loss += loss.item()
        avg_val_loss = total_val_loss / len(val_loader)
        scheduler.step(avg_val_loss)

        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)

        marker = ""
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            marker = " *best*"

        print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}{marker}", flush=True)

    elapsed_time = time.time() - start_time
    print(f"Training completed in {elapsed_time:.2f} seconds.", flush=True)

    if best_state is not None:
        model.load_state_dict(best_state)

    # Evaluate BLEU on the full tst2013 test set.
    print("Evaluating BLEU score on the full tst2013 test set...")
    hypotheses = []
    references = []
    smooth = SmoothingFunction().method4

    sample_translations = []

    for i, (en_sent, vi_sent) in enumerate(zip(test_en, test_vi)):
        pred_vi = translate_greedy_seq2seq(model, en_sent, src_vocab, trg_vocab, device)
        hypotheses.append(trg_vocab.tokenize(pred_vi))
        references.append([trg_vocab.tokenize(vi_sent)])
        if i < 5:
            sample_translations.append({'src': en_sent, 'ref': vi_sent, 'pred': pred_vi})

    bleu_score = corpus_bleu(references, hypotheses, smoothing_function=smooth) * 100
    print(f"Seq2Seq Attention BLEU Score on tst2013 (full): {bleu_score:.2f}")

    history['final_bleu'] = bleu_score
    history['elapsed_seconds'] = elapsed_time
    history['sample_translations'] = sample_translations

    # Save model & metrics
    torch.save({
        'model_state_dict': model.state_dict(),
        'src_vocab': src_vocab,
        'trg_vocab': trg_vocab
    }, os.path.join(models_dir, 'seq2seq_model.pt'))

    with open(os.path.join(logs_dir, 'seq2seq_metrics.json'), 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=2, ensure_ascii=False)

    print("Saved Seq2Seq Attention model checkpoint and metrics successfully.")

if __name__ == '__main__':
    main()
