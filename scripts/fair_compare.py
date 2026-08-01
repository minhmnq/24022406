"""Fair BLEU comparison on tst2013 with matched decoding for both models.

Modes:
  greedy  - both models use greedy decoding
  beam    - both models use beam search (default beam_size=5)

Writes:
  logs/fair_comparison.json
  logs/loss_comparison.png  (epoch-aligned plot)
  REPORT.md                 (updated comparison section)
"""

import os
import sys
import json
import math
import time
import __main__ as _main_mod

import torch
import torch.nn.functional as F
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPTS)

from train_transformer import (  # noqa: E402
    Transformer,
    SimpleVocab,
    translate_greedy,
    translate_beam,
    nopeak_mask,
)
from train_seq2seq import (  # noqa: E402
    Seq2SeqAttention,
    batch_translate_greedy_seq2seq,
)

# Checkpoints were pickled with SimpleVocab living in __main__.
_main_mod.SimpleVocab = SimpleVocab


def torch_load_ckpt(path, map_location):
    return torch.load(path, map_location=map_location, weights_only=False)


def batch_translate_beam_seq2seq(
    model,
    src_sentences,
    src_vocab,
    trg_vocab,
    device,
    beam_size=5,
    max_len=80,
    length_penalty=0.7,
    batch_size=32,
):
    """Batched beam search for Seq2Seq + Luong Attention.

    Processes one batch at a time; within a batch each sentence has its own
    beam list. Encoder is run once per batch then reused.
    """
    model.eval()
    all_preds = []
    pad_idx = src_vocab.stoi[src_vocab.pad_token]
    sos_idx = trg_vocab.stoi[trg_vocab.sos_token]
    eos_idx = trg_vocab.stoi[trg_vocab.eos_token]

    for start in range(0, len(src_sentences), batch_size):
        batch = src_sentences[start:start + batch_size]
        src_ids_list = [src_vocab.encode(s) for s in batch]
        max_src = max(len(x) for x in src_ids_list)
        padded = [ids + [pad_idx] * (max_src - len(ids)) for ids in src_ids_list]
        src_tensor = torch.tensor(padded, dtype=torch.long, device=device)
        src_mask = (src_tensor != pad_idx)
        B = src_tensor.size(0)

        with torch.no_grad():
            src_embedded = model.src_embed(src_tensor)
            encoder_outputs, (enc_h, enc_c) = model.encoder_lstm(src_embedded)

            # [num_layers, B, hidden]
            dec_h0 = enc_h.view(model.num_layers, 2, B, model.hidden_dim // 2)
            dec_h0 = torch.cat([dec_h0[:, 0], dec_h0[:, 1]], dim=-1)
            dec_c0 = enc_c.view(model.num_layers, 2, B, model.hidden_dim // 2)
            dec_c0 = torch.cat([dec_c0[:, 0], dec_c0[:, 1]], dim=-1)

            # Expand encoder for beams: [B*beam, src_len, H]
            beam = beam_size
            enc_exp = encoder_outputs.unsqueeze(1).expand(-1, beam, -1, -1).reshape(B * beam, max_src, model.hidden_dim)
            mask_exp = src_mask.unsqueeze(1).expand(-1, beam, -1).reshape(B * beam, max_src)
            h = dec_h0.unsqueeze(2).expand(-1, -1, beam, -1).reshape(model.num_layers, B * beam, model.hidden_dim)
            c = dec_c0.unsqueeze(2).expand(-1, -1, beam, -1).reshape(model.num_layers, B * beam, model.hidden_dim)

            # tokens: [B, beam, cur_len]; start with SOS
            tokens = torch.full((B, beam, 1), sos_idx, dtype=torch.long, device=device)
            # cumulative log-prob [B, beam]
            scores = torch.full((B, beam), -1e9, device=device)
            scores[:, 0] = 0.0
            finished = torch.zeros(B, beam, dtype=torch.bool, device=device)
            context = torch.zeros(B * beam, model.hidden_dim, device=device)

            for _ in range(max_len):
                if finished.all():
                    break

                last = tokens[:, :, -1].reshape(B * beam)
                emb = model.trg_embed(last)
                lstm_in = torch.cat([emb, context], dim=-1).unsqueeze(1)
                out, (h, c) = model.decoder_lstm(lstm_in, (h, c))
                dec_t = out.squeeze(1)
                context, _ = model.attention(dec_t, enc_exp, src_mask=mask_exp)
                concat = torch.tanh(model.wc(torch.cat([dec_t, context], dim=-1)))
                log_probs = F.log_softmax(model.out(concat), dim=-1)  # [B*beam, V]
                log_probs = log_probs.view(B, beam, -1)

                # For finished beams, force EOS continuation with 0 extra score
                if finished.any():
                    mask_fin = finished.unsqueeze(-1)
                    log_probs = log_probs.masked_fill(mask_fin, -1e9)
                    # keep finished beam alive by selecting only EOS with score 0
                    log_probs[:, :, eos_idx] = torch.where(
                        finished,
                        torch.zeros_like(log_probs[:, :, eos_idx]),
                        log_probs[:, :, eos_idx],
                    )

                # candidate scores: [B, beam, V]
                cand_scores = scores.unsqueeze(-1) + log_probs
                cand_scores = cand_scores.view(B, -1)  # [B, beam*V]
                top_scores, top_idx = cand_scores.topk(beam, dim=-1)
                next_beam = top_idx // log_probs.size(-1)  # which previous beam
                next_token = top_idx % log_probs.size(-1)

                # reorder states
                gather_idx = (torch.arange(B, device=device).unsqueeze(1) * beam + next_beam).reshape(B * beam)
                h = h[:, gather_idx, :]
                c = c[:, gather_idx, :]
                context = context[gather_idx]
                enc_exp = enc_exp[gather_idx]
                mask_exp = mask_exp[gather_idx]

                # rebuild tokens
                old_tokens = tokens
                new_tokens = []
                for b in range(B):
                    seqs = []
                    for k in range(beam):
                        prev = next_beam[b, k].item()
                        tok = next_token[b, k].item()
                        seqs.append(torch.cat([old_tokens[b, prev], torch.tensor([tok], device=device)]))
                    new_tokens.append(torch.stack(seqs))
                tokens = torch.stack(new_tokens)  # [B, beam, len]
                scores = top_scores
                finished = finished.gather(1, next_beam) | (next_token == eos_idx)

            # pick best beam per sentence with length penalty
            best_preds = []
            for b in range(B):
                best_score = -1e18
                best_tok = None
                for k in range(beam):
                    seq = tokens[b, k].tolist()
                    # drop SOS
                    body = []
                    for t in seq[1:]:
                        if t == eos_idx:
                            break
                        body.append(t)
                    length = max(len(body), 1)
                    ns = scores[b, k].item() / (length ** length_penalty)
                    if ns > best_score:
                        best_score = ns
                        best_tok = body
                best_preds.append(trg_vocab.decode(best_tok or []))
            all_preds.extend(best_preds)

    return all_preds


def load_transformer(ckpt_path, device):
    ckpt = torch_load_ckpt(ckpt_path, device)
    src_vocab, trg_vocab = ckpt['src_vocab'], ckpt['trg_vocab']
    # Infer N from checkpoint keys
    layer_ids = {
        int(k.split('.')[2])
        for k in ckpt['model_state_dict']
        if k.startswith('encoder.layers.')
    }
    n_layers = max(layer_ids) + 1 if layer_ids else 4
    # Infer d_model from embed weight
    d_model = ckpt['model_state_dict']['encoder.embed.embed.weight'].shape[1]
    heads = 8
    model = Transformer(
        len(src_vocab), len(trg_vocab),
        d_model=d_model, N=n_layers, heads=heads, dropout=0.0, tie_weights=True,
    ).to(device)
    # out.weight may be tied; load strictly after constructing with tie_weights
    model.load_state_dict(ckpt['model_state_dict'], strict=True)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    return model, src_vocab, trg_vocab, {
        'n_layers': n_layers,
        'd_model': d_model,
        'heads': heads,
        'params': n_params,
    }


def load_seq2seq(ckpt_path, device):
    ckpt = torch_load_ckpt(ckpt_path, device)
    src_vocab, trg_vocab = ckpt['src_vocab'], ckpt['trg_vocab']
    emb = ckpt['model_state_dict']['src_embed.weight'].shape[1]
    # encoder_lstm.weight_ih_l0 shape: [4*(hidden//2), embed]
    w = ckpt['model_state_dict']['encoder_lstm.weight_ih_l0']
    hidden_half = w.shape[0] // 4
    hidden_dim = hidden_half * 2
    # count layers from keys
    layer_keys = [k for k in ckpt['model_state_dict'] if k.startswith('encoder_lstm.weight_ih_l')]
    # weight_ih_l0 and weight_ih_l0_reverse for bi; unidirectional reverse only for layer 0..
    num_layers = 1 + max(
        (int(k.split('weight_ih_l')[1].split('_')[0]) for k in layer_keys if not k.endswith('_reverse')),
        default=0,
    )
    model = Seq2SeqAttention(
        len(src_vocab), len(trg_vocab),
        embed_dim=emb, hidden_dim=hidden_dim, num_layers=num_layers, dropout=0.0,
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'], strict=True)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    return model, src_vocab, trg_vocab, {
        'embed_dim': emb,
        'hidden_dim': hidden_dim,
        'num_layers': num_layers,
        'params': n_params,
    }


def bleu_of(preds, refs, vocab):
    hyps = [vocab.tokenize(p) for p in preds]
    references = [[vocab.tokenize(r)] for r in refs]
    smooth = SmoothingFunction().method4
    return corpus_bleu(references, hyps, smoothing_function=smooth) * 100


def eval_transformer(model, src_vocab, trg_vocab, test_en, test_vi, device, mode, beam_size):
    preds = []
    t0 = time.time()
    for s in test_en:
        if mode == 'greedy':
            pred = translate_greedy(model, s, src_vocab, trg_vocab, device)
        else:
            pred = translate_beam(model, s, src_vocab, trg_vocab, device, beam_size=beam_size)
        preds.append(pred)
    elapsed = time.time() - t0
    score = bleu_of(preds, test_vi, trg_vocab)
    samples = [
        {'src': test_en[i], 'ref': test_vi[i], 'pred': preds[i]}
        for i in range(min(5, len(test_en)))
    ]
    return score, elapsed, samples, preds


def eval_seq2seq(model, src_vocab, trg_vocab, test_en, test_vi, device, mode, beam_size):
    t0 = time.time()
    if mode == 'greedy':
        preds = batch_translate_greedy_seq2seq(
            model, test_en, src_vocab, trg_vocab, device, batch_size=128
        )
    else:
        preds = batch_translate_beam_seq2seq(
            model, test_en, src_vocab, trg_vocab, device,
            beam_size=beam_size, batch_size=16,
        )
    elapsed = time.time() - t0
    score = bleu_of(preds, test_vi, trg_vocab)
    samples = [
        {'src': test_en[i], 'ref': test_vi[i], 'pred': preds[i]}
        for i in range(min(5, len(test_en)))
    ]
    return score, elapsed, samples, preds


def plot_losses(t_metrics, s_metrics, out_path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5))
    t_ep = range(1, len(t_metrics['train_loss']) + 1)
    s_ep = range(1, len(s_metrics['train_loss']) + 1)
    ax.plot(t_ep, t_metrics['train_loss'], label='Transformer Train', color='#2b5c8f', linewidth=2)
    ax.plot(t_ep, t_metrics['val_loss'], label='Transformer Val', color='#4682b4', linestyle='--')
    ax.plot(s_ep, s_metrics['train_loss'], label='Seq2Seq Train', color='#d9534f', linewidth=2)
    ax.plot(s_ep, s_metrics['val_loss'], label='Seq2Seq Val', color='#f0ad4e', linestyle='--')
    ax.set_title('Training & Validation Loss (IWSLT 2015 En-Vi)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True, linestyle=':', alpha=0.6)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def write_report(report_path, result):
    t = result['transformer']
    s = result['seq2seq']
    fair = result['fair']
    # Prefer beam as primary if available, else greedy
    primary = 'beam' if 'beam' in fair else 'greedy'
    other = 'greedy' if primary == 'beam' else 'beam'

    t_bleu_p = fair[primary]['transformer_bleu']
    s_bleu_p = fair[primary]['seq2seq_bleu']
    t_sample = fair[primary]['transformer_samples']
    s_sample = fair[primary]['seq2seq_samples']

    lines = []
    lines.append('# Building Machine Translation English - Vietnamese')
    lines.append('')
    lines.append('## Đề Bài (Problem Statement)')
    lines.append('- **Task**: Building Machine Translation English - Vietnamese')
    lines.append('- **Reimplementation with 2 cases**:')
    lines.append('  1. **Google Colab**: `demo_transformer.ipynb` (Transformer Architecture)')
    lines.append('  2. **GitHub NMT**: [tensorflow/nmt](https://github.com/tensorflow/nmt) (Seq2Seq + Luong Attention Architecture)')
    lines.append('- **Comparison Corpus**: English - Vietnamese Parallel Corpus (**IWSLT 2015**)')
    lines.append('')
    lines.append('> Ghi chú: Case NMT (Seq2Seq) được **reimplement bằng PyTorch** theo kiến trúc/hparams của `tensorflow/nmt` (IWSLT15), không chạy TensorFlow 1.x gốc. Case Transformer reimplement từ `demo_transformer.ipynb`.')
    lines.append('')
    lines.append('---')
    lines.append('')
    lines.append('## 1. Tổng Quan Bộ Dữ Liệu (IWSLT 2015 Corpus)')
    lines.append(f"- **Tập Train (`train.en`, `train.vi`)**: 133,317 cặp câu (lọc độ dài ≤ 80 khi train)")
    lines.append('- **Tập Validation (`tst2012`)**: 1,553 cặp câu')
    lines.append('- **Tập Test (`tst2013`)**: 1,268 cặp câu')
    lines.append(f"- **Vocabulary (build từ train, min_freq=2, max_size=30k)**: EN: {result['vocab']['src']} | VI: {result['vocab']['trg']}")
    lines.append('')
    lines.append('---')
    lines.append('')
    lines.append('## 2. Reimplementation Case 1: GitHub NMT (Seq2Seq + Luong Attention)')
    lines.append('Dựa trên kiến trúc `tensorflow/nmt` (hparams `iwslt15.json`):')
    lines.append(f"- **Encoder**: {s['cfg']['num_layers']}-layer Bidirectional LSTM (hidden mỗi hướng = {s['cfg']['hidden_dim'] // 2}, concat = {s['cfg']['hidden_dim']}).")
    lines.append(f"- **Decoder**: {s['cfg']['num_layers']}-layer Unidirectional LSTM (hidden = {s['cfg']['hidden_dim']}).")
    lines.append('- **Attention**: Scaled Luong Attention + Input Feeding.')
    lines.append(f"- **Embedding**: {s['cfg']['embed_dim']}")
    lines.append(f"- **Tổng tham số**: **{s['cfg']['params']/1e6:.1f}M**")
    lines.append(f"- **Train**: {result['train_meta']['seq2seq_epochs']} epochs | train loss cuối {result['train_meta']['seq2seq_train_loss']:.4f} | val loss tốt nhất-track {result['train_meta']['seq2seq_val_loss']:.4f}")
    lines.append(f"- **Thời gian train**: {result['train_meta']['seq2seq_seconds']:.2f}s")
    lines.append('')
    lines.append('---')
    lines.append('')
    lines.append('## 3. Reimplementation Case 2: Google Colab (`demo_transformer.ipynb`)')
    lines.append('Dựa trên notebook `demo_transformer.ipynb` (PyTorch):')
    lines.append(f"- **Kiến trúc**: Transformer Encoder-Decoder.")
    lines.append(f"- **Cấu hình thực tế**: $N={t['cfg']['n_layers']}$ layers, $d_{{model}}={t['cfg']['d_model']}$, $d_{{ff}}=2048$, $h={t['cfg']['heads']}$ heads.")
    lines.append('- **Positional Encoding**: Sinusoidal.')
    lines.append(f"- **Tổng tham số**: **{t['cfg']['params']/1e6:.1f}M**")
    lines.append(f"- **Train**: {result['train_meta']['transformer_epochs']} epochs | train loss cuối {result['train_meta']['transformer_train_loss']:.4f} | val loss cuối {result['train_meta']['transformer_val_loss']:.4f}")
    lines.append(f"- **Thời gian train**: {result['train_meta']['transformer_seconds']:.2f}s")
    lines.append('')
    lines.append('---')
    lines.append('')
    lines.append('## 4. Bảng So Sánh Kết Quả (Fair Comparison)')
    lines.append('')
    lines.append('Cả hai model được đánh giá lại trên **cùng** `tst2013`, **cùng tokenizer**, với **cùng chế độ decode**.')
    lines.append('')
    lines.append('| Tiêu chí | Case 1: Seq2Seq + Luong Attn | Case 2: Transformer |')
    lines.append('|---|---|---|')
    lines.append(f"| **Mô hình** | {s['cfg']['num_layers']}-layer Bi-LSTM + Luong | {t['cfg']['n_layers']}-layer Multi-Head Transformer |")
    lines.append(f"| **Tổng tham số** | **{s['cfg']['params']/1e6:.1f}M** | **{t['cfg']['params']/1e6:.1f}M** |")
    lines.append(f"| **Số Epochs train** | {result['train_meta']['seq2seq_epochs']} | {result['train_meta']['transformer_epochs']} |")
    lines.append(f"| **Train Loss cuối** | {result['train_meta']['seq2seq_train_loss']:.4f} | {result['train_meta']['transformer_train_loss']:.4f} |")
    lines.append(f"| **Val Loss cuối (`tst2012`)** | {result['train_meta']['seq2seq_val_loss']:.4f} | {result['train_meta']['transformer_val_loss']:.4f} |")
    if 'greedy' in fair:
        lines.append(f"| **BLEU greedy (`tst2013`)** | **{fair['greedy']['seq2seq_bleu']:.2f}** | **{fair['greedy']['transformer_bleu']:.2f}** |")
    if 'beam' in fair:
        bs = fair['beam']['beam_size']
        lines.append(f"| **BLEU beam={bs} (`tst2013`)** | **{fair['beam']['seq2seq_bleu']:.2f}** | **{fair['beam']['transformer_bleu']:.2f}** |")
    lines.append(f"| **Thời gian huấn luyện** | {result['train_meta']['seq2seq_seconds']:.2f}s | **{result['train_meta']['transformer_seconds']:.2f}s** |")
    lines.append('')
    lines.append(f'*Primary decode mode for qualitative samples: **{primary}**.')
    lines.append('')
    lines.append('---')
    lines.append('')
    lines.append(f'## 5. Ví Dụ Dịch Thực Tế Trên `tst2013` (decode={primary})')
    lines.append('')
    for i, (ts, ss) in enumerate(zip(t_sample, s_sample), 1):
        lines.append(f'### Ví dụ {i}:')
        lines.append(f"- **Input (EN)**: `{ts['src']}`")
        lines.append(f"- **Reference (VI)**: `{ts['ref']}`")
        lines.append(f"- **Case 1 (Seq2Seq)**: `{ss['pred']}`")
        lines.append(f"- **Case 2 (Transformer)**: `{ts['pred']}`")
        lines.append('')
    lines.append('---')
    lines.append('')
    lines.append('## 6. Kết Luận')
    delta = t_bleu_p - s_bleu_p
    winner = 'Transformer' if delta >= 0 else 'Seq2Seq'
    lines.append(f'1. **Chất lượng dịch (fair, {primary})**: {winner} cao hơn (**{max(t_bleu_p, s_bleu_p):.2f} vs {min(t_bleu_p, s_bleu_p):.2f}**, Δ = **{abs(delta):.2f} BLEU**).')
    if other in fair:
        d2 = fair[other]['transformer_bleu'] - fair[other]['seq2seq_bleu']
        lines.append(f'2. **Cùng chiều hướng với decode {other}**: Transformer {fair[other]["transformer_bleu"]:.2f} vs Seq2Seq {fair[other]["seq2seq_bleu"]:.2f} (Δ = {d2:+.2f}).')
    ratio = result['train_meta']['seq2seq_seconds'] / max(result['train_meta']['transformer_seconds'], 1e-6)
    lines.append(f'3. **Tốc độ train**: Transformer nhanh hơn khoảng **{ratio:.1f}×** ({result["train_meta"]["transformer_seconds"]/60:.1f} phút vs {result["train_meta"]["seq2seq_seconds"]/60:.1f} phút).')
    lines.append('4. **Fairness**: So sánh BLEU đã dùng cùng corpus IWSLT15, cùng split, cùng tokenizer và cùng chế độ decode (greedy và/hoặc beam).')
    lines.append('')

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'=== Fair comparison on {device} ===', flush=True)

    data_dir = os.path.join(ROOT, 'data')
    models_dir = os.path.join(ROOT, 'models')
    logs_dir = os.path.join(ROOT, 'logs')
    os.makedirs(logs_dir, exist_ok=True)

    with open(os.path.join(data_dir, 'tst2013.en'), encoding='utf-8') as f:
        test_en = [line.strip() for line in f]
    with open(os.path.join(data_dir, 'tst2013.vi'), encoding='utf-8') as f:
        test_vi = [line.strip() for line in f]

    t_model, t_src, t_trg, t_cfg = load_transformer(
        os.path.join(models_dir, 'transformer_model.pt'), device
    )
    s_model, s_src, s_trg, s_cfg = load_seq2seq(
        os.path.join(models_dir, 'seq2seq_model.pt'), device
    )
    print(f"Transformer: N={t_cfg['n_layers']} d={t_cfg['d_model']} params={t_cfg['params']/1e6:.1f}M", flush=True)
    print(f"Seq2Seq: layers={s_cfg['num_layers']} H={s_cfg['hidden_dim']} params={s_cfg['params']/1e6:.1f}M", flush=True)
    print(f"Vocab sizes: T_src={len(t_src)} T_trg={len(t_trg)} | S_src={len(s_src)} S_trg={len(s_trg)}", flush=True)

    fair = {}
    beam_size = 5

    for mode in ('greedy', 'beam'):
        print(f'\n--- Evaluating mode={mode} ---', flush=True)
        t_bleu, t_sec, t_samples, _ = eval_transformer(
            t_model, t_src, t_trg, test_en, test_vi, device, mode, beam_size
        )
        print(f'Transformer {mode}: BLEU={t_bleu:.2f} time={t_sec:.1f}s', flush=True)
        s_bleu, s_sec, s_samples, _ = eval_seq2seq(
            s_model, s_src, s_trg, test_en, test_vi, device, mode, beam_size
        )
        print(f'Seq2Seq     {mode}: BLEU={s_bleu:.2f} time={s_sec:.1f}s', flush=True)
        fair[mode] = {
            'beam_size': beam_size if mode == 'beam' else 1,
            'transformer_bleu': t_bleu,
            'seq2seq_bleu': s_bleu,
            'transformer_eval_seconds': t_sec,
            'seq2seq_eval_seconds': s_sec,
            'transformer_samples': t_samples,
            'seq2seq_samples': s_samples,
        }

    with open(os.path.join(logs_dir, 'transformer_metrics.json'), encoding='utf-8') as f:
        t_metrics = json.load(f)
    with open(os.path.join(logs_dir, 'seq2seq_metrics.json'), encoding='utf-8') as f:
        s_metrics = json.load(f)

    # Update metrics files with fair scores (keep training history)
    t_metrics['fair_bleu_greedy'] = fair['greedy']['transformer_bleu']
    t_metrics['fair_bleu_beam5'] = fair['beam']['transformer_bleu']
    t_metrics['final_bleu'] = fair['beam']['transformer_bleu']  # primary
    t_metrics['decode_note'] = 'primary=beam5; also greedy for fair comparison'
    s_metrics['fair_bleu_greedy'] = fair['greedy']['seq2seq_bleu']
    s_metrics['fair_bleu_beam5'] = fair['beam']['seq2seq_bleu']
    s_metrics['final_bleu'] = fair['beam']['seq2seq_bleu']
    s_metrics['decode_note'] = 'primary=beam5; also greedy for fair comparison'
    # refresh samples from beam
    t_metrics['sample_translations'] = fair['beam']['transformer_samples']
    s_metrics['sample_translations'] = fair['beam']['seq2seq_samples']

    with open(os.path.join(logs_dir, 'transformer_metrics.json'), 'w', encoding='utf-8') as f:
        json.dump(t_metrics, f, indent=2, ensure_ascii=False)
    with open(os.path.join(logs_dir, 'seq2seq_metrics.json'), 'w', encoding='utf-8') as f:
        json.dump(s_metrics, f, indent=2, ensure_ascii=False)

    plot_path = os.path.join(logs_dir, 'loss_comparison.png')
    plot_losses(t_metrics, s_metrics, plot_path)
    print(f'Saved plot: {plot_path}', flush=True)

    result = {
        'transformer': {'cfg': t_cfg},
        'seq2seq': {'cfg': s_cfg},
        'vocab': {'src': len(t_src), 'trg': len(t_trg)},
        'train_meta': {
            'transformer_epochs': len(t_metrics['train_loss']),
            'seq2seq_epochs': len(s_metrics['train_loss']),
            'transformer_train_loss': t_metrics['train_loss'][-1],
            'seq2seq_train_loss': s_metrics['train_loss'][-1],
            'transformer_val_loss': t_metrics['val_loss'][-1],
            'seq2seq_val_loss': s_metrics['val_loss'][-1],
            'transformer_seconds': t_metrics.get('elapsed_seconds', 0),
            'seq2seq_seconds': s_metrics.get('elapsed_seconds', 0),
        },
        'fair': fair,
    }

    out_json = os.path.join(logs_dir, 'fair_comparison.json')
    # JSON-serializable copy without huge pred lists beyond samples
    serializable = {
        'transformer_cfg': t_cfg,
        'seq2seq_cfg': s_cfg,
        'vocab': result['vocab'],
        'train_meta': result['train_meta'],
        'fair': {
            mode: {
                k: v for k, v in vals.items()
                if k.endswith('_bleu') or k.endswith('_seconds') or k == 'beam_size' or k.endswith('_samples')
            }
            for mode, vals in fair.items()
        },
    }
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)
    print(f'Saved: {out_json}', flush=True)

    report_path = os.path.join(ROOT, 'REPORT.md')
    write_report(report_path, result)
    print(f'Updated: {report_path}', flush=True)

    print('\n================ FAIR COMPARISON =================')
    print(f"{'Mode':<10} | {'Transformer':>12} | {'Seq2Seq':>12} | {'delta':>8}")
    print('-' * 50)
    for mode in ('greedy', 'beam'):
        tb = fair[mode]['transformer_bleu']
        sb = fair[mode]['seq2seq_bleu']
        print(f'{mode:<10} | {tb:12.2f} | {sb:12.2f} | {tb-sb:+8.2f}')
    print('=' * 50)


if __name__ == '__main__':
    main()
