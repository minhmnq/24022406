import sys
import os
import site
site.addsitedir(r'C:\Users\minhn\AppData\Roaming\Python\Python313\site-packages')

import json
import matplotlib.pyplot as plt
import pandas as pd

def main():
    logs_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'logs')
    artifacts_dir = os.path.dirname(os.path.abspath(__file__))
    
    t_path = os.path.join(logs_dir, 'transformer_metrics.json')
    s_path = os.path.join(logs_dir, 'seq2seq_metrics.json')
    
    if not os.path.exists(t_path) or not os.path.exists(s_path):
        print("[ERROR] Training metrics files not found yet.")
        return

    with open(t_path, encoding='utf-8') as f:
        t_metrics = json.load(f)
    with open(s_path, encoding='utf-8') as f:
        s_metrics = json.load(f)

    # 1. Plot Loss Curves (each model may have a different epoch count)
    t_epochs = range(1, len(t_metrics['train_loss']) + 1)
    s_epochs = range(1, len(s_metrics['train_loss']) + 1)

    plt.figure(figsize=(10, 5))
    plt.plot(t_epochs, t_metrics['train_loss'], label='Transformer Train Loss', color='#2b5c8f', linewidth=2)
    plt.plot(t_epochs, t_metrics['val_loss'], label='Transformer Val Loss', color='#4682b4', linestyle='--')

    plt.plot(s_epochs, s_metrics['train_loss'], label='Seq2Seq-Attention Train Loss', color='#d9534f', linewidth=2)
    plt.plot(s_epochs, s_metrics['val_loss'], label='Seq2Seq-Attention Val Loss', color='#f0ad4e', linestyle='--')

    plt.title('Training & Validation Loss Comparison (IWSLT 2015 En-Vi)', fontsize=14, fontweight='bold')
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()

    plot_path = os.path.join(logs_dir, 'loss_comparison.png')
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Saved loss comparison plot to {plot_path}")

    # Prefer fair BLEU keys when present
    t_bleu_g = t_metrics.get('fair_bleu_greedy', t_metrics.get('final_bleu'))
    s_bleu_g = s_metrics.get('fair_bleu_greedy', s_metrics.get('final_bleu'))
    t_bleu_b = t_metrics.get('fair_bleu_beam5')
    s_bleu_b = s_metrics.get('fair_bleu_beam5')

    # 2. Print Comparison Summary
    print("\n=================== MODEL COMPARISON REPORT ===================")
    print(f"{'Metric':<28} | {'Transformer':<18} | {'Seq2Seq Luong Attn':<18}")
    print("-" * 72)
    print(f"{'Final Train Loss':<28} | {t_metrics['train_loss'][-1]:<18.4f} | {s_metrics['train_loss'][-1]:<18.4f}")
    print(f"{'Final Val Loss':<28} | {t_metrics['val_loss'][-1]:<18.4f} | {s_metrics['val_loss'][-1]:<18.4f}")
    print(f"{'BLEU greedy (tst2013)':<28} | {t_bleu_g:<18.2f} | {s_bleu_g:<18.2f}")
    if t_bleu_b is not None and s_bleu_b is not None:
        print(f"{'BLEU beam=5 (tst2013)':<28} | {t_bleu_b:<18.2f} | {s_bleu_b:<18.2f}")
    print(f"{'Training Time (s)':<28} | {t_metrics['elapsed_seconds']:<18.2f} | {s_metrics['elapsed_seconds']:<18.2f}")
    print("=" * 72)

if __name__ == '__main__':
    main()
