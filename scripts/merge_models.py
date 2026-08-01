import os
import glob

def merge_files(model_prefix):
    part_files = sorted(glob.glob(f"{model_prefix}.part*"), key=lambda x: int(x.split('.part')[-1]))
    if not part_files:
        return
    out_file = model_prefix
    print(f"Merging {len(part_files)} parts into {out_file}...")
    with open(out_file, 'wb') as outfile:
        for pf in part_files:
            with open(pf, 'rb') as infile:
                outfile.write(infile.read())
            print(f"  Merged {pf}")
    print(f"Successfully restored {out_file} ({os.path.getsize(out_file) / (1024*1024):.2f} MB)")

if __name__ == "__main__":
    merge_files("models/seq2seq_model.pt")
    merge_files("models/transformer_model.pt")
