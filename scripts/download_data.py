import os
import urllib.request

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
os.makedirs(DATA_DIR, exist_ok=True)

BASE_URL = "https://nlp.stanford.edu/projects/nmt/data/iwslt15.en-vi/"
FILES = [
    "train.en", "train.vi",
    "tst2012.en", "tst2012.vi",
    "tst2013.en", "tst2013.vi",
    "vocab.en", "vocab.vi"
]

def download_file(filename):
    url = BASE_URL + filename
    dest_path = os.path.join(DATA_DIR, filename)
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 0:
        print(f"[SKIP] {filename} already exists ({os.path.getsize(dest_path)} bytes)")
        return dest_path
    
    print(f"Downloading {filename} from {url}...")
    opener = urllib.request.build_opener()
    opener.addheaders = [('User-Agent', 'Mozilla/5.0')]
    urllib.request.install_opener(opener)
    urllib.request.urlretrieve(url, dest_path)
    print(f"[DONE] Saved {filename} ({os.path.getsize(dest_path)} bytes)")
    return dest_path

def main():
    print("=== Downloading IWSLT 2015 English-Vietnamese Dataset ===")
    for f in FILES:
        download_file(f)
    print("=== Dataset download complete ===")

if __name__ == '__main__':
    main()
