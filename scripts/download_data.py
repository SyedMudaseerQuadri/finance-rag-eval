"""Download BEIR FiQA-2018 into data/fiqa (corpus.jsonl, queries.jsonl, qrels/test.tsv)."""
import sys
import urllib.request
import zipfile
from pathlib import Path

URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/fiqa.zip"
DATA = Path(__file__).resolve().parents[1] / "data"


def main() -> None:
    target = DATA / "fiqa"
    if (target / "corpus.jsonl").exists():
        print("already downloaded:", target)
        return
    DATA.mkdir(exist_ok=True)
    zpath = DATA / "fiqa.zip"
    print("downloading", URL)
    urllib.request.urlretrieve(URL, zpath)
    with zipfile.ZipFile(zpath) as z:
        z.extractall(DATA)
    zpath.unlink()
    print("extracted to", target)


if __name__ == "__main__":
    sys.exit(main())
