from pathlib import Path
import csv
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)
archive = DATA / "sms+spam+collection.zip"
url = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
if not archive.exists():
    urllib.request.urlretrieve(url, archive)
with zipfile.ZipFile(archive) as bundle:
    raw = bundle.read("SMSSpamCollection").decode("utf-8")
out = DATA / "sms_spam_uci.csv"
with out.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.writer(handle)
    writer.writerow(["text", "label"])
    for line in raw.splitlines():
        label, text = line.split("\t", 1)
        writer.writerow([text, 1 if label == "spam" else 0])
print(out)
