import csv
from pathlib import Path
import whisper

IN_CSV  = Path("voice-dev/data/segments.csv")
OUT_CSV = Path("voice-dev/data/transcripts.csv")

model = whisper.load_model("small")

rows_out = []
with IN_CSV.open("r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for r in reader:
        wav = r["lofi_path"] or r["hq_path"]
        result = model.transcribe(wav, language="uk", fp16=False, verbose=False)
        text = result["text"].strip()
        rows_out.append({
            "idx": r["idx"], "file": wav,
            "start": float(r["start"]), "end": float(r["end"]),
            "text": text
        })

with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["idx","file","start","end","text"])
    w.writeheader(); w.writerows(rows_out)

print(f"OK -> {OUT_CSV} ({len(rows_out)} рядків)")