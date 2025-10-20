from faster_whisper import WhisperModel
from pathlib import Path
import csv

AUDIO_DIR=Path("voice-dev/data/clean_hq"); OUT="voice-dev/data/metadata.csv"
model=WhisperModel("medium", device="cpu", compute_type="int8")

with open(OUT,"w",newline="",encoding="utf-8") as f:
    w=csv.writer(f,delimiter="|"); n=0
    for wav in sorted(AUDIO_DIR.glob("*.wav")):
        text=[]
        segments, info = model.transcribe(str(wav), language="uk", vad_filter=False, beam_size=5)
        for s in segments: text.append(s.text.strip())
        line=" ".join(text).strip()
        if len(line)<3: continue
        w.writerow([str(wav), line])
        n+=1
print("Wrote rows:", n, "to", OUT)