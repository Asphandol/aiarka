from pathlib import Path
import numpy as np, soundfile as sf, subprocess, csv
import webrtcvad

SRC_WAV = Path("voice-dev/data/raw/yarka_voice_softdenoise.wav")
TMP_16K = SRC_WAV.parent / "_tmp_16k_mono.wav"
HQ_DIR  = Path("voice-dev/data/clean_hq")
LOFI_DIR= Path("voice-dev/data/clean_16k")
CSV_OUT = Path("voice-dev/data/segments.csv")

SAMPLE_RATE = 16000
FRAME_MS = 30
AGGR = 2
PAD_MS = 150
MIN_MS = 800

def run(cmd):
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def to_mono16k_for_vad(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    run(["ffmpeg","-y","-i",str(src),"-ac","1","-ar",str(SAMPLE_RATE),"-sample_fmt","s16",str(dst)])

def gen_frames_int16(x_int16: np.ndarray, sr, ms):
    n = int(sr*ms/1000)
    total = (len(x_int16)//n)*n
    x_int16 = np.ascontiguousarray(x_int16[:total])
    for i in range(0,total,n):
        yield x_int16[i:i+n].tobytes()

def main():
    HQ_DIR.mkdir(parents=True, exist_ok=True)
    LOFI_DIR.mkdir(parents=True, exist_ok=True)

    to_mono16k_for_vad(SRC_WAV, TMP_16K)


    x, sr = sf.read(TMP_16K, dtype="int16")
    if x.ndim>1: x = x[:,0]
    vad = webrtcvad.Vad(AGGR)
    frames = list(gen_frames_int16(x, sr, FRAME_MS))
    voiced = [vad.is_speech(fb, sr) for fb in frames]

    segs = []
    cur = None
    for i,v in enumerate(voiced):
        if v and cur is None:
            cur = i
        elif not v and cur is not None:
            segs.append((cur, i-1)); cur=None
    if cur is not None:
        segs.append((cur, len(voiced)-1))

    fr_n = int(sr*FRAME_MS/1000)
    pad  = PAD_MS/1000.0
    min_dur = MIN_MS/1000.0

    time_segs = []
    for s,e in segs:
        a = max(0.0, (s*fr_n)/sr - pad)
        b = ( (e+1)*fr_n )/sr + pad
        if (b-a) >= min_dur:
            time_segs.append((a,b))

    with open(CSV_OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["idx","start","end","hq_path","lofi_path"])
        for i,(a,b) in enumerate(time_segs):
            hq = HQ_DIR / f"yarka_{i:05d}.wav"
            lofi = LOFI_DIR / f"yarka_{i:05d}.wav"
            run(["ffmpeg","-y","-ss", f"{a:.3f}", "-to", f"{b:.3f}",
                 "-i", str(SRC_WAV), "-c","copy", str(hq)])
            run(["ffmpeg","-y","-i", str(hq),
                 "-ac","1","-ar", str(SAMPLE_RATE), "-sample_fmt","s16", str(lofi)])
            w.writerow([i, f"{a:.3f}", f"{b:.3f}", str(hq), str(lofi)])

    TMP_16K.unlink(missing_ok=True)
    print(f"Saved {len(time_segs)} segments.\nHQ: {HQ_DIR}\n16k: {LOFI_DIR}\nCSV: {CSV_OUT}")

if __name__ == "__main__":
    main()