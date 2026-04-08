"""Run group analysis in two batches, saving intermediate results."""
import sys, json, time, pickle
from pathlib import Path

sys.path.insert(0, str(Path("/sessions/dazzling-vigilant-fermat/mnt/20260331 - Elerding et al., VTA GABA neurons")))
from fp_glm_group_analysis import load_h5_data, run_subject_glm

SAVE_DIR = Path("/sessions/dazzling-vigilant-fermat/mnt/20260331 - Elerding et al., VTA GABA neurons")
H5 = SAVE_DIR / "streams_peth_multi_mix_all.h5"

batch = int(sys.argv[1])  # 1 or 2

print(f"Loading data...")
df = load_h5_data(H5)
subjects = sorted(df["subject"].unique())

if batch == 1:
    batch_subjects = subjects[:16]
else:
    batch_subjects = subjects[16:]

print(f"Batch {batch}: {len(batch_subjects)} subjects")
results = []
t0 = time.time()

for si, subj in enumerate(batch_subjects):
    ch = df[df["subject"] == subj]["channel_id"].iloc[0]
    t_subj = time.time()
    print(f"[{si+1}/{len(batch_subjects)}] {subj} ({ch})...", end=" ", flush=True)
    res = run_subject_glm(df, subj, n_shifts=5000, alpha=0.01, seed=42+si+(batch-1)*16)
    results.append(res)
    elapsed = time.time() - t_subj
    sig = [res["chunk_names"][k] for k in range(len(res["chunk_names"]))
           if res["full_glm"]["significant"][k] and res["chunk_names"][k] != "session"]
    print(f"{elapsed:.1f}s  R2={res['full_glm']['r2_full']:.3f}  sig: {sig if sig else 'none'}")

print(f"\nBatch {batch} done in {time.time()-t0:.0f}s")
with open(SAVE_DIR / f"_batch{batch}_results.pkl", "wb") as f:
    pickle.dump(results, f)
print(f"Saved to _batch{batch}_results.pkl")
