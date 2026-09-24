# /// script
# requires-python = ">=3.11,<3.13"
# dependencies = ["torch==2.13.0", "numpy", "coreai-torch==0.4.2", "coreai-core==1.0.0b2", "torchcrepe", "resampy", "soundfile", "hmmlearn"]
# ///
"""CREPE (Kim, Salamon, Li, Bello — ISMIR 2018, MIT) → tune-crepe-<capacity>-float32.aimodel, plus fixtures.

    uv run Tools/export_crepe.py [--capacity full|tiny] [--batch 256] [--out Tools/exports] [--install] [clip=path.wav …]

The network is torchcrepe's port of the original Keras weights: 1024 normalised samples at 16 kHz in,
360 sigmoid activations over 20-cent bins out. It is exported at a fixed batch of frames; the host pads
the last batch. Everything around the network is the ORIGINAL crepe pipeline (marl/crepe core.py, MIT),
vendored below rather than torchcrepe's, because torchcrepe dithers its cents with random noise and
squashes the activations through a second sigmoid; the original is deterministic:
  centre zero-pad 512, hop 160 (10 ms), frames normalised by mean and (biased) std clipped at 1e-8,
  cents = weighted average of the activation over the 9 bins around the argmax (to_local_average_cents),
  confidence = max activation, frequency = 10·2^(cents/1200),
  viterbi = a 360-state HMM on the argmax bins (triangular transition of width 12, self-emission 0.1),
  then the local average around the path bin.
Fixtures: per clip the 16 kHz audio the model saw, its activations, both decodings and the confidence;
for 16 kHz synthetic clips the normalised frames too."""
import argparse, os, sys, json, time, shutil
from pathlib import Path
import numpy as np, torch, torch.nn as nn, soundfile as sf
import torchcrepe
from numpy.lib.stride_tricks import as_strided
import coreai_torch
from coreai.runtime import AIModelAssetMetadata
HERE = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument("--capacity", default="full", choices=["full", "tiny"]); ap.add_argument("--batch", type=int, default=256)
ap.add_argument("--out", default=str(HERE / "exports")); ap.add_argument("--fixtures", default=str(HERE.parent / "Tests/PitchTrackerTests/Fixtures"))
ap.add_argument("--install", action="store_true"); ap.add_argument("clips", nargs="*", help="name=path.wav")
args = ap.parse_args()
SR, WINDOW, HOP, BINS = 16000, 1024, 160, 360
CENTS = np.linspace(0, 7180, BINS) + 1997.3794084376191

# ---- the original crepe pipeline (marl/crepe core.py), vendored ----
def frames_of(audio16k):
    audio = np.pad(audio16k.astype(np.float32), 512, mode="constant", constant_values=0)
    n = 1 + int((len(audio) - WINDOW) / HOP)
    frames = as_strided(audio, shape=(WINDOW, n), strides=(audio.itemsize, HOP * audio.itemsize)).transpose().copy()
    frames -= np.mean(frames, axis=1)[:, np.newaxis]
    frames /= np.clip(np.std(frames, axis=1)[:, np.newaxis], 1e-8, None)
    return frames
def local_average_cents(salience, center=None):
    if center is None: center = int(np.argmax(salience))
    start, end = max(0, center - 4), min(len(salience), center + 5)
    s = salience[start:end]
    return float(np.sum(s * CENTS[start:end]) / np.sum(s))
def viterbi_cents(activation):
    from hmmlearn import hmm
    starting = np.ones(BINS) / BINS
    xx, yy = np.meshgrid(range(BINS), range(BINS))
    transition = np.maximum(12 - abs(xx - yy), 0); transition = transition / np.sum(transition, axis=1)[:, None]
    self_emission = 0.1
    emission = np.eye(BINS) * self_emission + np.ones((BINS, BINS)) * ((1 - self_emission) / BINS)
    model = hmm.CategoricalHMM(BINS, starting, transition)
    model.startprob_, model.transmat_, model.emissionprob_ = starting, transition, emission
    observations = np.argmax(activation, axis=1)
    path = model.predict(observations.reshape(-1, 1), [len(observations)])
    return np.array([local_average_cents(activation[i], path[i]) for i in range(len(observations))]), path
def viterbi_path_numpy(observations):
    """The same HMM decoded exactly as hmmlearn's C++ viterbi does it — what the Swift port implements, asserted equal.
    Forward: lattice[t,i] = max_j(lattice[t-1,j] + logT[j,i]) + logE[t,i]. Backtrack by RECOMPUTATION, not stored
    pointers: where_from = argmax_i(lattice[t,i] + logT[i,where_from]) — and on a tie the LAST index, because hmmlearn
    takes std::max over (value, index) pairs; the final state alone is the FIRST maximum (std::max_element). Ties are
    common in unvoiced stretches (equal-probability bridges), so this matters."""
    xx, yy = np.meshgrid(range(BINS), range(BINS))
    transition = np.maximum(12 - abs(xx - yy), 0); transition = transition / np.sum(transition, axis=1)[:, None]
    with np.errstate(divide="ignore"): logT = np.log(transition)
    self_emission = 0.1
    log_self, log_other = np.log(self_emission + (1 - self_emission) / BINS), np.log((1 - self_emission) / BINS)
    n = len(observations)
    lattice = np.empty((n, BINS))
    lattice[0] = np.log(1 / BINS) + np.where(np.arange(BINS) == observations[0], log_self, log_other)
    for t in range(1, n):
        lattice[t] = np.max(lattice[t - 1][:, None] + logT, axis=0) + np.where(np.arange(BINS) == observations[t], log_self, log_other)
    path = np.zeros(n, dtype=np.int64)
    where = int(np.argmax(lattice[n - 1])); path[n - 1] = where
    for t in range(n - 2, -1, -1):
        v = lattice[t] + logT[:, where]
        where = int(BINS - 1 - np.argmax(v[::-1])); path[t] = where      # last maximum
    return path

# ---- the network ----
model = torchcrepe.Crepe(args.capacity)
model.load_state_dict(torch.load(Path(torchcrepe.__file__).parent / "assets" / f"{args.capacity}.pth", map_location="cpu"))
model.eval()
print(f"crepe {args.capacity}: {sum(p.numel() for p in model.parameters())/1e6:.1f} M parameters, batch {args.batch}", flush=True)
def activations(frames):
    out = []
    with torch.no_grad():
        for i in range(0, len(frames), args.batch):
            out.append(model(torch.from_numpy(frames[i:i + args.batch])).numpy())
    return np.concatenate(out)

fx = Path(args.fixtures); fx.mkdir(parents=True, exist_ok=True)
manifest = {"capacity": args.capacity, "batch": args.batch, "clips": {}}
for spec in args.clips:
    name, path = spec.split("=", 1)
    y, sr = sf.read(path, dtype="float32", always_2d=True); y = y.mean(1).astype(np.float32)
    if sr != SR:
        from resampy import resample
        y = resample(y, sr, SR).astype(np.float32)
    frames = frames_of(y)
    act = activations(frames).astype(np.float32)
    # torchcrepe's own preprocess differs only in the std (unbiased): report, do not assert.
    tc = torchcrepe.preprocess(torch.from_numpy(y)[None], SR, HOP, batch_size=len(frames)+1, pad=True)
    tcf = next(tc).numpy()
    local = np.array([local_average_cents(a) for a in act])
    vit, path = viterbi_cents(act)
    assert np.array_equal(path, viterbi_path_numpy(np.argmax(act, axis=1))), f"{name}: numpy viterbi != hmmlearn"
    conf = act.max(axis=1)
    sf.write(fx / f"{name}_16k.wav", y, SR, subtype="FLOAT")
    act.tofile(fx / f"{name}_{args.capacity}_activations.f32")
    local.astype(np.float64).tofile(fx / f"{name}_{args.capacity}_local.f64"); vit.astype(np.float64).tofile(fx / f"{name}_{args.capacity}_viterbi.f64")
    conf.astype(np.float32).tofile(fx / f"{name}_{args.capacity}_confidence.f32")
    if sr == SR and len(frames) <= 256: frames.astype(np.float32).tofile(fx / f"{name}_frames.f32")
    hz = 10 * 2 ** (local / 1200)
    manifest["clips"][name] = {"frames": int(len(frames)), "seconds": len(y) / SR, "source_rate": int(sr)}
    print(f"{name}: {len(y)/SR:.2f} s, {len(frames)} frames; torchcrepe frames max |diff| {np.abs(frames - tcf[:len(frames)]).max():.2e}; "
          f"median f0 {np.median(hz[conf > 0.5]) if (conf > 0.5).any() else float('nan'):.2f} Hz, confidence median {np.median(conf):.3f}; viterbi path == numpy", flush=True)
json.dump(manifest, open(fx / f"manifest_{args.capacity}.json", "w"), indent=1)

class Net(nn.Module):
    def __init__(self, m): super().__init__(); self.m = m
    def forward(self, frames): return self.m(frames)
net = Net(model).eval()
t0 = time.time()
with torch.no_grad():
    ex = torch.export.export(net, (torch.randn(args.batch, WINDOW),)).run_decompositions(coreai_torch.get_decomp_table())
converter = coreai_torch.TorchConverter(mode=coreai_torch.TorchConverter.Mode.RELEASE)
converter.add_exported_program(ex, input_names=["frames"], output_names=["activations"], entrypoint_name="main")
program = converter.to_coreai(); program.optimize()
meta = AIModelAssetMetadata()
meta.author = "Kim, Salamon, Li, Bello — CREPE (ISMIR 2018); PyTorch weights via torchcrepe (Max Morrison); Core AI export by PitchTracker"
meta.license = "MIT"
meta.model_description = f"CREPE {args.capacity}: [{args.batch}, 1024] normalised 16 kHz frames → [{args.batch}, 360] sigmoid activations over 20-cent bins from 1997.38 cents (31.7 Hz). Framing and decoding in the host as marl/crepe does them."
meta.creation_date = int(time.time())
out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
asset = out / f"tune-crepe-{args.capacity}-float32.aimodel"
if asset.exists(): shutil.rmtree(asset)
program.save_asset(asset, meta)
print(f"saved {asset} ({sum(f.stat().st_size for f in asset.rglob('*'))/1e6:.0f} MB) in {time.time()-t0:.0f} s", flush=True)
if args.install:
    dest = Path.home() / "Library/Application Support/tune"; dest.mkdir(parents=True, exist_ok=True)
    d = dest / asset.name
    if d.exists(): shutil.rmtree(d)
    shutil.copytree(asset, d); print(f"installed into {dest}")
