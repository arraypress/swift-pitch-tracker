# PitchTracker

On-device **CREPE** for Swift — the pitch of a recording every 10 ms, in cents, hertz and MIDI
note number, with the network's confidence that the frame is pitched at all. CREPE (Kim,
Salamon, Li and Bello, ISMIR 2018; MIT) runs on Core AI; the framing, both decoders and the
confidence are the original `crepe` pipeline in Swift. Nothing leaves the machine.

```swift
import PitchTracker

let tracker = try await CrepeTracker(contentsOf: assetURL)            // tune-crepe-full-float32.aimodel
let track = try await tracker.track(contentsOf: url)                  // any file AVFoundation decodes
for frame in track.voiced() {                                         // confidence ≥ 0.5
    frame.time, frame.frequency, frame.cents, frame.midi, frame.confidence
}
let viterbi = try await tracker.track(contentsOf: url, method: .viterbi)   // crepe's viterbi=True
```

## What it is exactly

The network is CREPE's, from torchcrepe's port of the original Keras weights (`full`, 22 M
parameters, and `tiny`, 0.5 M): 1024 normalised samples at 16 kHz in, 360 sigmoid activations
out over bins 20 cents apart from 1997.38 cents above 10 Hz (31.70 Hz) to 9177.38 (1975.5 Hz).
It is exported at a fixed batch of 256 frames; the host pads the last batch.

Everything else is `marl/crepe`'s `core.py`, not torchcrepe's, because torchcrepe dithers its
cents with random noise and squashes the activations through a second sigmoid, and the original
is deterministic: centre zero-padding of 512 samples, a frame every 160 samples (10 ms), each
frame normalised by its mean and biased standard deviation clipped at 1e-8 (`CrepeFrontEnd`);
cents as the activation-weighted mean of the nine bins around the argmax
(`PitchDecoder.localAverage`, crepe's default); or `viterbi`, a 360-state HMM on the argmax bins
with a triangular transition of width 12 and self-emission 0.1, then the local average around
each path bin; confidence as the largest activation; frequency `10·2^(cents/1200)`.

The Viterbi decoder reproduces hmmlearn's C++ exactly, tie rule included — the last index on
a tie during backtracking, the first for the final state — because unvoiced stretches are full
of equal-probability bridges and the tie rule decides the path.

A 16 kHz file is read sample for sample; other rates go through AVAudioConverter at its best
quality (upstream uses resampy), which is a resampler difference, not a model one.

## Measured against upstream

`Tools/export_crepe.py` writes both assets and, per clip, the 16 kHz audio, the activations,
both decodings and the confidence from the vendored original pipeline; `swift test` compares.
The original Keras `crepe` package was run on the same clips as a second reference.

| Stage | Agreement |
|---|---|
| Keras crepe vs the exported network (3 clips) | 145–151 dB PSNR on the activations, argmax identical |
| Vendored decoders vs `crepe.predict` | 0.001 cents (float32 vs float64) |
| Framing, 3 clips | > 120 dB vs crepe's `get_activation` frames |
| Network on Core AI, full, 4 clips | 141–145 dB on the activations |
| Whole clips end to end, 7 clips × 2 decoders | max 0.001 cents on every voiced frame; confidence to 4e-6 |
| Network on Core AI, tiny, 2 clips | 145 dB |
| Viterbi path vs hmmlearn | identical on every clip and on random sequences |

Synthetic tones as ground truth: a steady A3 + 13 cents reads A3 + 11 cents at confidence 0.94;
C4 with ±40-cent vibrato at 6 Hz reads a mean of 60.017 with a 73-cent swing; a two-second
glide from E2 to E3 reads 41.17 → 50.85 (expected 41.2 → 50.8). On an M3 Max the full model
takes about 0.2 s per ten seconds of audio.

## The models

Not bundled. Export them from torchcrepe's weights, or download:

```sh
uv run Tools/export_crepe.py --capacity full --install     # and --capacity tiny
hf download arraypress/tune-crepe --local-dir models
```

The command-line tool is [`tune`](https://github.com/arraypress/swift-tune-cli).

## Requirements

- macOS 27+ (Core AI), Apple silicon
- Swift 6

## License

MIT — see [LICENSE](LICENSE). CREPE and its weights are MIT (Kim et al.); the PyTorch weights
are torchcrepe's (Max Morrison, MIT).
