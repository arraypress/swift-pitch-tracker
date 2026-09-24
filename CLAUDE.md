# CLAUDE.md — swift-pitch-tracker

CREPE on Core AI: `CrepeTracker` (the network, fixed batch of 256 frames),
`CrepeFrontEnd` (16 kHz framing + normalisation, file loading),
`PitchDecoder` (local average, Viterbi), `PitchTrack`. Module `PitchTracker`;
the CLI is `../swift-tune-cli` (binary `tune`). Added 2026-09-24.

## Build & test
```bash
swift build && swift test        # framing + decoders need no model
```
Model-backed tests run when `~/Library/Application Support/tune/` holds
`tune-crepe-full-float32.aimodel` (and `-tiny-`): `uv run Tools/export_crepe.py
--capacity full --install name=clip.wav …` writes the asset and the fixtures
(probe venv: torchcrepe, hmmlearn, resampy). A Keras cross-check venv is at
`scratchpad/crepe/.tfvenv` (tensorflow 2.21 + the original `crepe` package,
installed with `setuptools<80` and `--no-build-isolation`).

## Traps (all measured)
- **Two "originals".** torchcrepe (the PyTorch weights) DITHERS its cents
  with triangular noise (`convert.dither`) and applies sigmoid twice in
  `weighted_argmax`; its Viterbi uses softmax emissions. The port follows
  marl/crepe's `core.py` (deterministic, the paper's) for everything
  outside the network — and the two frame normalisations differ (torch's
  unbiased std, floor 1e-10 vs numpy's biased std, floor 1e-8): 2e-3 on
  normal frames, 16 on near-silent ones. The network itself is identical to
  Keras (145–151 dB).
- **hmmlearn's tie rule.** The HMM has many equal-probability paths through
  unvoiced stretches. hmmlearn's C++ backtracks by recomputation and takes
  `std::max` over (value, index) pairs → the LAST index on a tie; the final
  state uses `std::max_element` → the FIRST. Stored back-pointers with
  first-max gave 21 different frames on one clip at identical likelihood.
- Bin 0 is 31.70 Hz (1997.38 cents above 10 Hz), 53 cents under C1 — not
  32.70.
- `InferenceFunction` exposes no input shapes (the swiftinterface is a
  stub), so the batch is an init parameter that must match the export.
- Same `AVAudioFile` read loop as SampleSearch (short reads, eofErr on
  compressed files); a 16 kHz file never touches AVAudioConverter.
