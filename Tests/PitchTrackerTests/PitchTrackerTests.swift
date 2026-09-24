//
//  PitchTrackerTests.swift
//  PitchTrackerTests
//
//  Created by David Sherlock on 2026.
//
//  The CREPE port against the original crepe pipeline: framing and both
//  decoders on fixtures the export wrote (no model needed), then — with the
//  asset installed under ~/Library/Application Support/tune — the network's
//  activations and whole clips end to end, and synthetic tones of known
//  pitch as ground truth.
//

import Foundation
import Testing
@testable import PitchTracker

private func fixtureURL(_ name: String, _ ext: String) throws -> URL {
    try #require(Bundle.module.url(forResource: name, withExtension: ext, subdirectory: "Fixtures"))
}
private func floats(_ name: String) throws -> [Float] {
    try Data(contentsOf: fixtureURL(name, "f32")).withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
}
private func doubles(_ name: String) throws -> [Double] {
    try Data(contentsOf: fixtureURL(name, "f64")).withUnsafeBytes { Array($0.bindMemory(to: Double.self)) }
}
private func psnr(_ reference: [Float], _ got: [Float]) -> Double {
    precondition(reference.count == got.count)
    var err = 0.0, lo = Double.infinity, hi = -Double.infinity
    for i in reference.indices {
        let d = Double(reference[i]) - Double(got[i]); err += d * d
        lo = min(lo, Double(reference[i])); hi = max(hi, Double(reference[i]))
    }
    let mse = err / Double(reference.count)
    return mse == 0 ? .infinity : 20 * log10((hi - lo) / mse.squareRoot())
}
private let installed = URL(fileURLWithPath: NSHomeDirectory()).appendingPathComponent("Library/Application Support/tune")
private func assetURL(_ c: CrepeTracker.Capacity) -> URL { installed.appendingPathComponent(CrepeTracker.assetName(c)) }
private var fullInstalled: Bool { FileManager.default.fileExists(atPath: assetURL(.full).path) }
private var tinyInstalled: Bool { FileManager.default.fileExists(atPath: assetURL(.tiny).path) }

@Suite struct FrontEndTests {

    @Test("frames match crepe's get_activation framing and normalisation", arguments: ["steady", "vibrato", "glide"])
    func frames(clip: String) throws {
        let samples = try CrepeFrontEnd.load(fixtureURL("\(clip)_16k", "wav"))
        let reference = try floats("\(clip)_frames")
        let frames = CrepeFrontEnd.frames(samples)
        #expect(frames.count == reference.count, "\(frames.count / 1024) frames vs \(reference.count / 1024)")
        let db = psnr(reference, frames)
        print("frames \(clip): \(String(format: "%.1f", db)) dB, max |diff| \(zip(reference, frames).map { abs($0 - $1) }.max()!)")
        #expect(db > 120)
    }

    @Test("frame count is 1 + samples / 160")
    func count() {
        #expect(CrepeFrontEnd.frameCount(samples: 32_000) == 201)
        #expect(CrepeFrontEnd.frames([Float](repeating: 0, count: 1600)).count == 11 * 1024)
        // silence normalises to zeros, not NaN
        #expect(CrepeFrontEnd.frames([Float](repeating: 0, count: 1600)).allSatisfy { $0 == 0 })
    }
}

@Suite struct DecoderTests {

    @Test("bin cents are crepe's cents_mapping")
    func mapping() {
        #expect(abs(PitchDecoder.binCents[0] - 1997.3794084376191) < 1e-9)
        #expect(abs(PitchDecoder.binCents[359] - (7180 + 1997.3794084376191)) < 1e-9)
        #expect(abs(PitchDecoder.binCents[1] - PitchDecoder.binCents[0] - 20) < 1e-9)
        #expect(abs(PitchDecoder.frequency(cents: PitchDecoder.binCents[0]) - 31.700) < 0.001)   // 10·2^(1997.3794/1200): crepe starts 53 cents under C1
    }

    @Test("local average and viterbi reproduce crepe on fixture activations", arguments: ["steady", "vibrato", "glide", "steady441", "bass", "vocal", "lead"])
    func decoders(clip: String) throws {
        let activations = try floats("\(clip)_full_activations")
        let local = PitchDecoder.decode(activations, method: .localAverage)
        let viterbi = PitchDecoder.decode(activations, method: .viterbi)
        let refLocal = try doubles("\(clip)_full_local"), refViterbi = try doubles("\(clip)_full_viterbi"), refConf = try floats("\(clip)_full_confidence")
        #expect(local.cents.count == refLocal.count)
        let dl = zip(local.cents, refLocal).map { abs($0 - $1) }.max()!
        let dv = zip(viterbi.cents, refViterbi).map { abs($0 - $1) }.max()!
        let dc = zip(local.confidence, refConf).map { abs($0 - $1) }.max()!
        print("decode \(clip): local max |Δ| \(dl) cents, viterbi max |Δ| \(dv) cents, confidence max |Δ| \(dc)")
        #expect(dl < 1e-3 && dv < 1e-3 && dc == 0)
        #expect(viterbi.confidence == local.confidence)
    }

    @Test("viterbi keeps a jump-free path through an octave error")
    func viterbiSmooths() {
        var obs = [Int](repeating: 100, count: 30)
        obs[15] = 160   // one frame an octave up
        let path = PitchDecoder.viterbiPath(observations: obs)
        #expect(path.allSatisfy { $0 == 100 })
        #expect(PitchDecoder.viterbiPath(observations: []) == [])
    }
}

@Suite(.serialized) struct ModelTests {

    @Test("activations match torchcrepe's full model", .enabled(if: fullInstalled), arguments: ["steady", "vibrato", "glide", "vocal"])
    func activations(clip: String) async throws {
        let tracker = try await CrepeTracker(contentsOf: assetURL(.full))
        let samples = try CrepeFrontEnd.load(fixtureURL("\(clip)_16k", "wav"))
        let got = try await tracker.activations(frames: CrepeFrontEnd.frames(samples))
        let reference = try floats("\(clip)_full_activations")
        #expect(got.count == reference.count)
        let db = psnr(reference, got)
        print("activations \(clip): \(String(format: "%.1f", db)) dB over \(got.count / 360) frames")
        #expect(db > 80)
    }

    @Test("whole clips decode to crepe's cents", .enabled(if: fullInstalled), arguments: ["steady", "vibrato", "glide", "steady441", "bass", "vocal", "lead"])
    func endToEnd(clip: String) async throws {
        let tracker = try await CrepeTracker(contentsOf: assetURL(.full))
        let samples = try CrepeFrontEnd.load(fixtureURL("\(clip)_16k", "wav"))
        for method in PitchDecoder.Method.allCases {
            let track = try await tracker.track(samples16k: samples, method: method)
            let ref = try doubles("\(clip)_full_\(method == .localAverage ? "local" : "viterbi")")
            let refConf = try floats("\(clip)_full_confidence")
            let voiced = zip(track.frames, refConf).filter { $0.1 > 0.5 }
            let worst = voiced.map { abs($0.0.cents - ref[track.frames.firstIndex { $0.time == $0.time }!]) }
            _ = worst
            let diffs = track.frames.indices.filter { refConf[$0] > 0.5 }.map { abs(track.frames[$0].cents - ref[$0]) }
            let confDiff = track.frames.indices.map { abs(track.frames[$0].confidence - Double(refConf[$0])) }.max()!
            print("end-to-end \(clip) \(method): max |Δ| \(String(format: "%.3f", diffs.max() ?? 0)) cents over \(diffs.count) voiced frames; confidence max |Δ| \(String(format: "%.2e", confDiff))")
            #expect((diffs.max() ?? 0) < 1, "\(clip) \(method)")
        }
    }

    @Test("tiny activations match torchcrepe's tiny model", .enabled(if: tinyInstalled), arguments: ["steady", "vocal"])
    func tiny(clip: String) async throws {
        let tracker = try await CrepeTracker(contentsOf: assetURL(.tiny), capacity: .tiny)
        let samples = try CrepeFrontEnd.load(fixtureURL("\(clip)_16k", "wav"))
        let got = try await tracker.activations(frames: CrepeFrontEnd.frames(samples))
        let reference = try floats("\(clip)_tiny_activations")
        let db = psnr(reference, got)
        print("tiny activations \(clip): \(String(format: "%.1f", db)) dB")
        #expect(db > 80)
    }

    @Test("synthetic tones of known pitch", .enabled(if: fullInstalled))
    func groundTruth() async throws {
        let tracker = try await CrepeTracker(contentsOf: assetURL(.full))
        // A3 + 13 cents = 221.66 Hz, steady for two seconds
        let steady = try await tracker.track(contentsOf: fixtureURL("steady_16k", "wav"))
        let mid = steady.frames.filter { $0.time > 0.2 && $0.time < 1.8 }
        let medianMidi = mid.map(\.midi).sorted()[mid.count / 2]
        let (note, cents) = PitchTrack.nearest(midi: medianMidi)
        print("steady: \(PitchTrack.name(midi: note)) \(String(format: "%+.1f", cents)) cents, confidence \(String(format: "%.2f", mid.map(\.confidence).reduce(0, +) / Double(mid.count)))")
        #expect(note == 57 && abs(cents - 13) < 5)
        // C4 with ±40-cent vibrato at 6 Hz: mean on C4, swing about 80 cents peak to peak
        let vibrato = try await tracker.track(contentsOf: fixtureURL("vibrato_16k", "wav"))
        let v = vibrato.frames.filter { $0.time > 0.2 && $0.time < 1.8 }.map(\.midi)
        let mean = v.reduce(0, +) / Double(v.count)
        print("vibrato: mean \(String(format: "%.3f", mean)) (C4 = 60), swing \(String(format: "%.0f", (v.max()! - v.min()!) * 100)) cents")
        #expect(abs(mean - 60) < 0.05 && (v.max()! - v.min()!) * 100 > 60)
        // E2 → E3 over two seconds: one octave, monotonic in the middle
        let glide = try await tracker.track(contentsOf: fixtureURL("glide_16k", "wav"))
        let g = glide.frames.filter { $0.time > 0.2 && $0.time < 1.8 }
        print("glide: \(String(format: "%.2f → %.2f", g.first!.midi, g.last!.midi)) over \(g.count) frames")
        #expect(abs(g.first!.midi - (40 + 0.2 * 12 / 2)) < 0.3 && abs(g.last!.midi - (40 + 1.8 * 12 / 2)) < 0.3)
    }
}
