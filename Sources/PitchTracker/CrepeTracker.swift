//
//  CrepeTracker.swift
//  PitchTracker
//
//  Created by David Sherlock on 2026.
//
//  CREPE (Kim, Salamon, Li, Bello — ISMIR 2018, MIT) on Core AI: the
//  network in the .aimodel at a fixed batch of frames, everything around it
//  the original crepe pipeline in Swift. `full` is the paper's model;
//  `tiny` is the small one for speed.
//

import CoreAI
import Foundation

public final class CrepeTracker: @unchecked Sendable {

    public enum Capacity: String, Sendable, CaseIterable { case full, tiny }

    public static func assetName(_ capacity: Capacity) -> String { "tune-crepe-\(capacity.rawValue)-float32.aimodel" }

    public let url: URL
    public let capacity: Capacity
    /// Frames per model run — the export's `--batch` (256 by default); the last batch is zero-padded.
    public let batch: Int
    private let function: InferenceFunction

    public init(contentsOf url: URL, capacity: Capacity = .full, batch: Int = 256) async throws {
        guard FileManager.default.fileExists(atPath: url.path) else { throw PitchTrackerError.modelNotFound(url.path) }
        self.url = url
        self.capacity = capacity
        self.batch = batch
        var options = SpecializationOptions(preferredComputeUnitKind: .gpu)
        options.expectFrequentReshapes = true
        let model: AIModel
        do { model = try await AIModel(contentsOf: url, options: options) } catch {
            throw PitchTrackerError.modelUnavailable("could not load \(url.lastPathComponent): \(error)")
        }
        guard let function = try model.loadFunction(named: "main") else {
            throw PitchTrackerError.modelUnavailable("no main function in \(url.lastPathComponent)")
        }
        self.function = function
    }

    /// Activations `[frames][360]` for normalised frames `[frames][1024]`, the last batch zero-padded.
    public func activations(frames: [Float]) async throws -> [Float] {
        let window = CrepeFrontEnd.window, bins = PitchDecoder.bins
        let n = frames.count / window
        var out = [Float]()
        out.reserveCapacity(n * bins)
        var start = 0
        while start < n {
            let count = min(batch, n - start)
            var input = Array(frames[(start * window)..<((start + count) * window)])
            if count < batch { input.append(contentsOf: [Float](repeating: 0, count: (batch - count) * window)) }
            var result = NDArray(shape: [batch, bins], scalarType: .float32)
            var views = InferenceFunction.MutableViews()
            views.insert(result.mutableRawView(), for: "activations")
            do {
                _ = try await function.run(inputs: ["frames": Self.array(input, shape: [batch, window])],
                                           states: InferenceFunction.MutableViews(), outputViews: consume views)
            } catch { throw PitchTrackerError.modelUnavailable("inference failed: \(error)") }
            out.append(contentsOf: Self.floats(result).prefix(count * bins))
            start += count
            try Task.checkCancellation()
        }
        return out
    }

    /// The pitch track of 16 kHz mono samples.
    public func track(samples16k: [Float], method: PitchDecoder.Method = .localAverage) async throws -> PitchTrack {
        guard !samples16k.isEmpty else { throw PitchTrackerError.emptyAudio }
        let activations = try await activations(frames: CrepeFrontEnd.frames(samples16k))
        let (cents, confidence) = PitchDecoder.decode(activations, method: method)
        let frames = cents.indices.map { f -> PitchTrack.Frame in
            let hz = PitchDecoder.frequency(cents: cents[f])
            return .init(time: Double(f) * Double(CrepeFrontEnd.hop) / Double(CrepeFrontEnd.sampleRate), cents: cents[f],
                         frequency: hz, midi: PitchTrack.midi(frequency: hz), confidence: Double(confidence[f]))
        }
        return PitchTrack(frames: frames, method: method, samples: samples16k.count)
    }

    /// Any audio file: decoded, mixed to mono, resampled to 16 kHz, tracked.
    public func track(contentsOf url: URL, method: PitchDecoder.Method = .localAverage) async throws -> PitchTrack {
        try await track(samples16k: CrepeFrontEnd.load(url), method: method)
    }

    static func array(_ values: [Float], shape: [Int]) -> NDArray {
        var a = NDArray(shape: shape, scalarType: .float32)
        let view = a.mutableView(as: Float.self)
        view.withUnsafeMutablePointer { p, _, _ in values.withUnsafeBufferPointer { p.update(from: $0.baseAddress!, count: values.count) } }
        return a
    }

    static func floats(_ array: NDArray) -> [Float] {
        let count = array.shape.reduce(1, *)
        var out = [Float](repeating: 0, count: count)
        array.view(as: Float.self).withUnsafePointer { p, _, _ in out.withUnsafeMutableBufferPointer { $0.baseAddress!.update(from: p, count: count) } }
        return out
    }
}
