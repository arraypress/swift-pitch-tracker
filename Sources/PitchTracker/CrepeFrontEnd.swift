//
//  CrepeFrontEnd.swift
//  PitchTracker
//
//  Created by David Sherlock on 2026.
//
//  CREPE's input as marl/crepe's `get_activation` builds it: 16 kHz mono,
//  zero-padded 512 samples each side so a frame is centred on its timestamp,
//  1024-sample frames every 160 samples (10 ms), each frame normalised to
//  zero mean and unit (biased) standard deviation, the deviation clipped at
//  1e-8 so silence stays finite.
//

import AVFoundation
import Foundation

public enum CrepeFrontEnd {

    public static let sampleRate = 16_000
    public static let window = 1024
    /// 10 ms.
    public static let hop = 160
    static let floor: Float = 1e-8

    /// How many frames `count` samples give, padded.
    public static func frameCount(samples count: Int) -> Int { 1 + (count + window - window) / hop }

    /// `[frames][1024]` normalised frames for 16 kHz mono samples.
    public static func frames(_ samples: [Float]) -> [Float] {
        let half = window / 2
        var padded = [Float](repeating: 0, count: samples.count + window)
        padded.replaceSubrange(half..<(half + samples.count), with: samples)
        let n = frameCount(samples: samples.count)
        var out = [Float](repeating: 0, count: n * window)
        var frame = [Float](repeating: 0, count: window)
        for f in 0..<n {
            let start = f * hop
            var mean: Float = 0
            for i in 0..<window { frame[i] = padded[start + i]; mean += frame[i] }
            mean /= Float(window)
            var variance: Float = 0
            for i in 0..<window { frame[i] -= mean; variance += frame[i] * frame[i] }
            let std = max(floor, (variance / Float(window)).squareRoot())
            for i in 0..<window { out[f * window + i] = frame[i] / std }
        }
        return out
    }

    /// Any audio file AVFoundation decodes: mixed to mono by the mean of its channels, then
    /// resampled to 16 kHz with AVAudioConverter at its best quality when it is not 16 kHz already.
    public static func load(_ url: URL) throws -> [Float] {
        let file: AVAudioFile
        do { file = try AVAudioFile(forReading: url) } catch { throw PitchTrackerError.audioUnreadable("\(url.lastPathComponent): \(error.localizedDescription)") }
        let format = file.processingFormat
        let channels = Int(format.channelCount)
        guard file.length > 0, let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: 1 << 16) else { throw PitchTrackerError.emptyAudio }
        var mono = [Float]()
        mono.reserveCapacity(Int(file.length))
        let scale = 1 / Float(channels)
        while file.framePosition < file.length {
            do { try file.read(into: buffer, frameCount: buffer.frameCapacity) } catch {
                if mono.isEmpty { throw PitchTrackerError.audioUnreadable("\(url.lastPathComponent): \(error.localizedDescription)") }
                break   // a compressed file's length is an estimate; end of stream after samples is the end
            }
            let frames = Int(buffer.frameLength)
            guard frames > 0, let data = buffer.floatChannelData else { break }
            for i in 0..<frames {
                var v = data[0][i]
                if channels > 1 { for c in 1..<channels { v += data[c][i] }; v *= scale }
                mono.append(v)
            }
        }
        guard !mono.isEmpty else { throw PitchTrackerError.emptyAudio }
        return format.sampleRate == Double(sampleRate) ? mono : resample(mono, from: format.sampleRate, to: Double(sampleRate))
    }

    static func resample(_ input: [Float], from source: Double, to destination: Double) -> [Float] {
        guard source != destination, !input.isEmpty,
              let inFormat = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: source, channels: 1, interleaved: false),
              let outFormat = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: destination, channels: 1, interleaved: false),
              let converter = AVAudioConverter(from: inFormat, to: outFormat),
              let inBuffer = AVAudioPCMBuffer(pcmFormat: inFormat, frameCapacity: AVAudioFrameCount(input.count))
        else { return input }
        converter.sampleRateConverterQuality = AVAudioQuality.max.rawValue
        inBuffer.frameLength = AVAudioFrameCount(input.count)
        input.withUnsafeBufferPointer { inBuffer.floatChannelData![0].update(from: $0.baseAddress!, count: input.count) }
        let capacity = AVAudioFrameCount(Double(input.count) * destination / source) + 16
        guard let outBuffer = AVAudioPCMBuffer(pcmFormat: outFormat, frameCapacity: capacity) else { return input }
        nonisolated(unsafe) var delivered = false
        let status = converter.convert(to: outBuffer, error: nil) { _, outStatus in
            if delivered { outStatus.pointee = .endOfStream; return nil }
            delivered = true; outStatus.pointee = .haveData; return inBuffer
        }
        guard status != .error, let data = outBuffer.floatChannelData else { return input }
        return Array(UnsafeBufferPointer(start: data[0], count: Int(outBuffer.frameLength)))
    }
}

public enum PitchTrackerError: Error, CustomStringConvertible {
    case modelNotFound(String), modelUnavailable(String), audioUnreadable(String), emptyAudio
    public var description: String {
        switch self {
        case .modelNotFound(let p): return "no model at \(p)"
        case .modelUnavailable(let m): return m
        case .audioUnreadable(let m): return m
        case .emptyAudio: return "the audio is empty"
        }
    }
}
