//
//  PitchTrack.swift
//  PitchTracker
//
//  Created by David Sherlock on 2026.
//
//  What a tracked recording is: a frame every 10 ms with its pitch in cents
//  above 10 Hz, in hertz, as a MIDI note number with fraction, and the
//  network's confidence that the frame is pitched at all.
//

import Foundation

public struct PitchTrack: Sendable {

    public struct Frame: Sendable {
        /// Seconds from the start; frame k is centred on k × 10 ms.
        public let time: Double
        /// Cents above 10 Hz, crepe's unit.
        public let cents: Double
        public let frequency: Double
        /// Fractional MIDI note number at A4 = 440: 69 + 12·log2(f/440).
        public let midi: Double
        /// The largest activation, 0…1.
        public let confidence: Double

        public init(time: Double, cents: Double, frequency: Double, midi: Double, confidence: Double) {
            self.time = time; self.cents = cents; self.frequency = frequency; self.midi = midi; self.confidence = confidence
        }
    }

    public let frames: [Frame]
    public let method: PitchDecoder.Method
    /// The 16 kHz sample count the model saw.
    public let samples: Int

    public init(frames: [Frame], method: PitchDecoder.Method, samples: Int) {
        self.frames = frames; self.method = method; self.samples = samples
    }

    public var seconds: Double { Double(samples) / Double(CrepeFrontEnd.sampleRate) }

    /// Frames whose confidence reaches `threshold` (crepe's convention is 0.5 for voicing).
    public func voiced(threshold: Double = 0.5) -> [Frame] { frames.filter { $0.confidence >= threshold } }

    public static func midi(frequency: Double) -> Double { 69 + 12 * log2(frequency / 440) }
    public static let noteNames = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

    /// "A3" for 57.
    public static func name(midi number: Int) -> String { "\(noteNames[((number % 12) + 12) % 12])\(number / 12 - 1)" }

    /// The nearest note and the signed distance to it in cents, for a fractional note number.
    public static func nearest(midi: Double) -> (note: Int, cents: Double) {
        let note = Int(midi.rounded())
        return (note, (midi - Double(note)) * 100)
    }
}
