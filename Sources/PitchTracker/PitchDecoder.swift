//
//  PitchDecoder.swift
//  PitchTracker
//
//  Created by David Sherlock on 2026.
//
//  From 360 activations per frame to cents, as marl/crepe decodes them.
//  `localAverage` is the default: the activation-weighted mean of the cents
//  of the nine bins around the argmax. `viterbi` first smooths the argmax
//  sequence with a 360-state HMM — a triangular transition of width 12 that
//  rewards continuity, self-emission 0.1 — and then takes the local average
//  around each path bin. Confidence is the largest activation. Frequency is
//  10·2^(cents/1200), so bin 0 is 1997.38 cents = 31.70 Hz (53 cents under C1) and bin 359
//  is 9177.38 cents = 1975.5 Hz (a little under B6).
//

import Foundation

public enum PitchDecoder {

    public static let bins = 360
    public static let centsPerBin = 20.0
    /// The cents of bin 0: `1200·log2(32.70/10)`.
    public static let firstBinCents = 1997.3794084376191
    /// crepe's `cents_mapping`: `linspace(0, 7180, 360) + 1997.379…`.
    public static let binCents: [Double] = (0..<bins).map { firstBinCents + 7180.0 * Double($0) / Double(bins - 1) }

    public enum Method: String, Sendable, CaseIterable { case localAverage, viterbi }

    public static func frequency(cents: Double) -> Double { 10 * pow(2, cents / 1200) }
    public static func cents(frequency: Double) -> Double { 1200 * log2(frequency / 10) }

    /// crepe's `to_local_average_cents` for one frame's activations, around `center` (the argmax unless given).
    public static func localAverageCents(_ activation: ArraySlice<Float>, center: Int? = nil) -> Double {
        let base = activation.startIndex
        var argmax = 0
        var best = -Float.infinity
        for i in 0..<bins where activation[base + i] > best { best = activation[base + i]; argmax = i }
        let c = center ?? argmax
        let start = max(0, c - 4), end = min(bins, c + 5)
        var product = 0.0, weight = 0.0
        for i in start..<end {
            let s = Double(activation[base + i])
            product += s * binCents[i]; weight += s
        }
        return product / weight
    }

    /// Cents and confidence per frame for `[frames][360]` activations.
    public static func decode(_ activations: [Float], method: Method) -> (cents: [Double], confidence: [Float]) {
        let n = activations.count / bins
        var confidence = [Float](repeating: 0, count: n)
        var argmax = [Int](repeating: 0, count: n)
        for f in 0..<n {
            var best = -Float.infinity, bi = 0
            for i in 0..<bins where activations[f * bins + i] > best { best = activations[f * bins + i]; bi = i }
            confidence[f] = best; argmax[f] = bi
        }
        let centers: [Int?]
        switch method {
        case .localAverage: centers = [Int?](repeating: nil, count: n)
        case .viterbi: centers = viterbiPath(observations: argmax).map { Optional($0) }
        }
        let cents = (0..<n).map { f in localAverageCents(activations[(f * bins)..<((f + 1) * bins)], center: centers[f]) }
        return (cents, confidence)
    }

    /// crepe's `to_viterbi_cents` HMM: uniform start, transition ∝ max(12 − |i − j|, 0) row-normalised,
    /// emission 0.1 + 0.9/360 for the observed bin and 0.9/360 for every other — decoded exactly as
    /// hmmlearn's C++ `viterbi` does it, because ties are common in unvoiced stretches and the tie rule
    /// decides the path: the lattice is `max_j(lattice[t−1][j] + logT[j][i]) + logE[t][i]`; the final
    /// state is the FIRST maximum of the last row (`std::max_element`); every earlier state is recomputed
    /// as `argmax_i(lattice[t][i] + logT[i][next])` taking the LAST maximum on a tie (`std::max` over
    /// (value, index) pairs). Asserted equal to hmmlearn in the export on real and random sequences.
    public static func viterbiPath(observations: [Int]) -> [Int] {
        let n = observations.count
        guard n > 0 else { return [] }
        var logT = [Double](repeating: -.infinity, count: bins * bins)
        for i in 0..<bins {
            var row = 0.0
            for j in 0..<bins { row += Double(max(12 - abs(i - j), 0)) }
            for j in max(0, i - 11)...min(bins - 1, i + 11) { logT[i * bins + j] = log(Double(12 - abs(i - j)) / row) }
        }
        let selfEmission = 0.1
        let logSelf = log(selfEmission + (1 - selfEmission) / Double(bins)), logOther = log((1 - selfEmission) / Double(bins))
        let logStart = log(1 / Double(bins))
        var lattice = [Double](repeating: 0, count: n * bins)
        for i in 0..<bins { lattice[i] = logStart + (i == observations[0] ? logSelf : logOther) }
        for t in 1..<n {
            let prev = (t - 1) * bins
            for i in 0..<bins {
                var best = -Double.infinity
                for j in max(0, i - 11)...min(bins - 1, i + 11) { best = max(best, lattice[prev + j] + logT[j * bins + i]) }
                lattice[t * bins + i] = best + (i == observations[t] ? logSelf : logOther)
            }
        }
        var path = [Int](repeating: 0, count: n)
        var next = 0
        var best = -Double.infinity
        for i in 0..<bins where lattice[(n - 1) * bins + i] > best { best = lattice[(n - 1) * bins + i]; next = i }   // first maximum
        path[n - 1] = next
        for t in stride(from: n - 2, through: 0, by: -1) {
            var bestValue = -Double.infinity, bestIndex = 0
            for i in 0..<bins {
                let v = lattice[t * bins + i] + logT[i * bins + next]
                if v >= bestValue { bestValue = v; bestIndex = i }      // last maximum on a tie
            }
            next = bestIndex
            path[t] = next
        }
        return path
    }
}
