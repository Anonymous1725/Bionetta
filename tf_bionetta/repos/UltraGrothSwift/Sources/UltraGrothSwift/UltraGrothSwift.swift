import Foundation

import UltraGrothLib

private let ERROR_SIZE = UInt64(256)
private let PROOF_SIZE = UInt64(4 * 1024 * 1024)
private let PUB_SIGNALS_SIZE = UInt64(4 * 1024 * 1024)

/// A class that provides methods for generating zk-SNARK proofs using the Groth16 protocol.
public class Groth16 {
    /// Generates a zk-SNARK proof using the Groth16 protocol.
    ///
    /// - Parameters:
    ///   - zkey: The zero-knowledge key data.
    ///   - wtns: The witness data.
    ///
    /// - Returns: A tuple containing the proof data and public signals data encoded in JSON.
    public static func groth16Prover(_ zkey: Data, _ wtns: Data) throws -> (proof: Data, pubSignals: Data) {
        var proofSize = PROOF_SIZE
        var pubSignalsSize = PUB_SIGNALS_SIZE
        
        let proofBuffer = UnsafeMutablePointer<UInt8>.allocate(capacity: Int(PROOF_SIZE))
        let pubSignalsBuffer = UnsafeMutablePointer<UInt8>.allocate(capacity: Int(PUB_SIGNALS_SIZE))
        
        let errorBuffer = UnsafeMutablePointer<UInt8>.allocate(capacity: Int(ERROR_SIZE))
        
        let result = groth16_prover(
            (zkey as NSData).bytes, UInt64(zkey.count),
            (wtns as NSData).bytes, UInt64(wtns.count),
            proofBuffer, &proofSize,
            pubSignalsBuffer, &pubSignalsSize,
            errorBuffer, ERROR_SIZE
        )
        
        try handleGroth16ProverError(result, errorBuffer)
        
        var proof = Data(bytes: proofBuffer, count: Int(proofSize))
        var pubSignals = Data(bytes: pubSignalsBuffer, count: Int(pubSignalsSize))
        
        let proofNullIndex = proof.firstIndex(of: 0x00)!
        let pubSignalsNullIndex = pubSignals.firstIndex(of: 0x00)!
        
        proof = proof[0..<proofNullIndex]
        pubSignals = pubSignals[0..<pubSignalsNullIndex]
        
        return (proof: proof, pubSignals: pubSignals)
    }
}

/// A class that provides methods for generating zk-SNARK proofs using the UltraGroth protocol.
public class UltraGroth {
    /// Generates a zk-SNARK proof using the UltraGroth protocol.
    ///
    /// - Parameters:
    ///   - zkey: The zero-knowledge key data.
    ///   - wtns: The witness data.
    ///
    /// - Returns: A tuple containing the proof data and public signals data encoded in JSON.
    public static func ultraGrothProver(_ zkey: Data, _ wtns: Data) throws -> (proof: Data, pubSignals: Data) {
        var proofSize = PROOF_SIZE
        var pubSignalsSize = PUB_SIGNALS_SIZE
        
        let proofBuffer = UnsafeMutablePointer<UInt8>.allocate(capacity: Int(PROOF_SIZE))
        let pubSignalsBuffer = UnsafeMutablePointer<UInt8>.allocate(capacity: Int(PUB_SIGNALS_SIZE))
        
        let errorBuffer = UnsafeMutablePointer<UInt8>.allocate(capacity: Int(ERROR_SIZE))
        
        let result = ultra_groth_prover(
            (zkey as NSData).bytes, UInt64(zkey.count),
            (wtns as NSData).bytes, UInt64(wtns.count),
            proofBuffer, &proofSize,
            pubSignalsBuffer, &pubSignalsSize,
            errorBuffer, ERROR_SIZE
        )
        
        try handleGroth16ProverError(result, errorBuffer)
        
        var proof = Data(bytes: proofBuffer, count: Int(proofSize))
        var pubSignals = Data(bytes: pubSignalsBuffer, count: Int(pubSignalsSize))
        
        let proofNullIndex = proof.firstIndex(of: 0x00)!
        let pubSignalsNullIndex = pubSignals.firstIndex(of: 0x00)!
        
        proof = proof[0..<proofNullIndex]
        pubSignals = pubSignals[0..<pubSignalsNullIndex]
        
        return (proof: proof, pubSignals: pubSignals)
    }
}

private func handleGroth16ProverError(
    _ result: Int32,
    _ errorBuffer: UnsafeMutablePointer<UInt8>
) throws {
    if result == PROVER_ERROR {
        let errorMsg = String(bytes: Data(bytes: errorBuffer, count: Int(ERROR_SIZE)), encoding: .utf8)!
            .replacingOccurrences(of: "\0", with: "")
        
        throw UltraGrothError.ProofGenerationError(errorMsg)
    }
    
    if result == PROVER_ERROR_SHORT_BUFFER {
        throw UltraGrothError.ProofGenerationBuffersTooShort
    }
}
