import Foundation

public enum UltraGrothError: Error {
    case ProofGenerationError(String)
    case ProofGenerationBuffersTooShort
}
