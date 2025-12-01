# UltraGrothSwift

A Swift package providing easy-to-use interfaces for Groth16 and UltraGroth zero-knowledge proof generation.

## Overview

UltraGrothSwift is a Swift wrapper around the UltraGroth library, offering both standard Groth16 and optimized UltraGroth proof generation capabilities. This package enables iOS applications to generate zero-knowledge proofs efficiently using precompiled witness and zkey files.

## Features

- **Groth16 Proof Generation**: Standard Groth16 zkSNARK proof generation
- **UltraGroth Proof Generation**: Optimized proof generation with UltraGroth

## Installation

### Swift Package Manager

Add UltraGrothSwift to your project by adding the following dependency to your `Package.swift`:

```swift
dependencies: [
    .package(url: "https://github.com/rarimo/UltraGrothSwift.git", from: "1.0.0")
]
```

Or add it through Xcode:

1. File → Add Package Dependencies
2. Enter the repository URL: `https://github.com/rarimo/UltraGrothSwift.git`
3. Select the version range and add to your target

## Usage

### Importing the Library

```swift
import UltraGrothSwift
```

### Groth16 Proof Generation

Generate a standard Groth16 proof using a zkey file and witness data:

```swift
do {
    let zkeyData = // Load your zkey file data
    let witnessData = // Load your witness data
    
    let result = try Groth16.groth16Prover(zkeyData, witnessData)
    
    let proof = result.proof        // Generated proof data
    let pubSignals = result.pubSignals // Public signals data
    
    print("Proof generated successfully")
    print("Proof size: \(proof.count) bytes")
    print("Public signals size: \(pubSignals.count) bytes")
    
} catch UltraGrothError.ProofGenerationError(let message) {
    print("Proof generation failed: \(message)")
} catch UltraGrothError.ProofGenerationBuffersTooShort {
    print("Buffer size insufficient for proof generation")
} catch {
    print("Unexpected error: \(error)")
}
```

### UltraGroth Proof Generation

Generate an optimized UltraGroth proof:

```swift
do {
    let zkeyData = // Load your zkey file data
    let witnessData = // Load your witness data
    
    let result = try UltraGroth.ultraGrothProver(zkeyData, witnessData)
    
    let proof = result.proof        // Generated proof data
    let pubSignals = result.pubSignals // Public signals data
    
    print("UltraGroth proof generated successfully")
    
} catch UltraGrothError.ProofGenerationError(let message) {
    print("UltraGroth proof generation failed: \(message)")
} catch UltraGrothError.ProofGenerationBuffersTooShort {
    print("Buffer size insufficient for proof generation")
} catch {
    print("Unexpected error: \(error)")
}
```

## API Reference

### `Groth16` Class

The main class providing proof Groth16 generation functionality.

#### Groth16 Static Methods

##### `groth16Prover(_:_:)`

```swift
public static func groth16Prover(_ zkey: Data, _ wtns: Data) throws -> (proof: Data, pubSignals: Data)
```

Generates a standard Groth16 proof.

**Parameters:**

- `zkey`: The circuit's zkey file data
- `wtns`: The witness data

**Returns:**

- A tuple containing the generated proof and public signals as raw JSON data

**Throws:**

- `UltraGrothError.ProofGenerationError`: When proof generation fails
- `UltraGrothError.ProofGenerationBuffersTooShort`: When internal buffers are insufficient

### `UltaGroth` Class

The main class providing proof generation functionality.

#### UltaGroth Static Methods

##### `ultraGrothProver(_:_:)`

```swift
public static func ultraGrothProver(_ zkey: Data, _ wtns: Data) throws -> (proof: Data, pubSignals: Data)
```

Generates an optimized UltraGroth proof.

**Parameters:**

- `zkey`: The circuit's zkey file data
- `wtns`: The witness data

**Returns:**

- A tuple containing the generated proof and public signals as raw JSON data

**Throws:**

- `UltraGrothError.ProofGenerationError`: When proof generation fails
- `UltraGrothError.ProofGenerationBuffersTooShort`: When internal buffers are insufficient
  
## Requirements

- iOS 16.0+
- Swift 6.0+
- Xcode 15.0+

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
