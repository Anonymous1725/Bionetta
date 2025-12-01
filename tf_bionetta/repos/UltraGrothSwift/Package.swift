// swift-tools-version: 6.0
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "UltraGrothSwift",
    products: [
        // Products define the executables and libraries a package produces, making them visible to other packages.
        .library(
            name: "UltraGrothSwift",
            targets: ["UltraGrothSwift", "UltraGrothFramework"]
        ),
    ],
    targets: [
        // Targets are the basic building blocks of a package, defining a module or a test suite.
        // Targets can depend on other targets in this package and products from dependencies.
        .target(
            name: "UltraGrothSwift",
            dependencies: ["UltraGrothFramework"],
            linkerSettings: [.linkedFramework("SystemConfiguration")]
        ),
        .binaryTarget(
            name: "UltraGrothFramework",
            url: "https://github.com/rarimo/ultragroth/releases/download/v0.1.1/UltraGroth.xcframework.zip",
            checksum: "9eb221e26a6568e863cf0524174b1445112887f8ea26ff3ee663d91e4d04850e"
        ),
        .testTarget(
            name: "UltraGrothSwiftTests",
            dependencies: ["UltraGrothSwift"]
        ),
    ]
)
