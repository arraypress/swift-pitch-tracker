// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "swift-pitch-tracker",
    platforms: [
        .macOS("27.0"),
    ],
    products: [
        .library(name: "PitchTracker", targets: ["PitchTracker"]),
    ],
    targets: [
        .target(name: "PitchTracker"),
        .testTarget(
            name: "PitchTrackerTests",
            dependencies: ["PitchTracker"],
            resources: [.copy("Fixtures")]
        ),
    ]
)
