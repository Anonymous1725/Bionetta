#!/bin/bash

set -e

cargo build --release --target aarch64-apple-ios --lib
cargo build --release --target aarch64-apple-ios-sim --lib

XCFWNAME="Wtns{{ framework_name }}"
FWNAME="Wtns{{ framework_name }}Lib"

function create_framework() {
    for fw in "$@"; do
        copy_framework_files "${fw}"
    done

    local fw_paths=()
    for fw in "$@"; do
        fw_paths+=("-framework" "Frameworks/${fw}/$FWNAME.framework")
    done

    for fw in "$@"; do
        {
        echo "<?xml version=\"1.0\" encoding=\"UTF-8\"?>"
        echo "<!DOCTYPE plist PUBLIC \"-//Apple//DTD PLIST 1.0//EN\" \"http://www.apple.com/DTDs/PropertyList-1.0.dtd\">"
        echo "<plist version=\"1.0\">"
        echo "<dict>"
        echo "    <key>CFBundlePackageType</key>"
        echo "    <string>FMWK</string>"
        echo "    <key>CFBundleIdentifier</key>"
        echo "    <string>Rarilabs.$FWNAME</string>"
        echo "    <key>CFBundleExecutable</key>"
        echo "    <string>$FWNAME</string>"
        echo "    <key>CFBundleShortVersionString</key>"
        echo "    <string>1.0.0</string>"
        echo "    <key>CFBundleVersion</key>"
        echo "    <string>3</string>"
        echo "    <key>MinimumOSVersion</key>"
        echo "    <string>100</string>"
        echo "</dict>"
        echo "</plist>"
        } > "Frameworks/$fw/$FWNAME.framework/Info.plist"
    done

    rm -rf "Frameworks/$XCFWNAME.xcframework"
    xcrun xcodebuild -create-xcframework \
        "${fw_paths[@]}" \
        -output "Frameworks/$XCFWNAME.xcframework"

    
    if [ -n "$CODE_SIGNER" ]; then
        codesign --timestamp -s "$CODE_SIGNER" "Frameworks/$XCFWNAME.xcframework"
    fi
}

function copy_framework_files() {
    local FRAMEWORK_PATH="Frameworks/$1/$FWNAME.framework"

    mkdir -p "$FRAMEWORK_PATH/Headers"

    cp ../header.h "$FRAMEWORK_PATH/Headers/$FWNAME.h"

    mkdir -p $FRAMEWORK_PATH/Modules
    {
    echo "framework module $FWNAME {"
    echo "    umbrella header \"$FWNAME.h\""
    echo "    export *"
    echo "    module * { export * }"
    echo "}"
    } > $FRAMEWORK_PATH/Modules/module.modulemap

    cp target/$1/release/lib{{ model_name }}.a $FRAMEWORK_PATH/$FWNAME
}

rm -rf Frameworks

strip -x target/aarch64-apple-ios/release/lib{{ model_name }}.a target/aarch64-apple-ios-sim/release/lib{{ model_name }}.a

frameworks=("aarch64-apple-ios" "aarch64-apple-ios-sim")
create_framework "${frameworks[@]}"

pushd "Frameworks"
zip -X -9 -r "$XCFWNAME.xcframework.zip" "$XCFWNAME.xcframework" -i */$FWNAME -i *.plist -i *.h -i *.modulemap
popd

swift package compute-checksum "Frameworks/$XCFWNAME.xcframework.zip" > Frameworks/$XCFWNAME.xcframework.zip.checksum

echo "$XCFWNAME.xcframework.zip checksum: $(cat Frameworks/$XCFWNAME.xcframework.zip.checksum)"
