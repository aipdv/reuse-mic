#!/bin/sh
# Builds ReuseMic.app (menu bar) around ../reuse and ../reuse_rt.bin. Ad-hoc signed.
set -e
cd "$(dirname "$0")"
A=ReuseMic.app; rm -rf "$A"; mkdir -p "$A/Contents/MacOS" "$A/Contents/Resources"
swiftc -O -swift-version 5 -module-cache-path "${TMPDIR:-/tmp}/swift-mc" -o "$A/Contents/MacOS/ReuseMic" main.swift
cp ../reuse ../reuse_rt.bin "$A/Contents/Resources/"
cat > "$A/Contents/Info.plist" <<'PL'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>CFBundleName</key><string>ReuseMic</string>
  <key>CFBundleIdentifier</key><string>local.reusemic</string>
  <key>CFBundleExecutable</key><string>ReuseMic</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleShortVersionString</key><string>1.0</string>
  <key>LSMinimumSystemVersion</key><string>13.0</string>
  <key>LSUIElement</key><true/>
  <key>NSMicrophoneUsageDescription</key><string>ReuseMic cleans your microphone on this Mac and sends it to BlackHole 2ch.</string>
</dict></plist>
PL
codesign --force --deep -s - "$A"
echo "built $(pwd)/$A"
