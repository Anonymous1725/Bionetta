#!/bin/bash

FILE_ID="1tiCLJ9l1fU60e0eRP7uBIoKUOU6iUOhL"
ARCHIVE_NAME="saved_models.tar.xz"
TARGET_DIR="examples"

echo ">>> Downloading file using gdown..."
gdown --fuzzy "https://drive.google.com/uc?id=$FILE_ID" -O "$ARCHIVE_NAME"

if [ ! -s "$ARCHIVE_NAME" ]; then
    echo "❌ Download failed or file is empty!"
    exit 1
fi

echo ">>> Extracting..."
mkdir -p "$TARGET_DIR"
tar -xf "$ARCHIVE_NAME" -C "$TARGET_DIR"

echo ">>> Removing archive..."
rm "$ARCHIVE_NAME"

echo "✅ Done! Extracted to $TARGET_DIR"
