#!/bin/bash

# Base URL of the GitHub repository
BASE_URL="https://raw.githubusercontent.com/lichess-org/chess-openings/master"
# Target folder
TARGET_DIR="warehouse/lichess-chess-openings"
# Ensure target directory exists
mkdir -p "$TARGET_DIR"

# Uncomment the line below to change to the project root directory,
# in case you're running this script from inside the scripts/ folder.
# cd "$(dirname "$0")/.."

# Loop from a to e
for LETTER in a b c d e; do
    FILENAME="${LETTER}.tsv"
    echo "Downloading $FILENAME..."
    curl -sSL "${BASE_URL}/${FILENAME}" -o "${TARGET_DIR}/${FILENAME}"
done

echo "Done. Files saved to: $TARGET_DIR"
