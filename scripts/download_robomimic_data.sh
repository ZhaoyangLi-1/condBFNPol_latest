#!/bin/bash
# ==============================================================================
# Download Robomimic Datasets
# ==============================================================================
# Downloads the robomimic image dataset for Diffusion Policy
# Source: https://diffusion-policy.cs.columbia.edu/data/training/
# 
# This downloads ALL robomimic tasks (lift, can, square) in one zip file
# ==============================================================================

set -e

PROJECT_DIR="$PROJECT_ROOT"
DATA_DIR="${PROJECT_DIR}/data"

URL="https://diffusion-policy.cs.columbia.edu/data/training/robomimic_image.zip"
ZIP_FILE="${DATA_DIR}/robomimic_image.zip"

echo "========================================"
echo "Robomimic Dataset Download"
echo "========================================"
echo ""

# Check current disk usage
echo "=== Current Disk Status ==="
df -h $PROJECT_ROOT | tail -1
echo ""

# Create data directory
mkdir -p "$DATA_DIR"
cd "$DATA_DIR"

# Check if already extracted
if [ -d "${DATA_DIR}/robomimic" ]; then
    echo "Robomimic data folder already exists!"
    echo "Contents:"
    find ${DATA_DIR}/robomimic -name "*.hdf5*" -o -name "*.zarr*" 2>/dev/null | head -10
    echo ""
    read -p "Re-download and overwrite? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Keeping existing data."
        exit 0
    fi
fi

echo "=== Downloading robomimic_image.zip ==="
echo "URL: $URL"
echo "This file is approximately 6-10 GB, please be patient..."
echo ""

# Download with progress
wget --progress=bar:force -O "$ZIP_FILE" "$URL"

if [ $? -ne 0 ]; then
    echo "ERROR: Download failed!"
    rm -f "$ZIP_FILE"
    exit 1
fi

echo ""
echo "=== Download complete ==="
ls -lh "$ZIP_FILE"
echo ""

echo "=== Extracting ==="
unzip -o "$ZIP_FILE"

if [ $? -eq 0 ]; then
    echo ""
    echo "=== Extraction complete ==="
    
    # Clean up zip file to save space
    rm -f "$ZIP_FILE"
    echo "Removed zip file to save space"
    
    # Show what was extracted
    echo ""
    echo "=== Extracted contents ==="
    find robomimic -type f -name "*.hdf5*" -o -name "*.zarr*" 2>/dev/null | head -20
    echo ""
    
    # Show disk usage
    du -sh robomimic/
else
    echo "ERROR: Extraction failed!"
    exit 1
fi

echo ""
echo "========================================"
echo "Done! Data is in: ${DATA_DIR}/robomimic/"
echo "========================================"
