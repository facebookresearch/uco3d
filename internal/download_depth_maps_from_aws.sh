#!/bin/bash

# Create base directory for depths
BASE_DIR="/fsx-repligen/shared/datasets/uCO3D/temp_depth_check"
mkdir -p "$BASE_DIR"

# Array of folder names
folders=(
    "83-7645-37766"
    "78-2534-50232"
    "31-59665-80922"
    "1-28317-25169"
    "811-12126-68807"
    "415-99774-76371"
    "1-73927-62462"
    "137-92015-3319"
    "39-89601-51876"
    "150-12801-22937"
    "465-65822-46025"
    "22-47118-33897"
    "124-50644-76953"
    "100-66834-80865"
    "5131-10720-4933"
)

# Process each folder
for folder in "${folders[@]}"; do
    echo "Processing $folder..."
    
    # Create target directory
    target_dir="$BASE_DIR/$folder"
    mkdir -p "$target_dir"
    
    # Sync from S3
    aws s3 sync "s3://genai-project-repligen-archive/xingchenliu/sfm/dense-only-tar/$folder/mapper_output/0/depths/" "$target_dir/"
    
    echo "Completed syncing $folder"
done

echo "All sync operations completed!"