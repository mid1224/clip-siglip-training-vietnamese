# Convert UIT-ViIC captions.txt to COCO format JSON
# UIT-ViIC dataset: https://huggingface.co/datasets/ThucPD/UIT-ViIC

import json
import os

def create_coco_json(captions_file, output_file):
    images = []
    annotations = []
    image_id_map = {}
    annotation_id = 1
    
    # Read captions.txt
    with open(captions_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for line in lines:
        if not line.strip():
            continue
            
        # Separate image path and caption (by tab in original captions.txt)
        parts = line.strip().split('\t')
        if len(parts) < 2:
            continue
            
        image_path = parts[0]
        caption = parts[1]
        
        # Get image file name from image path
        file_name = os.path.basename(image_path)
        
        # Create image id if not exists
        if file_name not in image_id_map:
            image_id = len(image_id_map) + 1
            image_id_map[file_name] = image_id
            images.append({
                "id": image_id,
                "file_name": file_name
            })
        
        # Add annotation
        annotations.append({
            "id": annotation_id,
            "image_id": image_id_map[file_name],
            "caption": caption
        })
        annotation_id += 1
        
    coco_format = {
        "images": images,
        "annotations": annotations,
        "type": "captions",
        "info": {"description": "UIT-ViIC Vietnamese Image-Caption Dataset"}
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(coco_format, f, ensure_ascii=True, indent=4)
    print(f"Created {output_file} with {len(images)} images and {len(annotations)} annotations.")

# Path to captions.txt file
create_coco_json("src/data/UIT-ViIC/dataset/train/captions.txt", "src/data/UIT-ViIC/dataset/train/train.json")