import os
import json
import random
import xml.etree.ElementTree as ET
from sklearn.model_selection import train_test_split

# Step 1: Read the image paths from the text file
with open('/home/ailab/AUE8088-PA2/datasets/kaist-rgbt/train-all-04.txt', 'r') as file:
    image_paths = file.readlines()

# Step 2: Split the image paths into training and validation sets
train_paths, val_paths = train_test_split(image_paths, test_size=0.2, random_state=42)

def parse_annotation(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    annotations = []
    
    for obj in root.findall('object'):
        name = obj.find('name').text
        bbox = obj.find('bndbox')
        x = int(bbox.find('x').text)
        y = int(bbox.find('y').text)
        w = int(bbox.find('w').text)
        h = int(bbox.find('h').text)
        annotations.append({
            'category_id': 0 if name == 'person' else 1,  # Simplified for example
            'bbox': [x, y, w, h],
            'height': h,
            'occlusion': int(obj.find('occlusion').text),
            'ignore': int(obj.find('difficult').text),
        })
    
    return annotations

def create_json_and_txt(image_paths, output_json, output_txt):
    images = []
    annotations = []
    image_id = 0
    annotation_id = 0
    
    with open(output_txt, 'w') as txt_file:
        for path in image_paths:
            path = path.strip()
            im_name = os.path.basename(path)
            # Adjust im_name format
            im_name = im_name.replace('_', '/').replace('.jpg', '')
            
            xml_file = os.path.join('datasets/kaist-rgbt/train/labels-xml', im_name.replace('/', '_') + '.xml')
            
            # Parse XML file
            annotation_list = parse_annotation(xml_file)
            
            # Add image info
            images.append({
                'id': image_id,
                'im_name': im_name,
                'height': 512,  # Example height
                'width': 640,   # Example width
            })
            
            # Add annotations
            for annotation in annotation_list:
                annotation['id'] = annotation_id
                annotation['image_id'] = image_id
                annotations.append(annotation)
                annotation_id += 1
            
            image_id += 1
            
            # Write the image path to the txt file
            dataset_folder = im_name.split('_')[0]
            txt_file.write(f'datasets/kaist-rgbt/train/images/{{}}/{im_name.replace("/", "_")}.jpg\n')
    
    dataset = {
        'images': images,
        'annotations': annotations,
        'categories': [
            {'id': 0, 'name': 'person'},
            {'id': 1, 'name': 'cyclist'},
            {'id': 2, 'name': 'people'},
            {'id': 3, 'name': 'person?'},
        ],
    }
    
    with open(output_json, 'w') as json_file:
        json.dump(dataset, json_file, indent=4)

# Step 3: Create JSON files and txt files for training and validation sets
create_json_and_txt(train_paths, 'train_set.json', 'train.txt')
create_json_and_txt(val_paths, 'val_set.json', 'val.txt')
