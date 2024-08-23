import numpy as np
from mmpose.apis import MMPoseInferencer
import json

# Initialize the inferencer with your AK model
inferencer = MMPoseInferencer(
    pose2d='td-hm_hrnet-w32_8xb32-300e_animalkingdom_P3_bird-256x256',
    pose2d_weights='https://download.openmmlab.com/mmpose/v1/animal_2d_keypoint/topdown_heatmap/animal_kingdom/td-hm_hrnet-w32_8xb32-300e_animalkingdom_P3_bird-256x256-566feff5_20230519.pth',
    device='cpu'
)

# Path to the input image
img_path = 'tests/data/ak/1077-Bernheim_newpartner.jpg'

# Perform inference
result_generator = inferencer(img_path)
result = next(result_generator)

# Initialize a list to store annotations
annotations = []

# Iterate over each detected instance in the predictions
for i, instance_data in enumerate(result['predictions']):
    keypoints = instance_data[i]['keypoints'] 
    bbox = instance_data[i]['bbox'][0]  
    keypoint_scores = instance_data[i]['keypoint_scores'] 

    # Convert keypoints to the required format (x, y, confidence)
    formatted_keypoints = []
    for j, keypoint in enumerate(keypoints):
        if keypoint_scores[j] > 0.5:  # keypoints with a confidence score > 0.5
            formatted_keypoints.extend([int(keypoint[0]), int(keypoint[1]), int(2)])
        else:
            formatted_keypoints.extend([int(keypoint[0]), int(keypoint[1]), int(0)])

    # Calculate the number of keypoints
    # num_keypoints = sum(1 for score in keypoint_scores if score > 0.5)  # Consider keypoints with a confidence score > 0.5
    num_keypoints = len(keypoint_scores)

    # Calculate area (area of the bounding box)
    area = int(bbox[2] * bbox[3])

    # Generate a unique annotation ID (in practice, you should manage this ID across your dataset)
    annotation_id = i + 1  # Simply using index + 1 for unique ID

    # Example COCO annotation dictionary
    annotation = {
        "segmentation": [],
        "area": area,
        "iscrowd": 0,
        "image_id": 23364344,  # Replace with your actual image ID
        "bbox": [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])],
        "category_id": 1,  # Assuming category ID 1 corresponds to "animal"
        "id": annotation_id,
        "keypoints": formatted_keypoints,
        "num_keypoints": num_keypoints
    }
    annotations.append(annotation)

# Print the resulting annotations
print(json.dumps(annotations, indent=4))