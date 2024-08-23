# from mmpose.apis import init_pose_model, inference_top_down_pose_model

# # Initialize your model with the config file and the trained weights
# config_file = 'configs/animal_2d_keypoint/topdown_heatmap/ak/w32_256x256_adam_lr1e-3_ak_vhr_s.py'
# checkpoint_file = 'checkpoints/VHR-BirdPose-S.pth'

# model = init_pose_model(config_file, checkpoint_file, device='cpu')

# # Path to the input image
# img = 'tests/data/ak/1077-Bernheim_newpartner.jpg'
# person_bbox = [50, 50, 150, 150]  # example bounding box
# pose_results = inference_top_down_pose_model(model, img, person_bbox, format='xywh')

# # Use the pose_results
# print(pose_results)

# ##############
import numpy as np
from mmpose.apis import MMPoseInferencer
import json

from mmpose.models.vhr_birdpose import VHRBirdPose 

# Initialize the inferencer with your custom model
inferencer = MMPoseInferencer(
    pose2d='configs/animal_2d_keypoint/topdown_heatmap/ak/w32_256x256_adam_lr1e-3_ak_vhr_s.py',
    pose2d_weights='checkpoints/VHR-BirdPose-S.pth',
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
