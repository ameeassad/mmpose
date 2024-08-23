from mmpose.apis import MMPoseInferencer

# Initialize the inferencer with your AK model
inferencer = MMPoseInferencer(
    pose2d='td-hm_hrnet-w32_8xb32-300e_animalkingdom_P3_bird-256x256',
    pose2d_weights='https://download.openmmlab.com/mmpose/v1/animal_2d_keypoint/topdown_heatmap/animal_kingdom/td-hm_hrnet-w32_8xb32-300e_animalkingdom_P3_bird-256x256-566feff5_20230519.pth'
)

# Access the dataset meta information
dataset_meta = inferencer.inferencer.model.dataset_meta

# Extract keypoints and skeleton information
keypoint_names = [name for _, name in sorted(dataset_meta['keypoint_id2name'].items())]
skeleton_links = dataset_meta['skeleton_links']

# Convert skeleton indices to 1-based index (COCO uses 1-based indices)
skeleton_coco_format = [[link[0] + 1, link[1] + 1] for link in skeleton_links]

# Create the "categories" dictionary
categories = [
    {
        "supercategory": "bird",  # or "person" depending on your use case
        "id": 1,  # You can adjust the ID based on your dataset
        "name": "bird",  # Adjust based on your dataset
        "keypoints": keypoint_names,
        "skeleton": skeleton_coco_format
    }
]

# Example COCO dataset dictionary
coco_categories = {
    "categories": categories
}

# Print the resulting COCO dataset structure with categories
import json
print(json.dumps(coco_categories, indent=4))
