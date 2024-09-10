"""
A simple COCO annotation visualization script using matplotlib. NO predictions are made, only visualized. 
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

# Load your image
image_path = 'tests/data/ak/1077-Bernheim_newpartner.jpg' 
image = Image.open(image_path)

# Your annotation data
annotations = [
    {
        "segmentation": [],
        "area": 466344,
        "iscrowd": 0,
        "image_id": 23364344,
        "bbox": [
            0.0,
            0.0,
            612.0,
            762.0
        ],
        "category_id": 1,
        "id": 1,
        "keypoints": [
            175,
            139,
            2,
            153,
            168,
            2,
            146,
            161,
            2,
            86,
            191,
            2,
            138,
            183,
            2,
            138,
            183,
            2,
            86,
            191,
            2,
            257,
            325,
            2,
            146,
            332,
            2,
            265,
            407,
            2,
            146,
            466,
            0,
            503,
            503,
            2,
            562,
            474,
            2,
            369,
            310,
            2,
            317,
            488,
            2,
            205,
            481,
            2,
            354,
            570,
            2,
            198,
            563,
            2,
            339,
            682,
            2,
            190,
            652,
            2,
            436,
            384,
            2,
            480,
            421,
            2,
            585,
            526,
            2
        ],
        "num_keypoints": 23
    }
]

# Create a matplotlib figure
fig, ax = plt.subplots(1)
ax.imshow(image)

# Draw the bounding box
bbox = annotations[0]['bbox']
rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], linewidth=2, edgecolor='r', facecolor='none')
ax.add_patch(rect)

# Draw the keypoints
keypoints = annotations[0]['keypoints']
for i in range(0, len(keypoints), 3):
    x, y, v = keypoints[i], keypoints[i+1], keypoints[i+2]
    if v > 0:  # If visibility flag is > 0, draw the keypoint
        ax.plot(x, y, 'bo')  # Blue dot for keypoints

# Show the image with annotations
plt.axis('off')
plt.show()
