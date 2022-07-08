# 3D Data Processing
## Introduction 
The goal of this laboratory is to segment a point cloud taken from the famous Semantic-Kitti dataset using PointNet.
The original dataset counts about 30 labels, but for this assignment we remap them to only 3:
• Traversable (road, parking, sidewalk, ecc.)
• Not-Traversable (cars, trucks, fences, trees, people, objects)
• Unknown (outliers)

## Implementation Details
I followed the original implementation that you can find [here](https://github.com/charlesq34/pointnet)

![image](https://user-images.githubusercontent.com/62805357/177939094-d47ccb91-77a5-4b5a-9cfa-2d25386f3427.png)
