# 3D Data Processing - Cloud Registration
## Goal
The goal of this laboratory is: given a source and a target point cloud, find the alignment transformation of the source to
the target cloud

## Implementation Details
The implementation has been done in C++ by using Open3D library.
In order to complete the methods:
* Registration(...) : constructor.
* drawregistrationresult(...) : method to visualize target and source with two different colors.
* preprocess(...): method to downsample, estimate normals and compute FPFH features.
* executeglobalregistration(): method to execute the global registration. I decide to use (and keep in the code)
RANSAC for global registration but I also tried Fast global registration.
* executeicpregistration(...): method to refine the result


## Results
![image](https://user-images.githubusercontent.com/62805357/177940654-ee262842-7087-4cd1-b0c7-0f8561b6847a.png)
