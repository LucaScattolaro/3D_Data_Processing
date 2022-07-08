# 3D Data processing - Stereo Matching Lab
## Main Goal
The goal of the homework is to compute disparity maps of stereo images using Patch Matching algorithm.
I extend and provided C++ software with the patch match core functionalities: disparity propagation and
random search.

## Implementation
The goal is to extend the process() method to perform:
1. Spatial propagation
2. Random search around the current disparity
3. View propagation
Let’s now focus on the idea applied for each of the aforementioned method

### Spatial propagation
we evaluate whether assigning to p the disparity dq of spatial neighbor pixel q decrease the matching costs.

### Random search around the current disparity
we should perturb the disparity at position (x,y) by a factor of deltaz where deltaz ∈ [enddz, maxdeltaz]

### View propagation
we check all pixels p′ of the second view that have our current pixel p as a matching point according to their current disparity.


## Results
![image](https://user-images.githubusercontent.com/62805357/177941158-43412bab-1f79-496a-a7fb-279d6e71d304.png)
