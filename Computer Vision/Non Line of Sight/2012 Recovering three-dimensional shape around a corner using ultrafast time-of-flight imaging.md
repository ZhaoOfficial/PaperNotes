# Recovering three-dimensional shape around a corner using ultrafast time-of-flight imaging

## Abstract

## Introduction

在图像传感器上检测到的光由直接从光源传播到传感器视线中的物体的直接光，和在撞击视线上的物体之前与场景的其他部分相互作用的间接光组成。来自视线外物体的光通过多次反射（或反弹）作为间接光到达传感器。在传统成像中，如果反射或反弹是漫射的，则很难利用这种非视线光。
The light detected on an image sensor is composed of direct light, that travels directly from the light source to an object in the line of sight of the sensor, and indirect light that interacts with other parts of the scene before striking an object in the line of sight. Light from objects outside the line of sight reaches the sensor as indirect light, via multiple reflections (or bounces). In conventional imaging, it is difficult to exploit this non-line-of-sight light, if the reflections or bounces are diffuse.

我们使用一个光源一次照亮一个场景点，并记录与场景交互后的反射光。
We use a light source to illuminate one scene spot at a time and record the reflected light after its interaction with the scene.

我们用短脉冲照亮场景，并使用返回光的飞行时间作为分析来自场景的直接光和散射光的手段。
We illuminate the scene with a short pulse and use the time-of-flight of returning light as a means to analyze direct and scattered light from the scene.

## Results

### Imaging process

我们不知道相机接收到的哪些脉冲来自目标物体的哪个表面点。
We do not know which pulses received by the camera came from which surface point on the target object.

我们扫描并连续改变漫射壁上激光光斑的位置。 记录的双曲线的形状和位置相应地变化。条纹图像中的每个像素对应于墙上的一个有限区域和一个 2 ps 的时间间隔，一个离散的时空。
We scan and successively change the position of the laser spot on the diffuse wall. The shape and position of the recorded hyperbolic curve varies accordingly. Each pixel in a streak image corresponds to a finite area on the wall and a 2-ps time interval, a discretized space-time bin.

我们将笛卡尔空间离散为体素并计算体素在隐藏表面上的可能性。
We discretize the Cartesian space into voxels and compute the likelihood of the voxel being on a hidden surface.

单条纹图像中所有像素的加权强度总和为目标补丁创建了一个近似的热图。对漫射器壁上的许多激光位置重复该过程，并使用来自相应条纹图像的像素提供更好的近似值。在实践中，我们使用约 60 个激光位置。传统的反投影需要一个高通滤波步骤。我们使用沿着体素网格的 z 方向并近似垂直于墙壁的数据的二阶导数作为有效过滤器，并恢复图 3e 中的隐藏表面补丁。由于热图中体素处的值是对大量条纹图像像素求和的结果，因此热图包含低噪声，并且与二阶导数滤波器相关的噪声放大是可以接受的。
Because values at the voxels in the heatmap are the result of summing a large number of streak image pixels, the heatmap contains low noise and the noise amplification associated with a second-derivative filter is acceptable. The summation of weighted intensities from all pixels in a single streak image creates an approximate heatmap for the target patch. Repeating the process for many laser positions on the diffuser wall, and using pixels from the corresponding streak images provides a better approximation. In practice, we use ~60 laser positions. Traditional backprojection requires a high-pass filtering step. We use the second derivative of the data along the z direction of the voxel grid and approximately perpendicular to the wall as an effective filter and recover the hidden surface patch in Fig. 3e.

### Algorithm









