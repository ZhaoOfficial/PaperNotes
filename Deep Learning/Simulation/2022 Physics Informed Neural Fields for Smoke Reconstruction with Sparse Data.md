# Physics Informed Neural Fields for Smoke Reconstruction with Sparse Data

## 0 Abstract

由于底层物理的复杂性以及捕获数据中的严重遮挡和复杂照明，从稀疏的多视图 RGB 视频中对动态流体进行高保真重建仍然是一项艰巨的挑战。
High-fidelity reconstruction of dynamic fluids from sparse multiview RGB videos remains a formidable challenge, due to the complexity of the underlying physics as well as the severe occlusion and complex lighting in the captured data.  

我们提出了第一种通过在不考虑光照条件、几何信息或边界条件作为输入的情况下从一组稀疏视频帧进行端到端优化的控制物理学（即 Navier -Stokes 方程）来重建动态流体现象的方法。
We present the first method to reconstruct dynamic fluid phenomena by leveraging the governing physics (ie, Navier -Stokes equations) in an end-to-end optimization from a mere set of sparse video frames without taking lighting conditions, geometry information, or boundary conditions as input.

## 1 Introduction

我们首先在时间维度上增强空间场景表示，并学习一个时变的神经辐射场。以具有动态流体的场景的 RGB 视频作为输入，时变 NeRF 学习使用多层感知器 (MLP) 对时空辐射场进行编码。
We first augment the spatial scene representation in the time dimension and learn a time-varying neural radiance field. Taking RGB videos of a scene with dynamic fluid as input, the time-varying NeRF learns to encode a spatiotemporal radiance field with a Multi-Layered Perceptron (MLPs).  

我们建议使用基于物理的深度学习技术，并使用另一个 MLP 来表示连续的时空速度场。
We propose to use physics-informed deep learning technologies and use another MLP to represent the continuous spatiotemporal velocity field.   
