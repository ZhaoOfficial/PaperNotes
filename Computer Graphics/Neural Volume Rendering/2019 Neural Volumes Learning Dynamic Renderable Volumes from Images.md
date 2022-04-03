# Neural Volumes: Learning Dynamic Renderable Volumes from Images

## 0 Abstract

受断层成像中使用的积分投影模型启发，新的动态对象的表示方法。
Representing dynamic objects inspired by the integral projection model used in tomographic imaging.

该方法在多视图捕获设置中直接从 2D 图像监督，不需要显式重建或跟踪对象。
The approach is supervised directly from 2D images in a multi-view capture setting and does not require explicit reconstruction or tracking of the object.

我们的方法有两个主要组成部分：将**输入图像转换为 3D 体积表示的编码器-解码器网络**，以及支持端到端训练的**可微分光线前进操作**。
Our method has two primary components: **an encoder-decoder network that transforms input images into a 3D volume representation**, and **a differentiable ray-marching operation** that enables end-to-end training.

## 1 Introduction

由于基于网格的表示在很大程度上依赖于重建的质量来产生引人注目的渲染，因此它们不适合处理此类情况（薄结构、不断演变的拓扑、半透明、散射、遮挡和生物运动）。
Since mesh-based representations rely heavily on the quality of reconstruction to produce compelling renderings, they are ill-suited to handle such cases (thin structures, evolving topology, translucency, scattering, occlusion, and biological motion).

由于他们的神经架构不是 3D 感知的，因此这些方法不会以与现实世界一致的方式推断出新的视角。
Since their neural architectures are not 3D aware, the methods do not extrapolate to novel viewpoints in a way that is consistent with the real world.

在对近场场景进行建模时，该问题会更加严重。与远场捕获相比，当用户与近场场景中的对象交互时，视角的变化更为常见，而远场捕获的交互性较少，且观察者主要是静止的。
The problem is exacerbated when modelling near-field scenes, where variation in viewpoint is more common as a user interacts with objects in the scene, compared with far-field captures where there is less interactivity and the viewer is mainly stationary.

这项工作的一个重要见解是，如果可以**同时学习几何和外观变化**，则可以对几何变化可解释的现象进行建模，从而更好地泛化不同的视角。
An important insight in this work is that if **both geometry and appearance variations can be learned simultaneously**, phenomena explainable by geometric variations may be modelled as such, leading to better generalization across viewpoints.





















