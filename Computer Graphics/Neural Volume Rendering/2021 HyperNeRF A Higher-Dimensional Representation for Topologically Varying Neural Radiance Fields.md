# HyperNeRF: A Higher-Dimensional Representation for Topologically Varying Neural Radiance Fields

## 0 Abstract

一种重建这种非刚性场景的常用方法是使用学习得到的形变场映射，这种映射输入是每个输入图像中的坐标，输出是规范模板坐标空间。
A common approach to reconstruct such non-rigid scenes is through the use of a learned deformation field mapping from coordinates in each input image into a canonical template coordinate space.

然而，这些基于形变的方法难以对拓扑变化进行建模，因为拓扑变化需要形变场的不连续性，但这些形变场必然是连续的。
However, these deformation-based approaches struggle to model changes in topology, as topological changes require a discontinuity in the deformation field, but these deformation fields are necessarily continuous.

我们通过将 NeRF 提升到更高维度的空间来解决这个限制，并将与每个单独的输入图像对应的 5D 辐射场表示为穿过这个“超空间”的切片。
We address this limitation by lifting NeRFs into a higher dimensional space, and by representing the 5D radiance field corresponding to each individual input image as a slice through this “hyper-space”.

## 1 Introduction

首先，虽然经典等值面添加了单个环境维度，但我们可以添加**任意数量的环境维度**以提供更多自由度。其次，不像传统的那样将等值面**限制在超平面上**，而是允许通过 MLP 表示的通用**弯曲切片流形**。
First, whereas classical level sets add a single ambient dimension, we can add **any number of ambient dimensions** to provide more degrees of freedom. Second, rather than restrict level sets to **hyper planes**, as is traditional, we allow general, **curved slicing manifolds**, represented through MLPs.

以前使用更高维输入的方法需要大量的正则化或额外的监督。相比之下，我们的方法保留了一个形变场，该场先前已证明具有融合不同角度观察信息的强大能力，并且我们用一种优化策略来鼓励更高维度的平滑行为，而不是用正则化器。
Previous methods using higher dimensional inputs require either substantial regularization or additional supervision. In contrast, our method retains a deformation field, which has previously demonstrated strong ability to fuse information across observations, and instead of regularizers, we use an optimization strategy that encourages smooth behaviour in the higher dimensions.

## 2 Related Work

### 2.1 Non-Rigid Reconstruction

一种非刚性重建技术中的常用方法，是将场景分解为场景几何的规范模型（跨帧固定）和扭曲规范场景几何以再现每个输入图像的形变模型。
A common approach in non-rigid reconstruction techniques is to decompose a scene into a canonical model of scene geometry (which is fixed across frames) and a deformation model that warps the canonical scene geometry to reproduce each input image.

在光谱的另一端，使用多个相机和深度传感器的复杂且昂贵的捕获设置可用于过度约束问题，从而允许登记和融合 3D 扫描以产生高质量的结果。
On the opposite end of the spectrum, complicated and expensive capture setups using multiple cameras and depth sensors can be used to over-constrain the problem, thereby allowing 3D scans to be registered and fused to produce high quality results.

### 2.2 Neural Rendering

直到最近，这一领域的主流范式一直将合成图像的任务框定为某种 "图像到图像的翻译 "任务，其中神经网络被训练为将一个场景的某些表示映射到该场景的图像中。
The dominant paradigm in this field has, until recently, been framing the task of synthesizing an image as a sort of “image to image translation” task, in which a neural network is trained to map some representation of a scene into an image of that scene.

最近，神经渲染领域的研究已经开始从这种 "图像到图像的翻译 "范式转向 "神经场景表示 "范式。
Research within neural rendering has recently begun to shift away from this “image to image translation” paradigm and towards a “neural scene representation” paradigm.

注意到，在NeRF中，神经网络并不是用来渲染图像的--而是用一个基于物理的解析的体积渲染引擎来渲染一个场景，而这个场景的几何和辐射度恰好是由一个神经网络来设定参数。
Note that, in NeRF, a neural network is not used to render an image—instead, an analytical physics-based volumetric rendering engine is used to render a scene whose geometry and radiance happen to be parameterized by a neural network.

尽管 NeRF 在所有内容都是静态的场景中产生了令人信服的结果，但在有移动物体的情况下，它的失败是灾难性的。基于形变的方法将空间变化的形变应用于一些规范的辐射场，而基于调制的方法则直接将场景的辐射场限定在输入图像的某些属性上并对其进行相应的修改。
Though NeRF produces compelling results on scenes in which all content is static, it fails catastrophically in the presence of moving objects. Deformation-based approaches apply a spatially-varying deformation to some canonical radiance field, and modulation-based approaches directly condition the radiance field of the scene on some property of the input image and modify it accordingly.


我们的方法可以被认为是基于变形和基于调制的方法的结合：我们使用变形来模拟场景中的运动，从而产生一个良好的优化，但我们也扩展了NeRF的三维输入坐标空间，以接受额外的高维坐标作为输入，并允许沿高维以及空间维度进行变形。
Our method can be thought of as a combination of deformation-based and modulation-based approaches: we use deformations to model motion in the scene, resulting in a well-behaved optimization, but we also extend NeRF's 3D input coordinate space to take additional higher-dimension coordinates as input, and allow for deformations along the higher dimensions as well as the spatial dimensions.









