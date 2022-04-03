# Volumetric Video - Acquisition, Compression, Interaction and Perception

## Abstract

体积视频、自由视角视频或 4D 重建指的是使用多视角设置随时间推移重建 3D 内容的过程。
Volumetric video, free-viewpoint video or 4D reconstruction refer to the process of reconstructing 3D content over time using a multi-view setup.

## 1 Course Overview

捕获和分发体积内容的困难在于，必须为每一帧重建一个三维场景。
The difficulty in capturing and distributing volumetric content is that a 3D scene must be reconstructed for every frame.

从单一摄像机视角进行三维重建是非常不理想的。
3D reconstruction from a single camera view is highly ill-posed.

体积视频的格式大不相同，范围包括点云、三维网格或体积表示法，例如由有符号距离函数（SDF）编码。
Formats for volumetric video vary largely and ranges from point clouds, 3D meshes or a volumetric representation e.g. encoded by a signed distance function (SDF).

对于像人体这样的实体物体，由于更高的压缩率和更高的视觉质量，在几帧内具有一致顶点集的三维网格通常比点云更受欢迎。同时，网格表示并不总是理想的，特别是对于薄的物体或头发。
For solid objects, like human bodies, 3D meshes with a consistent vertex set over a few frames is often preferred over point clouds due to higher compression ratios and higher visual quality. At the same time a mesh representation is not always ideal, especially for thin objects or hair.

体积视频的一个可能的独特属性是，它既可以被视为像电影或戏剧一样的线性媒体，也可以被视为类似于游戏的互动媒体，在短的动画周期之间的无缝过渡模拟了生命般的互动。
A probably unique property of volumetric video is that it can be considered as both, a linear medium like cinema or theatre or an interactive medium similar to games, where a seamless transition between short animation cycles simulates life-like interactions.

首先，它应该提供一个捕获和分发体积内容所需的概述，并成为相关文献和开源项目的起点。本教程的第二个核心内容是提高对相关开源项目的认识，以促进复制体积视频管道，获得论文或教科书中描述的主题的实现，并鼓励为开源项目作出贡献，以提高个人在该领域的声誉。
First, it should provide an overview of aspects required to capture and distribute volumetric content and be a starting point for related literature and open sourced projects. The second central aspect of this tutorial is to raise awareness for related open source projects to facilitate replicating volumetric video pipelines, get access to an implementation of topics described in papers or textbooks, and encourage to contribute to open source projects to increase the personal reputation within the field.

## 4 Extended Abstracts

### 4.1 Photogrammetry pipeline

快速生成物体的 3D 模型的能力是许多应用中的一个关键问题，从 3D 打印的逆向工程到混合现实应用和视觉特效（VFX）的内容创作。
The ability to quickly generate a 3D model of an object is a key problem in many applications, from the reverse engineering for 3D printing to the content creation for Mixed Reality applications and Visual Special Effects (VFX).

### 4.2 Low-cost volumetric video with consumer grade sensors

最后，该教程将介绍这种低成本的体积捕捉系统所支持和促成的工作，展示潜在的用途。更重要的是，考虑到正在进行的数据驱动的革命，这种系统可用于轻松收集体积数据集，HUMAN4D 就是一个例子。
Finally, the tutorial will present the work that has been supported and enabled by this low-cost volumetric capture system, showcasing potential uses. More importantly, taking into account the ongoing data-driven revolution, such systems can be used for easily collecting volumetric datasets, with an example being HUMAN4D.

### 4.3 Poisson surface reconstruction

几十年来，从扫描的三维点重建表面一直是一个持续的研究领域，而且随着商品三维扫描仪的普及而变得更加重要。经典的表面重建方法是以三种方式之一来处理这个问题的。(1)将一个简单的复合体拟合到点云上，并将简单点标记为内部或外部；(2)演化一个基面，使其适合这些点；或(3)将一个隐含函数拟合到点云上，并使用 Marching Cubes 等算法提取一个适当的水平集。
Reconstructing surfaces from scanned 3D points has been an ongoing research area for several decades, and has only become more important as commodity 3D scanners have become ubiquitous. Classically, approaches for surface reconstruction have approached the problem in one of three ways: (1) fitting a simplicial complex to the point cloud and labelling simplices as either interior or exterior; (2) evolving a base surface so that it fits the points; or (3) fitting an implicit function to the point cloud and extracting an appropriate level-set using algorithm like Marching Cubes.

### 4.4 4D compression and streaming

### 4.5 Towards Animations Volumetric Video


这些方法超越了自由视点体积视频的应用，通过(i)用语义和动画属性丰富捕获的数据和(ii)应用基于几何和视频的混合动画方法，允许直接对高质量的数据本身进行动画，而不是创建一个与捕获的数据相似的可动画的模型，来重新激活和改变演员的表演。
Going beyond the application of free-viewpoint volumetric video, these methods allow re-animation and alteration of an actor’s performance through (i) the enrichment of the captured data with semantics and animation properties and (ii) applying hybrid geometry- and video-based animation methods that allow a direct animation of the high-quality data itself instead of creating an animatable model that resembles the captured data.

### 4.6 Perceptual aspects on volumetric video quality
