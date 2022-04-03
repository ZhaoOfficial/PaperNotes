# Unstructured Lumigraph Rendering

## 0 Abstract

我们描述了一种基于图像的渲染方法，它概括了当前使用的许多基于图像的渲染算法，包括光场渲染和视角相关的纹理映射。
We describe an image based rendering approach that generalizes many image based rendering algorithms currently in use including light field rendering and view-dependent texture mapping.

它允许从一组输入相机进行光场风格渲染，这些相机不限于平面或任何特定的流形。
It allows for lumigraph style rendering from a set of input cameras that are not restricted to a plane or to any specific manifold.

在常规和平面输入相机位置的情况下，我们的算法简化为典型的光场方法。在更少的相机和良好的近似几何的情况下，我们的算法表现得像视图相关的纹理映射。
In the case of regular and planar input camera positions, our algorithm reduces to a typical lumigraph approach. In the case of fewer cameras and good approximate geometry, our algorithm behaves like view-dependent texture mapping.

## 1 Introduction

两种有效的 IBR 方法是视图相关纹理映射和光场方法。
Two effective IBR methods are view-dependent texture mapping (VDTM) and the light field/lumigraph approaches.

光场渲染需要来自相机的大量图像，这些图像的中心位于均匀采样的二维面上，但它对场景的几何形状几乎没有假设。相比之下，VDTM 假设一个相对准确的几何模型，但只需要来自输入相机的少量纹理，这些纹理可以处于一般位置。
Light field rendering requires a large collection of images from cameras whose centers lie on a regularly sampled two-dimensional patch, but it makes few assumptions about the geometry of the scene. In contrast, VDTM assumes a relatively accurate geometric model, but requires only a small number of textures from input cameras that can be in general position.























