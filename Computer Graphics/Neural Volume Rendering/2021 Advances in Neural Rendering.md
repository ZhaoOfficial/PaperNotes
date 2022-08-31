# Advances in Neural Rendering

## 0 Abstract

传统上，场景的合成图像是使用光栅化或光线追踪等渲染算法生成的，这些算法将具体定义的几何表达和材料属性作为输入。这些输入共同定义了实际场景和渲染的内容，并被称为场景表达（其中场景由一个或多个对象组成）。
Traditionally, synthetic images of a scene are generated using rendering algorithms such as rasterization or ray tracing, which take specifically defined representations of geometry and material properties as input. Collectively, these inputs define the actual scene and what is rendered, and are referred to as the scene representation (where a scene consists of one or more objects).



## 3 Fundamentals of Neural Rendering

### 3.1 Scene Representations

通常，体积表达可以表达表面，但反之则不行。体积表达存储体积属性，例如密度、不透明度或占用率，但它们也可以存储多维特征，例如颜色或辐射度。与体积表达相比，表面表达存储与物体的表面有关的属性。 。它们不能用于模拟体积物质，例如烟雾（除非它是粗略的近似值）。
In general, volumetric representations can represent surfaces, but not vice versa. Volumetric representations store volumetric properties such as densities, opacities or occupancies, but they can also store multidimensional features such as colors or radiance. In contrast to volumetric representations, surface representations store properties w.r.t. the surface of an object. They cannot be used to model volumetric matter, such as smoke (unless it is a coarse approximation).

对于表面和体积表达，都有连续的和离散的对应物。神经渲染方法对于连续表达特别感兴趣，因为它们可以提供解析的梯度。对于表面表达，有两种不同的方法来表达表面--显式或隐式。
For both surface and volumetric representations, there are continuous and discretized counterparts. The continuous representations are particularly interesting for neural rendering approaches since they can provide analytic gradients. For surface representations, there are two different ways to represent the surface – explicitly or implicitly.

在神经渲染中，使用神经网络来拟合表面或体积表示函数的场景表示称为神经场景表示。
In the context of neural rendering, a scene representation that is using a neural network to approximate the surface or volumetric representation function is called neural scene representation.

