# Acquiring the Reflectance Field of a Human Face

## Abstract

我们首先使用灯光舞台在入射照明方向的密集采样下从一小组视点获取面部图像。
We first acquire images of the face from a small set of viewpoints under a dense sampling of incident illumination directions using a light stage.

## Introduction

将人脸照片纹理映射到几何模型上的传统方法通常在光照、视点和表情的变化下无法显得逼真。
The traditional approach of texture-mapping a photograph of a face onto a geometric model usually fails to appear realistic under changes in lighting, viewpoint, and expression.

皮肤以漫反射和镜面反射两种方式反射光，并且这两种反射分量都在空间上发生变化。
skin reflects light both diffusely and specularly, and both of these reflection components are spatially varying.

### Definition of the Reflectance Field

我们可以将在每个可能的入射照明场下从 A 发出的辐射光场表示为一个八维反射场
We can represent the radiant light field from A under every possible incident field of illumination as an eight-dimensional reflectance field:
$$
R=R(R_i,R_r)=R(u_i,v_i,\theta_i,\phi_i;u_r,v_r,\theta_r,\phi_r)
$$
$R_i$ 是入射光场，$R_r$ 是反射光场。反射场和双向散射表面反射分布函数等价。

获取真实物体反射场的直接方法是获取物体 $R_r(u_r, v_r, \theta_r, \phi_r)$ 的一组光场，对从方向 $(\theta_i,\phi_i)$ 到达表面 A 处 $(u_i, v_i)$ 的入射光束进行密集采样。
A direct method to acquire the reflectance field of a real object would be to acquire a set of light fields of an object $R_r(u_r, v_r, \theta_r, \phi_r)$ for a dense sampling of incident beams of illumination from direction $(\theta_i,\phi_i)$ arriving at the surface A at $(u_i, v_i)$.

我们仅获取入射照明场源于远离 A 的非局部反射场，因此所有光源都照到一点上。 因此，非局部反射场可以表示为 $R'=R'(\theta_i,\phi_i;u_r,v_r,\theta_r,\phi_r)$。
We acquire only non-local reflectance fields where the incident illumination field originates far away from A so that $R_i(u_i,v_i,\theta_i,\phi_i)=R_i(u'_i,v'_i,\theta_i,\phi_i)$ for all $u_i,v_i,u'_i,v'_i$. Thus a non-local reflectance field can be represented as $R'=R'(\theta_i,\phi_i;u_r,v_r,\theta_r,\phi_r)$.

## Re-illuminating Faces

### Constructing Reflectance Functions

对应于穿过该像素的光线的反射函数 $R_{xy}(\theta,\phi)$。
A reflectance function $R_{xy}(\theta,\phi)$ corresponding to the ray through that pixel.

如果我们让具有照明方向 $(\theta,\phi)$ 的图像中 $(x,y)$ 位置的像素值表示为 $L_{\theta,\phi}(x,y)$
If we let the pixel value at location $(x,y)$ in the image with illumination direction$(\theta,\phi)$ be represented as $L_{\theta,\phi}(x,y)$,
$$
R_{xy}(\theta,\phi)=L_{\theta,\phi}(x,y)
$$

### Re-illuminating the Face

由于每个 $R_{xy}(\theta,\phi)$ 表示由于来自方向 $(\theta,\phi)$ 的照明，每个像素 $(x,y)$ 向相机反射的光量，并且由于光是可加的，我们可以在原始光源 $L_i(\theta,\phi)$ 的任意组合下计算人脸 $\hat{L}(x,y)$ 的图像：
Since each $R_{xy}(\theta,\phi)$ represents how much light is reflected toward the camera by pixel $(x,y)$ as a result of illumination from direction $(\theta, \phi)$, and since light is additive, we can compute an image of the face $\hat{L}(x,y)$ under any combination of the original light source $L_i(\theta,\phi)$:
$$
\hat{L}(x,y)=\sum_{\theta,\phi}R_{xy}(\theta,\phi)L_i(\theta,\phi)
$$


由于光源对视球进行密集采样，因此我们可以使用此基础表示任何形式的采样入射照明。在这种情况下，需要考虑每个原始光照方向所覆盖的立体角 $\delta A$： 
Since the light sources densely sample the viewing sphere, we can represent any form of sampled incident illumination using this basis. In this case, it is necessary to consider the solid angle $\delta A$ covered by each of the original illumination directions:
$$
\hat{L}(x,y)=\sum_{\theta,\phi}R_{xy}(\theta,\phi)L_i(\theta,\phi)\delta A(\theta,\phi)
$$
首先，入射光照图被 $\delta A$ 的映射归一化。然后，生成的映射乘以像素的反射函数。最后，将该乘积的像素值相加以计算重新照明的像素值。
First, the map of incident illumination is normalized by the map of $\delta A$. Then, the resulting map is multiplied by the pixel's reflectance function. Finally, the pixel values of this product are summed to compute the re-illuminated pixel value.

### Discussion

由于每个渲染图像也可以表示为原始图像的线性组合，因此保留了所有非漫反射、相互照明、半透明和次表面散射的效果。
Since each rendered image can also be represented as a linear combination of the original images, all of the proper effects of non diffuse reflectance, mutual illumination, translucence, and subsurface scattering are preserved。

反射函数变得混叠的地方是存在自阴影的地方；这样做的结果是，在恶劣的照明条件下，人们会看到有点阶梯状的阴影。
The place where the reflectance functions do become aliased is where there is self-shadowing; the expected result of this is that one would see somewhat stair-stepped shadows in harsh lighting situations.

如果我们需要改变人物服装的颜色，我们可以记录主体两次，一次穿着白色衣服，一次穿着黑色衣服。从第一个图像中减去第二个图像，得到衣服间接照明的图像，然后可以将其着色为任何所需的颜色并添加回黑色衣服拍摄的图像中。
If we need to change the color of the person’s clothing, we can record the subject twice, once wearing white clothing and once with black clothing. Subtracting the second image from the first yields an image of the indirect illumination from the clothing, which can then be tinted to any desired color and added back in to the image taken with the black clothing.

## Changing the Viewpoint

### Investigating Skin Reflectance

我们注意到，从皮肤反射的光可以分解为两个分量：由在空气-油界面处的折射率过渡处立即反射的光组成的镜面反射分量，以及由非朗伯漫反射分量组成的非朗伯漫反射分量。通过空气-油界面传输的光，经过一定数量的次表面散射相互作用后，从油层传输到空气。
we note that the light reflected from the skin can be decomposed into two components: a specular component consisting of light immediately reflected at the index of refraction transition at the air-oil interface, and a non-Lambertian diffuse component consisting of light transmitted through the air-oil interface that, after some number of subsurface scattering interactions, is transmitted from the oil layer to air.

从皮肤镜面反射的光将保持入射光的偏振；然而，从表面下方发出的光将被散射相互作用去极化。利用这一事实，我们可以通过在光源和相机上放置线性偏振器来分离反射分量。
Light which reflects specularly off the skin will maintain the polarization of the incident light; however, light which emerges from below the surface will have been depolarized by scattering interactions. Taking advantage of this fact, we can separate the reflection components by placing linear polarizers on both the light source and the camera.

首先，对于较大的 $\theta_i$ 或 $\theta_r$ 值，镜面反射分量变得更强，并表现出非镜面反射。
First, the specular component becomes much stronger for large values of $\theta_i$ or $\theta_r$ and exhibits off-specular reflection.








