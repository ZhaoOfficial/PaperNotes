# Eikonal Fields for Refractive Novel-View Synthesis

## 0 Abstract

我们针对 3D 变化的折射率 (IoR) 场进行优化，并根据光 eikonal 传输定律跟踪通过该场的光线朝向所述折射率的空间梯度弯曲。
We optimize for a field of 3D-varying index of refraction (IoR) and trace light through it that bends toward the spatial gradients of said IoR according to the laws of light eikonal transport.

## 1 Introduction



## 2 Related Work



## 3 Light Transport ODE Zoo

完整的模型处理折射和非折射场景，但仅适用于文献中的合成场景。发射吸收模型可用于逆渲染，但不包括折射。我们的 eikonal 模型与发射吸收模型相结合，可以在实际的逆渲染中处理折射透明度。
The complete one handles refractive and non-refractive scenes, but was only applied to synthetic scenes in the literature. The emission-absorption one can be used for inverse rendering, but excludes refraction. Our eikonal one, in combination with the emission-absorption one, can handle refractive transparency in practical inverse rendering.

|      Symbol       |               Description                |
| :---------------: | :--------------------------------------: |
| $s\in[0,+\infty[$ | the distance along a (curved) light path |
|      $n(s)$       |         the index of refraction          |
|     $\sigma$      |          extinction coefficient          |
|    $q/\sigma$     |             source function              |
|                   |                                          |



### 3.1 Complete model

Radiative transfer equation:
$$
n(s)^2\frac{\mathrm{d}(L/n^2)}{\mathrm{d}s}=-\sigma(s)L(s)+q(s)
$$












