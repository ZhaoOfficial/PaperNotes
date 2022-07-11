# Eikonal Fields for Refractive Novel-View Synthesis

## 0 Abstract

我们针对 3D 变化的折射率 (IoR) 场进行优化，并根据光 eikonal 传输定律跟踪通过该场的光线朝向所述折射率的空间梯度弯曲。
We optimize for a field of 3D-varying index of refraction (IoR) and trace light through it that bends toward the spatial gradients of said IoR according to the laws of light eikonal transport.

## 1 Introduction


用于学习 3D 场捕捉场景几何的现有解决方案是基于沿直线路径的不透明或透明光传输。然而，在存在透明物体的情况下，光线会弯曲，即它会改变其方向。光路弯曲的精确方式取决于在 IoR 场的空间梯度上运行的某个 eikonal 方程，我们展示了该方程可以通过适当的公式在实践中求解 - 并在学习中微分。
Existing solutions to learn 3D fields capturing scene geometry are based on opaque or transparent light transport along straight paths. In the presence of transparent objects, however, light bends, i.e., it changes its direction. The precise way in which light paths are curved depends on a certain eikonal equation operating on spatial gradients of the IoR field which we show can be solved - and differentiated over in learning - in practice with the appropriate formulation.

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












