# Chapter 5 Stochastic Path-Tracing Algorithm

## 5.2 Ray-Tracing Set-Up

$$
L_{\text{pixel}}=\int_{\text{image plane}}L(p\to\text{eye})h(p)dp
$$

$p$ 是图像平面上的点，$h(p)$ 是权重函数，以下是伪代码：

```python
def computeImage():
    for pixel in image:
        radiance = 0
        H = integral(h(p))
        for ray in viewing_rays:
            pick uniform sample point p such that h(p) != 0
            construct ray at origin eye, direction p to eye
            radiance = radiance + rad(ray) * h(p)
        radiance = radiance / (number_of_viewing_rays * H)
def rad(ray)
    find closest intersection point x of ray with scene
    computeRadiance(x, eye to x)
```

## 5.3 Simple Stochastic Ray Tracing

### 5.3.1 Truly Random Paths

假设我们想要得到表面上 $x$ 处的辐射：
$$
\begin{align*}
L(x\to\Theta)&=L_e(x\to\Theta)+L_r(x\to\Theta)\\
&=L_e(x\to\Theta)+\int_{\Omega_x}f_r(x,\Psi\to\Theta)L(x\gets\Psi)\cos(N_x,\Psi)d\omega_{\Psi}
\end{align*}
$$
这个积分可以用 Monte Carlo 方法来计算。首先在半球上随机采样 $N$ 个满足概率密度分布 $p(\Psi)$（可能未知可能已知）的方向 $\Psi_i$。估计函数就是：
$$
\langle L_r(x\to\Theta)\rangle=\frac{1}{N}\sum_{i=1}^{n}\frac{f_r(x,\Psi\to\Theta)L(x\gets\Psi_i)\cos(N_x,\Psi_i)}{p(\Psi_i)}
$$
如果我们对场景建模了，那么我们可以知道各点的 BRDF 和对应的 $\cos$ 值。而未知的是入射辐射 $L(x\gets\Psi_i)$，因为：
$$
L(x\gets\Psi_i)=L(r(x,\Psi_i)\to-\Psi_i)
$$
我们就要追踪从 $x$ 点出发，方向为 $\Psi$ 的光线，直到最近的相交点出。于是我们就开始了新一轮的光线追踪。

理论上，选取的概率密度分布 $p(\Psi)$ 与各点的 BRDF 和对应的 $\cos$ 值相近比较好。

### 5.3.2 Russian Roulette

简单随机光线追踪算法中描述的递归路径生成器需要一个停止条件。否则，生成的路径将是无限长的，算法不会停止。 The recursive path generator described in the simple stochastic ray-tracing algorithm needs a stopping condition. Otherwise, the generated paths would be of inﬁnite length and the algorithm would not come to a halt.

添加停止条件时，必须小心不要对最终图像引入任何偏差。从理论上讲，光在场景中会无限反射，我们不能忽略这些长长的光路，这可能非常重要。因此，我们必须找到一种方法来限制路径的长度，但仍然能够获得正确的解决方案。
When adding a stopping condition, one has to be careful not to introduce any bias to the ﬁnal image. Theoretically, light reﬂects inﬁnitely in the scene, and we cannot ignore these light paths of a long length, which might be potentially very important. Thus, we have to ﬁnd a way to limit the length of the paths but still be able to obtain a correct solution.

第一种技术是在固定数量的评估之后切断递归评估。这为需要追踪的光线数量设置了上限，但重要的光传输可能已被忽略。因此，图像是有偏的。
A ﬁrst technique is cutting oﬀ the recursive evaluations after a ﬁxed number of evaluations. This puts an upper bound on the number of rays that need to be traced, but important light transport might have been ignored. Thus, the image will be biased.

另一种方法是使用自适应截止长度。当路径碰到光源时，在光源处发现的辐射仍需要乘以所有先前交点处的所有余弦因子和 BRDF 评估值（并除以所有 PDF 值），然后才能将其添加到最终像素辐射的估计值中。这个累积的乘法因子可以与加长路径一起存储。如果该因子低于某个阈值，则停止递归路径生成。最后的图像还是有偏的。(PlenOctree)
Another approach is to use an adaptive cut-oﬀ length. When a path hits a light source, the radiance found at the light source still needs to be multiplied by all cosine factors and BRDF evaluations (and divided by all PDF values) at all previous intersection points before it can be added to the ﬁnal estimate of the radiance through the pixel. This accumulating multiplication factor can be stored along with the lengthening path. If this factor falls below a certain threshold, recursive path generation is stopped. The ﬁnal image will still be biased. 

假设我们的积分是：
$$
I=\int_{0}^{1}f(x)dx
$$
我们用 $x/P(P\le1)$ 换元：
$$
I=\int_{0}^{P}\frac{1}{P}f(\frac{x}{P})dx
$$
因此积分范围会缩紧。

如果 $f(x)$ 将是另一个递归积分（如渲染方程中的情况），则应用俄罗斯轮盘赌对于每个点是递归停止的结果的概率等于 $\alpha=1-P$ 。$\alpha$ 称为吸收概率。
If $f(x)$ would be another recursive integral (as is the case in the rendering equation), the result of applying Russian roulette is that recursion stops with a probability equal to $\alpha=1-P$ for each evaluation point. $\alpha$ is called the absorption probability.

## 5.4 Direct Illumination

### 5.4.1 Direct and Indirect Illumination

渲染方程
$$
\begin{align*}
L(x\to\Theta)&=L_e(x\to\Theta)+L_r(x\to\Theta)\\
&=L_e(x\to\Theta)+\int_{\Omega_x}f_r(x,\Psi\to\Theta)L(x\gets\Psi)\cos(N_x,\Psi)d\omega_{\Psi}\\
&=L_e(x\to\Theta)+\int_{\Omega_x}f_r(x,\Psi\to\Theta)L(r(x,\Psi)\to-\Psi)\cos(N_x,\Psi)d\omega_{\Psi}\\
\end{align*}
$$
将外来光源 $L(r(x,\Psi)\to-\Psi)$ 也表示为自发光和反光部分，
$$
\begin{align*}
L_r(x\to\Theta)&=\int_{\Omega_x}f_r(x,\Psi\to\Theta)L_{e}(r(x,\Psi)\to-\Psi)\cos(N_x,\Psi)d\omega_{\Psi}\\
&+\int_{\Omega_x}f_r(x,\Psi\to\Theta)L_{r}(r(x,\Psi)\to-\Psi)\cos(N_x,\Psi)d\omega_{\Psi}\\
&=L_{\text{direct}}(x\to\Theta)+L_{\text{indirect}}(x\to\Theta)
\end{align*}
$$
其中：
$$
\begin{align*}
L_{\text{direct}}(x\to\Theta)&=\int_{A}f_r(x,\overrightarrow{xy}\to\Theta)L_{e}(y\to\overrightarrow{yx})G(x,y)V(x,y)dA_{y}\\
\end{align*}
$$
通过在光源区域上生成表面点，我们确信，如果光源对点 $x$ 可见，则将非零贡献添加到 $x$ 的蒙特卡罗估计量中。如果 $x$ 在阴影中，则对像素值的贡献等于 0。
By generating surface points on the area of the light sources, we are sure that, if the light source is visible to the point $x$, a nonzero contribution is added to the Monte Carlo estimator for $x$. If $x$ is in a shadow, the contribution to the estimator equals 0.

阴影光线的总数。增加阴影光线的数量将产生更好的估计。
Total number of shadow rays. Increasing the number of shadow rays will produce a better estimate.

每个光源的阴影光线。根据重要性采样原则，每个光源的阴影光线数应该与光源对 $x$ 的照度的相对贡献成正比。
Shadow rays per light source. According to the principle of importance sampling, the number of shadow rays per light source should be proportional to the relative contribution of the light source to the illumination of $x$.

光源内阴影光线的分布。光源中对直接照射影响较大的部分，应产生更多的阴影光线。例如，大面积光源将具有靠近要照明的表面点的区域。这些区域应该接收更多的阴影光线以获得更准确的直接照明估计量。
Distribution of shadow rays within a light source. More shadow rays should be generated for the parts of the light source that have a greater impact on the direct illumination. For example, large area light sources will have areas that are close to the surface points to be illuminated. These areas should receive more shadow rays to obtain a more accurate estimator for direct illumination.

### 5.4.2 Single Light Source Illumination

Apply Monte Carlo:
$$
\langle L_{\text{direct}}(x\to\Theta)\rangle=\frac{1}{N}\sum_{i=1}^{N}\frac{f_r(x,\overrightarrow{xy_i}\to\Theta)L_e(y_i\to\overrightarrow{y_ix})G(x,y_i)V(x,y_i)}{p(y_i)}
$$
对于在光源上采样的每个表面点 $y_i$，需要计算 $y_i$ 和 $x$ 之间的能量转移。
For each surface point $y_i$ sampled on the light source, the energy transfer between $y_i$ and $x$ needs to be computed.

**Uniform sampling of light source area**

**Uniform sampling of solid angle subtended by light source**

### 5.4.3 Multiple Light Source Illumination

First pick the light source, the generate rays:
$$
\langle L_{\text{direct}}(x\to\Theta)\rangle=\frac{1}{N}\sum_{i=1}^{N}\frac{f_r(x,\overrightarrow{xy_i}\to\Theta)L_e(y_i\to\overrightarrow{y_ix})G(x,y_i)V(x,y_i)}{p_{L}(k_i)p(y_i|k_i)}
$$

## 5.5 Environment Map Illumination

环境映射对单个点周围方向的半球上存在的总照明进行编码。
An environment map encodes the total illumination present on the hemisphere of directions around a single point.

环境映射可以在数学上描述为分段连续函数，其中每个像素对应于环境图中心点 $x$ 周围的小立体角 $\Delta\Omega$。
An environment map can be described mathematically as a stepwisecontinuous function, in which each pixel corresponds to a small solid angle $\Delta\Omega$ around the point $x$ at which the environment map is centered.

### 5.5.1 Capturing Environment Maps

### 5.5.2 Parameterizations

经纬参数化：优点是倾斜角 $\Theta$ 的分布相等，但是在两个极点周围都有一个奇点，在地图中用线表示。其他问题是地图中的像素不占据相等的立体角，并且 $\phi=0$ 和 $\phi=2\pi$ 角度不是连续映射的。
Latitude-longitude parameterization: The advantage is an equal distribution of the tilt angle $\Theta$, but there is a singularity around both poles, which are represented as lines in the map. Additional problems are that the pixels in the map do not occupy equal solid angles and that the $\phi=0$ and $\phi=2\pi$ angles are not mapped continuously next to each other.

Projected-disk parameterization

Concentric-map parameterization

### 5.5.3 Sampling Environment Maps

渲染方程和之前的一模一样，只是对环境映射采样。

## 5.6 Indirect Illumination

### 5.6.1 Uniform Sampling for Indirect Illumination

与使用直接照明方程所做的不同，我们无法将此积分重新表示为较小的积分域。$L_r(r(x,\Psi),-\Psi)$ 具有（在封闭环境中）所有 $(x,\Psi)$ 对的非零值。因此，需要将整个半球视为积分域，并需要进行相应的采样。
Unlike what was done with the direct illumination equation, we cannot reformulate this integral to a smaller integration domain. $L_r(r(x,\Psi),-\Psi)$ has (in a closed environment) a nonzero value for all $(x,\Psi)$ pairs. So, the entire hemisphere needs to be considered as the integration domain and needs to be sampled accordingly.
$$
\langle L_{\text{indirect}}(x\to\Theta)\rangle=\frac{1}{N}\sum_{i=1}^{N}\frac{f_r(x,\Psi_i\to\Theta)L_r(r(x,\Psi_i)\to-\Psi_i)\cos(N,\Psi_i)}{p(\Psi_i)}
$$

### 5.6.2 Importance Sampling for Indirect Illumination

#### Cosine Sampling

采样方向与法线 $N_x$ 周围的余弦成正比可防止在 $\cos(\Psi, N_x)$ 等于 0 的半球水平方向附近采样太多。
Sampling directions proportional to the cosine lobe around the normal $N_x$ prevents too many directions from being sampled near the horizon of the hemisphere where $\cos(\Psi, N_x)$ equals 0.
$$
p(\Psi_i)=\frac{\cos(\Psi,N_x)}{\pi}
$$

$$
\langle L_{\text{indirect}}(x\to\Theta)\rangle=\frac{\pi}{N}\sum_{i=1}^{N}f_r(x,\Psi_i\to\Theta)L_r(r(x,\Psi_i)\to-\Psi_i)
$$

#### BRDF Sampling

理想情况下，应该更频繁地对具有高 BRDF 值的方向进行采样。当存在光泽或高度镜面反射 BRDF 时，对 BRDF 采样是一种很好的降噪技术。
Ideally, directions with a high BRDF value should be sampled more often. BRDF sampling is a good noise-reducing technique when a glossy orhighly specular BRDF is present.

更好的是尝试与 BRDF 和余弦项的乘积成正比地采样。
Even better would be trying to sample proportional to the product of the BRDF and the cosine term.

---

原版 Phong BRDF：
$$
f_r(x, \Psi\rightarrow\Theta)=\underbrace{k_s\frac{(R\cdot\Theta)^n}{N_{x}\cdot\Phi}}_{\text{Specular}}+\underbrace{k_d}_{\text{Diffuse}}
$$
改编：
$$
f_r(x, \Psi\rightarrow\Theta)=\underbrace{k_s\cos^{n}(\Psi,\Theta_s)}_{\text{Specular}}+\underbrace{k_d}_{\text{Diffuse}}
$$
$\Theta_s$ 是关于 $N_x$ 的镜面反射方向。将上式带入渲染方程，可得到镜面反射和漫反射两部分的积分。
$$
\begin{align*}
f_r(x, \Psi\rightarrow\Theta)&=\int_{\Omega_x}L_r(r(x,\Psi)\to-\Psi)k_d\cos(N_x,\Psi)d\omega_{\Psi}\\
&+\int_{\Omega_x}L_r(r(x,\Psi)\to-\Psi)k_s\cos^{n}(\Psi,\Theta_s)\cos(N_x,\Psi)d\omega_{\Psi}\\
\end{align*}
$$
然后从概率为 $q_1$ 的采样漫反射部分，概率为 $q_2$ 的采样镜面反射部分，概率为 $1-q_1-q_2$ 的不采样的三个事件采样。合并后得到：
$$
\langle L_{\text{indirect}}(x\to\Theta)\rangle=\frac{1}{N}\sum_{i=1}^{N}\frac{L_r(r(x,\Psi_i)\to-\Psi_i)(k_d+k_s\cos^{n}(\Psi_i,\Theta_s))\cos(N,\Psi_i)}{q_1p_1(\Psi_i)+q_2p_2(\Psi_i)}
$$
$q_1,q_2$ 可以选取和他们分布有关的取值，即：
$$
q_1=\pi k_d\\
q_2=\frac{2\pi}{n+2}k_s
$$

---

#### Incident Radiance Field Sampling

### 5.6.3 Area Sampling

$$
L_{\text{indirect}}(x\to\Theta)=\int_{A_{\text{Scene}}}L_r(y\to\overrightarrow{yx})f_r(x,\overrightarrow{xy}\to\Theta)G(x,y)V(x,y)dA_y
$$

离散化后为：
$$
\langle L_{\text{indirect}}(x\to\Theta)\rangle=\frac{1}{N}\sum_{i=1}^{N}\frac{L_r(y_i\to\overrightarrow{y_ix})f_r(x,\overrightarrow{xy_i}\to\Theta)G(x,y_i)V(x,y_i)}{p(y_i)}
$$

### 5.6.4 Putting It All Together

**每个像素的观察光线数**：更多的观察光线 $N_p$ 可消除混叠并减少噪点。
**Number of viewing rays per pixel**: A higher number of viewing rays $N_p$ eliminates aliasing and decreases noise.

**直接光照**：从每个点 $x$ 投射的阴影光线 $N_d$ 的总数。如何从每个阴影光线的所有可用光源中选择单个光源。阴影光线在单个光源区域上的分布。
**Direct illumination**: The total number of shadow rays $N_d$ cast from each point $x$. How a single light source is selected from among all the available light sources for each shadow ray. The distribution of the shadow ray over the area of a single light source.

**间接光照**：分布在半球 $\Omega_x$ 上的间接照明光线 $N_i$ 的数量。这些射线在半球的精确分布。俄罗斯轮盘赌的吸收概率以阻止递归。
**Indirect illumination**: Number of indirect illumination rays $N_i$ distributed over the hemisphere $\Omega_x$. Exact distribution of these rays over the hemisphere. Absorption probabilities for Russian roulette in order to stop the recursion.

### 5.6.5 Classic Ray Tracing

## 5.7 Light Tracing
