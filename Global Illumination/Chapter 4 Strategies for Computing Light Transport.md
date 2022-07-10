# Chapter 4 Strategies for Computing Light Transport

## 4.1 Formulation of the Rendering Equation

Original Rendering Equation:
$$
L(x\to\Theta)=L_e(x\to\Theta)+\int_{\Omega_x}f_r(x,\Psi\to\Theta)L(x\gets\Psi)\cos(N_x,\Psi)d\omega_{\Psi}
$$

### 4.1.1 Exitant Radiance, Integration over the Hemisphere

经典形式的渲染方程中的入射辐射被最近可见点 $y=r(x,\Psi)$ 的等效出射辐射代替。
The incident radiance in the classic form of the rendering equation is replaced by the equivalent exitant radiance at the nearest visible point $y=r(x,\Psi)$.
$$
L(x\to\Theta)=L_e(x\to\Theta)+\int_{\Omega_x}f_r(x,\Psi\to\Theta)L(y\to-\Psi)\cos(N_x,\Psi)d\omega_{\Psi}
$$
![](image/exitant_hemisphere.png)

在基于此公式设计算法时，将在半球上进行积分，作为积分域中每个点的函数评估的一部分，将投射一条射线并定位最近的交点。
When designing an algorithm based on this formulation, one will integrate over the hemisphere, and as part of the function evaluation for each point in the integration domain, a ray will be cast and the nearest intersection point located.

### 4.1.2 Exitant Radiance, Integration over Surfaces

半球方程转换为对所有表面点的积分：
The hemispherical equation is transformed to an integral over all surface points:
$$
L(x\to\Theta)=L_e(x\to\Theta)+\int_{A}f_r(x,\Psi\to\Theta)L(y\to\overrightarrow{yx})V(x,y)G(x,y)dA_{y}\\
G(x,y)=\frac{\cos(N_x,\Psi)\cos(N_y,-\Psi)}{r^2_{xy}}
$$
![](image/exitant_surface.png)

与先前公式的主要区别在于，$x$ 处的入射辐射被视为源于场景中的所有表面，而不仅仅是来自半球 $\Omega_x$。使用此公式的算法将需要检查可见性 $V (x, y)$，这与从 $x$ 向 $\Theta$ 方向投射射线略有不同。
The main difference with the previous formulation is that incident radiance at $x$i s seen as originating at all surfaces in the scene and not only at the hemisphere $\Omega_x$. Algorithms using this formulation will need to check the visibility $V (x, y)$, which is slightly different than casting a ray from $x$ in a direction $\Theta$.

### 4.1.3 Incident Radiance, Integration over the Hemisphere

Take inverse of 4.1.1, $y=r(x,\Theta)$.
$$
L(x\gets\Theta)=L_e(x\gets\Theta)+\int_{\Omega_x}f_r(y,\Psi\to-\Theta)L(y\gets\Psi)\cos(N_y,\Psi)d\omega_{\Psi}
$$
![](image/incident_hemisphere.png)

### 4.1.4 Incident Radiance, Integration over Surfaces

Take inverse of 4.1.2, $y=r(x,\Theta)$.
$$
L(x\gets\Theta)=L_e(x\gets\Theta)+\int_{A}f_r(y,\overrightarrow{yz}\to-\Theta)L(y\gets\overrightarrow{yz})V(y,z)G(y,z)dA_z
$$
![](image/incident_surface.png)

### 4.1.5 Radiant Flux

大多数全局照明算法旨在计算一些选定的点和方向的集合中的平均辐射。计算一组平均辐射值的一种可能方法是计算该组的辐射通量。假设辐射在整个集合中缓慢变化，可以通过将通量除以集合的总面积和总立体角来获得平均辐射值。
Most global illumination algorithms are aimed at computing the average radiance over some chosen sets of points and directions. One possible way of computing the average radiance value over a set is to compute the radiant flux over that set. By assuming the radiance to be changing slowly over the set, an average radiance value can be obtained by dividing the flux by the total area and total solid angle of the set.











