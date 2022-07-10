# Appendix B Hemispherical Coordinates

## B.1 Hemispherical Coordinates

Note that we are in the hemispherical coordinate, which means that $\phi\in[0,2\pi],\theta\in[0,\pi/2]$. 
$$
\begin{align*}
x&=r\cos\phi\sin\theta\\
y&=r\sin\phi\sin\theta\\
z&=r\cos\theta\\
r&=\sqrt{x^2+y^2+z^2}\\
\phi&=\arctan(y/x)\\
\theta&=\arccos(y)
\end{align*}
$$

## B.2 Solid Angle

一个有限立体角 $\Omega$ 定义为半球上的一块区域的面积除以半球半径的平方。单位为 (sr)。
A finite solid angle $\Omega$ subtended by an area on the hemisphere is defined as the total area divided by the squared radius of the hemisphere. Solid angles are dimensionless but are expressed in steradians (sr). 
$$
\Omega=\frac{A}{r^2}
$$
为了计算空间中任意曲面或物体所围成的立体角，我们首先将曲面或物体投影到半球上，然后计算投影的立体角。
To compute the solid angle subtended by an arbitrary surface or object in space, we first project the surface or object on the hemisphere and compute the solid angle of the projection.
对于面积微元：
For small surfaces:
$$
d\omega=\frac{dA\cos\alpha}{l^2}
$$
其中 $\alpha$ 是面积微元的法向量和面积微元到球心连线方向的夹角，$l$ 是连线距离。

## B.3 Integrating over the Hemisphere

在 $\Theta=(\theta,\phi)$ 附近的立体角微元：
A differential solid angle, centered around direction $\Theta$:
$$
d\omega_{\Theta}=\sin\theta d\theta d\phi
$$
半球积分为：
$$
\int_{\Omega}f(\Theta)d\omega_{\Theta}=\int^{2\pi}_{0}\int^{\pi/2}_{0}f(\theta,\phi)\sin\theta d\theta d\phi
$$
例子：
$$
\begin{align*}
\int_{\Omega}\cos^{N}(\Theta,N_{x})d\omega_{\Theta}&=\int_{0}^{2\pi}\int_{0}^{\pi/2}\cos^{N}\theta\sin\theta d\theta d\phi\\
&=\int _{0}^{2\pi}\left[-\frac{\cos^{N+1}\theta}{N+1}\right]\mid_{0}^{\pi/2}d\phi\\
&=\int _{0}^{2\pi}\frac{1}{N+1}d\phi\\
&=\frac{2\pi}{N+1}
\end{align*}
$$

## B.4 Hemisphere-Area Transformation

将半球上的积分表示为从 $x$ 看到的可见表面上的积分。
Express an integral over the hemisphere as an integral over visible surfaces seen from $x$.
$$
d\omega_{\Theta}=\frac{\cos\theta_ydA_y}{r_{xy}^2}
$$
其中 $\theta_y$ 是表面法向 $N_y$ 和视线的连线的夹角，$A_y$ 是面积微元，$r_{xy}^2$ 是连线距离。

因此将对立体角的积分转化到了对面积的积分：
$$
\int_{\Omega}f(\Theta)d\omega_{\Theta}=\int_Af(y)\frac{\cos\theta_y}{r_{xy}^2}dA_y
$$
