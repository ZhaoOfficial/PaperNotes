# Radon Transformation

#### 问题描述

![problem](D:\Paper\Note Image\radon.png)

左侧的线状发射源可以均匀发射垂直于发射源的射线，射线没有折射和反射地经过物体且有不同程度的衰减，到达感应器后可以检测射线强度。假设物体上每一点对应射线的衰减程度为函数 $f(x,y)$。根据感应器感应的射线强度我们可以得到某个方向上射线的衰减程度。我们的目的是通过不同方向检测的衰减程度得到每一点的函数值。

#### 问题抽象

直线 $L$ 穿过物体得到的强度就是函数 $f(x,y)$ 在 $L$ 上的线积分
$$
\mathcal{R}_L=\int_Lf(x, y)ds
$$
直线 $L$ 的表示。设直线法向量为 $\vec{n}=(\cos\theta, \sin\theta)$。
$$
\begin{align*}
y&=kx+b\\
y&=x\tan\phi+b\quad(\phi=\theta+\frac{\pi}{2})\\
y\cos\phi&=x\sin\phi+b\cos\phi\\
x\sin(\theta+\frac{\pi}{2})-y\cos(\theta+\frac{\pi}{2})&=-b\cos(\theta+\frac{\pi}{2})\\
x\cos\theta+y\sin\theta&=b\sin\theta\\
x\cos\theta+y\sin\theta&=\rho\\
\end{align*}
$$
故线积分可以表示为二重积分，这就是 Radon 变换
$$
\begin{align*}
\mathcal{R}_L&=\int_Lf(x, y)ds\\
\mathcal{R}(\theta,\rho)&=\int_{(\theta,\rho)}f(x, y)ds\\
\mathcal{R}(\theta,\rho)&=\int_{x\cos\theta+y\sin\theta=\rho}f(x, y)ds\\
\mathcal{R}(\theta,\rho)&=\iint f(x, y)\cdot\delta(\rho-x\cos\theta-y\sin\theta)dxdy\\
\end{align*}
$$
然后我们对 $\rho$ 做 Fourier 变换
$$
\begin{align*}
&\qquad\mathcal{F}\{\mathcal{R}(\theta,\rho)\}\\
&=\int[\iint f(x, y)\delta(\rho-x\cos\theta-y\sin\theta)dxdy]e^{-j\omega\rho}d\rho\\
&=\int[\iint f(x, y)\delta(\rho-x\cos\theta-y\sin\theta)dxdy]e^{-j\omega(\rho-x\cos\theta-y\sin\theta+x\cos\theta+y\sin\theta)}d\rho\\
&=\iint f(x, y)e^{-j\omega(x\cos\theta+y\sin\theta)}[\int\delta(\rho-x\cos\theta-y\sin\theta)e^{-j\omega(\rho-x\cos\theta-y\sin\theta)}d\rho]dxdy\\
&=\hat{\delta}(\omega)\iint f(x, y)e^{-j\omega(x\cos\theta+y\sin\theta)}\\
&=\hat{\delta}(\omega)\hat{f}(\omega\cos\theta,\omega\sin\theta)\\
&=\hat{f}(\omega\cos\theta,\omega\sin\theta)
\end{align*}
$$
即，二元函数的 Radon 变换的 Fourier 变换是函数的二维 Fourier 变换。

二维 Fourier 变换公式
$$
F(u,v)=\int^{\infty}_{-\infty}\int^{\infty}_{-\infty}f(x,y)e^{-j(ux+vy)}dxdy
$$
可以得到，对 $\theta$ 方向的检测结果进行 Fourier 变换得到的数据填充到过原点的 $\theta$ 角度的那条线上，最后进行反傅立叶变换即可得到原函数形式。

### 中心切片定理 Central Slice Theorem

















