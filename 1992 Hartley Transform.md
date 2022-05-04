# Appendix A Hartley Transform

## A.1 连续 Hartley 变换 Continuous Hartley Transform

Hartley 变换的公式如下：
$$
\text{cas}\ x=\cos x+\sin x
$$

$$
\begin{align*}
H(\omega)&=\frac{1}{\sqrt{2\pi}}\int_{-\infty}^{\infty}f(t)\text{cas}(\omega t)dt\\
&=\frac{1}{\sqrt{2\pi}}\int_{-\infty}^{\infty}f(t)\cos(\omega t)dt+\frac{1}{\sqrt{2\pi}}\int_{-\infty}^{\infty}f(t)\sin(\omega t)dt\tag{1}\\
f(t)&=\frac{1}{\sqrt{2\pi}}\int_{-\infty}^{\infty}H(\omega)\text{cas}(\omega t)d\omega\\
&=\frac{1}{\sqrt{2\pi}}\int_{-\infty}^{\infty}H(\omega)\cos(\omega t)d\omega+\frac{1}{\sqrt{2\pi}}\int_{-\infty}^{\infty}H(\omega)\sin(\omega t)d\omega\tag{2}
\end{align*}\\
$$
与 Fourier 变换的公式比较：
$$
e^{jx}=\cos x+j\sin x
$$

$$
\begin{align*}
F(\omega)&=\frac{1}{\sqrt{2\pi}}\int_{-\infty}^{\infty}f(t)e^{-j\omega t}dt\\
&=\frac{1}{\sqrt{2\pi}}\int_{-\infty}^{\infty}f(t)\cos(\omega t)dt-j\frac{1}{\sqrt{2\pi}}\int_{-\infty}^{\infty}f(t)\sin(\omega t)dt\\
f(t)&=\frac{1}{\sqrt{2\pi}}\int_{-\infty}^{\infty}F(\omega)e^{j\omega t}d\omega\\
&=\frac{1}{\sqrt{2\pi}}\int_{-\infty}^{\infty}F(\omega)\cos(\omega t)d\omega+j\frac{1}{\sqrt{2\pi}}\int_{-\infty}^{\infty}F(\omega)\sin(\omega t)d\omega
\end{align*}\\
$$
可以发现

1. 对于任意实信号 $f(t)$，Hartley 变换的结果是实数，而 Fourier 变换的结果可能是复数。
2. Hartley 正变换和逆变换的式子结构完全一致。

---

### $\text{cas}$ 函数正交性证明 Orthogonal Base

$$
\begin{align*}
&\quad\ \int_{-\pi}^{\pi}\text{cas}(mx)\text{cas}(nx)dx\\
&=\int_{-\pi}^{\pi}[\cos(mx)+\sin(mx)][\cos(nx)+\sin(nx)]dx\\
&=\int_{-\pi}^{\pi}\cos(mx)\cos(nx)+\sin(mx)\cos(nx)+\cos(mx)\sin(nx)+\sin(mx)\sin(nx)dx\\
&=2\pi\delta_{mn}
\end{align*}
$$

---

### Hartley 变换和 Fourier 变换的关系

定义：
$$
\begin{align*}
H_e(\omega)&=\frac{1}{2}[H(\omega)+H(-\omega)]\\
H_o(\omega)&=\frac{1}{2}[H(\omega)-H(-\omega)]
\end{align*}
$$
那么：
$$
\begin{align*}
H_e(\omega)&=\mathcal Re\{F(\omega)\}\\
H_o(\omega)&=-\mathcal Im\{F(\omega)\}\\
F(\omega)&=H_e(\omega)-jH_o(\omega)\\
H(\omega)&=\mathcal Re\{F(\omega)\}-\mathcal Im\{F(\omega)\}
\end{align*}
$$

### 性质 Property

|      |                            $f(t)$                            |                         $H(\omega)$                          |
| :--: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| odd  | $\frac{1}{\sqrt{2\pi}}\int_{-\infty}^{\infty}H(\omega)\sin(\omega t)d\omega$ | $\frac{1}{\sqrt{2\pi}}\int_{-\infty}^{\infty}f(t)\sin(\omega t)dt$ |
| even | $\frac{1}{\sqrt{2\pi}}\int_{-\infty}^{\infty}H(\omega)\cos(\omega t)d\omega$ | $\frac{1}{\sqrt{2\pi}}\int_{-\infty}^{\infty}f(t)\cos(\omega t)dt$ |

令：
$$
H(\omega)=\mathcal H\{f(t)\}
$$
[线性性质 Linearity]

[时移性质 Time shifting]
$$
\begin{align*}
\mathcal H(f(t-t_0))&=\mathcal Re\{e^{-j\omega t_0}F(\omega)\}-\mathcal Im\{e^{-j\omega t_0}F(\omega)\}\\
&=\mathcal Re\{[\cos\omega t_0-j\sin\omega t_0][H_e(\omega)-jH_o(\omega)]\}\\
&\quad-\mathcal Im\{[\cos\omega t_0-j\sin\omega t_0][H_e(\omega)-jH_o(\omega)]\}\\
&=\cos\omega t_0[H_e(\omega)+H_o(\omega)]+\sin\omega t_0[H_e(\omega)-H_o(\omega)]
\end{align*}
$$
[时间伸缩 Time Scaling]
$$
\begin{align*}
\mathcal H(f(at))&=\frac{1}{|a|}\mathcal Re\{F(\frac{\omega}{a})\}-\frac{1}{|a|}\mathcal Im\{F(\frac{\omega}{a})\}\\
&=\frac{1}{|a|}H(\frac{\omega}{a})
\end{align*}
$$
[卷积性质 Convolution]
$$
\begin{align*}
\mathcal H\{f(t)*g(t)\}&=\mathcal Re\{F_1(\omega)F_2(\omega)\}-\mathcal Im\{F_1(\omega)F_2(\omega)\}\\
&=\mathcal Re\{[H_{1e}(\omega)-jH_{1o}(\omega)][H_{2e}(\omega)-jH_{2o}(\omega)]\}\\
&\quad-\mathcal Im\{[H_{1e}(\omega)-jH_{1o}(\omega)][H_{2e}(\omega)-jH_{2o}(\omega)]\}\\
&=H_{1e}(\omega)H_{2e}(\omega)-H_{1o}(\omega)H_{2o}(\omega)+H_{1e}(\omega)H_{2o}(\omega)+H_{1o}(\omega)H_{2e}(\omega)\\
&=H_{1e}(\omega)H_{2}(\omega)+H_{1o}(\omega)H_{2}(-\omega)\\
&=\frac{1}{2}[H_1(\omega)+H_1(-\omega)]H_{2}(\omega)+\frac{1}{2}[H_1(\omega)-H_1(-\omega)]H_{2}(-\omega)\\
&=\frac{1}{2}[H_1(\omega)H_{2}(\omega)+H_1(-\omega)H_{2}(\omega)+H_1(\omega)H_{2}(-\omega)-H_1(-\omega)H_{2}(-\omega)]
\end{align*}
$$
特别的：
$$
\begin{align*}
\mathcal H\{f(t)*g(t)\}&=\begin{cases}
H_1(\omega)H_{2}(\omega)&\text{one of }f(t),g(t)\text{ is even}\\
H_1(-\omega)H_{2}(\omega)&f(t)\text{ is odd}\\
H_1(\omega)H_{2}(-\omega)&g(t)\text{ is odd}\\
H_1(-\omega)H_{2}(-\omega)&f(t),g(t)\text{ a}
\end{cases}
\end{align*}
$$
[相关性质 Correlation]
$$
\begin{align*}
\mathcal H\{f(t)**g(t)\}&=\mathcal Re\{F_1(\omega)F_2(\omega)\}-\mathcal Im\{F_1(\omega)F_2(\omega)\}\\
&=\frac{1}{2}[H_1(\omega)H_{2}(\omega)+H_1(-\omega)H_{2}(\omega)-H_1(\omega)H_{2}(-\omega)+H_1(-\omega)H_{2}(-\omega)]
\end{align*}
$$
[微分性质 Differential]
$$
\begin{align*}
\mathcal H\{\frac{d}{dt}f(t)\}&=\mathcal Re\{j\omega [H_e(\omega)-jH_o(\omega)]\}-\mathcal Im\{j\omega [H_e(\omega)-jH_o(\omega)]\}\\
&=\omega H_o(\omega)-\omega H_e(\omega)\\
&=-\omega H(-\omega)\\
\mathcal H\{\frac{d^2}{dt^2}f(t)\}&=-\omega^2 H(\omega)\\
\end{align*}
$$

[能量性质 Energy]

## A.2 离散 Hartley 变换 Discrete Hartley Transform

`TODO`

## A.3 2D Hartley 变换 2D Hartley transform

### 第一种形式 The first form

$$
H(u, v)=\frac{1}{2\pi}\int_{-\infty}^{\infty}\int_{-\infty}^{\infty}f(x, y)\text{cas}(ux+vy)\ dxdy\\
f(x, y)=\frac{1}{2\pi}\int_{-\infty}^{\infty}\int_{-\infty}^{\infty}H(u,v)\text{cas}(ux+vy)\ dudv\\
$$

类似一维情况：
$$
\begin{align*}
H_e(u, v)&=\frac{1}{2}[H(u, v)+H(-u, -v)]\\
H_o(u, v)&=\frac{1}{2}[H(u, v)-H(-u, -v)]\\
F(u, v)&=H_e(u, v)-jH_o(u, v)\\
H(u, v)&=\mathcal Re\{F(u, v)\}-\mathcal Im\{F(u, v)\}
\end{align*}
$$
[卷积性质 Convolution]
$$
\begin{align*}
&\quad\ \mathcal H\{f(x, y)*g(x, y)\}\\
&=\mathcal Re\{F_1(u, v)F_2(u, v)\}-\mathcal Im\{F_1(u, v)F_2(u, v)\}\\
&=\frac{1}{2}[H_1(u, v)H_{2}(u, v)+H_1(-u, -v)H_{2}(u, v)\\&\quad+H_1(u, v)H_{2}(-u, -v)-H_1(-u, -v)H_{2}(-u, -v)]
\end{align*}
$$
对于 Fourier 变换来说，它的基 $e^{-j(ux+vy)}=e^{-jux}e^{-jvy}$ 可以分离，而上述形式的 $\text{cas}(ux+vy)\ne\text{cas}(ux)\text{cas}(vy)$，因此导出下述形式。

### 第二种形式 The second form

$$
H(u, v)=\frac{1}{2\pi}\int_{-\infty}^{\infty}\int_{-\infty}^{\infty}f(x, y)\text{cas}(ux)\text{cas}(vy)\ dxdy\\
$$

因此它满足：
$$
\begin{align*}
H(u, v)&=\frac{1}{2\pi}\int_{-\infty}^{\infty}\int_{-\infty}^{\infty}f(x, y)[\cos(ux-vy)+\sin(ux+vy)]\ dxdy\\
&=\frac{1}{2\pi}\int_{-\infty}^{\infty}\int_{-\infty}^{\infty}f(x, y)[\mathcal{R}e\{\cos(ux-vy)-j\sin(ux-vy)\}\\
&\quad-\mathcal{I}m\{\cos(ux+vy)-j\sin(ux+vy)\}]\ dxdy\\
&=\mathcal{R}e\{F(u, -v)\}-\mathcal{I}m\{F(u,v)\}\\
F(u, v)&=\frac{1}{2\pi}\int_{-\infty}^{\infty}\int_{-\infty}^{\infty}f(x, y)e^{-j(ux+vy)}dxdy\\
&=\frac{1}{2\pi}\int_{-\infty}^{\infty}\int_{-\infty}^{\infty}f(x, y)[\cos(ux+vy)-j\sin(ux+vy)]dxdy\\
&=\frac{1}{2\pi}\int_{-\infty}^{\infty}\int_{-\infty}^{\infty}f(x, y)[\frac{1}{2}\text{cas}(ux)\text{cas}(-vy)+\frac{1}{2}\text{cas}(-ux)\text{cas}(vy)\\
&\quad-\frac{j}{2}\text{cas}(ux)\text{cas}(vy)-\frac{j}{2}\text{cas}(-ux)\text{cas}(-vy)]dxdy\\
&=\frac{1}{2}[H(u,-v)+H(-u,v)]-\frac{j}{2}[H(u,v)-H(-u,-v)]
\end{align*}
$$
[卷积性质 Convolution]
$$
\begin{align*}
&\quad\ \mathcal H\{f(x, y)*g(x, y)\}\\
&=\frac{1}{4}\mathcal Re\{[[H_1(u,-v)+H_1(-u,v)]-j[H_1(u,v)-H_1(-u,-v)]]\\
&\quad\ [[H_2(u,-v)+H_2(-u,v)]-j[H_2(u,v)-H_2(-u,-v)]]\}\\
&\quad-\frac{1}{4}\mathcal Im\{[[H_1(u,-v)+H_1(-u,v)]-j[H_1(u,v)-H_1(-u,-v)]]\\
&\quad\ [[H_2(u,-v)+H_2(-u,v)]-j[H_2(u,v)-H_2(-u,-v)]]\}\\
&=\frac{1}{4}[H_1(u, -v)H_{2}(u, -v)+H_1(-u, v)H_{2}(u, -v)\\
&\quad+H_1(u, -v)H_{2}(-u, v)+H_1(-u, v)H_{2}(-u, v)\\
&\quad-H_1(u, v)H_{2}(u, v)+H_1(u, v)H_{2}(-u, -v)\\
&\quad-H_1(-u, -v)H_{2}(u, v)+H_1(-u, -v)H_{2}(-u, -v)\\
&\quad+H_1(u, -v)H_{2}(u, v)+H_1(-u, v)H_{2}(u, v)\\
&\quad-H_1(u, -v)H_{2}(-u, -v)-H_1(-u, v)H_{2}(-u, -v)\\
&\quad+H_1(u, v)H_{2}(u, -v)+H_1(u, v)H_{2}(-u, v)\\
&\quad-H_1(-u, -v)H_{2}(u, -v)-H_1(-u, -v)H_{2}(u, -v)
]
\end{align*}
$$
卷积和相关计算，如果利用 Hartley 变换，式子右端要进行 4 个乘积运算和 3 个求和运算，而 Fourier 变换只需要一个乘积运算，这样 Hartley 变换效率会较低。
