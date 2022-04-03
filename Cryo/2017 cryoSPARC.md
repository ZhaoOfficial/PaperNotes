# SGD

## 目标函数 Objective Function

给定 $N$ 张图片求 $K$ 个 3D 结构的最大后验。
$$
\begin{align*}
&\quad\arg\max_{\mathbf V}\log P(\mathbf V\mid\mathbf X)\tag{1}\\
&=\arg\max_{\mathbf V}[\log P(\mathbf X\mid\mathbf V)+\log P(\mathbf V)]\tag{2}\\
&=\arg\max_{\mathbf V}[\sum_{i=1}^{N}\log P(X_i\mid\mathbf V)+\sum_{k=1}^{K}\log P(V_k)]\tag{3}\\
&=\arg\max_{\mathbf V}f(\mathbf V)
\end{align*}
$$
$f(\mathbf V)$ 是在给定观测图像下异构结构的对数后验概率。(2) 式中的先验结构信息可以设置，而且假设结构信息之间相互独立。
$$
f(\mathbf V)=\sum_{i=1}^{N}\log P(X_i\mid\mathbf V)+\sum_{k=1}^{K}\log P(V_k)
$$
其中
$$
P(X_i\mid\mathbf V)=\sum_{k=1}^{K}\pi_kP(X_i\mid V_k)\equiv U_i\tag{4}
$$
是在给定所有 3D 结构的条件下观察到图像 $X_i$ 的似然，每个结构都带有观察到图像 $X_i$ 的可能性，故带有权重，一般认为是均匀分布的。故令 $\pi_k=1/K$。
$$
P(X_i\mid V_k)=\int P(X_i\mid\phi,V_k)P(\phi)d\phi\tag{5}
$$
是在特定的 3D 结构下观察到图像 $X_i$ 的似然，全概率了 3D 结构的角度。其中角度的先验一般认为是均匀分布的，故令 $p(\phi)=1$。而 $P(X_i\mid V_j)$ 的获得稍后给出。

所以
$$
f(\mathbf V)=\sum_{i=1}^{N}\log [\sum_{k=1}^{K}\pi_k\int P(X_i\mid\phi,V_k)P(\phi)d\phi]+\sum_{k=1}^{K}\log P(V_k)\tag{6}
$$

## 梯度 Gradient

目标函数的梯度
$$
\frac{\partial f}{\partial V_k}=\sum_{i=1}^{N}\frac{1}{P(X_i\mid\mathbf V)}\pi_k\int \frac{\partial}{\partial V_k}P(X_i\mid\phi,V_k)P(\phi)d\phi+\frac{\partial}{\partial V_k}\log P(V_k)\tag{7}
$$
近似梯度
$$
\frac{\partial f}{\partial V_k}\approx G_k=\frac{N}{M}\sum_{i\in\mathbf M}\frac{1}{P(X_i\mid\mathbf V)}\pi_k\int \frac{\partial}{\partial V_k}P(X_i\mid\phi,V_k)P(\phi)d\phi+\frac{\partial}{\partial V_k}\log P(V_k)\tag{8}
$$
其中 $\mathbf M$ 是每次选取的图像的集合，内容和大小都可以变化。$N/M$ 是修正到 $N$ 个图像的梯度。

## 矩更新 SGD SGD update rule with momentum

SGD 维护了现有的结构 $V_k$ 的状态，记第 $n$ 次迭代是 $V_{k}^{(n)}$，更新为 $dV_{k}^{(n)}$，迭代公式为
$$
dV_{k}^{(n)}=(\mu)dV_{k}^{(n-1)}+(1-\mu)(\eta_{k})G_k^{(n)}\\
V_{k}^{(n+1)}=V_{k}^{(n)}+dV_{k}^{(n)}
$$
其中 $\mu$ 是比例系数，固定为 $0.9$，$\eta_k$ 是步长，每次迭代可变。

## 步长 Step-sizes

$\eta_k$ 和 $f(\mathbf V)$ 的二阶曲率。原则上 $f(\mathbf V)$ 是非凸光滑可微函数，可以计算 Hessian 矩阵，但是计算起来消耗太大。

首先，Hessian 矩阵在 Fourier 空间是对角化的。Fourier 切片定理。

其次，用一个替代的凸二次函数作为 $f(\mathbf V)$ 某点附近的近似，并计算该函数的二阶导。

Fourier 空间中所有维度的最大曲率直接用作每个结构的步长的倒数。 
$$
\frac{1}{\eta_k}=\|\sum_{i\in\mathbf M}\pi_{k}\int P(\phi\mid X_i,V_k)\mathbf P_{\phi}^{T}\frac{C_i^2}{\sigma^2}d\phi\|_{\infty}
$$
$\mathbf P$ 是 $\phi$ 的投影算符，$C_i$ 是图像的 CTF 值，$\sigma^2$ 是噪声的方差。

## 样本数 Mini-batch sizes

太小的样本数使得更新变慢，一般一开始设置 $30\times K$ 的大小，随后分辨率到 20 $\AA$ 时提高到 $100\times K$。

## 噪声模型 Noise model

噪声是动态计算得到的
$$
\sigma^2_{l,k}=\frac{w_{l,k}\bar{\sigma}^2_{l,k}+\tilde{w}\tilde{\sigma}^2+\hat{w}_k\hat{\sigma}^2}{w_{l,k}+\tilde{w}+\hat{w}_k}
$$
$w_{l,k}=\sum_{i=1}^{k}\gamma^{k-1}M_iC_{l,i}^2$，$M_i$ 是第 $i$ 次迭代的样本数，$C_{l,i}^2$ 是第 $i$ 次迭代时频率为 $l$ 平均平方 CTF 值。

$\bar{\sigma}^2_{l,k}=\sum_{i=1}^{k}\gamma^{k-i}M_ie^2_{l,i}$，$e^2_{l,i}$ 是第 $i$ 次迭代时频率为 $l$ 平均误差。

$\gamma=0.9999$ 为衰减率。

$\tilde{w}=50$ 为先验权重。

$\tilde{\sigma}^2$ 是从图像角落得到的初始白噪声的方差。

$\hat{w}_k=2500\gamma^k$ 是膨胀的初始噪声权重。

$\hat{\sigma}^2=8\tilde{\sigma}^2$ 是膨胀的初始噪声的方差。

## 随机初始化 Random initialization

# Branch and Bound Search

姿态 $\phi$ 被分解为 3D 旋转和 2D 平移 $r,t$，在 Fourier 域中
$$
P(X_i\mid\phi,V_k)=P(X_i\mid r,t,V_k)=\frac{1}{Z}\exp(\sum_{l}\frac{-1}{2\sigma^{2}_{l}}|C_lY_l(r)-S_l(t)X_l|^2)
$$
这里，$X_i,V_k$ 都在 Fourier 域中。

$Y_l(r)=\Theta_l(r)V$ 表示了结构 $V$ 在旋转为 $r$ 时候频率为 $l$ 的系数。

$\Theta_l(r)$ 是线性投影算子，3D 切片操作。

$C_l$ 是 CTF 函数。

$S_l(t)$ 是 2D 平移 $t$ 对应的 2D 相移。

$\sigma_l^2$ 是 Gaussian 误差。

$Z$ 可以被忽略。

## 直觉 Intuition behind a lower bound

构建这个易于计算且含义丰富的下界函数是个挑战。

低频部分对于整个误差函数的贡献远大于高频部分。所以下界函数选取低频部分。

## 导出 Derivation of a lower bound

$$
E(r,t)=\sum_{\|l\|\le L}\frac{1}{2}|C_lY_l(r)-S_l(t)X_l|^2+\sum_{\|l\|> L}\frac{1}{2}|C_lY_l(r)-S_l(t)X_l|^2
$$

$L$ 一开始设置为比较小的值，随后每次迭代都增加直到达到 Nyquist 频率。

把高频部分分为三部分
$$
\begin{align*}
B(r, t) &= \sum_{\|l\|> L}\frac{1}{2}|C_lY_l(r)-S_l(t)X_l|^2\\
&=\sum_{\|l\|> L}\frac{1}{2}|X_l|^2+\sum_{\|l\|> L}\frac{1}{2}C_l^2|Y_l(r)|^2-\sum_{\|l\|> L}C_l\mathfrak R_e[Y_l^*(r)S_l(t)X_l]
\end{align*}
$$
$|S_l(t)|=1$。

CTF 函数是实值函数。

第一项三图像高频部分的能量，第二项是高频切片的能量，第三部分是切片和图像的相关性。

图像的组成
$$
X_l=C_l\tilde{X}_l+\epsilon_l\\
\epsilon_l\thicksim\mathcal{C}\mathcal{N}(0, \sigma_l^2/2)
$$
$\tilde{X}_l$ 是真实值。

$\epsilon_l$ 是满足复正态分布的噪声。

带入第三项
$$
\sum_{\|l\|> L}C_l^2\mathfrak R_e[Y_l^*(r)S_l(t)\tilde{X}_l]+\sum_{\|l\|> L}C_l\mathfrak R_e[Y_l^*(r)S_l(t)\epsilon_l]\\
\le\sum_{\|l\|> L}C_l^2|Y_l(r)||\tilde{X}_l|+\sum_{\|l\|> L}C_l\mathfrak R_e[Y_l^*(r)S_l(t)\epsilon_l]\\
$$
后一项可以化简为
$$
\begin{align*}
H&=\sum_{\|l\|> L}C_l\mathfrak R_e[Y_l^*(r)S_l(t)\epsilon_l]\\
&=\sum_{\|l\|> L}C_l\mathfrak R_e[Y_l^*(r)\epsilon_l]\\
&=\sum_{\|l\|> L}C_l\mathfrak R_e[Y_l^*(r)\mathcal{C}\mathcal{N}(0, \sigma_l^2/2)]\\
&=\sum_{\|l\|> L}C_l\mathfrak R_e[\mathcal{C}\mathcal{N}(0, \sigma_l^2|Y_l(r)|^2/2)]\\
&=\sum_{\|l\|> L}\mathcal{N}(0, \sigma_l^2C_l^2|Y_l(r)|^2/2)\\
&=\mathcal{N}(0, \sum_{\|l\|> L}\sigma_l^2C_l^2|Y_l(r)|^2/2)\\
\end{align*}
$$
带入二三项
$$
\begin{align*}
B_2-B_3&=\sum_{\|l\|> L}\frac{1}{2}C_l^2|Y_l(r)|^2-\sum_{\|l\|> L}C_l\mathfrak R_e[Y_l^*(r)S_l(t)X_l]\\
&\ge\sum_{\|l\|> L}\frac{1}{2}C_l^2|Y_l(r)|^2-\sum_{\|l\|> L}C_l^2|Y_l(r)||\tilde{X}_l|-H\\
&=\sum_{\|l\|> L}\frac{1}{2}[C_l^2|Y_l(r)|^2-2C_l^2|Y_l(r)||\tilde{X}_l|]-H\\
\end{align*}
$$
其中 $Q=\sum_{\|l\|> L}\frac{1}{2}[C_l^2|Y_l(r)|^2-2C_l^2|Y_l(r)||\tilde{X}_l|]$ 关于 $Y_l(r)$ 的正定二次型。
$$
\begin{align*}
Q&=\sum_{\|l\|> L}\frac{1}{2}[C_l^2(|Y_l(r)|^2-2|Y_l(r)||\tilde{X}_l|)]\\
&=\sum_{\|l\|> L}\frac{1}{2}[C_l^2(|Y_l(r)|^2-2|Y_l(r)||\tilde{X}_l|+|\tilde{X}_l|^2-|\tilde{X}_l|^2)]\\
&=\sum_{\|l\|> L}\frac{1}{2}[C_l^2(|Y_l(r)|-|\tilde{X}_l|)^2-C_l^2|\tilde{X}_l|^2]\\
&\ge\sum_{\|l\|> L}-\frac{1}{2}C_l^2|\tilde{X}_l|^2\\

\end{align*}
$$
由于我们并不知道 $\tilde{X}_l$ 真实值，所以 $Q$ 的最小值不能得到。但是我们可以假设它是由某个模型 $V$ 的某个姿势 $r^*$ 投影而来的。
$$
\tilde{X}_l = Y_l(r^*)
$$
因此
$$
Q\ge\sum_{\|l\|> L}-\frac{1}{2}C_l^2|Y_l(r^*)|^2
$$
$r^*$ 应该是全部姿势中的一个姿势，因此 $Q$ 的下界是
$$
\begin{align*}
Q&\ge\sum_{\|l\|> L}-\frac{1}{2}C_l^2|Y_l(r^*)|^2\\
&\ge\min_{r}\sum_{\|l\|> L}-\frac{1}{2}C_l^2|Y_l(r)|^2\\
&=-\max_{r}\sum_{\|l\|> L}\frac{1}{2}C_l^2|Y_l(r)|^2\\
&\equiv\hat{Q}
\end{align*}
$$
记 $\hat{r}$ 和 $\hat{Y}$ 是使上式去到最值的姿势和对应的图像。

将上述结果代入 $B(r,t)$
$$
B(r,t)\ge\sum_{\|l\|> L}\frac{1}{2}|X_l|^2-\sum_{\|l\|> L}\frac{1}{2}C_l^2|\hat{Y_l}|^2-H\\
$$
再代入 $E(r, t)$
$$
E(r, t)\ge\sum_{\|l\|\le L}\frac{1}{2}|C_lY_l(r)-S_l(t)X_l|^2+\sum_{\|l\|> L}\frac{1}{2}|X_l|^2-\sum_{\|l\|> L}\frac{1}{2}C_l^2|\hat{Y_l}|^2-H
$$
$H$ 的出现在 $4\sigma_H$ 的概率中是 0.999936，即
$$
\begin{align*}
H&\le 4\sigma_h\\
&= 4\sqrt{\sum_{\|l\|> L}\frac{1}{2}\sigma_l^2C_l^2|Y_l(r)|^2}\\
&\le 4\max_r\sqrt{\sum_{\|l\|> L}\frac{1}{2}\sigma_l^2C_l^2|Y_l(r)|^2}\\
&= 4\sqrt\max_r{\sum_{\|l\|> L}\frac{1}{2}\sigma_l^2C_l^2|Y_l(r)|^2}\\
&= 4\sqrt{\sum_{\|l\|> L}\frac{1}{2}\sigma_l^2C_l^2|\hat{Y}_l|^2}\\
\end{align*}
$$
最终 $E(r, t)$ 的下界是
$$
E(r, t)\ge\sum_{\|l\|\le L}\frac{1}{2}|C_lY_l(r)-S_l(t)X_l|^2+\sum_{\|l\|> L}\frac{1}{2}|X_l|^2\\-\sum_{\|l\|> L}\frac{1}{2}C_l^2|\hat{Y_l}|^2-4\sqrt{\sum_{\|l\|> L}\frac{1}{2}\sigma_l^2C_l^2|\hat{Y}_l|^2}
$$

## 计划拆分 Subdivision scheme

上述计算过程中，我们可以丢弃部分计算结果的区域，并且用 2D 和 3D 的直角坐标系分别保留候选的位移信息和姿势信息。

## 近似 Approximations





