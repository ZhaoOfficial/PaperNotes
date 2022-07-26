# Chapter 3 Monte Carlo Methods

## 3.2 Why Are Monte Carlo Techniques Useful?

Monte Carlo 方法的一大优势在于其概念简单；Monte Carlo 技术的另一个优点是它们可以应用于广泛的问题。
One major strength of the Monte Carlo approach lies in its conceptual simplicity; another advantage of Monte Carlo techniques is that they can be applied to a wide range of problems.

Monte Carlo 技术的一个缺点是它们相对较慢的收敛速度 $1/\sqrt{N}$，其中 $N$ 是样本数。
One disadvantage of Monte Carlo techniques is their relatively slow convergence rate of $1/\sqrt{N}$, where $N$ is the number of samples.

但是存在 Monte Carlo 方法是唯一可行的求解技术的问题：高维积分和具有非光滑被积函数的积分等。
But there are problems for which Monte Carlo methods are the only feasible solution technique: higher-dimensional integrals and integrals with non-smooth integrands, among others.

## 3.3 Review of Probability Theory

## 3.4 Monte Carlo Integration

Estimate
$$
I=\int_a^bf(x)\mathrm{d}x
$$

### 3.4.1 Weighted Sum of Random Variables

$$
G=\sum_{i=1}^Nw_ig_i
$$

$g_i$ are i.i.d random variables and $\sum_{i=1}^Nw_i=1$. Let $w_i=1/N$, then the expecation of $G$ is:
$$
E[G(x)]=\frac{1}{N}\sum_{i=1}^NE[g_i(x)]=\frac{1}{N}\sum_{i=1}^NE[g(x)]=E[g(x)]
$$
and the variance of $G$ is:
$$
\begin{align*}
\mathrm{Var}[G(x)]&=\mathrm{Var}\left[\frac{1}{N}\sum_{i=1}^Ng_i(x)\right]\\
&=\sum_{i=1}^N\mathrm{Var}\left[\frac{1}{N}g_i(x)\right]\quad(\text{i.i.d})\\
&=\frac{1}{N^2}\sum_{i=1}^N\mathrm{Var}\left[g_i(x)\right]\\
&=\frac{1}{N}\mathrm{Var}\left[g_i(x)\right]
\end{align*}
$$
因此，随着 $N$ 的增加，$G$ 的方差随着 $N$ 而减小，使得 $G$ 越来越好地估计 $E[g(x)]$。标准差 $\sigma$ 减小速度是 $\sqrt{N}$。
Thus, as $N$ increases, the variance of $G$ decreases with $N$, making $G$ an increasingly good estimator of $E[g(x)]$. The standard deviation $\sigma$ decreases as $\sqrt{N}$.

### 3.4.2 Estimator

Estimator:
$$
\langle{I}\rangle=\frac{1}{N}\sum_{i=1}^{N}\frac{f(x_i)}{p(x_i)}
$$
这个估计器的方差（样本均值的方差）应该是：
$$
\begin{align*}
\mathrm{Var}_{p}[I]&=\mathrm{Var}_{p}\left[\frac{1}{N}\sum_{i=1}^{N}\frac{f(x_i)}{p(x_i)}\right]\\
&=\frac{1}{N^2}\sum_{i=1}^{N}\mathrm{Var}_{p}\left[\frac{f(x_i)}{p(x_i)}\right]\\
&=\frac{1}{N}\left[\int_a^b\left(\frac{f(x)}{p(x)}\right)^2p(x)\mathrm{d}x-\left(\int_a^bf(x)\mathrm{d}x\right)^2\right]\\
&=\frac{1}{N}\int_a^b\left(\frac{f(x)}{p(x)}-I\right)^2p(x)\mathrm{d}x
\end{align*}
$$

### 3.4.3 Bias

### 3.4.4 Accuracy

### 3.4.5 Estimating the Variance

### 3.4.6 Deterministic Quadrature versus Monte Carlo

## 3.5 Sampling Random Variables

### 3.5.1 Inverse Cumulative Distribution Function

### 3.5.2 Rejection Sampling

### 3.5.3 Look-Up Table

## 3.6 Variance Reduction

### 3.6.1 Importance Sampling

通过基于要积分的函数的信息明智地选择概率分布，可以减少计算的方差。
The variance of the computation can be reduced by choosing the probability distribution wisely based on information about the function to be integrated.

重要性采样的难点在于选择一个 $p(x)$ 使得方差最小。事实上，一个完美的估计器会使方差为零。
The difficulty of importance sampling is to choose a $p(x)$ such that the variance is minimized. In fact, a perfect estimator would have the variance be zero.

通过泛函分析，我们可以得知，当
$$
p(x)=\frac{|f(x)|}{\int_{D}f(x)dx}
$$
此时的方差会达到 0。但是这样的 $p(x)$ 需要我们提前知道积分的值，而这正是我们需要求的东西。

### 3.6.2 Stratified Sampling

分层抽样的基本思想是将集成域拆分为不相交的子域（也称为层），并分别用一个或多个样本评估每个子域中的积分值 $m$。
The basic idea in stratified sampling is to split the integration domain into disjoint subdomains (also called strata) and evaluate the integral $m$ in each of the subdomains separately with one or more samples.
$$
\int_{a}^{b}f(x)dx=\sum_{i=0}^{m-1}\int_{c_i}^{c_{i+1}}f(x)dx
$$
与盲蒙特卡罗积分方法相比，分层抽样通常会导致较小的方差。
Stratified sampling often leads to a smaller variance as compared to a blind Monte Carlo integration method.















