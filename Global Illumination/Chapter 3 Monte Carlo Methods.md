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

### 3.4.1 Weighted Sum of Random Variables

### 3.4.2 Estimator

这个估计的方差（样本均值的方差）应该是：**?**
$$
\begin{align*}
\text{Var}_{p}[I]&=\text{Var}_{p}\left[\frac{1}{N}\sum_{i=1}^{N}\frac{f(x_i)}{p(x_i)}\right]\\
&=\frac{1}{N}\int\left(\frac{f(x)}{p(x)}-I\right)^2p(x)dx
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















