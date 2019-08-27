# 拐点检测常用算法介绍

最近在学习拐点检测的相关问题， 发现 [C.Truong 的论文](https://arxiv.org/pdf/1801.00718.pdf) 对拐点检测的整个流程和目前主流的一些算法介绍的比较清楚，所以在这里进行了一些记录以及总结，并且对 **Truong** 发布的 **ruptures** 库做了一些简单的介绍。如果想要进行更深入的研究，请参考[原论文](https://arxiv.org/pdf/1801.00718.pdf)和 [ruptures](https://github.com/deepcharles/ruptures)。

## 问题定义

拐点检测名为 **change point detection**，对于一条不平缓的时间序列曲线，认为存在一些时间点 $(t_{1},t_{2},...,t_{k})$，使得曲线在这些点对应的位置发生突变，这些时间点对应的曲线点称为拐点，在连续的两个拐点之间，曲线是平稳的。

![changepoint](C:\Users\w50005335\Pictures\changepoint.PNG)

拐点检测算法的质量通过算法输出拐点与实际观测到的拐点的差值绝对值除以样本数来评估。
$$
\forall k, \quad\left|\hat{t}_{k}-t_{k}^{\star}\right| / T,
$$
理想情况下，当样本数T无穷大时，误差应该减少到0，这种性质称为满足渐近一致性 (asymptotic consistency.)
$$
\max _{k=1, \ldots, K^{*}}\left|\hat{t}_{k}-t_{k}^{\star}\right| / T \stackrel{p}{\longrightarrow} 0
$$

## 符号定义

$y_{a..b}$ 表示时间点 a 和 b 之间的时间序列，因此完整信号为 $y_{0..T}$。

对于给定的拐点索引 $t$，它的关联分数 **associate fraction** 称为拐点分数 **change point fractions** ，公式为 ：
$$
\tau :=t / T \in(0,1].
$$

拐点分数的集合 $\boldsymbol{\tau}=\left\{\tau_{1}, \tau_{2}, \ldots\right\}$ 写作 $|\boldsymbol{\tau}$|

## 研究方法

一般思路是构造一个对照函数 **contrast function**，目标是将对照函数的值最小化。
$$
V(\mathbf{t}, y) :=\sum_{k=0}^{K} c\left(y_{t_{k} \ldots t_{k+1}}\right)，
$$
其中 $c(\cdot)$ 表示用来测量拟合度 **goodness-of-fit** 的损失函数 **cost function**, 损失函数的值在均匀的子序列上较低，在不均匀的子序列上较高。

基于离散优化问题 **discrete optimization problem**，拐点的总数量记为 $K$ ：

如果 $K$ 是固定值，估算的拐点值为：
$$
\operatorname{min}_{|\mathbf{t}|=K} V(\mathbf{t}),
$$
如果 $K$ 不是固定值，估算的拐点值为：
$$
\min _{t} V(\mathbf{t})+\operatorname{pen}(\mathbf{t}).
$$
其中 $pen(t)$为对 $t$ 的惩罚项项

在这种方法论下，拐点检测的算法包含以下三个元素：

1. 选择合适的损失函数来测算子序列的均匀程度 **homogeneity**，这与要检测的变化类型有关
2. 解决离散优化问题
3. 合理约束拐点的数量，确定使用固定的 $K$ 还是用 $pen()$ 来惩罚 **penalizing** 不固定的数量

##  损失函数

损失函数与被检测拐点的变化类型有关，下文介绍的损失函数使用在子序列 $y_{a..b}$ 上，其中 $1 <= a < b <= T$

### Piecewise i.i.d. signals

首先介绍一种使用 $i.i.d.$ 变量对信号 $y$ 进行建模的一般参数模型。

$i.i.d.$ **Independent and identically distributed**，独立同分布，指一组随机变量中每个变量的概率分布都相同，且这些随机变量互相独立。

假设 $y_{1},...,y_{T}$ 是独立随机变量，概率分布 **probability distribution**  $f\left(y_{t} | \theta\right)$ 依赖于向量值参数 $\theta$ **vector-value parameter**，$\theta$ 表示预测到发生突然变化的兴趣点。假设存在一系列真实的拐点 $\mathrm{t}^{\star}=\left\{t_{1}^{\star}, \ldots\right\}$，并且满足：
$$
y_{t} \sim \sum_{i=0}^{K} f\left(\cdot | \theta_{k}\right) , \left(t_{k}^{\star}<t \leq t_{k+1}^{\star}\right).
$$
在这种场景下，拐点检测通过最大似然估计 **maximum likelihood estimation** 实施，相关的损失函数是 **negative log-likelihood**：
$$
c_{\mathrm{i.i.d}}\left(y_{a .  . b}\right) :=-\sup _{\theta} \sum_{t=a+1}^{b} \log f\left(y_{t} | \theta\right).
$$
分布的选择取决于对数据的先验知识。 历史上，高斯分布通常用于模拟均值和极差（Scale = 最大值 - 最小值）变化。后来的大部分文献选用了其他参数分布，最突出的是关于一般指数族的分布。

#### 例一：Mean-shifts for normal data

均值漂移模型是拐点检测文献研究最多的模型之一，$y_{1}, \dots, y_{T}$ 为具有分段常数均值和常数方差的一系列独立正态随机变量。这种情况下，损失函数 $c_{i.i.d.}(\cdot)$ 为：
$$
c_{L_{2}}\left(y_{a . . b}\right) :=\sum_{t=a+1}^{b}\left\|y_{t}-\overline{y}_{a . . b}\right\|_{2}^{2}
$$
其中 $\overline{y}_{a..b}$ 是子信号 $y_{a..b}$ 的经验均值 **empirical mean**，这种损失函数一般也称为平方误差损失 **quadratic error loss**。一般应用于DNA序列和地理信号处理。

#### 例二：Mean-shifts and scale-shifts for normal data

这是一种对均值漂移模型的自然拓展，这种模型也允许方差发生突然变化，$y_{1}, \dots, y_{T}$ 为具有分段常数均值和常数方差的一系列独立正态随机变量。这种情况下，损失函数 $c_{i.i.d.}(\cdot)$ 为：
$$
c_{\Sigma}\left(y_{a .. b}\right) :=\log \operatorname{det} \widehat{\Sigma}_{a .. b}
$$
其中 $\widehat{\Sigma}_{a .. b}$ 是子信号 $y_{a..b}$ 的经验协方差矩阵 **empirical covariance matrix**，这种损失函数可用于检测随机（不要求高斯）变量的前两个时刻的变化，即使它是加入 $c_{i.i.d.}(\cdot)$ 的高斯似然。一般应用于股票市场时间序列和医药信息处理。

#### 例三：Count data

$y_{1}, \dots, y_{T}$ 为具有分段常数速率参数的一系列独立泊松分布 **Poisson distributed** 随机变量，这种模型的损失函数 $c_{\text {poisson}}(\cdot)$ 为：
$$
c_{\text { Poisson }}\left(y_{a . . b}\right) :=-(b-a) \overline{y}_{a .. b} \log \overline{y}_{a .. b}
$$
这种模型常应用于建模统计数据。

从理论上讲，变化点检测将一致性属性视为最大似然估计方法。实际上，已经表明，一般情况下，随着样本数量增长为无穷大，估测的最优拐点在概率上收敛于真实的拐点。拐点检测在 $c_{i.i.d.}(\cdot)$ 下满足渐进一致性。总体而言，参数化的损失函数是有用的并且理论上满足渐进一致性，然而，如果数据不能很好的近似于有效的参数化分布，拐点检测的效果也会受到很大的影响，这迫使我们使用其他的模型。

### Linear model

当发生突然变化的变量之间存在线性关系时，可以使用线性模型，这种变化在相关文献中通常被称为结构化变化。这个公式引出了几个著名的模型比如自回归模型 **autoregressive (AR) model** 和多元回归模型 **multiple regressions** 等。

在数学上，信号 y 被视为单变量响应变量，两种协变量的信号 $x=\left\{x_{t}\right\}_{t=1}^{T}$ 和 $z=\left\{z_{t}\right\}_{t=1}^{T}$ 分别记为 $\mathbb{R}^{p} -valued$ 和 $\mathbb{R}^{q}-valued$，假设有一系列真实的拐点 $\mathbf{t}^{\star}=\left\{t_{1}^{\star}, \ldots, t_{K}^{\star}\right\}$，一般线性模型为：
$$
\forall t, t_{k}^{\star}<t \leq t_{k+1}^{\star}, \quad y_{t}=x_{t}^{\prime} u_{k}+z_{t}^{\prime} v+\varepsilon_{t} \quad(k=0, \ldots, K).
$$
其中 $v \in \mathbb{R}^{q}$ 和 $u_{k} \in \mathbb{R}^{p}$ 为未知回归参数，$\varepsilon_{t}$ 为噪声。此模型也被称为部分结构化变化模型 **partial structural change model**，y 和 x 的线性关系发生突变，而 y 和 z 的关系是一个常量，如果删除项 $z^{\prime}_{t} v$ 则为纯结构化变化模型 **pure structural change model**。相关的损失函数基于最小二乘残差，公式为：
$$
c_{\text { linear }}\left(y_{a . . b}\right) :=\min _{u \in \mathbb{R}^{p}, v \in \mathbb{R}^{q}} \sum_{t=a+1}^{b}\left(y_{t}-x_{t}^{\prime} u-z_{t}^{\prime} v\right)^{2}.
$$
存在一个封闭形式的公式，因为它是一个简单的最小二乘回归。

理论上，对于协变量分布 $(x_{t},z_{t})$，噪声分布 $\varepsilon_{t}$，拐点间的距离为 $t_{k+1}^{\star}-t_{k}^{\star}$，估测的拐点在温和假设下概率上渐进回归于真实的拐点。还有其他的渐近性质，例如拐点位置的渐近分布和 $V(\hat{t}_{K})$ 的限制分布（可用于设计统计检验）。

上述公式（13）有时也会替换为最小绝对偏差之和，理论上也满足渐近一次性：
$$
c_{\text {linear}, L_{1}}\left(y_{a \ldots b}\right) :=\min _{u \in \mathbb{R}^{p}, v \in \mathbb{R}^{q}} \sum_{t=a+1}^{b}\left|y_{t}-x_{t}^{\prime} u-z_{t}^{\prime} v\right|.
$$
线性模型一般应用于金融数据处理。

#### 例四：Autoregressive model

分段自回归模型 **Piecewise autoregressive models** 也属于线性模型，通过将公式（12）中的参数 $x_{t}$ 设定为 $x_{t}=\left[y_{t-1}, y_{t-2}, \dots, y_{t-p}\right]$ 并删除项 $z^{\prime}_{t} v$，信号 y 实际上是对有序 p 的分段自回归。由此产生的损失函数 $c_{AR}(\cdot)$ 能够检测过程中的自回归系数变化。
$$
c_{AR}(y_{a..b}) := \min _{u \in \mathbb{R} P} \sum_{t=a+1}^{b}\left(y_{t}-x_{t}^{\prime} u\right)^{2}.
$$
分段自回归过程是一种特殊的随时间变化 time-varying 的 ARMA 过程。理论上，即使引入了噪声 $\varepsilon_{t}$（例如动态平均过程），也能保持一致性。因此，在实践中，拐点检测可以应用于完整的ARMA过程。

分段回归模型一般应用于分段弱平稳信号，比如 EEG/ECG 时间序列，fMRI 时间序列（functional magnetic resonance imaging）和 语音识别任务。

### Kernel change point detection

可以将原始信号通过核函数 **kernel function** 映射到高维空间中执行拐点检测，这种技术在机器学习的很多技术中都得到了应用，如 **SVM**（support vector machine）和 聚类 **clustering**。

原始信号 y 可以通过一个核函数 $k(\cdot, \cdot) : \mathbb{R}^{d} \times \mathbb{R}^{d} \rightarrow \mathbb{R}$ 映射到一个 RKHS（reproducing Hilbert space） 空间 $\mathcal{H}$，相关的映射函数 $\phi : \mathbb{R} \rightarrow \mathcal{H}$ 对任何样本 $y_{s}, y_{t} \in \mathbb{R}^{d}$ 隐式定义为：
$$
\phi\left(y_{t}\right)=k\left(y_{t}, \cdot\right) \in \mathcal{H} \quad \text { and } \quad\left\langle\phi\left(y_{s}\right) | \phi\left(y_{t}\right)\right\rangle_{\mathcal{H}}=k\left(y_{s}, y_{t}\right).
$$
**RKHS norm** $\|\cdot\|_{\mathcal{H}}$ 隐式定义为：
$$
\left\|\phi\left(y_{t}\right)\right\|_{\mathcal{H}}^{2}=k\left(y_{t}, y_{t}\right).
$$
直觉上，信号被映射到高位空间 $\mathcal{H}$ 上会变成分段常数，于是目标变为检测嵌入信号中的均值漂移，相关的衡量“平均分散” **average scatter** 的损失函数为：
$$
c_{\text { kernel }}\left(y_{a \ldots b}\right) :=\sum_{t=a+1}^{b}\left\|\phi\left(y_{t}\right)-\overline{\mu}_{a . . b}\right\|_{\mathcal{H}}^{2}
$$
其中 $\overline{\mu}_{a . . b} \in \mathcal{H}$ 是嵌入信号的经验均值 $\left\{\phi\left(y_{t}\right)\right\}_{t=a+1}^{b}$，由于众所周知的“核技巧” **kernel trick,**，不需要显式计算映射数据样本。实际上，在简单的代数操作之后，kernel 损失函数可以重写为：
$$
c_{\text { kernel }}\left(y_{a \ldots b}\right):=\sum_{t=a+1}^{b} k\left(y_{t}, y_{t}\right)-\frac{1}{b-a} \sum_{s, t=a+1}^{b} k\left(y_{s}, y_{t}\right).
$$
任何 kernel 都可以插入到这个公式中，但最常用的 kernel 为高斯核 **Gaussian kernel**（也称为径向基函数 **radial basis function**）：
$$
c_{\mathrm{tbf}}\left(y_{a . . b}\right) :=(b-a)-\frac{1}{b-a} \sum_{s, t=a+1}^{b} \exp \left(-\gamma\left\|y_{s}-y_{t}\right\|^{2}\right).
$$
其中 $\gamma>0$，被称为带宽参数 **bandwidth parameter**，在该情况下使用线性 kernel 在形式上等同于对分段常数信号使用 $c_{L_{2}}(\cdot)$。

总体而言，kernel 拐点检测是一种非参数化的免模型 **model-free** 方法，可以用于没有先验条件，不满足基础分部形式的分段 ***i.i.d.*** 信号。常被应用于生理信号（如脑电波）以及音频时间序列分割任务。

### Mahalanobis-type metric

在拐点检测场景中，Mahalanobis-type metric 可以与度量学习算法一起使用来度量信号，也常见于聚类方法。

对于对于任何对称正半定矩阵 **symmetric positive semi-definite matrix** $M \in \mathbb{R}^{d \times d}$，对于样本 $y_{s},y_{t}$，相关的伪度量 **pseudo-metric** $\|\cdot\|_{M}$ 为：
$$
\left\|y_{t}-y_{s}\right\|_{M}^{2} :=\left(y_{t}-y_{s}\right)^{\prime} M\left(y_{t}-y_{s}\right)
$$
设 $\overline{y}_{a . . b}$ 为子信号 $y_{a . . b}$ 的经验均值，对应的损失函数定义为：
$$
c_{M}\left(y_{a . . b}\right) :=\sum_{t=a+1}^{b}\left\|y_{t}-\overline{y}_{a . . b}\right\|_{M}^{2}
$$
本质上，度量矩阵 $M$ 为协方差矩阵的逆矩阵，记为 **Mahalanobis metric**：
$$
M=\widehat{\Sigma}^{-1}
$$
其中 $\widehat{\Sigma}$ 为信号 y 的经验协方差矩阵。

如果把矩阵分解为 $M=U^{\prime} U$，可以得到：
$$
\left\|y_{t} -y_{s}\right\|_{M}^{2}=\left\|U y_{t}-U y_{s}\right\|^{2}
$$
使用 Mahalanobis-type metric 仅涉及样本的线性处理，引入非线性的一种方法是将其与 kernel-based 转换相结合。对于给定的核函数 $k(\cdot, \cdot) : \mathbb{R}^{d} \times \mathbb{R}^{d} \mapsto \mathbb{R}$ 以及相关的映射 $\phi(\cdot) : \mathbb{R}^{d} \mapsto \mathcal{H}$，在特征空间 $\mathcal{H}$ 上的 Mahalanobis-type metric $\|_{\mathcal{H}, M}\|$ 为：
$$
\left\|\phi\left(y_{s}\right)-\phi\left(y_{t}\right)\right\|_{\mathcal{H}, M}^{2}=\left(\phi\left(y_{s}\right)-\phi\left(y_{t}\right)\right)^{\prime} M\left(\phi\left(y_{s}\right)-\phi\left(y_{t}\right)\right).
$$
其中 M 是一个定义在 $\mathcal{H}$ 上的对称正半定矩阵，得到的成本函数为：
$$
c_{\mathcal{H}, M}\left(y_{a \ldots b}\right) :=\sum_{t=a+1}^{b}\left\|y_{t}-\overline{y}_{a . b}\right\|_{\mathcal{H}, M}^{2}
$$


### 损失函数总结

![costFunction](C:\Users\w50005335\Pictures\costFunction.PNG)



## 应用于固定 K 的查找算法

下面展示当拐点数量K是固定值的时候，如何使用离散优化算法进行拐点检测。指定 $\hat{\mathbf{t}}_{K}(y)$ 为最小化对照函数 $V(\cdot)$ 的方法，写作：
$$
\hat{\mathbf{t}}_{K}(y) :=\underset{|\mathbf{t}|=K}{\arg \min } V(\mathbf{t}, y).
$$

根据信号的上下文信息观测出的最优拐点集合记作 $\hat{\mathbf{t}}_{K}$。

### 最优检测 Optimal Detection

#### 简介

应用于固定 K 的拐点查找算法是在索引集 $\{1, \ldots, T\}$ 上的离散优化问题，所有可能的组合基数为 $\left(\begin{array}{c}{T-1} \\ {K}\end{array}\right)$ ，数量太多使得无法进行全面的遍历所有可能组合，但是却可以有效的使用动态规划 **dynamic programming** 的方法，本文将此算法记作 **Opt** 。

实际上，动态规划依赖于目标函数 $V(\cdot)$ 的附加性质。粗略地说，它包括递归地解决子问题，并满足于以下观察结果：
$$
\begin{aligned} \min _{|t|=K} V\left(\mathbf{t}, y=y_{0 .. T}\right) &=\min _{0=t_{0}<t_{1}<\cdots<t_{K}<t_{K+1}=T} \sum_{k=0}^{K} c\left(y_{t_{k} .. t_{k+1}}\right) \\ &=\min _{t \leq T-K}\left[c\left(y_{0 . . t}\right)+\min _{t=t_{0}<t_{1}<\cdots<t_{K-1}<t_{K}=T} \sum_{k=0}^{K-1} c\left(y_{t_{k}.. t_{k+1}}\right)\right] \\ &=\min _{t \leq T-K}\left[c\left(y_{0 . . t}\right)+\min _{|\mathbf{t}|=K-1} V\left(\mathbf{t}, y_{t .. T}\right)\right]. \end{aligned}
$$
直观来看，公式表明如果由 K-1 个元素组成的最优子信号 $\left\{y_{s}\right\}_{t}^{T}$ 已知，那么计算出第一个拐点是容易的。接下来就可以通过递归的执行上述观察结果来得出完整的集合。结合对二次误差损失函数 **quadratic error cost function** 的计算，**Opt** 算法的复杂度为 $\mathcal{O}\left(K T^{2}\right)$。使用的损失函数越复杂，计算的复杂度就越高。由于 **Opt** 算法的复杂度是二次的，因此适合应用于比较短的信号，一般包含一百个样本点左右。

有兴趣的朋友可以阅读[Rigaill](https://arxiv.org/pdf/1004.0887.pdf)和[Yann](https://link.springer.com/content/pdf/10.1007%2Fs00180-013-0422-9.pdf)在论文中提供的几种降低 **Opt** 算法计算负担的方法。

#### 伪代码

![Opt](C:\Users\w50005335\Pictures\Opt.PNG)

### 滑动窗口 Window sliding

#### 简介

滑动窗口算法（下文记为 **Win**）是一种快速近似的 **Opt** 的替代算法，该方案依赖于单个变化点检测程序并将其扩展以找到多个变化点。算法实施时，两个相邻的窗口沿着信号滑动。 计算第一窗口和第二窗口之间的差异。 对给定的损失函数，两个子信号间的差异为：
$$
d\left(y_{a . . t}, y_{t . . b}\right)=c\left(y_{a .. b}\right)-c\left(y_{a . . t}\right)-c\left(y_{t .. b}\right) \quad(1 \leq a<t<b \leq T).
$$
当这两个窗口包含不相似的片段时，计算得的差异值将会很大， 可以检测到一个拐点。在离线设置中，计算完整的差异曲线并执行峰值搜索过程以找到拐点索引。

![Win](C:\Users\w50005335\Pictures\Win.PNG)

类似的，还有一种方法叫双样本检验（或同质性检验）**two-sample test (or homogeneity test)** ，这是一种统计假设检验程序，旨在评估两个样本群体的分布是否相同。

**Win** 的最大好处是复杂性低（与 $c_{L_{2}}$ 搭配时为线性时间），易于实现，此外，任何单一拐点检测方法都可以加入到该方案中。有大量可用于差异的渐近分布的参考文献（用于几种成本函数），这对于在峰值搜索过程中找到合适的阈值很有用。然而，**Win** 通常是不稳定的，因为单个变化点检测是在小区域（窗口）上进行的，降低了它的统计功效。

#### 伪代码

![WinAlg](C:\Users\w50005335\Pictures\WinAlg.PNG)

### 二分分割 Binary segmentation

#### 简介

二分分割（下文记为 **BinSeg**）也是一种快速近似的 **Opt** 的替代算法，在拐点检测的应用场景下，BinSeg是最常用的方法之一，因为它概念简单，易于实现。

**BinSeg** 是一种顺序贪心算法 **greedy sequential algorithm**，在每次迭代中，执行单个变化点的检测并产生估计$\hat{t}^{(k)}$ 。在下文中，上标 <sup>k</sup> 指的是顺序算法的第 k 步。算法的示意图如下：

![BinSeg](C:\Users\w50005335\Pictures\BinSeg.PNG)

第一个估计的拐点 $\hat{t}^{(1)}$ 为：
$$
\hat{t}^{(1)} :=\underset{1 \leq t<T-1}{\arg \min } \underbrace{[c\left(y_{0 . . t}\right)+c\left(y_{t . . T}\right)]}_{V(\mathrm{t}=\{t\})}.
$$
这个操作是贪心的，目的是找出使得总体损失最小的拐点，然后在 $\hat{t}^{(1)}$ 的位置将信号分成了两部分，再对得到的子信号进行相同的操作，直到找不到更多的拐点为止。表达的正式一些，在 **BinSeg** 的 k 次迭代以后，下一个估算的拐点 $\hat{t}^{(k+1)}$ 属于原始信号的 k + 1 个子片段之一，计算公式为：
$$
\hat{t}^{(k+1)} :=\underset{t \notin\left\{\hat{t}^{(1)}, \ldots, \hat{t}^{(k)}\right\}}{\arg \min } V\left(\left\{\hat{t}^{(1)}, \ldots, \hat{t}^{(k)}\right\} \cup\{t\}\right).
$$
由于 K 的总数是已知的，因此 **BinSeg** 需要执行 K 步来估算所有的拐点，贪心检测的过程可以设计为一个双样本检验的过程。在使用二次损失误差 **quadratic error loss** $C_{L_{2}}$的情况下，**BinSeg** 的复杂度为 $\mathcal{O}(T \log T)$。

从理论上讲，**BinSeg** 被证明可以在均值模型下用二次误差 $C_{L_{2}}$ 产生渐近一致的估计。更近期的结果表明，**BinSeg** 只有当任何两个相邻变化点之间的最小间距大于 $T^{3 / 4}$ 时一致。一般来说，BinSeg的输出只是最优解的近似值，尤其是不能精确地检测到接近的拐点，原因在于估计的拐点 $\hat{t}^{(k)}$ 不是从均匀分段估计的，并且每个估计都取决于先前的分析。 

#### 伪代码

![BinSegAlg](C:\Users\w50005335\Pictures\BinSegAlg.PNG)

目前有许多算法可以用于提高 **BinSeg** 的准确性，然而这些提高一般都会牺牲算法原有的简易性。感兴趣的朋友可以参考 [Circular binary segmentation](https://watermark.silverchair.com/kxh008.pdf?token=AQECAHi208BE49Ooan9kkhW_Ercy7Dm3ZL_9Cf3qfKAc485ysgAAAlMwggJPBgkqhkiG9w0BBwagggJAMIICPAIBADCCAjUGCSqGSIb3DQEHATAeBglghkgBZQMEAS4wEQQMyj4SMZfEsDbsTWx4AgEQgIICBkPVv1LrDRiWmQLPUN24VkLkHk5OyfaWXrcS87RFNuik5Ajo1TXC4Zx22XSw6qnz2m9F1jaG7zDbXJTjciNii8WlWCe43Jf-yy2HKRTKpl_ynpLLPDhqSWXCg5K0G4SQhYyCd-GmdswpTj6bZq6LJVuvP2yLPAiKajKjkfzXaUr_1NCFiacyJ47QRlztT1YY5ofyHK8x-1EO8V2o8uLshLOWL7ymL87zauoi180FSrPKDDV78WKc5YdSJNNwD6pJNb6fhI3NDh4cYmZugBDA54pV6s3RKc-m-_boL0GtVpu38rXvM27l_wQLgnVIyaWNQNHeFhDTfTXsv-YJ0c79ekXNOimmictCFfpKm3oI9p8d6PmWDEW6nXZ6YdVzr_rk6bPgkHAgOTX9NpUII7bqAno8z_8a0PByMUjF-HdP61wQS69iP5hjt-rn5mdRNbpqQCw5VX2zswreaiSnvGJhDHOSjdtuF8HY4AVxL76-Ilm_0dLiJ-ECfcndMEWuU4gL1tRHpAMbaEZJvXIkCGp3P_7Wg7e0XKrf_KgkdrWtrOl2NVFUFiXRvD7ML2uQBtcQzCnLYGQZ2qwW_o2FUKGvRAPRlHIQWd1G1ScyFTYtdAG39iNglf_yQkDsE_wyVk4G77xSdH9eBkC8HFnYdqmmmwhDhjQePh23EjCMJpOyATrp7C3N3Rnu) 和 [wild binary segmentation algorithm](https://projecteuclid.org/download/pdfview_1/euclid.aos/1413810727)。

### 自底向上分割 Bottom-up segmentation

#### 简介

自底向上分割是 **BinSeg**  的自然对应，下文记作 **BotUp**，这种算法同样也是一种近似算法。跟 **BinSeg** 相反，**BotUp** 最开始就把源信号分割成很多子信号，然后序列化的把它们合并直到剩下 K 个拐点。 在每个步骤中，所有潜在的变化点（分隔相邻子段的索引）按它们分开的段之间的差异进行排序。差异性最小的拐点被删除，由它分割的两个子段被合并。跟 **BinSeg** 的贪心算法相反，**BotUp** 被称为一种“慷慨”的算法。

算法示意图为：

![BotUp](C:\Users\w50005335\Pictures\BotUp.PNG)

对于给定的损失函数，两个子信号的差异性为：
$$
d\left(y_{a . . t}, y_{t . . b}\right)=c\left(y_{a .. b}\right)-c\left(y_{a . . t}\right)-c\left(y_{t .. b}\right) \quad(1 \leq a<t<b \leq T).
$$
**BotUp** 的优势在于线性复杂度（对T个样本使用二次损失函数$C_{L_{2}}$）以及概念的简单性。但是它也存在一些问题：

1. 如果实际的拐点不在最开始划分出来的点集中，那么 **BotUp** 也永远不会考虑它。
2. 在第一次迭代中，融合过程可能是不可靠的，因为计算只是在小的区段上进行，统计意义比较小。
3. 在现有文献中，**BotUp** 的研究不如 **BinSeg**，没有理论收敛性研究 **theoretical convergence study** 可用。

然而，[An online algorithm for segmenting time series](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=989531) 的作者发现 **BotUp** 在十多个数据集上的表现优于 **BinSeg**。

#### 伪代码

![BotUpAlg](C:\Users\w50005335\Pictures\BotUpAlg.PNG)



## 应用于可变 K 值的查找算法

接下来我们讨论拐点总数未知情况下的算法，在这种情况下，我们的主要注意力在于选择惩罚函数 **penalty function** **pen()** 以及惩罚等级 **penalty level** $\beta$。要解决的优化问题为：
$$
\min _{t} V(t)+\operatorname{pen}(t)
$$

### $l_{0}$ 惩罚项  （$l_{0}$ penalty）

#### 简介

$l_{0}$ 惩罚项也被称为线性惩罚项，被认为是最流行的惩罚项，一般而言，$l_{0}$ 惩罚项记作 $\mathrm{pen}_{l_{0}}$，公式为：
$$
\operatorname{pen}_{l_{0}}(\mathrm{t}) :=\beta|\mathrm{t}|.
$$
其中 $\beta > 0$， 负责权衡复杂性和拟合度 (controls the trade-off between complexity and goodness-of-fit)，直观而言，当出现的拐点越来越多时，它平衡了总成本 $V(\cdot)$ 减少的幅度。有很多利用这个规则的例子，最著名的是 **BIC (Bayesian Information Criterion)** 和 **AIC (Akaike Information Criterion)**。

假设对信号 **y** 有一个分段 ***i.i.d.*** 模型，设定 $c=c_{i . i . d}$，对于给定的一组拐点 **t**，引出 **BIC** 的约束似然公式为：
$$
2 V(\mathrm{t})+p \log T|\mathrm{t}|.
$$
其中 **p** 为参数空间 **parameter space** 的维度（如 $p=1$ 表示平均值的单变量变化，$p = 2$ 表示平均值和极差的单变量变化等等），因此，**BIC** 等同于当 $\beta=p / 2 \log T$ 时的线性惩罚拐点检测：
$$
\operatorname{pen}_{B I C}(\mathrm{t}) :=\frac{p}{2} \log T|\mathrm{t}|.
$$
类似的，在固定方差的均值飘逸模型下 **mean-shift model with fixed variance**，$c=c_{L_{2}}$，**BIC** 惩罚项为：
$$
\operatorname{pen}_{B I C, L_{2}}(\mathrm{t}) :=\sigma^{2} \log T|\mathrm{t}|
$$
**AIC** 也是类似的，公式为：
$$
\operatorname{pen}_{A I C, L_{2}}(\mathrm{t}) :=\sigma^{2}|\mathrm{t}|
$$
从理论上讲，拐点估计的一致性在某些情况下是可以证明的。 研究最多的模型是均值漂移模型，当满足下列假设时，其估计的拐点数和拐点分数概率渐近收敛：

- 在无噪声设置中，如果惩罚值以适当的速率收敛到零，
- 如果噪声是高斯白噪声 **Gaussian white noise** 并且 $\mathrm{pen}=\mathrm{pen}_{\mathrm{BIC}, L_{2}}$，
- 如果噪声是二阶静止过程（适当降低自相关功能）并且惩罚级别慢慢发散到无穷大（慢于T）

最近的研究表明对于 $c = c_{kernel}$ 的线性惩罚项的拐点检测估计的一致性会以 $1/T$ 的速率收敛到 0。对于其他损失函数（最值得注意的是 $c_{i.i.d.}$）的理论结果仅涉及成本总和到其真实值的收敛，没有关于一致性的结果。

在各种假设下渐近地收敛于概率：

#### Pelt 算法

实施线性惩罚项拐点检测的一个朴素的实施方案是对足够大的 $K_{max}$，对 $K = 1, ... ,K_{max}$ 都进行执行 **Opt**，然后找出使得惩罚结果最小的部分，但是这种方案的平方复杂度是难以接受的，因此我们需要实施其他复杂度更低的方法。

**Pelt** 算法 **Pruned Exact Linear Time** 可以用于找出 $pen = pen_{l_{0}}$ 的准确解，该方法按顺序考虑每个样本，并基于明确的修剪规则决定是否从潜在的拐点集中丢弃它。假设片段长度是从均匀分布中随机选取的，**Pelt** 的复杂度是 $\mathcal{O}(T)$，尽管这种设定是不现实的(会产生短片段)，但 **Pelt** 依然比上述的朴素启发式快几个数量级。

使用 **Pelt** 时唯一需要调整的参数只有惩罚级别 $\beta$，$\beta$ 的设定至少与样本数和信号维度有关，$\beta$ 值越小，算法对变化的感知越敏感，可以选用较小的值来找到更多的拐点，也可以使用较大的值来仅关注显著变化。

#### 伪代码

![Pelt](C:\Users\w50005335\Pictures\Pelt.PNG)



### $l_{1}$ 惩罚项 （ $l_{1}$ penalty）

为了进一步降低具有线性惩罚的变化点检测的计算成本，已经提出了一种将 $l_{0}$ 惩罚项转换为 $l_{1}$ 惩罚项的替代公式。同样的理由也在机器学习的许多发展中起到了至关重要的作用，如稀疏回归，压缩感知，稀疏PCA等，在数值分析和图像去噪中，这种惩罚项也称为总变异正则项。

引入该策略的目的是检测带高斯噪声 **Gaussian noise** 的分段常量信号中的均值漂移现象，相关的损失函数是 $C_{L_{2}}$，在这种设定下，拐点检测优化问题被记作：
$$
\min _{\mathbf{t}} V(\mathbf{t})+\beta \sum_{k=1}^{|\mathbf{t}|}\left\|\overline{y}_{t_{k-1} .. t_{k}}-\overline{y}_{t_{k} .. t_{k+1}}\right\|_{1}
$$
其中 $\overline{y}_{t_{k-1} .. t_{k}}$ 是子信号 $y_{t_{k-1} .. t_{k}}$ 的经验均值，因此，$l_{1}$ 为：
$$
\operatorname{pen}_{l_{1}}(\mathrm{t}) :=\beta \sum_{k=1}^{|t|}\left\|\overline{y}_{t_{k-1} . . t_{k}}-\overline{y}_{t_{k} .. t_{k+1}}\right\|_{1}
$$
有[文献](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.727.8130&rep=rep1&type=pdf)指出它实际上等同于凸优化问题 **convex optimization problem**，对于矩阵 $Y:=\left[y_{1}, \ldots, y_{T}\right]^{\prime} \in \mathbb{R}^{T \times d}$，$S \in \mathbb{R}^{T \times T-1}$ 且：
$$
S_{i j} :=\left\{\begin{array}{l}{1 \text { if } i>j} \\ {0 \text { if } i \leq j}\end{array}\right.
$$
$\overline{Y}$ 和 $\overline{S}$ 表示对矩阵 $Y$ 和 $S$ 进行 **centering each column** 所得的矩阵，**centering each column** 表示矩阵的每一列的每个元素减去该列均值后得到的矩阵。那么拐点检测优化问题的公式等价为：
$$
\min _{\Delta \in \mathbb{R}^{(T-1) \times d}}\|\overline{Y}-\overline{S} \Delta\|^{2}+\beta \underbrace{\sum_{t=1}^{T-1}\left\|\Delta_{t, \bullet}\right\|_{1}}_{\|\Delta\|_{1,1}}
$$
估算的拐点为矩阵 $\Delta$ 的有效行 **support rows**（不为0的行），矩阵 $\Delta$ 为“跳跃矩阵” **jump matrix**，包含了信号 $\overline{S} \Delta$ 均值漂移的位置和幅度，这个公式是著名的 **Lasso Regression (“least absolute shrinkage and selection operator”)**。可以直观看出 $l_{1}$ 惩罚项将许多系数缩小为零，在这种情况下，它意味着它有利于 y 的分段常数近似 **piecewise constant approximations**。 惩罚项 $\beta$ 的值越高，允许得到的拐点越少。

在算法上，这种优化执行了最小角度回归 **the least-angle regression (Lars)**。当拐点总数的上限已知时，这种方法的复杂度为 $\mathcal{O}(T \log T)$。从理论角度来看，估算的拐点分数是渐近一致的。 该结果证明了惩罚值 $\beta$ 序列的适当收敛，拐点的数量也可以正确的预测。

###  其他惩罚项

改编的 **BIC** 标准是一种惩罚项，它不仅依赖于拐点数量，而且还依赖于拐点之间的距离，公式为：
$$
\operatorname{pen}_{\mathrm{mBIC}}(\mathrm{t}) :=3|\mathrm{t}| \log T+\sum_{k=0}^{|\mathrm{t}|+1} \log \left(\frac{t_{k+1}-t_{k}}{T}\right).
$$
$\operatorname{pen}_{\mathrm{mBIC}}(\mathrm{t})$ 的第二项适用于均匀分布的拐点，在索引非常近的时候取得最大值，例如 $t_{1}=1, t_{2}=2, \ldots$ 这种方式在实践中较难处理，因此一般用 $pen_{l_{0}}$ 近似。详细内容可阅读[文献](https://onlinelibrary.wiley.com/doi/pdf/10.1111/j.1541-0420.2006.00662.x#accessDenialLayout)。

还有一种惩罚项写作：
$$
\operatorname{pen}_{\mathrm{Leb}}(\mathrm{t}) :=\frac{|\mathrm{t}|+1}{T} \sigma^{2}\left(a_{1} \log \frac{|\mathrm{t}|+1}{T}+a_{2}\right).
$$
详细信息可阅读[文献](https://hal.inria.fr/inria-00071847/document)。



## 贝叶斯方法

贝叶斯方法是拐点检测文献中的重要方法，这些方法可以适应关于拐点位置分布的先验知识，被广泛应用于语音识别，大脑图像，音频分割和生物信息学等特定学科。

贝叶斯方法通常假设一个未观察到的离散状态变量 $s_{t}(t=1, \ldots, T)$，这个变量控制数据模型的产生。拐点为这个隐藏变量 $S_{t}$ 发生状态变化时的索引，假定信号 y 满足分段 ***i.i.d***，记作：
$$
y_{t} \sim f\left(\cdot | \theta_{s_{t}}\right)
$$
其中 $f(\cdot | \theta)$ 为对已观测到的参数 $\theta$ 的概率分布，$\theta_{i}$ 是已观测状态 i 的激发参数 **emission parameters**，观测的联合概率为：
$$
\mathbb{P}\left(y,\left[s_{1}, \ldots, s_{T}\right]\right)=\prod_{t=1}^{T} f\left(y_{t} | \theta_{s_{t}}\right)
$$
贝叶斯算法和其他方法不同，它通过假设状态变量 $S_{t}$ 的分布并计算出对观测状态序列的最佳解释。一种很著名的贝叶斯模型为隐马尔科夫模型 **Hidden Markov Model (HMM)**，状态变量被假设为马尔科夫链 **Markov chain**，也就是说 $S_{t}$ 的值只跟 $S_{t-1}$ 有关。概率可以完全用转移矩阵 **transition matrix** A 描述：
$$
A_{i j}=\mathbb{P}\left(s_{t}=j | s_{t-1}=i\right),
$$
**HMM** 由三个参数定义：转移矩阵 A，激发概率  $f(\cdot | \theta)$ 以及初始状态概率 $\mathbb{P}\left(s_{1}=\theta_{i}\right)$，这种情景下的的拐点检测相当于最大后验 **posteriori**（MAP）状态序列 $\hat{s}_{1}, \dots, \hat{s}_{T}$。如果提供参数的先验分布，则通常使用维特比算法 **Viterbi algorithm** 或通过随机采样方法（诸如蒙特卡罗马尔可夫链 **Monte Carlo Markov Chain**，**MCMC**）来执行计算。

**HMM** 的优势在于特殊场景下优秀的性能表现以及对任何分段 ***i.i.d.*** 信号的建模能力，是一种由很多实际应用的经过深思熟虑的方法。 然而，HMM依赖于在许多情况下未经验证的若干假设，例如观察被认为是独立的、激发分布是参数化的、通常是高斯混合、不适用于长段等。 也有几种相关算法被设计来克服这些限制。如：动态变换线性模型 **Switching linear dynamic models** 比独立激发模型具有更大的描述能力，它们适用于视觉跟踪的复杂任务。 然而，由于需要调整许多参数，因此校准步骤显得更加麻烦。半马尔可夫模型 **semi-Markov models** 放宽了马尔可夫假设，半马尔可夫过程跟踪当前段的长度，更适合用几个拐点对状态变量建模。**Dirichlet 隐马尔可夫模型 Dirichlet process hidden Markov models** 非常适合执行拐点检测，隐式计算状态的数量并且从后验分布采样使得计算上变得容易。



## ruptures 库简介

目前主要的用于拐点检测的库大部分都是基于 R 语言的，对于 python 而言，一个比较优秀的库是由 **Charles Truong** 等人开源的 [ruptures](https://github.com/deepcharles/ruptures) 。本文简单介绍了 **ruptures** 的基本特性，更多详细介绍以及使用手册请阅读[官方文档](http://ctruong.perso.math.cnrs.fr/ruptures-docs/build/html/index.html)。

**ruptures** 的流程结构如图所示：

 ![ruptures](C:\Users\w50005335\Pictures\ruptures.PNG)

**ruptures** 的主要特性如下：

- **Search methods**

  ​	**ruptures** 库包含了上文提到的主要查找算法，比如动态规划 **dynamic programming**，基于 ***l*<sub>0 </sub>** 约束的检测 **detection with a *l*<sub>0 </sub> constraint**，二元分割 **binary segmentation**，自下而上的分割 **bottom-up segmentation** 以及基于窗口的分割 **window-based segmentation**

- **Cost function**

  ​	在 **ruptures** 库中，可以选用有参损失函数 **parametric cost functions** 来检测标准统计量 **standard statistical quantities** 的变化，如均值 **mean**，规模 **scale**，维度间的线性关系 **linear relationship between dimensions**，自回归系数 **autoregressive coefficients** 等。也可以使用无参损失函数 **non-parametric cost functions (Kernel-based or Mahalanobis-type metric)** 来检测分布式变化。

- **Constraints**

  ​	无论拐点数量是否已知，上述方法都适用，需要注意的是，**ruptures** 在成本运算 **cost budget** 和线性惩罚项 **linear penalty term** 的基础上实现拐点检测。

- **Evaluation**

  ​	评估指标可用于定量比较分割，以及用可视化的方法评估算法性能

- **Input**

  ​	**ruptures** 能在可以转化为 ***Numpy array*** 的任何信号上实施，包括单因素信号和多因素信号 **univariate or multivariate signal**

- **Consistent interface and modularity**

  ​	离散优化方法和损失函数是拐点检测的两个主要组成部分，这些在 **ruptures** 中都是具体的对象，因此本库的模块化程度很高，一旦有新的算法产生，即可无缝的集成到 **ruptures** 中。

- **Scalability**  

  ​	数据探索通常需要使用不同的参数集多次运行相同方法。 为此，实现高速缓存以将中间结果保留在内存中，从而大大降低了在相同信号上多次运行相同算法的计算成本。 

  **ruptures** 还为具有速度限制的用户提供了信号采样和设置拐点最小距离的可能



## 总结

本文回顾了几种用于拐点检测的算法，描述了算法中的损失函数，搜索方法以及拐点数量约束。并简单介绍了基于 python 的拐点检测库 **ruptures**。拐点检测模块还有很多其他的优秀算法没有在文中展示，有兴趣更加深入的朋友可以参考 [C.Truong 的论文](https://arxiv.org/pdf/1801.00718.pdf) 第七章 **Summary Table** 中的文献整理或者阅读更多的论文，比如 [instance on Bayesian methods](https://link.springer.com/content/pdf/10.1007%2Fs11222-006-8450-8.pdf)，[in-depth theoretical survey](https://onlinelibrary.wiley.com/doi/pdf/10.1111/jtsa.12035) 和 [numerical comparisons in controlled settings](https://www.researchgate.net/profile/Rebecca_Killick/publication/260897758_Analysis_of_changepoint_models/links/0c9605329ccc8209f3000000.pdf)。

