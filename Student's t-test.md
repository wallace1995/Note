# Student's t-test

常作为检验一组来自正态分布总体的独立样本的期望值是否为一个实数，或者两组正态分布样本的期望值之差是否为某一个实数。可以用于检验两个样本集是否有显著的差异。

## 前提假设

大多数的 $t-test$  统计量形式为 $t=Z/s$，其中 $Z$ 与 $k$ 为已知数据的函数，$k$ 为尺度参数，

$t-test$ 的前提假设为：

- 样本满足正态分布，均值为 $\mu$，方差为 $\frac{\sigma^{2}}{n}$
- $s^{2}$ 满足置信度 $n-1$ 的卡方分布   $s^{2}$ follows a $x^2$ distribution with $n-1$ Degrees of freedom
- $Z$ 和 $s$ 相互独立 

**零假设 null hypothesis**：一般是希望被证明为错误的假设，如“两者无关联” 或 “两者非独立” 或 “没有变化”

**对立假设 Alternative hypothesis**：一般是希望能证明为正确的假设，如“两者有关联” 或 “两者独立” 或 “有变化”

## 主要类别

### One-sample *t*-test

可利用以下统计量 $t$ 对一组来自正态分配独立样本 $x_i$ 验证零假设总体期望值 $μ$ 为 $μ_0$ ：
$$
t=\frac{\overline{x}-\mu_{0}}{s / \sqrt{n}}
$$
其中：$i=1 \ldots n, \overline{x}=\frac{\sum_{i=1}^{n} x_{i}}{n}$ 为样本均值，$μ_0$ 为数学期望，$s=\sqrt{\frac{\sum_{i=1}^{n}\left(x_{i}-\overline{x}\right)^{2}}{n-1}}$ 为样本标准差，$n$ 为样本数量。

该统计量 $t$ 在零假设 $\mu=\mu_{0}$ 为真的条件下服从置信度为 $n-1$ 的 Student's *t*-distribution



### Dependent *t*-test for paired samples

与单样本检验类似，不过检验对象为两组正态分布独立样本之差

两组独立正态分布样本 $x_{1i}$ 与  $x_{2i}$ 之差为 $d_{i} = x_{1i} - x_{2i}$，可以利用以下统计量 *t* 检验 $d_{i}$ 的均值是否为 $\mu_{0}$。
$$
t=\frac{\overline{d}-\mu_{0}}{s_{d} / \sqrt{n}}
$$
其中：$i=1 \ldots n, \overline{d}=\frac{\sum_{i=1}^{n} d_{i}}{n}$ 为配对样本差值的平均数，$s_{d}=\sqrt{\frac{\sum_{i=1}^{n}\left(d_{i}-\overline{d}\right)^{2}}{n-1}}$，$n$ 为样本数量，该统计量 $t$ 在零假设 $\mu=\mu_{0}$ 为真的条件下服从置信度为 $n-1$ 的 Student's *t*-distribution



### Independent two-sample *t*-test

#### 样本数相等，方差相等

若两组独立正态分布样本 $x_{1i}$ 与 $x_{2i}$ 具有相同样本数 $n$，并且各自方差相等，则两组样本总体期望值差 $\mu_1 - \mu_2$ 是否为 $\mu_0$ 可利用以下统计量 $t$ 检验
$$
t=\frac{\overline{x}_{1}-\overline{x}_{2}-\mu_{0}}{\sqrt{2 s_{p}^{2} / n}}
$$
其中：$i=1 \ldots n, \overline{x}_{1}=\left(\sum_{i=1}^{n} x_{1 i}\right) / n$，$\overline{x}_{2}=\left(\sum_{i=1}^{n} x_{2 i}\right) / n$ 为两组样本各自的均值，

$s_{p}^{2}=\left(\sum_{i=1}^{n}\left(x_{1 i}-\overline{x}_{1}\right)^{2}+\sum_{i=1}^{n}\left(x_{2 i}-\overline{x}_{2}\right)^{2}\right) /(2 n-2)$ 两组样本的共同方差，该统计量 $t$ 在零假设 $\mu_{1} - \mu_{2} = \mu_{0}$ 为真的条件下服从置信度为 $2n-2$ 的 Student's *t*-distribution

#### 样本数相等，方差不相等

若两组独立正态分布样本 $x_{1i}$ 与 $x_{2i}$ 具有不同的样本数 $n_1$ 与 $n_2$，并且各自方差相等，则两组样本总体期望值差 $\mu_1 - \mu_2$ 是否为 $\mu_0$ 可利用以下统计量 $t$ 检验
$$
t=\frac{\overline{x}_{1}-\overline{x}_{2}-\mu_{0}}{\sqrt{s_{p}^{2} / n_{1}+s_{p}^{2} / n_{2}}}
$$
其中 $i=1 \ldots n_{1}, j=1 \ldots n_{2}, \overline{x}_{1}=\left(\sum_{i=1}^{n_1} x_{1 i}\right)/n_{1},\overline{x}_{2}=\left(\sum_{j=1}^{n_2} x_{2j}\right) / n_{2}$ 为两组样本各自的平均数，

$s_{p}^{2}=\left(\sum_{i=1}^{n}\left(x_{1 i}-\overline{x}_{1}\right)^{2}+\sum_{j=1}^{n}\left(x_{2 j}-\overline{x}_{2}\right)^{2}\right) /\left(n_{1}+n_{2}-2\right)$ 为两组样本共同的方差，该统计量 $t$ 在零假设 $\mu_{1} - \mu_{2} = \mu_{0}$ 为真的条件下服从置信度为 $n_{1}+n_{2}-2$ 的 Student's *t*-distribution

#### 样本数和方差都不相等

若两组独立正态分布样本 $x_{1i}$ 与 $x_{2i}$ 具有不同的样本数 $n_1$ 与 $n_2$，并且各自方差不相等，则两组样本总体期望值差 $\mu_1 - \mu_2$ 是否为 $\mu_0$ 可利用以下统计量 $t$ 检验
$$
t=\frac{\overline{x}_{1}-\overline{x}_{2}-\mu_{0}}{\sqrt{s_{1}^{2} / n_{1}+s_{2}^{2} / n_{2}}}
$$
其中 $i=1 \ldots n_{1}, j=1 \ldots n_{2}, \overline{x}_{1}=\left(\sum_{i=1}^{n_1} x_{1 i}\right)/n_{1},\overline{x}_{2}=\left(\sum_{j=1}^{n_2} x_{2j}\right) / n_{2}$ 为两组样本各自的平均数，

$s_{1}^{2}=\left(\sum_{i=1}^{n}\left(x_{1 i}-\overline{x}_{1}\right)^{2}\right) /\left(n_{1}-1\right) , s_{2}^{2}=\left(\sum_{j=1}^{n}\left(x_{2 j}-\overline{x}_{2}\right)^{2}\right) /\left(n_{2}-1\right)$ 为两组样本各自的方差。

该统计量 $t$ 在零假设 $\mu_{1} - \mu_{2} = \mu_{0}$ 为真的条件下服从置信度为 $df$ 的 Student's *t*-distribution
$$
d f=\frac{\left(s_{1}^{2} / n_{1}+s_{2}^{2} / n_{2}\right)^{2}}{\left(s_{1}^{2} / n_{1}\right)^{2} /\left(n_{1}-1\right)+\left(s_{2}^{2} / n_{2}\right)^{2} /\left(n_{2}-1\right)}
$$

