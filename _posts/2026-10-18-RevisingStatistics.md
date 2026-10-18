Over the past two days, I started revising my BSc Statistics course. This time, I am following my own syllabus and using a completely different book: All of Statistics by Larry Wasserman.

My goal is to build a deeper foundational understanding before moving into statistical learning and conducting self-directed research on Asian Strike options using my own Monte Carlo simulator.

Below are some of my handwritten notes from the past few days, digitized into Markdown using AI.

Have a good reading!

# Reminders

## Covariance

$$Cov(X, Y) = E((X - \mu_X)(Y - \mu_Y))$$

$\Leftrightarrow_{\text{Theorem}} Cov(X, Y) = E(XY) - E(X)E(Y)$  

## Variance-Covariance Matrix

$$V(X) = \begin{pmatrix} C(X_1, X_1) & C(X_1, X_2) & \cdots & C(X_1, X_k) \\ C(X_2, X_1) & C(X_2, X_2) & \cdots & C(X_2, X_k) \\ \vdots & \vdots & \ddots & \vdots \\ C(X_k, X_1) & C(X_k, X_2) & \cdots & C(X_k, X_k) \end{pmatrix}$$

where $X = \begin{pmatrix} X_1 \\ \vdots \\ X_k \end{pmatrix}$ and $C(X_i, X_i) = V(X_i)$.

## Conditional Expectations

$E(X\vert{}Y=y) = \begin{cases} \sum x f_{X\vert{}Y}(x\vert{}y) dx & \text{(Discrete)} \\ \int x f_{X\vert{}Y}(x\vert{}y) dx & \text{(Continuous)} \end{cases}$  

where:

- $f_{X\vert{}Y}(x\vert{}y) \overset{\text{(Discrete)}}{=} P(X=x\vert{}Y=y) = \frac{P(X=x, Y=y)}{P(Y=y)} = \frac{f_{X,Y}(x,y)}{f_Y(y)}$  
    
- $f_{X\vert{}Y}(x\vert{}y) \overset{\text{(Continuous)}}{=} \frac{f_{X,Y}(x,y)}{f_Y(y)}, f_Y(y) > 0$  
    

**Properties of** $f_{X,Y}$**:** $f_{X,Y}$ is $P(X=x, Y=y)$ if $X,Y$ are discrete. Else, it is such that: (i) $f_{X,Y}(x,y) \geq 0 \quad \forall (x,y)$ (ii) $\int_{\mathbb{R}} \int_{\mathbb{R}} f_{X,Y}(x,y) dx dy = 1$ (iii) $\forall A \subset \mathbb{R}^2 \left( P((X,Y) \in A) = \iint_A f_{X,Y} \right)$  

**Variance Reminder:** $E\left[(X-\mu)^2\right] = E(X^2) - E(X)^2$  

# One also benefits from reminding the CLT:

## Theorem (Central Limit Theorem)

$\{X_i\}_1^\infty$ iid with mean $\mu$ and variance $\sigma^2$. Let $\overline{X}_n = \frac{1}{n}\sum_{i=1}^n X_i$. Then,

$$Z_n := \frac{\overline{X}_n - \mu}{\sqrt{V(\overline{X}_n)}} = \frac{\sqrt{n}(\overline{X}_n - \mu)}{\sigma} \leadsto Z \sim N(0,1)$$

One can ask: _"How accurate are the approximations given by the CLT?"_

## Theorem (Berry-Esseen Inequality)

Suppose $E\vert{}X_1^3\vert{} < \infty$. Then

$$\sup_z \vert{}P(Z_n \leq z) - \Phi(z)\vert{} \leq \frac{33}{4} \frac{E\vert{}X_1 - \mu\vert{}^3}{\sqrt{n} \cdot \sigma^3}$$

If $Y_n$ has a limiting Normal distribution, the following method allows us to find the limiting distribution of $g(Y_n)$ where $g$ is any smooth function.

## Theorem (Delta Method)

Suppose that $\frac{\sqrt{n}(Y_n - \mu)}{\sigma} \leadsto N(0,1)$ and that $g$ is a differentiable function such that $g'(\mu) \neq 0$. Then $\frac{\sqrt{n}(g(Y_n) - g(\mu))}{\vert{}g'(\mu)\vert{}\sigma} \leadsto N(0,1)$. In other words,

$$Y_n \approx N(\mu, \sigma^2/n) \implies g(Y_n) \approx N\left(g(\mu), g'(\mu)^2 \frac{\sigma^2}{n}\right)$$

# Chapter 6 | Models, Statistical Inference and Learning

Statistical inference, or learning, as it is called in computer science, is the process of using data to infer the distribution that generated the data.

The typical question in this field of study would be: Given $X_1, \dots, X_n \sim F$, a sample, how do we infer $F$.

In some cases we might just infer some feature of $F$ such as its mean.

## $\S$ Parametric and Nonparametric Models $\S$  

A **statistical Model** $\mathcal{F}$ is a set of distributions (or densities, or regression functions).

A **parametric model** is a set $\mathcal{F}$ that can be parametrized by a finite number of parameters. For example, assuming the data follows/comes from a Normal distribution, one model is

$$\mathcal{F} = \left\{ f(x; \mu, \sigma) = \frac{1}{\sigma\sqrt{2\pi}} \exp\left(-\frac{1}{2\sigma^2}(x-\mu)^2\right), \mu \in \mathbb{R}, \sigma > 0 \right\}$$

Such model is a **two-parameter model**. In general, a parametric model has the form

$$\mathcal{F} = \{ f(x; \theta) \mid \theta \in \Theta \}$$

where $\theta$ is an unknown parameter (or vector of such object) that take values in $\Theta$. If $\theta = (\theta_1, \dots, \theta_n)$ and we are only interested in one component, say $\theta_i$, then the remaining parameters are called **nuisance parameters**.

A **nonparametric model** is a set $\mathcal{F}$ that cannot be parametrized by a finite number of parameters. For example, $\mathcal{F}_{ALL} = \{ \text{all CDF's} \}$ is nonparametric.

**Example:** Let $X_1, \dots, X_n \sim F$. Suppose that we mean to estimate $\mu = E(X_1) = \int x dF(x)$ assuming only that $\mu$ exists. The mean $\mu$ may be thought of as a function of $F$: $\mu = T(F) = \int x dF(x)$. In general, any function of $F$ is called a **statistical functional**.

Other examples would be the variance, $T(F) = \int x^2 dF(x) - \left(\int x dF(x)\right)^2$, and the median, $T(F) = F^{-1}(1/2)$.

**E.g.:** Suppose we observe pairs of data $(X_i, Y_i)_{i=1}^n$. Perhaps $X_i$ is the blood pressure and $Y_i$ is how long the subject lived. $X$ is called a **predictor** (or regressor). $Y$ is called the **outcome** (or the response variable). We call $r(x) = E(Y\vert{}X=x)$ the **regression function**. If we assume that $r \in \mathcal{F}$ where $\mathcal{F}$ is finite-dimensional, then we have a **parametric regression model**. If we assume the opposite, we have a **nonparametric regression model**.

Our goal is called **prediction** of $Y$ based on a new $X$ data point. If $Y$ is discrete, like $Y \in \{die, live\}$, then the prediction is then called **classification**. If our goal is to estimate the function $r$, then we call this a **regression** (or curve) estimation. Regression models are often written as

$$Y = r(X) + \epsilon \quad \text{where } E(\epsilon) = 0$$

Actually, one can always rewrite a regression model this way. $\left[ \epsilon = Y - r(X) \implies Y = Y - r(X) + r(X) = r(X) + \epsilon \text{. And } E(\epsilon) = E(E(\epsilon\vert{}X)) = E(E(Y-r(X)\vert{}X)) = E(E(Y\vert{}X) - r(X)) = E(r(X) - r(X)) = 0. \right]$  

**Notation:** If $\mathcal{F} = \{f(x; \theta) \mid \theta \in \Theta\}$ is a parametric model, we write $P_\theta(X \in A) = \int_A f(x; \theta) dx$ and $E_\theta(r(X)) = \int r(x) f(x; \theta) dx$. Similarly, $V_\theta = E_\theta(r(X)^2) - E_\theta(r(X))^2$.

## $\S$ Fundamental Concepts in Inference $\S$ 

Many (but not all) inferential problems can be identified as one of three types:

- $\to$ Estimation
    
- $\to$ Confidence Sets
    
- $\to$ Hypothesis Testing
    

_(These problems will be treated in detail later. For now, we will give a brief introduction to the ideas. Following the book.)_

### $\S$ Point Estimation $\S$ 

**Point estimation** refers to providing a single "best guess" of some quantity of interest. The quantity of interest could be a parameter in a parametric model, a CDF, a PDF, a regression function, or a prediction for a future value $Y$.

By convention, we denote a point estimate of $\theta$ by $\hat{\theta}$ or $\hat{\theta}_n$. Remember that $\theta$ is a fixed, unknown quantity. The estimate $\hat{\theta}$ depends on the data, so $\hat{\theta}$ is a **random variable**. More formally, let $X_1, \dots, X_n$ be $n$ IID data points from some distribution $F$. A point estimator $\hat{\theta}_n$ of $\theta$ is some function of $X_1, \dots, X_n$:

$$\hat{\theta}_n = g(X_1, \dots, X_n)$$

The bias of an estimator is defined by

$$\text{bias}(\hat{\theta}_n) = E_\theta(\hat{\theta}_n) - \theta$$

We say that $\hat{\theta}_n$ is **unbiased** if $\text{bias}(\hat{\theta}_n) = 0$. _(Unbiasedness used to get much attention, but these days, is considered less important; we will use many biased estimators.)_

A reasonable requirement for an estimator is that it should converge to the true parameter value as we collect more and more data. This requirement is quantified by the following definition.

**Definition:** A point estimator $\hat{\theta}_n$ of a parameter $\theta$ is **consistent** if $\hat{\theta}_n \xrightarrow{P} \theta$. //

The distribution of $\hat{\theta}_n$ is called the **sampling distribution**. The standard deviation of $\hat{\theta}_n$ is called the **standard error** ($se \equiv se(\hat{\theta}_n) := \sqrt{V(\hat{\theta}_n)}$). Often, the $se$ depends on the unknown $F$. In those cases, $se$ is an unknown quantity but we can, usually, estimate it. The estimated $se$ is denoted $\hat{se}$.

**Example:** Let $X_1, \dots, X_n \sim \text{Bernoulli}(p)$ and let $\hat{p}_n = n^{-1}\sum_{i=1}^n X_i$. Then $E(\hat{p}_n) = n^{-1}\sum_{i=1}^n E(X_i) = p$, so $\hat{p}_n$ is unbiased. The standard error is $se = \sqrt{V(\hat{p}_n)} = \sqrt{p(1-p)/n}$. Hence, $\hat{se} = \sqrt{\hat{p}(1-\hat{p})/n}$. //

The quality of a point estimate is sometimes assessed by the **mean squared error** (MSE)

$$MSE = E_\theta((\hat{\theta}_n - \theta)^2) \cdot \left[ \text{Reminder: } E_\theta((\hat{\theta} - \theta)^2) = \int_{\mathbb{R}^n} (\hat{\theta}_n(x) - \theta)^2 f(x;\theta)dx \right]$$

**Theorem:** $MSE = \text{bias}^2(\hat{\theta}_n) + V_\theta(\hat{\theta}_n)$. **Proof:** Let $\overline{\theta}_n = E_\theta(\hat{\theta}_n)$. Then $MSE = E_\theta((\hat{\theta}_n - \theta)^2) =$ $= E_\theta((\hat{\theta}_n - \overline{\theta}_n + \overline{\theta}_n - \theta)^2) =$ $= E_\theta((\hat{\theta}_n - \overline{\theta}_n)^2) + (\overline{\theta}_n - \theta)^2 =$ $= V_\theta(\hat{\theta}_n) + \text{bias}^2(\hat{\theta}_n) \quad \square$  

**Theorem:** If $\text{bias} \to 0$ and $se \to 0$ as $n \to \infty$ then $\hat{\theta}_n$ is consistent, that is, $\hat{\theta}_n \xrightarrow{P} \theta$. **Proof:** $(\text{bias} \to 0, se \to 0) \implies MSE \to 0 \implies \hat{\theta}_n \xrightarrow{mq} \theta \implies \hat{\theta}_n \xrightarrow{P} \theta$. $\quad \square$  

From this theorem and the latter example, $\hat{p}_n \xrightarrow{P} p$.

**Definition:** An estimator is **asymptotically Normal** if $\frac{\hat{\theta}_n - \theta}{se} \leadsto N(0,1)$.

### $\S$ Confidence Sets $\S$

A $1-\alpha$ **confidence interval** for a parameter $\theta$ is an interval $C_n = (a, b)$, where $a = a(X_1, \dots, X_n), b = b(X_1, \dots, X_n)$ are functions such that

$$P_\theta(\theta \in C_n) \geq 1 - \alpha, \quad \forall \theta \in \Theta$$

We call $1-\alpha$ the **coverage** of this interval. If $\theta$ is a vector, we use a **confidence set** (such as a sphere or an ellipse).

**Example:** Again, one falls into the Bernoulli setting. Let $C_n = (\hat{p}_n - \varepsilon_n, \hat{p}_n + \varepsilon_n)$, where $\varepsilon_n^2 = \log(2/\alpha)/2n$. By applying Hoeffding's inequality,

$$P(p \in C_n) \geq 1 - \alpha, \quad \forall p \in [0, 1].$$

Hence, $C_n$ is a $1-\alpha$ confidence interval. //

**Theorem:** Suppose that $\hat{\theta}_n \approx N(\theta, \hat{se}^2)$. Let $\Phi$ be the CDF of a standard Normal and let $z_{\alpha/2} = \Phi^{-1}(1 - (\alpha/2))$, that is, $P(Z > z_{\alpha/2}) = \alpha/2$ and $P(-z_{\alpha/2} \leq Z \leq z_{\alpha/2}) = 1 - \alpha$ where $Z \sim N(0,1)$. Let $C_n = (\hat{\theta}_n - z_{\alpha/2}\hat{se}, \hat{\theta}_n + z_{\alpha/2}\hat{se})$. Then

$$P_\theta(\theta \in C_n) \to 1 - \alpha \quad (\text{as } n \to \infty).$$

**Proof:** Let $Z_n := (\hat{\theta}_n - \theta) / \hat{se}$. By assumption, $Z_n \leadsto Z \sim N(0,1)$. Hence, $P_\theta(\theta \in C_n) = P_\theta(\hat{\theta}_n - z_{\alpha/2}\hat{se} < \theta < \hat{\theta}_n + z_{\alpha/2}\hat{se}) =$ $= P_\theta\left(-z_{\alpha/2} < \frac{\hat{\theta}_n - \theta}{\hat{se}} < z_{\alpha/2}\right) \to P(-z_{\alpha/2} < Z < z_{\alpha/2}) = 1 - \alpha \quad \square$  

### $\S$ Hypothesis Testing $\S$ 

In hypothesis testing, we start with some default theory, the **null hypothesis**, which is denoted $H_0$, and we ask:

**Q.:** _With this data, do we have enough evidence to reject my theory?_

If not, we retain/accept the null hypothesis. _(This is the main idea, we will detail each subsection.)_

One will now try to follow the following sequence of chapters:

![Chapter Diagram](_posts/files/Tikz_Oct18_2026.svg)

