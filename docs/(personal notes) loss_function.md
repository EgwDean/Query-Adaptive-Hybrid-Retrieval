# Loss Function and Training

## What the model is doing

For each query $q$ the router receives a feature vector $\mathbf{x} \in \mathbb{R}^{16}$
and must output a scalar $\hat{y} \in [0, 1]$, the predicted weight $\hat{\alpha}$
for the sparse retriever.

The soft label $y \in [0, 1]$ encodes how much sparse outperformed dense on that
query. So $y$ is not a hard class — it is a probability: the probability that
the sparse retriever is the better choice. Training is supervised on these
probability targets.

---

## Loss function: Binary Cross-Entropy

For a single query with true label $y$ and prediction $\hat{y}$:

$$\mathcal{L}(y,\, \hat{y}) = -\bigl[y \log \hat{y} + (1 - y) \log(1 - \hat{y})\bigr]$$

The total training loss over a dataset of $N$ queries is the mean:

$$\mathcal{L}_\text{total} = -\frac{1}{N} \sum_{i=1}^{N}
\bigl[y_i \log \hat{y}_i + (1 - y_i) \log(1 - \hat{y}_i)\bigr]$$

---

## Where the formula comes from

The label $y$ defines a Bernoulli distribution over two outcomes
("sparse wins" with probability $y$, "dense wins" with probability $1-y$).
The model's prediction $\hat{y}$ defines another Bernoulli distribution.

Cross-entropy $H(P, Q) = -\sum_x P(x) \log Q(x)$ measures how many bits are
needed to describe events from distribution $P$ using a code designed for $Q$.
The less $Q$ matches $P$, the more bits are wasted.

Plugging in $P = (y,\, 1-y)$ and $Q = (\hat{y},\, 1-\hat{y})$:

$$H(P, Q) = -[y \log \hat{y} + (1-y) \log(1-\hat{y})]$$

which is exactly the BCE formula. Minimising cross-entropy is the same as
minimising the KL divergence $D_\text{KL}(P \| Q) = H(P,Q) - H(P)$, since the
entropy of the labels $H(P)$ is a constant with respect to the model parameters.

It is also equivalent to **maximum likelihood estimation**: minimising BCE
is the same as maximising

$$\log \mathcal{L} = \sum_{i=1}^{N} \bigl[y_i \log \hat{y}_i + (1-y_i)\log(1-\hat{y}_i)\bigr]$$

the log-likelihood of the training labels under a Bernoulli model parameterised
by $\hat{y}$.

---

## Why not MSE?

Mean squared error measures the squared distance between prediction and label:

$$\mathcal{L}_\text{MSE}(y, \hat{y}) = (y - \hat{y})^2$$

Two properties make BCE preferable here.

**1. Asymmetric penalty for confident errors.**

Consider $y = 1$ (sparse strongly wins) and two bad predictions:

| Prediction $\hat{y}$ | BCE | MSE |
|---|---|---|
| 0.9 (correct, confident) | 0.105 | 0.01 |
| 0.5 (uncertain) | 0.693 | 0.25 |
| 0.1 (wrong, confident) | 2.303 | 0.81 |

BCE grows without bound as a confident wrong prediction approaches 0:
$-\log \hat{y} \to \infty$ as $\hat{y} \to 0$.
MSE only grows quadratically. A model that says "I am 90% sure dense wins"
when sparse clearly wins is penalised 28× more by BCE than by MSE.
This discourages overconfident wrong predictions, which are the most damaging
errors in a routing context.

**2. No gradient saturation.**

Model outputs are typically produced by a sigmoid:
$\hat{y} = \sigma(z) = 1/(1 + e^{-z})$.
The gradient of the loss with respect to the pre-activation $z$ is:

$$\frac{\partial \mathcal{L}_\text{BCE}}{\partial z} = \hat{y} - y$$

Clean and never zero. With MSE the gradient picks up an extra $\sigma'(z) = \hat{y}(1-\hat{y})$ factor, which approaches zero near $\hat{y} \in \{0,1\}$, making gradients vanish and training slow when predictions are already near the extremes.

**3. Natural calibration for ties.**

When $y = 0.5$ (both retrievers perform equally), the loss-minimising prediction
is exactly $\hat{y} = 0.5$. The BCE is minimised at $\hat{y} = y$ for any $y$,
so a tie in training data naturally produces a prediction of 0.5 — which is
exactly static RRF. MSE shares this property, but BCE rewards calibrated
uncertainty more strongly when the model is wrong.

---

## How the loss is minimised

### General principle: gradient descent

The model has parameters $\boldsymbol{\theta}$ (weights in a neural network,
tree structure in XGBoost). The loss $\mathcal{L}(\boldsymbol{\theta})$ is a
surface in parameter space. Gradient descent steps downhill along this surface:

$$\boldsymbol{\theta} \leftarrow \boldsymbol{\theta} - \eta \cdot \nabla_{\boldsymbol{\theta}} \mathcal{L}$$

where $\eta$ is the learning rate. The gradient $\nabla \mathcal{L}$ points in
the direction of steepest ascent; subtracting it moves toward a minimum.

The gradient of BCE with respect to $\hat{y}_i$ is:

$$\frac{\partial \mathcal{L}}{\partial \hat{y}_i} = -\frac{y_i}{\hat{y}_i} + \frac{1 - y_i}{1 - \hat{y}_i} = \frac{\hat{y}_i - y_i}{\hat{y}_i (1 - \hat{y}_i)}$$

The sign tells the optimiser whether to push the prediction up or down:
if $\hat{y}_i > y_i$ (over-predicted), the gradient is positive, so the update
reduces $\hat{y}_i$. If $\hat{y}_i < y_i$, it increases it. The magnitude
is larger when the prediction is far from the label and when $\hat{y}$ is
near 0.5 (denominator small when $\hat{y}$ is extreme, but numerator is larger there too — they balance out to the clean form $\hat{y} - y$ at the pre-activation level).

### XGBoost: gradient boosting

XGBoost does not update a fixed set of weights. Instead it builds an **additive
ensemble of trees** $F(\mathbf{x}) = \sum_{t=1}^{T} f_t(\mathbf{x})$, where each
new tree $f_t$ corrects the errors of the current ensemble.

At step $t$, the ensemble prediction is $F_{t-1}(\mathbf{x})$. XGBoost finds
the next tree $f_t$ by fitting it to the **negative gradient** of the loss — the
direction each training prediction should move to reduce the loss:

$$g_i = \frac{\partial \mathcal{L}(y_i, \hat{y}_i)}{\partial \hat{y}_i}
\quad\text{(first-order gradient)}$$

$$h_i = \frac{\partial^2 \mathcal{L}(y_i, \hat{y}_i)}{\partial \hat{y}_i^2}
\quad\text{(second-order, used for tree structure search)}$$

For BCE with $\hat{y}_i = \sigma(F_{t-1}(\mathbf{x}_i))$:

$$g_i = \hat{y}_i - y_i \qquad h_i = \hat{y}_i (1 - \hat{y}_i)$$

The new tree is built greedily: at each node it finds the feature and split
threshold that maximally reduces the weighted sum of squared gradients
$\sum_i g_i^2 / h_i$ (the gain criterion). The leaf value assigned to a
group of training examples is $-\sum_i g_i / \sum_i h_i$, the Newton step.

After all $T$ trees are built the final prediction is:

$$\hat{y} = \sigma\!\left(\sum_{t=1}^{T} \eta \cdot f_t(\mathbf{x})\right)$$

where $\eta$ is the learning rate (called `learning_rate` in config) that shrinks
each tree's contribution to avoid overfitting.

---

## Loss formula used in this project

For a training set of $N$ queries with soft labels $y_i \in [0,1]$ and model
predictions $\hat{y}_i \in [0,1]$:

$$\boxed{\mathcal{L} = -\frac{1}{N} \sum_{i=1}^{N}
\bigl[\,y_i \log \hat{y}_i + (1 - y_i) \log(1 - \hat{y}_i)\,\bigr]}$$

- **Objective in XGBoost config**: `binary:logistic`
- **Output**: probability $\hat{y} \in (0, 1)$, used directly as $\hat{\alpha}$
- **Labels**: soft values in $[0, 1]$ from the label formula in `routing_features.md`
- **Reduction**: mean over all queries in the training fold
