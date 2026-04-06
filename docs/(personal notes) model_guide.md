# Model Guide — Weak-Signal Router Candidates

This document describes the 9 model families evaluated in
`src/weak_signal_model_grid_search.py`, how each one works, and what
every tunable hyperparameter controls.

All models receive the same 16 query-level features (see
`docs/routing_features.md`) and predict a weight
$\hat{\alpha}(q) \in [0, 1]$ that is plugged into the wRRF fusion
formula.

---

## Training modes

| Mode | Models | Labels | Loss | Alpha source |
|---|---|---|---|---|
| **Classifier** | Logistic Regression, Gaussian NB, LDA | Binarised at 0.5 | Cross-entropy (native) | `predict_proba[:, 1]` |
| **Regressor (MSE)** | Random Forest, Extra Trees, AdaBoost, MLP | Soft $[0, 1]$ | Mean squared error | `predict` clipped to $[0, 1]$ |
| **Regressor (CE)** | XGBoost, LightGBM | Soft $[0, 1]$ | Binary cross-entropy | `predict` (sigmoid output) |

---

## 1. Logistic Regression

**Family:** linear model (classifier).

A single linear layer followed by a sigmoid maps the 16 features to
a probability.  Decision boundary is a hyperplane in feature space.

$$P(y = 1 \mid \mathbf{x}) = \sigma(\mathbf{w}^T \mathbf{x} + b)$$

Trained by minimising L2-regularised binary cross-entropy (log loss)
via the L-BFGS quasi-Newton solver.

### Parameters

| Parameter | Values | Description |
|---|---|---|
| `C` | 0.001 – 100 | Inverse regularisation strength.  Smaller $C$ = stronger L2 penalty on the weight vector, which shrinks coefficients toward zero and reduces overfitting.  $C$ relates to the weight-decay parameter $\lambda$ by $C = 1/\lambda$. |

Fixed: `penalty="l2"`, `solver="lbfgs"`, `max_iter=1000`.

---

## 2. Random Forest (Regressor)

**Family:** bagging ensemble of decision trees.

Trains $B$ independent decision trees, each on a bootstrap sample of
the training data and a random subset of features at every split.  The
final prediction is the arithmetic mean of all tree predictions.

Randomisation in both the row (bootstrap) and column (feature subset)
dimensions decorrelates the individual trees and reduces variance
compared to a single deep tree, with only a small increase in bias.

### Parameters

| Parameter | Values | Description |
|---|---|---|
| `n_estimators` | 50 – 500 | Number of trees $B$.  More trees reduce variance but increase training time linearly. |
| `max_depth` | 3, 5, 8, 12, None | Maximum depth of each tree.  `None` = grow until leaves are pure or contain `min_samples_leaf` samples.  Shallower trees = higher bias, lower variance. |
| `min_samples_split` | 2, 5, 10 | Minimum samples required to split an internal node.  Larger values prevent the tree from learning very specific patterns (regularisation). |
| `min_samples_leaf` | 1, 2, 4 | Minimum samples in a leaf.  Acts as a smoothing constraint: larger values produce more conservative predictions. |
| `max_features` | `"sqrt"`, `"log2"` | Number of features considered at each split.  `"sqrt"` = $\sqrt{p}$, `"log2"` = $\log_2(p)$ where $p = 16$.  Fewer features = more randomisation between trees. |

Fixed: `bootstrap=True`, `n_jobs=1` (outer parallelism used instead).

---

## 3. Extra Trees (Regressor)

**Family:** bagging ensemble of *extremely randomised* trees.

Very similar to Random Forest, with two key differences:

1. **No bootstrap:** each tree sees the full training set (by default).
2. **Random split thresholds:** instead of finding the best split among the
   candidate features, Extra Trees picks a random threshold for each
   candidate and chooses the best among those random splits.

These two changes make Extra Trees faster to train than RF (no sorting
for optimal splits) and inject more randomisation, which can reduce
variance further at the cost of slightly higher bias.

### Parameters

Same grid as Random Forest (see above).

---

## 4. XGBoost (Regressor, binary cross-entropy)

**Family:** gradient boosting over decision trees.

Builds an additive ensemble of shallow trees sequentially.  Each new
tree fits the *negative gradient* (pseudo-residuals) of the loss with
respect to the current ensemble's predictions.  XGBoost adds:

- **Newton boosting:** uses both first- and second-order gradient
  information for each split.
- **Regularisation terms** on leaf weights ($L_1$ and $L_2$).
- **Column subsampling** per tree.

With `objective="binary:logistic"` the loss is binary cross-entropy
and the model outputs a probability after a sigmoid transform — this
is equivalent to training on soft labels with log loss.

### Parameters

| Parameter | Values | Description |
|---|---|---|
| `n_estimators` | 50 – 300 | Number of boosting rounds (trees).  More rounds can overfit if the learning rate is high. |
| `max_depth` | 3, 4, 6, 8 | Maximum tree depth.  Controls interaction order: depth $d$ allows up to $d$-way feature interactions. |
| `learning_rate` | 0.01 – 0.3 | Shrinkage factor $\eta$ applied to each new tree.  Smaller $\eta$ needs more trees but generalises better. |
| `subsample` | 0.8, 1.0 | Fraction of training rows sampled per tree (without replacement).  Values $< 1$ add stochastic regularisation. |
| `colsample_bytree` | 0.8, 1.0 | Fraction of features sampled per tree.  Decorrelates successive trees. |
| `min_child_weight` | 1, 3, 5 | Minimum sum of instance-weight (Hessian) in a child node.  Larger values make the algorithm more conservative by requiring more evidence before creating a leaf. |

Fixed: `objective="binary:logistic"`, `eval_metric="logloss"`,
`verbosity=0`, `n_jobs=1`.

---

## 5. LightGBM (Regressor, binary cross-entropy)

**Family:** gradient boosting over decision trees (leaf-wise growth).

Functionally similar to XGBoost but uses a different tree-building
strategy:

- **Leaf-wise growth:** expands the leaf with the largest loss
  reduction, rather than growing level-by-level.  This often
  converges faster but can overfit on small datasets if `num_leaves`
  is too high.
- **Histogram-based splits:** bins continuous features into discrete
  histograms, which speeds up split finding.

With `objective="binary"` the loss is binary cross-entropy and
`predict()` returns post-sigmoid probabilities, same as XGBoost's
`binary:logistic`.

### Parameters

| Parameter | Values | Description |
|---|---|---|
| `n_estimators` | 50 – 300 | Number of boosting rounds. |
| `num_leaves` | 15, 31, 63 | Maximum number of leaves per tree.  This is the primary complexity control in leaf-wise growth (replaces `max_depth` as the main knob).  Rule of thumb: `num_leaves` $\leq 2^{\text{max\_depth}}$. |
| `learning_rate` | 0.01 – 0.3 | Shrinkage factor. |
| `subsample` | 0.8, 1.0 | Row subsampling fraction.  Requires `subsample_freq > 0` to take effect (set to 1 in the script). |
| `colsample_bytree` | 0.8, 1.0 | Feature subsampling fraction per tree. |
| `min_child_weight` | 1, 3, 5 | Minimum sum of Hessian in a leaf (same semantics as XGBoost). |

Fixed: `objective="binary"`, `subsample_freq=1`, `verbose=-1`,
`n_jobs=1`.

---

## 6. MLP (Regressor, MSE)

**Family:** feedforward neural network (multi-layer perceptron).

One or more fully-connected hidden layers with a non-linear activation,
followed by a linear output node.  Trained with the Adam optimiser
minimising mean squared error on the soft labels.

The MLP can model arbitrary non-linear decision boundaries, but is
more sensitive to feature scaling (z-score normalisation is applied)
and hyperparameter choices than tree-based models.

Early stopping on a 15 % validation split prevents overfitting:
training halts if the validation loss does not improve for 20
consecutive epochs.

### Parameters

| Parameter | Values | Description |
|---|---|---|
| `hidden_layer_sizes` | e.g. `[64, 32]` | Number and size of hidden layers.  `[64, 32]` = first hidden layer has 64 neurons, second has 32.  Deeper/wider networks have more capacity but are harder to train and more prone to overfitting. |
| `alpha` | 1e-4 – 0.1 | L2 regularisation penalty on all weights.  Larger values shrink weights toward zero, reducing overfitting. |
| `learning_rate_init` | 0.001, 0.01 | Initial step size for Adam.  Smaller values train more slowly but can find flatter minima. |
| `batch_size` | 16, 32 | Number of samples per gradient update.  Smaller batches add noise that can help escape local minima but slow down training. |
| `activation` | `"relu"`, `"tanh"` | Hidden-layer activation function.  ReLU: $\max(0, x)$, cheap and avoids vanishing gradients for positive inputs.  Tanh: $\tanh(x) \in (-1, 1)$, zero-centred, sometimes better for small networks. |

Fixed: `solver="adam"`, `max_iter=500`, `early_stopping=True`,
`validation_fraction=0.15`, `n_iter_no_change=20`.

---

## 7. Gaussian Naive Bayes (Classifier)

**Family:** generative probabilistic model.

Assumes that features within each class follow independent Gaussian
distributions.  Fits per-class mean $\mu_{ck}$ and variance
$\sigma^2_{ck}$ for every feature $k$, then applies Bayes' theorem:

$$P(y = c \mid \mathbf{x}) \propto P(y = c) \prod_{k=1}^{p} \mathcal{N}(x_k \mid \mu_{ck}, \sigma^2_{ck})$$

Despite the strong independence assumption, GNB trains instantly and
can serve as a useful lower-bound baseline.  Labels are binarised at
0.5; the output alpha is `predict_proba[:, 1]`.

### Parameters

| Parameter | Values | Description |
|---|---|---|
| `var_smoothing` | 1e-11 – 1e-7 | Portion of the largest per-feature variance that is added to all variances for numerical stability.  Larger values smooth the Gaussians more aggressively, making the classifier less sensitive to outlier features. |

---

## 8. AdaBoost (Regressor)

**Family:** adaptive boosting ensemble.

Trains a sequence of weak learners (shallow decision trees), where
each new tree focuses on the samples that previous trees predicted
poorly.  After each round the *sample weights* are increased for
misclassified instances and decreased for correct ones, so the next
tree concentrates on the hard cases.

Unlike gradient boosting (XGBoost/LightGBM), AdaBoost reweights
samples rather than fitting residuals, giving it a different
inductive bias.

### Parameters

| Parameter | Values | Description |
|---|---|---|
| `n_estimators` | 50 – 500 | Number of boosting rounds. |
| `learning_rate` | 0.01 – 1.0 | Shrinkage applied to each new tree's contribution.  There is a trade-off with `n_estimators`: lower learning rates need more rounds. |
| `base_max_depth` | 1, 2, 3 | Maximum depth of each base decision tree.  Depth 1 = decision stumps (single split), which is the classic AdaBoost configuration. Depth 2–3 allows limited feature interactions. |

Fixed: base estimator is `DecisionTreeRegressor`.

---

## 9. Linear Discriminant Analysis (Classifier)

**Family:** generative linear classifier.

Models each class as a multivariate Gaussian with a *shared* covariance
matrix $\Sigma$.  The decision boundary is linear because the
log-likelihood ratio between the two classes simplifies to a linear
function of $\mathbf{x}$ when both classes share $\Sigma$.

$$\delta_c(\mathbf{x}) = \mathbf{x}^T \Sigma^{-1} \boldsymbol{\mu}_c - \tfrac{1}{2} \boldsymbol{\mu}_c^T \Sigma^{-1} \boldsymbol{\mu}_c + \ln P(y = c)$$

Compared to logistic regression, LDA explicitly estimates the data
distribution rather than learning the boundary directly.  This can be
an advantage when the Gaussian assumption roughly holds, but a
disadvantage when it does not.

Labels are binarised at 0.5; the output alpha is `predict_proba[:, 1]`.

### Parameters

| Parameter | Values | Description |
|---|---|---|
| `solver` | `"svd"`, `"lsqr"`, `"eigen"` | Algorithm for fitting.  `"svd"` = singular value decomposition (does not compute the covariance matrix explicitly, most numerically stable).  `"lsqr"` = least-squares solution.  `"eigen"` = eigendecomposition of the covariance matrix. |
| `shrinkage` | `null`, `"auto"`, 0.1, 0.5, 0.9 | Regularisation of the covariance estimate.  Only used with `"lsqr"` or `"eigen"`.  `"auto"` uses the Ledoit–Wolf lemma to pick the optimal shrinkage coefficient automatically.  A numeric value $\in [0, 1]$ interpolates between the empirical covariance ($0$) and the identity matrix ($1$).  Shrinkage is critical when the number of samples is small relative to the number of features. |

Note: combinations with `solver="svd"` and `shrinkage != null` are
skipped automatically (SVD does not support explicit shrinkage).
