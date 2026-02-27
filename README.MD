# 📊 ml_visualizer

> One import. Call a function. See your graph.
> 82 static plots · 9 interactive Plotly charts · 10 math helpers
> Every function ships with built-in demo data — zero arguments needed.

```python
from ml_visualizer import *

plot_cost_3d()              # works instantly, no data needed
plot_cost_3d(x, y)          # or pass your own arrays
run_all_demos()             # gallery mode — see every plot
list_all_functions()        # print the full catalogue
```

---

## Setup

```bash
pip install numpy matplotlib seaborn scikit-learn scipy plotly umap-learn
```

| Library | Needed for |
|---|---|
| numpy, matplotlib, seaborn | Everything — required |
| scikit-learn | Demo data, models, metrics |
| scipy | Gaussian density, stats, signal |
| plotly | `iplot_*` interactive charts |
| umap-learn | `plot_umap()` |

Missing libraries print a one-line hint and skip gracefully.

---

## Core Ideas

**Zero-argument default** — every `plot_*` has demo data built-in.

**Pass your own data** — every parameter is optional, override only what you need.

**Pass any sklearn model** — `plot_decision_boundary`, `plot_feature_importance`,
`plot_error_analysis` etc. accept any fitted sklearn estimator.

**Static vs Interactive** — `plot_*` → matplotlib window.  `iplot_*` → Plotly in browser.

**Math helpers are real functions** — `sigmoid()`, `gradient_descent()` etc. usable in your own code.

---

## Quick Reference

### EDA — Run These First

```python
plot_missing_values(df)
plot_class_imbalance(y, class_names=["A","B"])
plot_feature_target_correlation(X, y, feature_names=[...])
plot_outlier_detection(X, method='iqr')
plot_pairplot(df, hue='target')
plot_train_test_distribution(X_train, X_test)
```

### Linear Regression

```python
plot_linear_regression(x, y)
plot_polynomial_regression(x, y, degrees=[1,4,9])
plot_predicted_vs_actual(y_true, y_pred)
plot_qq_residuals(y_true, y_pred)
```

### The Bowl & Gradient Descent

```python
plot_cost_3d(x, y)                          # 3D surface + contour
plot_cost_3d(x, y, elev=45, azim=60)        # rotate camera
iplot_cost_3d(x, y)                         # interactive rotation
plot_gradient_descent_path(x, y, alpha=0.01)
plot_learning_rate_comparison(x, y)
plot_optimizer_comparison(x, y)             # SGD vs Momentum vs Adam
plot_lr_schedule()                          # cosine / step / warmup
plot_lr_finder(losses, lrs)
```

### Logistic Regression

```python
plot_sigmoid()
plot_softmax(z)
plot_log_loss()
plot_decision_boundary(X, y, model)
```

### Neural Networks

```python
plot_nn_architecture([784, 256, 128, 10], ['ReLU','ReLU','Softmax'])
plot_training_history(train_loss, val_loss, train_acc, val_acc)
plot_activation_functions()
plot_dropout_effect(drop_rate=0.4)
plot_batch_normalization()
plot_vanishing_gradient(n_layers=10)
plot_weight_initialization()
plot_weight_distributions(model)
plot_loss_landscape()
plot_attention_heatmap(attn_matrix, tokens)
```

### Regularisation & Diagnostics

```python
plot_overfit_underfit(x, y)
plot_regularization_lambda(x, y)
plot_bias_variance_tradeoff()
plot_learning_curves(X, y)
plot_validation_curve(X, y, param_name='C')
plot_hyperparameter_heatmap(X, y)
```

### SVM

```python
plot_svm_margin(X, y, C=1.0)
plot_kernel_comparison(X, y)
```

### Trees & Ensembles

```python
plot_decision_tree(X, y, max_depth=4)
plot_ensemble_boundaries(X, y)
```

### Clustering

```python
plot_kmeans_steps(X, k=3)
plot_elbow_method(X)
plot_silhouette_score(X)
plot_dbscan(X, eps=0.3)
plot_dendrogram(X)
plot_gmm(X, n_components=3)
```

### PCA & Embeddings

```python
plot_pca_scree(X)
plot_pca_2d(X, y, labels=NAMES)
plot_tsne(X, y, perplexity=30)
plot_umap(X, y)
```

### Evaluation

```python
plot_confusion_matrix(y_true, y_pred, class_names=NAMES)
plot_roc_curve(y_true, y_prob)
plot_multiclass_roc(X, y, model)
plot_precision_recall(y_true, y_prob)
plot_f1_vs_threshold(y_true, y_prob)
plot_calibration_curve(y_true, y_prob)
plot_error_analysis(y_test, y_pred, X_test)
plot_feature_importance(model, feature_names=NAMES)
plot_partial_dependence(model, X_test)
plot_cv_folds(X, y, n_folds=5)
```

### Time Series

```python
plot_time_series(data, window=20)
plot_autocorrelation(data, max_lag=40)
```

### Interactive (Plotly)

```python
iplot_cost_3d()         # rotate the bowl
iplot_training_history(train_loss, val_loss)
iplot_pca_3d(X, y)      # rotate in 3D
iplot_confusion_matrix(y_true, y_pred)
iplot_decision_boundary(X, y, model)
iplot_roc_curve(y_true, y_prob)
iplot_scatter(x, y, color=labels)
iplot_kmeans(X, k=3)
```

---

## Math Helpers

```python
sigmoid(z)                          # 1 / (1 + e^{-z})
relu(z)                             # max(0, z)
leaky_relu(z, a=0.1)
elu(z); tanh_fn(z); swish(z); gelu(z); linear_fn(z)

compute_cost(x, y, w, b)            # MSE cost → scalar

w, b, hist = gradient_descent(x, y, alpha=0.01, n_iter=1000)
# hist = { 'J': [...], 'w': [...], 'b': [...] }
plot_convergence(hist['J'])
```

---

## Common Workflows

### Debugging a stuck model
```python
plot_learning_rate_comparison(x, y)
plot_cost_3d(x, y)
w, b, hist = gradient_descent(x, y, alpha=0.01)
plot_convergence(hist['J'])
plot_normalization_comparison(X)
```

### Bias vs Variance diagnosis
```python
plot_overfit_underfit(x, y)
plot_bias_variance_tradeoff()
plot_learning_curves(X, y)
plot_validation_curve(X, y, param_name='max_depth')
```

### Full post-training evaluation
```python
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]
plot_confusion_matrix(y_test, y_pred, class_names=NAMES)
plot_roc_curve(y_test, y_prob)
plot_f1_vs_threshold(y_test, y_prob)
plot_calibration_curve(y_test, y_prob)
plot_error_analysis(y_test, y_pred, X_test)
```

### Plug in any sklearn model
```python
from sklearn.ensemble import GradientBoostingClassifier
model = GradientBoostingClassifier().fit(X_train, y_train)
plot_decision_boundary(X_test, y_test, model)
plot_feature_importance(model, feature_names=NAMES)
plot_partial_dependence(model, X_test)
plot_error_analysis(y_test, model.predict(X_test), X_test)
```

---

## Andrew Ng Course Map

| Topic | Function |
|---|---|
| The Bowl J(w,b) | `plot_cost_3d()` |
| GD path on contour | `plot_gradient_descent_path()` |
| Feature scaling ellipses | `plot_feature_scaling_effect()` |
| Sigmoid | `plot_sigmoid()` |
| Log loss | `plot_log_loss()` |
| Underfit/overfit | `plot_overfit_underfit()` |
| Bias-Variance | `plot_bias_variance_tradeoff()` |
| K-Means animation | `plot_kmeans_steps()` |
| Anomaly detection | `plot_anomaly_detection()` |
| Gradient checking | `plot_gradient_checking()` |
| Dropout | `plot_dropout_effect()` |
| Batch norm | `plot_batch_normalization()` |
| Adam vs SGD | `plot_optimizer_comparison()` |
| Vanishing gradient | `plot_vanishing_gradient()` |
| Xavier/He init | `plot_weight_initialization()` |

---

## Tips

```python
# Resize all plots
plt.rcParams['figure.figsize'] = (12, 7)

# Save any plot
plt.savefig('plot.png', dpi=300, bbox_inches='tight')

# Skip slow plots
run_all_demos(skip=['plot_pairplot', 'plot_tsne'])

# Full gallery including Plotly
run_all_demos(interactive=True)
```

---

*4600+ lines · 82 static · 9 interactive · 10 math helpers*
*Every function: zero arguments required, fully customisable*

---

## What's in ml_cookbook.py

A companion learning file — teaches you HOW these plots are built from scratch:

- Chapter 1 — Basic plots (line, scatter, histogram, heatmap)
- Chapter 2 — Linear regression + the cost bowl + gradient descent
- Chapter 3 — Sigmoid + decision boundary
- Chapter 4 — Activations + NN diagram + training history
- Chapter 5 — Confusion matrix + ROC + F1 vs threshold
- Chapter 6 — K-Means steps + elbow
- Chapter 7 — PCA scree
- Chapter 8 — Plotly interactive
- Chapter 9 — Patterns: subplots, annotations, saving, colormaps
- Chapter 10 — Full workflows: EDA, evaluation, NN debugging

Each recipe: raw from-scratch code → then the one-liner from ml_visualizer.

```python
# Run all cookbook recipes
from ml_cookbook import run_all_recipes
run_all_recipes()
```