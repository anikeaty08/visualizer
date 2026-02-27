"""
╔══════════════════════════════════════════════════════════════════════════════╗
║   📚  ML VISUALIZATION COOKBOOK                                             ║
║   Learn how every graph type is built — from scratch to production          ║
╠══════════════════════════════════════════════════════════════════════════════╣
║   This file teaches you HOW to make ML visualizations yourself.             ║
║   Each section shows the concept, the minimal raw code, and then how        ║
║   to call the same thing from ml_visualizer with zero effort.               ║
║                                                                             ║
║   Structure per recipe:                                                     ║
║     CONCEPT   → what this graph shows and when to use it                   ║
║     BUILD IT  → write it from scratch (learning the internals)             ║
║     LIBRARY   → call it from ml_visualizer (production use)               ║
║     TIPS      → gotchas, variations, what to look for                      ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

# ─────────────────────────────────────────────────────────────────────────────
# IMPORTS — what the best libraries are and why
# ─────────────────────────────────────────────────────────────────────────────
"""
LIBRARY CHOICES AND WHY THEY'RE THE BEST:

numpy       — The bedrock. All math, arrays, linear algebra. Nothing competes.
matplotlib  — The universal plotting engine. Every other library builds on it.
              Direct control of every pixel. Use when you need exact layout.
seaborn     — Statistical plots with beautiful defaults. Built on matplotlib.
              Best for: heatmaps, distributions, pairplots, regression plots.
scikit-learn — The best ML library in Python. Clean API, consistent interface.
              Best for: models, preprocessing, metrics, datasets, pipelines.
plotly      — Interactive charts in the browser. Best for exploration.
              plotly.express is the high-level API — use this, not graph_objects.
scipy       — Scientific computing. Best for: stats, signal processing, distributions.
umap-learn  — UMAP dimensionality reduction. Faster + often better than t-SNE.

Install everything:
  pip install numpy matplotlib seaborn scikit-learn scipy plotly umap-learn
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL STYLE — set once, affects all plots
# ─────────────────────────────────────────────────────────────────────────────

sns.set_theme(style="darkgrid", palette="muted", font_scale=1.1)
plt.rcParams.update({
    "figure.dpi":       110,      # crisp on most monitors
    "axes.titlesize":   13,
    "axes.labelsize":   11,
    "legend.fontsize":  10,
    "axes.spines.top":  False,    # cleaner look
    "axes.spines.right":False,
})

# Colour palette — consistent across all plots
CB, CR, CG, CO, CP, CT, CGR = (
    "#4C72B0",   # blue
    "#DD4444",   # red
    "#2CA02C",   # green
    "#FF7F0E",   # orange
    "#9467BD",   # purple
    "#17BECF",   # teal
    "#7F7F7F",   # grey
)


# ══════════════════════════════════════════════════════════════════════════════
# CHAPTER 1 — BASIC PLOTS
# ══════════════════════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────────────────────────────────────
# 1.1  LINE PLOT
# ─────────────────────────────────────────────────────────────────────────────
"""
CONCEPT:
  The simplest plot. Use it for: training loss curves, metrics over time,
  any sequence of values. Add multiple lines to compare runs.

WHEN TO USE:
  - Loss / accuracy over epochs
  - Any metric that changes over an ordered sequence
  - Comparing two continuous signals
"""

def recipe_line_plot():
    x = np.linspace(0, 10, 300)
    y_train = np.exp(-0.3 * x) + 0.05 * np.random.randn(300)
    y_val   = np.exp(-0.25 * x) + 0.1

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(x, y_train, color=CB, lw=2.5, label="Train loss")
    ax.plot(x, y_val,   color=CR, lw=2.5, label="Val loss",   ls="--")
    ax.fill_between(x, y_train, y_val, alpha=0.1, color=CR)    # gap shading
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.set_title("Training Curves")
    ax.legend()
    plt.tight_layout(); plt.show()

# From ml_visualizer:
# from ml_visualizer import plot_line, plot_training_history
# plot_line(epochs, loss)
# plot_training_history(train_loss, val_loss, train_acc, val_acc)


# ─────────────────────────────────────────────────────────────────────────────
# 1.2  SCATTER PLOT
# ─────────────────────────────────────────────────────────────────────────────
"""
CONCEPT:
  Shows the relationship between two variables. The most useful EDA plot.
  Colour-code by a third variable to reveal group structure.

WHEN TO USE:
  - Feature vs feature (EDA)
  - Feature vs target
  - Embedding / PCA visualizations
  - Predicted vs actual
"""

def recipe_scatter():
    np.random.seed(0)
    X = np.random.randn(200, 2)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)   # fake binary labels

    fig, ax = plt.subplots(figsize=(7, 6))
    for cls, col, name in zip([0, 1], [CR, CB], ["Class 0", "Class 1"]):
        mask = y == cls
        ax.scatter(X[mask, 0], X[mask, 1],
                   color=col, s=50, alpha=0.8,
                   edgecolors="k", lw=0.3, label=name)
    ax.set_xlabel("Feature 1"); ax.set_ylabel("Feature 2")
    ax.set_title("Scatter — two classes")
    ax.legend()
    plt.tight_layout(); plt.show()

# From ml_visualizer:
# plot_scatter(X[:,0], X[:,1], c=y)


# ─────────────────────────────────────────────────────────────────────────────
# 1.3  HISTOGRAM + KDE
# ─────────────────────────────────────────────────────────────────────────────
"""
CONCEPT:
  Shows the distribution of a single variable.
  KDE (Kernel Density Estimate) is a smooth version of the histogram.

WHEN TO USE:
  - Checking if residuals are normally distributed
  - Comparing train vs test distributions
  - Understanding feature distributions before scaling
"""

def recipe_histogram():
    np.random.seed(1)
    normal_data = np.random.randn(500)
    skewed_data = np.random.exponential(1, 500)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Method 1 — matplotlib histogram with seaborn KDE overlay
    axes[0].hist(normal_data, bins=30, color=CB, alpha=0.7,
                 density=True, edgecolor="k", lw=0.3)
    from scipy.stats import gaussian_kde
    kde = gaussian_kde(normal_data)
    xs  = np.linspace(normal_data.min(), normal_data.max(), 300)
    axes[0].plot(xs, kde(xs), color=CR, lw=2.5, label="KDE")
    axes[0].set_title("Normal distribution")
    axes[0].legend()

    # Method 2 — seaborn histplot (easier)
    sns.histplot(skewed_data, bins=30, kde=True, ax=axes[1],
                 color=CO, alpha=0.75)
    axes[1].set_title("Skewed distribution")

    plt.tight_layout(); plt.show()

# From ml_visualizer:
# plot_histogram(data, bins=30, kde=True)


# ─────────────────────────────────────────────────────────────────────────────
# 1.4  HEATMAP
# ─────────────────────────────────────────────────────────────────────────────
"""
CONCEPT:
  Colour-encodes a 2D matrix. Use for: correlation matrices, confusion matrices,
  weight matrices, attention maps, grid search results.

KEY TRICK:
  Always set center=0 for symmetric data (correlations, weights).
  Use annot=True for small matrices (< 15×15), False for large ones.
"""

def recipe_heatmap():
    import pandas as pd

    # Build a correlation matrix
    np.random.seed(0)
    X  = np.random.randn(100, 5)
    X[:, 1] = X[:, 0] * 0.8 + np.random.randn(100) * 0.3   # correlated pair
    df = pd.DataFrame(X, columns=["Age", "Income", "Score", "Tenure", "Spend"])

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(df.corr(), annot=True, fmt=".2f",
                cmap="coolwarm", center=0,
                square=True, linewidths=0.5,
                cbar_kws={"shrink": 0.8}, ax=ax)
    ax.set_title("Correlation Matrix")
    plt.tight_layout(); plt.show()

# From ml_visualizer:
# plot_heatmap(df.corr(), title="Correlation Matrix")


# ══════════════════════════════════════════════════════════════════════════════
# CHAPTER 2 — LINEAR REGRESSION
# ══════════════════════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────────────────────────────────────
# 2.1  FITTED LINE + RESIDUALS
# ─────────────────────────────────────────────────────────────────────────────
"""
CONCEPT:
  The fitted line shows the model. Residuals show what the model missed.
  Residuals should be: random, centred at 0, no pattern vs x.

WHAT TO LOOK FOR:
  - Curved residual pattern → non-linear relationship, need polynomial features
  - Fan shape (residuals grow) → heteroscedasticity
  - Outlier bars → influential points
"""

def recipe_linear_regression():
    np.random.seed(3)
    x = np.sort(np.random.uniform(0, 10, 40))
    y = 2.5 * x + 5 + np.random.randn(40) * 4

    # Fit with numpy polyfit (no sklearn needed)
    w, b    = np.polyfit(x, y, 1)
    y_hat   = w * x + b
    resid   = y - y_hat
    x_line  = np.linspace(0, 10, 300)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Linear Regression   ŷ = {w:.2f}x + {b:.2f}", fontweight="bold")

    # Panel 1 — data + line + residual sticks
    axes[0].scatter(x, y, color=CR, s=60, zorder=5, label="Data")
    axes[0].plot(x_line, w * x_line + b, color=CB, lw=2.5, label="Fit")
    for xi, yi, ri in zip(x, y, resid):
        axes[0].plot([xi, xi], [yi, yi - ri], color=CGR, lw=1.0, ls="--", zorder=3)
    axes[0].set_xlabel("x"); axes[0].set_ylabel("y"); axes[0].legend()
    axes[0].set_title("Data + Fitted Line")

    # Panel 2 — residual plot
    axes[1].bar(range(len(resid)), resid,
                color=[CG if r >= 0 else CR for r in resid],
                edgecolor="k", alpha=0.8)
    axes[1].axhline(0, color="black", lw=1.5)
    axes[1].set_xlabel("Sample index"); axes[1].set_ylabel("Residual (y − ŷ)")
    axes[1].set_title("Residual Plot")

    plt.tight_layout(); plt.show()

# From ml_visualizer:
# plot_linear_regression(x, y)


# ─────────────────────────────────────────────────────────────────────────────
# 2.2  COST FUNCTION — THE BOWL 🥣
# ─────────────────────────────────────────────────────────────────────────────
"""
CONCEPT:
  J(w,b) = (1/2m) Σ(wx+b − y)²
  When plotted over all possible (w,b), it forms a bowl (convex surface).
  Gradient descent rolls the ball down to the minimum.

  The 3D surface and its top-down contour map are the most famous
  visualizations in Andrew Ng's courses. Understanding this = understanding GD.

KEY INSIGHT:
  - The bowl has one minimum (convex) — GD always finds it for linear regression
  - Elongated contours = features on different scales = slow zigzag descent
  - Circular contours = scaled features = fast descent
"""

def recipe_cost_3d():
    from mpl_toolkits.mplot3d import Axes3D  # noqa

    # Data
    x = np.array([1., 2., 3., 4., 5.])
    y = np.array([200., 400., 600., 800., 1000.])

    # Cost function
    def J(w, b):
        return np.mean((w * x + b - y) ** 2) / 2

    # Build grid
    opt_w, opt_b = np.polyfit(x, y, 1)
    w_arr  = np.linspace(opt_w - 200, opt_w + 200, 80)
    b_arr  = np.linspace(opt_b - 200, opt_b + 200, 80)
    W, B   = np.meshgrid(w_arr, b_arr)
    Jgrid  = np.vectorize(J)(W, B)
    min_pt = np.unravel_index(np.argmin(Jgrid), Jgrid.shape)

    fig = plt.figure(figsize=(16, 6))

    # 3D Surface
    ax1 = fig.add_subplot(121, projection="3d")
    ax1.plot_surface(W, B, Jgrid, cmap="viridis", alpha=0.85, edgecolor="none")
    ax1.scatter(W[min_pt], B[min_pt], Jgrid[min_pt],
                color="red", s=150, zorder=10, label="Minimum")
    ax1.set_xlabel("w"); ax1.set_ylabel("b"); ax1.set_zlabel("J(w,b)")
    ax1.set_title("3D Cost Surface — The Bowl")
    ax1.view_init(elev=30, azim=225)
    ax1.legend()

    # Contour
    ax2 = fig.add_subplot(122)
    cf  = ax2.contourf(W, B, Jgrid, levels=50, cmap="viridis", alpha=0.85)
    ax2.contour(W, B, Jgrid, levels=20, colors="white", alpha=0.2, lw=0.5)
    plt.colorbar(cf, ax=ax2)
    ax2.scatter(W[min_pt], B[min_pt], color="red", s=200, marker="*", zorder=5)
    ax2.set_xlabel("w"); ax2.set_ylabel("b")
    ax2.set_title("Contour Map — Top-down View")

    plt.tight_layout(); plt.show()

# From ml_visualizer:
# plot_cost_3d()
# iplot_cost_3d()   ← rotate in browser


# ─────────────────────────────────────────────────────────────────────────────
# 2.3  GRADIENT DESCENT PATH
# ─────────────────────────────────────────────────────────────────────────────
"""
CONCEPT:
  Gradient descent takes steps in the direction of steepest descent.
  Visualising the path on the contour map makes this concrete.

  ∂J/∂w = (1/m) Σ(wx+b − y)·x
  ∂J/∂b = (1/m) Σ(wx+b − y)
  w ← w − α·∂J/∂w
  b ← b − α·∂J/∂b

WHAT TO LOOK FOR:
  - Smooth spiral inward → good learning rate
  - Big zigzag → learning rate too high or unscaled features
  - Barely moves → learning rate too low
"""

def recipe_gradient_descent_path():
    x = np.array([1., 2., 3., 4., 5.])
    y = np.array([200., 400., 600., 800., 1000.])
    m = len(x)

    # Run GD manually
    w, b  = 0.0, 0.0
    alpha = 0.01
    w_hist, b_hist, J_hist = [w], [b], []

    for _ in range(400):
        err = w * x + b - y
        w  -= alpha * np.dot(err, x) / m
        b  -= alpha * np.sum(err)    / m
        w_hist.append(w); b_hist.append(b)
        J_hist.append(np.mean(err**2) / 2)

    # Build cost grid for contour
    opt_w, opt_b = np.polyfit(x, y, 1)
    W_g, B_g = np.meshgrid(np.linspace(opt_w-200, opt_w+200, 80),
                             np.linspace(opt_b-200, opt_b+200, 80))
    Jg = np.vectorize(lambda w_, b_: np.mean((w_*x+b_-y)**2)/2)(W_g, B_g)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Path on contour
    axes[0].contourf(W_g, B_g, Jg, levels=50, cmap="viridis", alpha=0.85)
    axes[0].plot(w_hist, b_hist, "w.-", lw=1.5, ms=3, label="GD path")
    axes[0].scatter(w_hist[0],  b_hist[0],  color="yellow", s=120, zorder=5, label="Start")
    axes[0].scatter(w_hist[-1], b_hist[-1], color="red",    s=120, zorder=5, marker="*", label="End")
    axes[0].set_xlabel("w"); axes[0].set_ylabel("b"); axes[0].legend()
    axes[0].set_title("Gradient Descent Path on Contour")

    # Convergence
    axes[1].plot(J_hist, color=CB, lw=2.5)
    axes[1].set_xlabel("Iterations"); axes[1].set_ylabel("J")
    axes[1].set_title("Convergence Curve")

    plt.tight_layout(); plt.show()

# From ml_visualizer:
# plot_gradient_descent_path(x, y, alpha=0.01)
# plot_optimizer_comparison(x, y)


# ══════════════════════════════════════════════════════════════════════════════
# CHAPTER 3 — LOGISTIC REGRESSION
# ══════════════════════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────────────────────────────────────
# 3.1  SIGMOID FUNCTION
# ─────────────────────────────────────────────────────────────────────────────
"""
CONCEPT:
  σ(z) = 1 / (1 + e^{-z})
  Squashes any real number to (0,1) — interpretable as a probability.
  The decision boundary is where σ(z) = 0.5, i.e. where z = 0.

IMPORTANT PROPERTIES:
  σ(-z) = 1 − σ(z)          (symmetric)
  σ'(z) = σ(z)(1 − σ(z))   (easy derivative — max 0.25, causes vanishing gradients!)
"""

def recipe_sigmoid():
    z   = np.linspace(-8, 8, 400)
    sig = 1 / (1 + np.exp(-z))         # the sigmoid
    der = sig * (1 - sig)               # its derivative

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].plot(z, sig, color=CB, lw=3, label="σ(z)")
    axes[0].axhline(0.5, color=CR, ls="--", lw=1.5, label="Threshold = 0.5")
    axes[0].fill_between(z, 0.5, sig, where=(sig > 0.5), alpha=0.15, color=CB, label="Predict 1")
    axes[0].fill_between(z, sig, 0.5, where=(sig < 0.5), alpha=0.15, color=CO, label="Predict 0")
    axes[0].set_ylim(-0.05, 1.05); axes[0].set_xlabel("z"); axes[0].set_ylabel("σ(z)")
    axes[0].set_title("Sigmoid Function"); axes[0].legend()

    axes[1].plot(z, der, color=CR, lw=3)
    axes[1].set_xlabel("z"); axes[1].set_ylabel("σ'(z)")
    axes[1].set_title("Sigmoid Derivative\n(max = 0.25 → vanishing gradient risk)")

    plt.tight_layout(); plt.show()

# From ml_visualizer:
# plot_sigmoid()


# ─────────────────────────────────────────────────────────────────────────────
# 3.2  DECISION BOUNDARY
# ─────────────────────────────────────────────────────────────────────────────
"""
CONCEPT:
  The decision boundary is the surface where P(y=1|x) = 0.5.
  For logistic regression: w·x + b = 0.
  Visualise as a probability heatmap — shows model confidence everywhere.

HOW TO BUILD IT:
  1. Create a dense grid covering the feature space
  2. Get predict_proba for every point on the grid
  3. Plot as contourf
  4. Draw the 0.5 contour line as the decision boundary
"""

def recipe_decision_boundary():
    from sklearn.datasets       import make_classification
    from sklearn.linear_model   import LogisticRegression

    X, y = make_classification(n_samples=200, n_features=2, n_redundant=0,
                                n_clusters_per_class=1, random_state=42)
    model = LogisticRegression().fit(X, y)

    # Step 1 — build grid
    h  = 0.02
    xx, yy = np.meshgrid(np.arange(X[:,0].min()-0.5, X[:,0].max()+0.5, h),
                          np.arange(X[:,1].min()-0.5, X[:,1].max()+0.5, h))

    # Step 2 — predict probability on every grid point
    Z = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:,1].reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(8, 6))
    # Step 3 — heatmap
    cf = ax.contourf(xx, yy, Z, levels=50, cmap="RdBu", alpha=0.75)
    plt.colorbar(cf, ax=ax, label="P(y=1)")
    # Step 4 — boundary line at 0.5
    ax.contour(xx, yy, Z, levels=[0.5], colors="black", linewidths=2.5)
    ax.scatter(X[:,0], X[:,1], c=y, cmap="RdBu", edgecolors="k", s=50, zorder=5)
    ax.set_title("Logistic Regression Decision Boundary")
    plt.tight_layout(); plt.show()

# From ml_visualizer:
# plot_decision_boundary(X, y, model)
# iplot_decision_boundary(X, y, model)


# ══════════════════════════════════════════════════════════════════════════════
# CHAPTER 4 — NEURAL NETWORKS
# ══════════════════════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────────────────────────────────────
# 4.1  ACTIVATION FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────
"""
CONCEPT:
  Activations add non-linearity to the network. Without them,
  stacking layers is just multiplying matrices — still linear.

WHICH TO USE:
  Hidden layers  → ReLU (default), Leaky ReLU (dying ReLU problem), GELU (transformers)
  Output (binary) → Sigmoid
  Output (multi)  → Softmax
  RNNs           → Tanh

VANISHING GRADIENT:
  Sigmoid and Tanh derivatives max out at 0.25 and 1.0.
  Multiplied across many layers → gradients → 0.
  ReLU derivative is 0 or 1 → no shrinkage (but can "die").
"""

def recipe_activations():
    z = np.linspace(-5, 5, 400)

    fns = {
        "Sigmoid":    1 / (1 + np.exp(-np.clip(z, -500, 500))),
        "Tanh":       np.tanh(z),
        "ReLU":       np.maximum(0, z),
        "Leaky ReLU": np.where(z > 0, z, 0.1 * z),
        "Swish":      z / (1 + np.exp(-z)),
        "GELU":       0.5*z*(1+np.tanh(np.sqrt(2/np.pi)*(z+0.044715*z**3))),
    }
    colors = [CB, CG, CR, CO, CP, CT]

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for ax, (name, vals), col in zip(axes.flatten(), fns.items(), colors):
        ax.plot(z, vals, color=col, lw=2.5)
        ax.axhline(0, color=CGR, lw=0.7, ls=":")
        ax.axvline(0, color=CGR, lw=0.7, ls=":")
        ax.set_title(name); ax.set_xlabel("z"); ax.set_ylabel("a")

    plt.suptitle("Activation Functions", fontsize=14, fontweight="bold")
    plt.tight_layout(); plt.show()

# From ml_visualizer:
# plot_activation_functions()
# plot_activation_derivatives()


# ─────────────────────────────────────────────────────────────────────────────
# 4.2  NETWORK ARCHITECTURE DIAGRAM
# ─────────────────────────────────────────────────────────────────────────────
"""
CONCEPT:
  Draw nodes (neurons) and edges (weights) to show the network structure.
  Helps communicate the model to others and check your own design.

TECHNIQUE:
  - Place nodes in columns (one column = one layer)
  - Center each column vertically
  - Draw lines between every node in layer l and every node in layer l+1
"""

def recipe_nn_diagram():
    layer_sizes = [3, 5, 4, 2]    # input → hidden1 → hidden2 → output
    max_n = max(layer_sizes)

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.axis("off")
    ax.set_xlim(-0.5, len(layer_sizes) - 0.5)
    ax.set_ylim(-0.5, max_n + 0.5)

    pos = {}
    for l, n in enumerate(layer_sizes):
        offset = (max_n - n) / 2
        for i in range(n):
            y = offset + i
            pos[(l, i)] = (l, y)
            col = CB if l == 0 else CO if l == len(layer_sizes)-1 else CG
            ax.add_patch(plt.Circle((l, y), 0.22, color=col, ec="black", lw=1.2, zorder=3))

        # Draw edges to next layer
        if l < len(layer_sizes) - 1:
            for i in range(n):
                for j in range(layer_sizes[l+1]):
                    x0, y0 = pos[(l, i)]
                    x1, y1 = pos[(l+1, j)]
                    ax.plot([x0, x1], [y0, y1], color=CGR, lw=0.5, alpha=0.5, zorder=1)

    layer_names = ["Input", "Hidden 1", "Hidden 2", "Output"]
    for l, name in enumerate(layer_names):
        ax.text(l, -0.3, name, ha="center", fontsize=10, fontweight="bold")
        ax.text(l, max_n+0.2, f"({layer_sizes[l]})", ha="center", fontsize=9, color=CGR)

    ax.set_title("Neural Network Architecture", fontsize=14, fontweight="bold")
    plt.tight_layout(); plt.show()

# From ml_visualizer:
# plot_nn_architecture([3, 5, 4, 2], ['ReLU', 'ReLU', 'Sigmoid'])


# ─────────────────────────────────────────────────────────────────────────────
# 4.3  TRAINING HISTORY
# ─────────────────────────────────────────────────────────────────────────────
"""
CONCEPT:
  Plot loss and accuracy vs epoch for both train and validation sets.

DIAGNOSE FROM THIS PLOT:
  train loss ↓, val loss ↓ together  → still learning, good
  train loss ↓, val loss ↑           → overfitting — add regularisation / dropout
  both losses plateau high            → underfitting — more capacity, more data
  val loss spikes                     → bad batch, high learning rate
  gap between train and val           → high variance (model memorising)
"""

def recipe_training_history():
    epochs     = np.arange(1, 101)
    train_loss = 2.5 * np.exp(-0.04*epochs) + 0.08 + 0.02*np.random.randn(100)
    val_loss   = 2.5 * np.exp(-0.035*epochs) + 0.18 + 0.05*np.random.randn(100)
    val_loss[60:] += np.linspace(0, 0.5, 40)    # simulate overfitting at epoch 60

    train_acc = np.clip(1 - train_loss/3, 0, 1)
    val_acc   = np.clip(1 - val_loss/3,   0, 1)

    # Detect overfit point
    overfit = np.argmax(np.diff(val_loss) > 0.02) + 1 if np.any(np.diff(val_loss) > 0.02) else None

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Training History", fontsize=13, fontweight="bold")

    axes[0].plot(epochs, train_loss, color=CB, lw=2.5, label="Train Loss")
    axes[0].plot(epochs, val_loss,   color=CR, lw=2.5, label="Val Loss")
    if overfit:
        axes[0].axvline(overfit, color=CO, ls="--", lw=2, label=f"Overfit ≈ ep{overfit}")
    axes[0].fill_between(epochs, train_loss, val_loss, alpha=0.1, color=CR)
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss"); axes[0].legend()
    axes[0].set_title("Loss Curves")

    axes[1].plot(epochs, train_acc, color=CB, lw=2.5, label="Train Acc")
    axes[1].plot(epochs, val_acc,   color=CR, lw=2.5, label="Val Acc")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy"); axes[1].legend()
    axes[1].set_title("Accuracy Curves")

    plt.tight_layout(); plt.show()

# From ml_visualizer:
# plot_training_history(train_loss, val_loss, train_acc, val_acc)
# iplot_training_history(train_loss, val_loss, train_acc, val_acc)


# ══════════════════════════════════════════════════════════════════════════════
# CHAPTER 5 — EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────────────────────────────────────
# 5.1  CONFUSION MATRIX
# ─────────────────────────────────────────────────────────────────────────────
"""
CONCEPT:
  For classification: rows = true class, columns = predicted class.
  Diagonal = correct. Off-diagonal = errors.

  True Positive  (TP) — predicted 1, actually 1
  True Negative  (TN) — predicted 0, actually 0
  False Positive (FP) — predicted 1, actually 0  (Type I error)
  False Negative (FN) — predicted 0, actually 1  (Type II error)

  Precision = TP / (TP + FP)  — of all predicted positives, how many are right?
  Recall    = TP / (TP + FN)  — of all actual positives, how many did we catch?
  F1        = 2 · P · R / (P + R)
"""

def recipe_confusion_matrix():
    from sklearn.datasets  import load_iris
    from sklearn.svm       import SVC
    from sklearn.model_selection import train_test_split
    from sklearn.metrics   import confusion_matrix

    iris = load_iris()
    Xtr, Xte, ytr, yte = train_test_split(iris.data, iris.target,
                                            test_size=0.3, random_state=0)
    model = SVC().fit(Xtr, ytr)
    cm    = confusion_matrix(yte, model.predict(Xte))

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=iris.target_names,
                yticklabels=iris.target_names,
                linewidths=0.5, ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    plt.tight_layout(); plt.show()

# From ml_visualizer:
# plot_confusion_matrix(y_test, model.predict(X_test), class_names=NAMES)


# ─────────────────────────────────────────────────────────────────────────────
# 5.2  ROC CURVE
# ─────────────────────────────────────────────────────────────────────────────
"""
CONCEPT:
  ROC = Receiver Operating Characteristic.
  Plots True Positive Rate vs False Positive Rate at every threshold.
  AUC (Area Under Curve) summarises it in one number.

  AUC = 0.5 → random classifier (diagonal line)
  AUC = 1.0 → perfect classifier
  AUC > 0.8 → generally good

WHEN TO PREFER PR CURVE INSTEAD:
  For highly imbalanced datasets (1% positive), ROC looks great even for
  bad models. Use Precision-Recall curve instead.
"""

def recipe_roc_curve():
    from sklearn.datasets        import make_classification
    from sklearn.linear_model    import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics         import roc_curve, auc

    X, y = make_classification(500, random_state=0)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0)
    model  = LogisticRegression().fit(Xtr, ytr)
    y_prob = model.predict_proba(Xte)[:, 1]    # probability of class 1

    fpr, tpr, thresholds = roc_curve(yte, y_prob)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color=CB, lw=2.5, label=f"Model  (AUC = {roc_auc:.3f})")
    ax.plot([0,1],[0,1], color=CGR, ls="--", lw=1.5, label="Random (AUC = 0.5)")
    ax.fill_between(fpr, tpr, alpha=0.12, color=CB)
    ax.set_xlabel("False Positive Rate (FPR)")
    ax.set_ylabel("True Positive Rate (TPR / Recall)")
    ax.set_title("ROC Curve"); ax.legend(loc="lower right")
    plt.tight_layout(); plt.show()

# From ml_visualizer:
# plot_roc_curve(y_test, model.predict_proba(X_test)[:,1])
# plot_multiclass_roc(X_test, y_test, model)


# ─────────────────────────────────────────────────────────────────────────────
# 5.3  F1 vs THRESHOLD
# ─────────────────────────────────────────────────────────────────────────────
"""
CONCEPT:
  The default 0.5 threshold is rarely optimal.
  This plot shows how Precision, Recall, and F1 change as you vary the threshold.
  Pick the threshold that maximises F1 (or whatever trade-off you need).

WHEN TO USE:
  - Medical diagnosis (maximise recall — missing a case is worse than false alarm)
  - Spam detection (maximise precision — false positives are annoying)
  - Fraud detection (depends on costs)
"""

def recipe_f1_threshold():
    from sklearn.datasets        import make_classification
    from sklearn.linear_model    import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics         import f1_score, precision_score, recall_score

    X, y = make_classification(500, random_state=2)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=2)
    y_prob = LogisticRegression().fit(Xtr, ytr).predict_proba(Xte)[:, 1]

    thresholds = np.linspace(0.01, 0.99, 200)
    precs, recs, f1s = [], [], []
    for t in thresholds:
        preds = (y_prob >= t).astype(int)
        precs.append(precision_score(yte, preds, zero_division=0))
        recs.append(recall_score(yte, preds, zero_division=0))
        f1s.append(f1_score(yte, preds, zero_division=0))

    best_t = thresholds[np.argmax(f1s)]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(thresholds, precs, color=CB, lw=2, label="Precision")
    ax.plot(thresholds, recs,  color=CR, lw=2, label="Recall")
    ax.plot(thresholds, f1s,   color=CG, lw=3, label="F1")
    ax.axvline(best_t, color=CO, ls="--", lw=2, label=f"Best = {best_t:.2f}")
    ax.axvline(0.5,    color=CGR, ls=":", lw=1.5, label="Default = 0.5")
    ax.set_xlabel("Threshold"); ax.set_ylabel("Score")
    ax.set_title("Precision / Recall / F1 vs Threshold"); ax.legend()
    plt.tight_layout(); plt.show()

# From ml_visualizer:
# plot_f1_vs_threshold(y_test, y_prob)


# ══════════════════════════════════════════════════════════════════════════════
# CHAPTER 6 — CLUSTERING
# ══════════════════════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────────────────────────────────────
# 6.1  K-MEANS STEP BY STEP
# ─────────────────────────────────────────────────────────────────────────────
"""
CONCEPT:
  K-Means algorithm:
  1. Randomly initialise K centroids
  2. Assign each point to nearest centroid
  3. Move each centroid to mean of its cluster
  4. Repeat until convergence

VISUALISING EACH STEP:
  Run with max_iter=step for each step to see centroid movement.
  This is the clearest way to build intuition for the algorithm.
"""

def recipe_kmeans_steps():
    from sklearn.datasets import make_blobs
    from sklearn.cluster  import KMeans

    X, _ = make_blobs(n_samples=200, centers=3, cluster_std=0.9, random_state=42)
    steps = [1, 2, 3, 5, 10, 30]

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle("K-Means: Centroid Evolution", fontsize=14, fontweight="bold")

    for ax, step in zip(axes.flatten(), steps):
        km = KMeans(n_clusters=3, max_iter=step, n_init=1,
                    init="random", random_state=42).fit(X)
        ax.scatter(X[:,0], X[:,1], c=km.labels_, cmap="tab10",
                   alpha=0.65, s=30, edgecolors="k", lw=0.2)
        ax.scatter(km.cluster_centers_[:,0], km.cluster_centers_[:,1],
                   color="red", s=220, marker="X", zorder=6, label="Centroids")
        ax.set_title(f"After {step} iteration(s)"); ax.legend(fontsize=8)

    plt.tight_layout(); plt.show()

# From ml_visualizer:
# plot_kmeans_steps(X, k=3, max_steps=6)
# plot_elbow_method(X)
# plot_silhouette_score(X)


# ─────────────────────────────────────────────────────────────────────────────
# 6.2  ELBOW METHOD
# ─────────────────────────────────────────────────────────────────────────────
"""
CONCEPT:
  Run K-Means for K=1 to K_max. Plot inertia (WCSS) vs K.
  The "elbow" — where adding more clusters stops helping much — is the optimal K.

LIMITATION:
  The elbow is sometimes ambiguous. Use silhouette score to confirm.
"""

def recipe_elbow():
    from sklearn.datasets import make_blobs
    from sklearn.cluster  import KMeans

    X, _ = make_blobs(n_samples=300, centers=3, random_state=42)
    k_range  = range(1, 11)
    inertias = [KMeans(k, random_state=42, n_init=10).fit(X).inertia_ for k in k_range]

    # Detect elbow via second derivative
    d2    = np.diff(np.diff(inertias))
    elbow = int(np.argmax(d2)) + 2

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(k_range, inertias, "b-o", lw=2.5, ms=8)
    ax.axvline(elbow, color=CR, ls="--", lw=2, label=f"Elbow at K={elbow}")
    ax.set_xlabel("K"); ax.set_ylabel("Inertia (WCSS)")
    ax.set_title("Elbow Method"); ax.legend()
    plt.tight_layout(); plt.show()

# From ml_visualizer:
# plot_elbow_method(X)


# ══════════════════════════════════════════════════════════════════════════════
# CHAPTER 7 — PCA & EMBEDDINGS
# ══════════════════════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────────────────────────────────────
# 7.1  PCA SCREE PLOT
# ─────────────────────────────────────────────────────────────────────────────
"""
CONCEPT:
  PCA finds directions of maximum variance in the data.
  The scree plot shows how much variance each principal component explains.
  Choose the number of components where cumulative variance reaches ~90-95%.

HOW PCA WORKS:
  1. Centre the data (subtract mean)
  2. Compute covariance matrix
  3. Eigen-decompose → principal components (directions of max variance)
  4. Project data onto top-k components
"""

def recipe_pca_scree():
    from sklearn.datasets    import load_wine
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA

    wine = load_wine()
    X    = StandardScaler().fit_transform(wine.data)
    pca  = PCA().fit(X)
    evr  = pca.explained_variance_ratio_
    cum  = np.cumsum(evr)
    n90  = np.searchsorted(cum, 0.90) + 1

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Scree plot
    idx = np.arange(1, len(evr)+1)
    axes[0].bar(idx, evr, color=CB, alpha=0.8, edgecolor="k", lw=0.5)
    axes[0].plot(idx, cum, "r-o", lw=2, ms=5, label="Cumulative")
    axes[0].axhline(0.90, color=CG, ls="--", lw=1.5, label="90% threshold")
    axes[0].axvline(n90,  color=CO, ls="--", lw=1.5, label=f"{n90} PCs → 90%")
    axes[0].set_xlabel("Principal Component"); axes[0].set_ylabel("Explained Variance")
    axes[0].set_title("Scree Plot"); axes[0].legend()

    # 2D projection
    X2 = PCA(2).fit_transform(X)
    axes[1].scatter(X2[:,0], X2[:,1], c=wine.target, cmap="Set1",
                    s=50, alpha=0.8, edgecolors="k", lw=0.3)
    axes[1].set_xlabel("PC 1"); axes[1].set_ylabel("PC 2")
    axes[1].set_title("2D PCA Projection")

    plt.tight_layout(); plt.show()

# From ml_visualizer:
# plot_pca_scree(X)
# plot_pca_2d(X, y, labels=CLASS_NAMES)
# plot_tsne(X, y)
# plot_umap(X, y)


# ══════════════════════════════════════════════════════════════════════════════
# CHAPTER 8 — INTERACTIVE CHARTS WITH PLOTLY
# ══════════════════════════════════════════════════════════════════════════════
"""
CONCEPT:
  plotly.express (px) is the fastest way to make interactive charts.
  Charts render in the browser — zoom, pan, hover on every point.

WHEN TO USE PLOTLY vs MATPLOTLIB:
  Matplotlib → publications, saving images, precise layout, subplots
  Plotly     → exploration, sharing with stakeholders, 3D rotation, hover info

GOLDEN RULE:
  Use px (plotly.express) not go (graph_objects) unless you need
  full manual control. px is cleaner, faster, and covers 95% of cases.
"""

def recipe_plotly_basics():
    try:
        import plotly.express as px
        import pandas as pd
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        print("pip install plotly"); return

    # 1 — Scatter
    np.random.seed(0)
    df = pd.DataFrame({
        "x":     np.random.randn(150),
        "y":     np.random.randn(150) * 0.5 + np.random.randn(150),
        "label": np.random.choice(["A","B","C"], 150),
        "size":  np.random.randint(5, 20, 150),
    })
    fig1 = px.scatter(df, x="x", y="y", color="label", size="size",
                      title="Interactive Scatter",
                      template="plotly_dark")
    fig1.show()

    # 2 — Line (training history)
    e = np.arange(1, 101)
    t_loss = np.exp(-0.04*e) + 0.05*np.random.randn(100)
    v_loss = np.exp(-0.03*e) + 0.1
    fig2 = make_subplots(rows=1, cols=1)
    fig2.add_trace(go.Scatter(x=e, y=t_loss, name="Train", line=dict(color="royalblue")))
    fig2.add_trace(go.Scatter(x=e, y=v_loss, name="Val",   line=dict(color="tomato")))
    fig2.update_layout(title="Training History", template="plotly_dark",
                        xaxis_title="Epoch", yaxis_title="Loss")
    fig2.show()

# From ml_visualizer:
# iplot_scatter(x, y, color=labels)
# iplot_training_history(train_loss, val_loss)
# iplot_cost_3d()         ← rotate the bowl in 3D
# iplot_pca_3d(X, y)     ← rotate PCA in 3D


# ══════════════════════════════════════════════════════════════════════════════
# CHAPTER 9 — COMMON PATTERNS & GOTCHAS
# ══════════════════════════════════════════════════════════════════════════════

"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PATTERN 1 — Subplots with shared axes
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
def recipe_subplots():
    # sharex / sharey link axes — great for comparing plots at same scale
    fig, axes = plt.subplots(2, 3, figsize=(15, 8),
                               sharex=False, sharey=False)
    # Access: axes[row, col]  or axes.flatten() to iterate

    # GridSpec gives more control over spacing
    fig2 = plt.figure(figsize=(14, 6))
    gs   = gridspec.GridSpec(2, 3, figure=fig2, hspace=0.4, wspace=0.3)
    ax_big  = fig2.add_subplot(gs[:, 0])        # tall left panel
    ax_top  = fig2.add_subplot(gs[0, 1:])       # wide top right
    ax_bot1 = fig2.add_subplot(gs[1, 1])
    ax_bot2 = fig2.add_subplot(gs[1, 2])

    ax_big.set_title("Tall Panel")
    ax_top.set_title("Wide Top Panel")
    plt.tight_layout(); plt.show()


"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PATTERN 2 — Decision boundary for ANY sklearn model
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
def recipe_any_model_boundary(model, X, y, ax=None):
    """
    Universal decision boundary recipe.
    Works with any sklearn model that has predict() or predict_proba().
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    h  = 0.02
    xx, yy = np.meshgrid(
        np.arange(X[:,0].min()-0.5, X[:,0].max()+0.5, h),
        np.arange(X[:,1].min()-0.5, X[:,1].max()+0.5, h))

    if hasattr(model, "predict_proba"):
        Z = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:,1]
    else:
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])

    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, levels=50, cmap="RdBu", alpha=0.7)
    ax.contour(xx, yy, Z, levels=[0.5], colors="black", linewidths=2)
    ax.scatter(X[:,0], X[:,1], c=y, cmap="RdBu", edgecolors="k", s=40, zorder=5)
    return ax


"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PATTERN 3 — Annotating points on a scatter
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
def recipe_annotate():
    np.random.seed(0)
    x = np.random.randn(10)
    y = np.random.randn(10)
    labels = [f"P{i}" for i in range(10)]

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(x, y, s=80, color=CB, edgecolors="k")
    for xi, yi, lbl in zip(x, y, labels):
        ax.annotate(lbl, (xi, yi),
                    textcoords="offset points", xytext=(6, 6),
                    fontsize=9, color=CR)
    ax.set_title("Annotated Scatter")
    plt.tight_layout(); plt.show()


"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PATTERN 4 — Save figure at high quality
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
def recipe_save_figure():
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3], [4, 5, 3])
    ax.set_title("My Plot")

    # PNG at 300 DPI — good for papers
    fig.savefig("plot.png", dpi=300, bbox_inches="tight", facecolor="white")

    # SVG — vector, infinitely scalable for presentations
    fig.savefig("plot.svg", bbox_inches="tight")

    # PDF — vector, for LaTeX
    fig.savefig("plot.pdf", bbox_inches="tight")

    plt.show()


"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PATTERN 5 — Colour maps: which to use when
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
"""
SEQUENTIAL (one direction, 0 → max):
  viridis  — default, perceptually uniform, colourblind-safe  ✅
  plasma   — warm alternative to viridis
  Blues    — clean for positive-only data (confusion matrices)

DIVERGING (negative / zero / positive):
  RdBu     — red→white→blue, great for decision boundaries and correlations ✅
  coolwarm — softer version of RdBu
  PiYG, PRGn — colourblind-safe diverging

QUALITATIVE (discrete categories):
  tab10    — 10 distinct colours, colourblind-safe ✅
  Set1     — vivid 9-colour set
  Set2     — softer pastel 8-colour set

AVOID:
  jet      — misleading, not perceptually uniform, distorts data
  rainbow  — same issues
"""


# ══════════════════════════════════════════════════════════════════════════════
# CHAPTER 10 — FULL WORKFLOWS
# ══════════════════════════════════════════════════════════════════════════════

"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WORKFLOW A — Complete EDA on a new dataset
  Run these in order before touching any model.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
def workflow_eda(df, X, y, feature_names, class_names, X_test=None):
    from ml_visualizer import (plot_missing_values, plot_class_imbalance,
                                plot_feature_target_correlation,
                                plot_outlier_detection, plot_pairplot,
                                plot_train_test_distribution, plot_heatmap)
    import pandas as pd

    print("Step 1 — Shape and types")
    print(f"  Rows: {len(df)}  Cols: {df.shape[1]}")
    print(df.dtypes.value_counts())

    print("\nStep 2 — Missing values")
    plot_missing_values(df)

    print("\nStep 3 — Target distribution")
    plot_class_imbalance(y, class_names=class_names)

    print("\nStep 4 — Feature importance (correlation)")
    plot_feature_target_correlation(X, y, feature_names=feature_names)

    print("\nStep 5 — Outliers")
    plot_outlier_detection(X, feature_names=feature_names)

    print("\nStep 6 — Feature relationships")
    plot_pairplot(df, hue='target')

    print("\nStep 7 — Correlation matrix")
    plot_heatmap(pd.DataFrame(X, columns=feature_names).corr(),
                 title="Feature Correlation Matrix")

    if X_test is not None:
        print("\nStep 8 — Covariate shift check")
        plot_train_test_distribution(X, X_test, feature_names=feature_names)


"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WORKFLOW B — After model training: full evaluation suite
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
def workflow_evaluate(model, X_test, y_test, class_names=None):
    from ml_visualizer import (plot_confusion_matrix, plot_roc_curve,
                                plot_precision_recall, plot_f1_vs_threshold,
                                plot_calibration_curve, plot_error_analysis,
                                plot_feature_importance, plot_partial_dependence)

    y_pred = model.predict(X_test)
    y_prob = (model.predict_proba(X_test)[:, 1]
              if hasattr(model, "predict_proba") else None)

    plot_confusion_matrix(y_test, y_pred, class_names=class_names)

    if y_prob is not None:
        plot_roc_curve(y_test, y_prob)
        plot_precision_recall(y_test, y_prob)
        plot_f1_vs_threshold(y_test, y_prob)
        plot_calibration_curve(y_test, y_prob)

    plot_error_analysis(y_test, y_pred, X_test, class_names=class_names)

    if hasattr(model, "feature_importances_"):
        plot_feature_importance(model)
        plot_partial_dependence(model, X_test)


"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WORKFLOW C — Debugging a neural network
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
def workflow_debug_nn(model=None, history=None, X=None, y=None):
    from ml_visualizer import (plot_training_history, plot_vanishing_gradient,
                                plot_weight_distributions, plot_lr_schedule,
                                plot_tsne, plot_loss_landscape)

    print("Check 1 — Training curves (look for divergence / plateau / overfit)")
    if history is not None:
        plot_training_history(
            history.get('loss'),     history.get('val_loss'),
            history.get('accuracy'), history.get('val_accuracy'))
    else:
        plot_training_history()   # demo data

    print("Check 2 — Vanishing gradients")
    plot_vanishing_gradient()

    print("Check 3 — Weight distributions (look for zeros / explosions)")
    if model is not None and hasattr(model, 'layers'):
        weights = [l.get_weights()[0] for l in model.layers if l.get_weights()]
        plot_weight_distributions(weights)
    else:
        plot_weight_distributions()

    print("Check 4 — LR schedule")
    plot_lr_schedule()

    print("Check 5 — Embedding quality (t-SNE)")
    if X is not None and y is not None:
        plot_tsne(X, y)

    print("Check 6 — Loss landscape")
    plot_loss_landscape()


# ══════════════════════════════════════════════════════════════════════════════
# RUN ALL RECIPES
# ══════════════════════════════════════════════════════════════════════════════

ALL_RECIPES = [
    (recipe_line_plot,              "1.1  Line Plot"),
    (recipe_scatter,                "1.2  Scatter Plot"),
    (recipe_histogram,              "1.3  Histogram + KDE"),
    (recipe_heatmap,                "1.4  Heatmap"),
    (recipe_linear_regression,      "2.1  Linear Regression"),
    (recipe_cost_3d,                "2.2  Cost Function — The Bowl"),
    (recipe_gradient_descent_path,  "2.3  Gradient Descent Path"),
    (recipe_sigmoid,                "3.1  Sigmoid Function"),
    (recipe_decision_boundary,      "3.2  Decision Boundary"),
    (recipe_activations,            "4.1  Activation Functions"),
    (recipe_nn_diagram,             "4.2  NN Architecture Diagram"),
    (recipe_training_history,       "4.3  Training History"),
    (recipe_confusion_matrix,       "5.1  Confusion Matrix"),
    (recipe_roc_curve,              "5.2  ROC Curve"),
    (recipe_f1_threshold,           "5.3  F1 vs Threshold"),
    (recipe_kmeans_steps,           "6.1  K-Means Steps"),
    (recipe_elbow,                  "6.2  Elbow Method"),
    (recipe_pca_scree,              "7.1  PCA Scree"),
    (recipe_plotly_basics,          "8.1  Plotly Interactive"),
    (recipe_subplots,               "9.1  Subplots"),
    (recipe_annotate,               "9.3  Annotate Scatter"),
    (recipe_save_figure,            "9.4  Save Figure"),
]


def run_all_recipes(skip=None):
    """
    Run every recipe in sequence. Close each window to move to the next.

    Usage:
        run_all_recipes()
        run_all_recipes(skip=['recipe_plotly_basics'])
    """
    skip = set(skip or [])
    total = len(ALL_RECIPES)
    print(f"\n📚  Running {total} cookbook recipes\n")
    for i, (fn, title) in enumerate(ALL_RECIPES, 1):
        if fn.__name__ in skip:
            print(f"  ⏭  Skipping [{i}] {title}")
            continue
        print(f"  ▶  [{i}/{total}]  {title}")
        try:
            fn()
        except Exception as e:
            print(f"      ⚠  Failed: {e}")
    print("\n✅  All recipes done!\n")


if __name__ == "__main__":
    # Uncomment what you want to run:
    # run_all_recipes()
    # recipe_cost_3d()
    # recipe_training_history()
    # recipe_decision_boundary()
    pass