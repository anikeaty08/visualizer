"""
╔══════════════════════════════════════════════════════════════════════════╗
║          🧠  ML VISUALIZER  — All Graphs in One File                   ║
║          From Linear Regression ➜ Deep Learning                        ║
╠══════════════════════════════════════════════════════════════════════════╣
║  HOW TO USE:                                                            ║
║    from ml_visualizer import *                                          ║
║    plot_sigmoid()                   ← works with NO data (uses demo)   ║
║    plot_cost_3d(x, y)               ← pass YOUR data                   ║
║    plot_decision_boundary(X, y, model)                                  ║
║                                                                         ║
║  EVERY function works with ZERO arguments (uses built-in demo data).   ║
║  Pass your own arrays to visualize your actual data.                    ║
╠══════════════════════════════════════════════════════════════════════════╣
║  SECTIONS                                                               ║
║  ─────────────────────────────────────────────────────────────────────  ║
║  [S1]  IMPORTS & STYLE CONFIG                                           ║
║  [S2]  MATH HELPERS  (sigmoid, relu, cost, gradient descent…)          ║
║  [S3]  BASIC PLOTS   (line, scatter, bar, hist, box, heatmap, pair)    ║
║  [S4]  LINEAR REGRESSION  (fit, residuals, multi-feature)              ║
║  [S5]  COST FUNCTION 1-D  (J vs w, J vs b)                            ║
║  [S6]  COST FUNCTION 3-D  (surface + contour  — THE BOWL)             ║
║  [S7]  GRADIENT DESCENT   (path, convergence, learning-rate compare)   ║
║  [S8]  FEATURE SCALING    (before/after contour, normalisation)        ║
║  [S9]  LOGISTIC REGRESSION (sigmoid, log-loss, decision boundary)      ║
║  [S10] ACTIVATION FUNCTIONS (relu, tanh, sigmoid, leaky, swish, gelu) ║
║  [S11] NEURAL NETWORK      (layer outputs, training curves)            ║
║  [S12] REGULARISATION      (underfit/overfit, λ sweep)                 ║
║  [S13] BIAS-VARIANCE        (tradeoff curve, learning curves)          ║
║  [S14] DECISION BOUNDARIES  (linear, non-linear, multiclass)           ║
║  [S15] CLUSTERING           (K-Means steps, elbow method)              ║
║  [S16] ANOMALY DETECTION    (Gaussian density + threshold)             ║
║  [S17] PCA                  (scree, 2-D projection, reconstruction)    ║
║  [S18] EVALUATION METRICS   (confusion matrix, ROC, PR curve)          ║
║  [S19] PLOTLY INTERACTIVE   (plotly.express equivalents of everything) ║
║  [S20] DEMO RUNNER          (run_all_demos() — see everything)         ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

# ════════════════════════════════════════════════════════════════════════════
# [S1]  IMPORTS & STYLE CONFIG
# ════════════════════════════════════════════════════════════════════════════

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# ── Optional: Plotly for interactive charts ──────────────────────────────────
try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_OK = True
except ImportError:
    PLOTLY_OK = False
    print("⚠  plotly not found → pip install plotly   (interactive plots disabled)")

# ── Optional: Scikit-learn for demo data & models ───────────────────────────
try:
    from sklearn.datasets import (make_classification, make_regression,
                                   make_blobs, make_circles, make_moons,
                                   load_iris, load_wine)
    from sklearn.linear_model  import LogisticRegression, LinearRegression
    from sklearn.preprocessing import StandardScaler, PolynomialFeatures
    from sklearn.decomposition import PCA
    from sklearn.cluster       import KMeans
    from sklearn.pipeline      import make_pipeline
    from sklearn.model_selection import train_test_split, learning_curve
    from sklearn.metrics import (confusion_matrix, roc_curve, auc,
                                  precision_recall_curve, average_precision_score)
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm     import SVC
    SK_OK = True
except ImportError:
    SK_OK = False
    print("⚠  scikit-learn not found → pip install scikit-learn")

# ── Optional: Scipy for Gaussian density ────────────────────────────────────
try:
    from scipy.stats import multivariate_normal
    SP_OK = True
except ImportError:
    SP_OK = False

# ── Global matplotlib style ──────────────────────────────────────────────────
sns.set_theme(style="darkgrid", palette="muted", font_scale=1.1)
plt.rcParams.update({
    "figure.dpi":        110,
    "axes.titlesize":    13,
    "axes.labelsize":    11,
    "legend.fontsize":   10,
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

# colour constants (feel free to edit these globally)
CB   = "#4C72B0"   # blue
CR   = "#DD4444"   # red
CG   = "#2CA02C"   # green
CO   = "#FF7F0E"   # orange
CP   = "#9467BD"   # purple
CT   = "#17BECF"   # teal
CGR  = "#7F7F7F"   # grey

print("✅  ml_visualizer ready!  Call any plot_*() or run run_all_demos()")


# ════════════════════════════════════════════════════════════════════════════
# [S2]  MATH HELPERS
#   Pure-numpy utilities used internally AND available to you.
# ════════════════════════════════════════════════════════════════════════════

def sigmoid(z):
    """σ(z) = 1 / (1 + e^-z)  — clips z to avoid overflow."""
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

def relu(z):        return np.maximum(0, z)
def leaky_relu(z, a=0.1): return np.where(z > 0, z, a * z)
def elu(z, a=1.0):  return np.where(z > 0, z, a * (np.exp(z) - 1))
def tanh_fn(z):     return np.tanh(z)
def swish(z):       return z * sigmoid(z)
def gelu(z):
    return 0.5 * z * (1 + np.tanh(np.sqrt(2/np.pi) * (z + 0.044715 * z**3)))
def linear_fn(z):   return z


def compute_cost(x, y, w, b):
    """
    MSE cost for 1-feature linear regression.
      J(w,b) = (1/2m) Σ (w·x + b − y)²

    Args:
        x, y : 1-D numpy arrays  (features, targets)
        w, b : scalars
    Returns:
        scalar float
    """
    m = len(x)
    return float(np.sum((w * x + b - y) ** 2) / (2 * m))


def gradient_descent(x, y, w_init=0.0, b_init=0.0,
                     alpha=0.01, n_iter=500):
    """
    Run gradient descent on linear regression cost.

    Args:
        x, y   : 1-D feature / target arrays
        w_init : starting weight  (float)
        b_init : starting bias    (float)
        alpha  : learning rate
        n_iter : iterations

    Returns:
        w_final, b_final, history_dict
        history_dict has keys → 'w', 'b', 'J'

    Example:
        w, b, h = gradient_descent(x_train, y_train, alpha=0.01)
        plot_convergence(h['J'])
    """
    m = len(x)
    w, b = float(w_init), float(b_init)
    hist = {"w": [w], "b": [b], "J": [compute_cost(x, y, w, b)]}
    for _ in range(n_iter):
        err  = w * x + b - y
        w   -= alpha * np.dot(err, x) / m
        b   -= alpha * np.sum(err)    / m
        hist["w"].append(w)
        hist["b"].append(b)
        hist["J"].append(compute_cost(x, y, w, b))
    return w, b, hist


def _make_grid(x, y, w_arr=None, b_arr=None, n=70):
    """Build (W, B, J) meshgrid for cost surface.  Internal helper."""
    opt_w, opt_b = np.polyfit(x, y, 1)
    if w_arr is None: w_arr = np.linspace(opt_w - 200, opt_w + 200, n)
    if b_arr is None: b_arr = np.linspace(opt_b - 200, opt_b + 200, n)
    W, B = np.meshgrid(w_arr, b_arr)
    J = np.vectorize(lambda w, b: compute_cost(x, y, w, b))(W, B)
    return W, B, J


def _show(fig=None):
    """Tight-layout + show.  Internal."""
    plt.tight_layout()
    plt.show()


# ════════════════════════════════════════════════════════════════════════════
# [S3]  BASIC PLOTS
# ════════════════════════════════════════════════════════════════════════════

def plot_line(x=None, y=None, label="f(x)", color=CB,
              xlabel="x", ylabel="y", title="Line Plot"):
    """
    Simple line plot.

    Args:
        x     : 1-D array  (default: 0→10)
        y     : 1-D array  (default: sin(x))
        label : legend label
        color : hex or named colour

    Example:
        plot_line()
        plot_line(epochs, loss, label="Train Loss", title="Loss Curve")
    """
    if x is None: x = np.linspace(0, 10, 300)
    if y is None: y = np.sin(x)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(x, y, color=color, linewidth=2.5, label=label)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title)
    ax.legend()
    _show(fig)


def plot_scatter(x=None, y=None, c=None, cmap="viridis",
                 xlabel="Feature", ylabel="Target",
                 title="Scatter Plot", alpha=0.7, s=60):
    """
    Scatter plot — optionally colour-coded by c.

    Args:
        x, y  : 1-D arrays
        c     : colour-per-point array (e.g. class labels)
        cmap  : matplotlib colormap
        s     : marker size

    Example:
        plot_scatter()
        plot_scatter(X[:,0], X[:,1], c=y_labels, title="Classes")
    """
    if x is None:
        np.random.seed(0)
        x = np.random.randn(120)
        y = 2*x + np.random.randn(120)*0.6
    fig, ax = plt.subplots(figsize=(8, 5))
    sc = ax.scatter(x, y, c=c, cmap=cmap, alpha=alpha,
                    s=s, edgecolors="k", linewidths=0.3)
    if c is not None: plt.colorbar(sc, ax=ax, label="Class / Value")
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title)
    _show(fig)


def plot_histogram(data=None, bins=30, kde=True, color=CB,
                   xlabel="Value", title="Histogram"):
    """
    Histogram with optional KDE overlay.

    Args:
        data : 1-D array  (default: standard normal sample)
        bins : number of bins
        kde  : overlay kernel density estimate

    Example:
        plot_histogram()
        plot_histogram(train_losses, bins=50, title="Loss Distribution")
    """
    if data is None: data = np.random.randn(600)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(data, bins=bins, kde=kde, ax=ax, color=color, alpha=0.75)
    ax.set_xlabel(xlabel); ax.set_title(title)
    _show(fig)


def plot_bar(categories=None, values=None,
             xlabel="Category", ylabel="Value", title="Bar Chart", color=CT):
    """
    Bar chart with value labels on top.

    Example:
        plot_bar()
        plot_bar(['Train','Val','Test'], [0.95, 0.89, 0.87], title="Accuracy")
    """
    if categories is None: categories = ["A", "B", "C", "D", "E"]
    if values     is None: values     = [4, 7, 5, 9, 3]
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(categories, values, color=color, edgecolor="black", alpha=0.85)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + max(values)*0.01,
                f"{v}", ha="center", fontsize=10)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title)
    _show(fig)


def plot_boxplot(data_list=None, labels=None, title="Box Plot", palette="Set2"):
    """
    Box plot — compare distributions across groups.

    Args:
        data_list : list of 1-D arrays, one per group
        labels    : list of group names

    Example:
        plot_boxplot()
        plot_boxplot([train_errors, val_errors], labels=["Train","Val"])
    """
    import pandas as pd
    if data_list is None:
        data_list = [np.random.randn(100), np.random.randn(100)+1.5,
                     np.random.randn(100)+3]
        labels    = ["Group A", "Group B", "Group C"]
    rows = []
    for lbl, arr in zip(labels, data_list):
        for v in arr:
            rows.append({"value": v, "group": lbl})
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(x="group", y="value", data=df, palette=palette, ax=ax)
    ax.set_title(title)
    _show(fig)


def plot_heatmap(matrix=None, annot=True, cmap="coolwarm",
                 title="Heatmap", fmt=".2f"):
    """
    General heatmap — works with any 2-D array (correlation matrix, confusion
    matrix, weights, etc.)

    Args:
        matrix : 2-D numpy array or pandas DataFrame
        annot  : show numbers in cells
        cmap   : colormap
        fmt    : number format string

    Example:
        plot_heatmap()
        plot_heatmap(corr_df, title="Correlation Matrix")
        plot_heatmap(weights_layer1, annot=False, title="Layer 1 Weights")
    """
    if matrix is None:
        matrix = np.corrcoef(np.random.randn(5, 80))
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(matrix, annot=annot, fmt=fmt, cmap=cmap,
                center=0, square=True, linewidths=0.4, ax=ax,
                cbar_kws={"shrink": 0.8})
    ax.set_title(title)
    _show(fig)


def plot_pairplot(df=None, hue=None, title="Pairplot — EDA"):
    """
    Seaborn pairplot — shows every feature-pair relationship.

    Args:
        df  : pandas DataFrame  (default: Iris dataset)
        hue : column name to colour by

    Example:
        plot_pairplot()
        plot_pairplot(my_df, hue="label")
    """
    import pandas as pd
    if df is None and SK_OK:
        iris = load_iris()
        df   = pd.DataFrame(iris.data, columns=iris.feature_names)
        df["species"] = [iris.target_names[i] for i in iris.target]
        hue = "species"
    elif df is None:
        df = pd.DataFrame(np.random.randn(80, 3), columns=["x","y","z"])
    g = sns.pairplot(df, hue=hue, diag_kind="kde",
                     plot_kws={"alpha": 0.55, "s": 35},
                     diag_kws={"fill": True, "alpha": 0.4})
    g.figure.suptitle(title, y=1.02, fontsize=13)
    plt.show()


# ════════════════════════════════════════════════════════════════════════════
# [S4]  LINEAR REGRESSION
# ════════════════════════════════════════════════════════════════════════════

def plot_linear_regression(x=None, y=None,
                            xlabel="x", ylabel="y",
                            title="Linear Regression"):
    """
    Scatter + fitted line + residual bars (2 panels).

    Args:
        x : 1-D feature array
        y : 1-D target array

    Example:
        plot_linear_regression()
        plot_linear_regression(x_train, y_train, xlabel="Size (sqft)",
                               ylabel="Price ($k)")
    """
    if x is None:
        x = np.array([1.0,1.7,2.0,2.5,3.0,3.2,3.8,4.2,4.7,5.0])
        y = np.array([250,300,480,430,630,730,800,880,950,1050])

    w, b   = np.polyfit(x, y, 1)
    x_line = np.linspace(x.min()-0.2, x.max()+0.2, 300)
    y_line = w*x_line + b
    resid  = y - (w*x + b)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    # Panel 1 — fit + residuals
    ax = axes[0]
    ax.scatter(x, y, color=CR, s=80, zorder=5, label="Data points")
    ax.plot(x_line, y_line, color=CB, lw=2.5,
            label=f"Fit: ŷ = {w:.1f}x + {b:.1f}")
    for xi, yi in zip(x, y):
        ax.plot([xi,xi], [yi, w*xi+b], color=CGR, ls="--", lw=1.1, zorder=3)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.set_title("Fitted Line\n(dashed = residuals)")
    ax.legend()

    # Panel 2 — residual bar chart
    colors = [CG if r >= 0 else CR for r in resid]
    axes[1].bar(np.arange(len(x)), resid, color=colors, edgecolor="k", alpha=0.8)
    axes[1].axhline(0, color="black", lw=1.2)
    axes[1].set_xlabel("Sample index")
    axes[1].set_ylabel("Residual  (y − ŷ)")
    axes[1].set_title("Residual Plot\n(green = over-predict, red = under)")

    _show(fig)


def plot_multi_feature_regression(X=None, y=None, feature_names=None):
    """
    Plot each feature vs target separately — for multi-feature linear regression.

    Args:
        X            : 2-D array  (n_samples × n_features)
        y            : 1-D target array
        feature_names: list of strings

    Example:
        plot_multi_feature_regression()
        plot_multi_feature_regression(X_train, y_train,
                                      feature_names=["Size","Rooms","Age"])
    """
    if X is None and SK_OK:
        X, y = make_regression(n_samples=120, n_features=3,
                                noise=15, random_state=0)
        feature_names = ["Feature 1", "Feature 2", "Feature 3"]
    elif X is None:
        X = np.random.randn(100,3)
        y = X @ [2,3,-1] + np.random.randn(100)
        feature_names = ["F1","F2","F3"]

    n = X.shape[1]
    if feature_names is None: feature_names = [f"F{i+1}" for i in range(n)]

    fig, axes = plt.subplots(1, n, figsize=(5*n, 5))
    fig.suptitle("Multi-Feature Regression — Each Feature vs Target",
                 fontsize=13, fontweight="bold")
    if n == 1: axes = [axes]

    for i, (ax, name) in enumerate(zip(axes, feature_names)):
        others = np.mean(X[:, [j for j in range(n) if j!=i]], axis=1)
        sc = ax.scatter(X[:,i], y, c=others, cmap="viridis",
                        alpha=0.7, s=45, edgecolors="k", lw=0.3)
        m_, b_ = np.polyfit(X[:,i], y, 1)
        xs = np.linspace(X[:,i].min(), X[:,i].max(), 200)
        ax.plot(xs, m_*xs+b_, color=CR, lw=2)
        plt.colorbar(sc, ax=ax, label="Other features")
        ax.set_xlabel(name); ax.set_ylabel("y"); ax.set_title(f"y vs {name}")

    _show(fig)


# ════════════════════════════════════════════════════════════════════════════
# [S5]  COST FUNCTION — 1-D views
# ════════════════════════════════════════════════════════════════════════════

def plot_cost_vs_w(x=None, y=None, b_fixed=0):
    """
    Plot J(w) with b held fixed — shows the parabolic dip.

    Args:
        x, y    : 1-D training arrays
        b_fixed : value of b to keep constant (float)

    Example:
        plot_cost_vs_w()
        plot_cost_vs_w(x_train, y_train, b_fixed=10)
    """
    if x is None:
        x = np.array([1.,2.,3.,4.,5.])
        y = np.array([200.,400.,600.,800.,1000.])

    opt_w, _ = np.polyfit(x, y, 1)
    w_arr     = np.linspace(opt_w - 220, opt_w + 220, 400)
    J_arr     = [compute_cost(x, y, w, b_fixed) for w in w_arr]
    best_w    = w_arr[np.argmin(J_arr)]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Cost Function — J(w) with b fixed", fontsize=13, fontweight="bold")

    axes[0].plot(w_arr, J_arr, color=CB, lw=2.5)
    axes[0].axvline(best_w, color=CR, ls="--",
                    label=f"Min at w ≈ {best_w:.1f}")
    axes[0].scatter([best_w], [min(J_arr)], color=CR, s=100, zorder=5)
    axes[0].set_xlabel("w"); axes[0].set_ylabel("J(w)")
    axes[0].set_title(f"Cost J vs w  (b = {b_fixed} fixed)")
    axes[0].legend()

    x_l = np.linspace(x.min()-0.3, x.max()+0.3, 200)
    axes[1].scatter(x, y, color=CR, s=80, zorder=5, label="Data")
    axes[1].plot(x_l, best_w*x_l + b_fixed, color=CB, lw=2.5,
                 label=f"Best w = {best_w:.1f}, b = {b_fixed}")
    axes[1].set_xlabel("x"); axes[1].set_ylabel("y")
    axes[1].set_title("Corresponding Fit")
    axes[1].legend()

    _show(fig)


def plot_cost_vs_wb(x=None, y=None):
    """
    Side-by-side: J(w) and J(b) — see both parameter contributions.

    Example:
        plot_cost_vs_wb()
        plot_cost_vs_wb(x_train, y_train)
    """
    if x is None:
        x = np.array([1.,2.,3.,4.,5.])
        y = np.array([200.,400.,600.,800.,1000.])

    opt_w, opt_b = np.polyfit(x, y, 1)
    w_arr = np.linspace(opt_w - 180, opt_w + 180, 400)
    b_arr = np.linspace(opt_b - 180, opt_b + 180, 400)
    J_w   = [compute_cost(x, y, w, opt_b) for w in w_arr]
    J_b   = [compute_cost(x, y, opt_w, b) for b in b_arr]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Cost Function — 1-D Slices through J(w,b)",
                 fontsize=13, fontweight="bold")

    for ax, arr, costs, label, opt, col in [
        (axes[0], w_arr, J_w, "w", opt_w, CB),
        (axes[1], b_arr, J_b, "b", opt_b, CG),
    ]:
        ax.plot(arr, costs, color=col, lw=2.5)
        ax.axvline(opt, color=CR, ls="--", label=f"Optimal {label} ≈ {opt:.1f}")
        ax.scatter([opt], [min(costs)], color=CR, s=100, zorder=5)
        ax.set_xlabel(label); ax.set_ylabel(f"J({label})")
        ax.set_title(f"Cost J vs {label}")
        ax.legend()

    _show(fig)


# ════════════════════════════════════════════════════════════════════════════
# [S6]  COST FUNCTION 3-D  — The Bowl 🥣
# ════════════════════════════════════════════════════════════════════════════

def plot_cost_3d(x=None, y=None, elev=30, azim=225, cmap="viridis"):
    """
    🎯 THE MOST ICONIC ANDREW NG PLOT.
    3-D surface + contour map of J(w,b) side by side.

    Args:
        x, y : 1-D training arrays
        elev : 3-D camera elevation (degrees)
        azim : 3-D camera azimuth  (degrees)
        cmap : matplotlib colormap

    Example:
        plot_cost_3d()
        plot_cost_3d(x_train, y_train)
        plot_cost_3d(x_train, y_train, elev=45, azim=60)  ← rotate the bowl
    """
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    if x is None:
        x = np.array([1.,2.,3.,4.,5.])
        y = np.array([200.,400.,600.,800.,1000.])

    W, B, J  = _make_grid(x, y)
    min_idx  = np.unravel_index(np.argmin(J), J.shape)

    fig = plt.figure(figsize=(16, 6))
    fig.suptitle("Cost Function J(w,b) — 3-D Surface & Contour",
                 fontsize=14, fontweight="bold")

    # 3-D surface
    ax1 = fig.add_subplot(121, projection="3d")
    surf = ax1.plot_surface(W, B, J, cmap=cmap, alpha=0.85, edgecolor="none")
    fig.colorbar(surf, ax=ax1, shrink=0.4, label="J(w,b)")
    ax1.scatter(W[min_idx], B[min_idx], J[min_idx],
                color="red", s=120, zorder=10, label="Minimum")
    ax1.set_xlabel("w"); ax1.set_ylabel("b"); ax1.set_zlabel("J(w,b)")
    ax1.set_title("3-D Surface — The Bowl 🥣")
    ax1.view_init(elev=elev, azim=azim)
    ax1.legend()

    # Contour (top-down)
    ax2 = fig.add_subplot(122)
    cf  = ax2.contourf(W, B, J, levels=50, cmap=cmap, alpha=0.85)
    ax2.contour(W, B, J, levels=20, colors="white", alpha=0.2, linewidths=0.6)
    fig.colorbar(cf, ax=ax2, label="J(w,b)")
    ax2.scatter(W[min_idx], B[min_idx],
                color="red", s=160, zorder=5, marker="*", label="Minimum")
    ax2.set_xlabel("w"); ax2.set_ylabel("b")
    ax2.set_title("Contour Map — Top-down View 🗺️")
    ax2.legend()

    _show(fig)


# ════════════════════════════════════════════════════════════════════════════
# [S7]  GRADIENT DESCENT
# ════════════════════════════════════════════════════════════════════════════

def plot_gradient_descent_path(x=None, y=None,
                                w_init=None, b_init=None,
                                alpha=0.01, n_iter=300):
    """
    Gradient descent path on contour + convergence curve.

    Args:
        x, y   : 1-D training arrays
        w_init : starting weight  (default: far from optimum)
        b_init : starting bias
        alpha  : learning rate
        n_iter : iterations

    Example:
        plot_gradient_descent_path()
        plot_gradient_descent_path(x_train, y_train, alpha=0.05)
    """
    if x is None:
        x = np.array([1.,2.,3.,4.,5.])
        y = np.array([200.,400.,600.,800.,1000.])

    opt_w, opt_b = np.polyfit(x, y, 1)
    if w_init is None: w_init = opt_w - 200
    if b_init is None: b_init = opt_b - 150

    _, _, hist = gradient_descent(x, y, w_init, b_init, alpha, n_iter)
    W, B, J    = _make_grid(x, y)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(f"Gradient Descent  (α={alpha}, {n_iter} steps)",
                 fontsize=13, fontweight="bold")

    # Contour + path
    cf = axes[0].contourf(W, B, J, levels=50, cmap="viridis", alpha=0.85)
    plt.colorbar(cf, ax=axes[0], label="J(w,b)")
    axes[0].plot(hist["w"], hist["b"], "w.-",
                 lw=1.4, ms=3, label="GD path", zorder=4)
    axes[0].scatter(hist["w"][0],  hist["b"][0],  color="yellow",
                    s=120, zorder=5, label="Start 🟡")
    axes[0].scatter(hist["w"][-1], hist["b"][-1], color="red",
                    s=120, zorder=5, marker="*", label="End 🔴")
    axes[0].set_xlabel("w"); axes[0].set_ylabel("b")
    axes[0].set_title("Path on Contour Map")
    axes[0].legend()

    # Convergence curve
    axes[1].plot(hist["J"], color=CB, lw=2.5)
    axes[1].set_xlabel("Iterations"); axes[1].set_ylabel("Cost J")
    axes[1].set_title("Convergence Curve")
    axes[1].annotate(f"Final J ≈ {hist['J'][-1]:.2f}",
                     xy=(n_iter, hist["J"][-1]),
                     xytext=(n_iter*0.55, hist["J"][5]*0.6),
                     arrowprops=dict(arrowstyle="->", color=CR),
                     color=CR, fontsize=11)
    _show(fig)


def plot_learning_rate_comparison(x=None, y=None,
                                   alphas=None, n_iter=250):
    """
    Compare multiple learning rates — Andrew Ng's classic demo.

    Args:
        x, y   : 1-D training arrays
        alphas : list of floats  (default: [0.001, 0.01, 0.05, 0.2])
        n_iter : iterations per alpha

    Example:
        plot_learning_rate_comparison()
        plot_learning_rate_comparison(x_train, y_train,
                                      alphas=[0.001, 0.01, 0.1, 0.5])
    """
    if x is None:
        x = np.array([1.,2.,3.,4.,5.])
        y = np.array([200.,400.,600.,800.,1000.])
    if alphas is None: alphas = [0.001, 0.01, 0.05, 0.2]

    cols = [CB, CG, CO, CR, CP, CT]
    n    = len(alphas)
    opt_w, opt_b = np.polyfit(x, y, 1)

    fig, axes = plt.subplots(1, n, figsize=(5*n, 5))
    fig.suptitle("Effect of Learning Rate α on Convergence",
                 fontsize=13, fontweight="bold")
    if n == 1: axes = [axes]

    for ax, a, col in zip(axes, alphas, cols):
        _, _, h = gradient_descent(x, y, opt_w-200, opt_b-150, a, n_iter)
        diverg  = h["J"][-1] > h["J"][0] * 2
        ax.plot(h["J"], color=col, lw=2.5)
        tag = "⚠️ Diverging!" if diverg else (
              "✅ Converged" if h["J"][-1] < h["J"][0] * 0.01 else "🔄 Converging")
        ax.set_title(f"α = {a}\n{tag}", fontsize=11)
        ax.set_xlabel("Iterations"); ax.set_ylabel("J")
        if diverg: ax.set_ylim(0, h["J"][0] * 3)

    _show(fig)


def plot_convergence(J_history, title="Convergence Curve"):
    """
    Plot cost vs iterations from any gradient descent run.

    Args:
        J_history : list or 1-D array of cost values
        title     : plot title

    Example:
        _, _, h = gradient_descent(x, y, alpha=0.01)
        plot_convergence(h['J'])

        # OR with your own training history list:
        plot_convergence(my_model_loss_history)
    """
    J = np.array(J_history)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(title, fontsize=13, fontweight="bold")

    axes[0].plot(J, color=CB, lw=2.5)
    axes[0].set_xlabel("Iterations"); axes[0].set_ylabel("J")
    axes[0].set_title("Linear scale")

    axes[1].plot(J, color=CO, lw=2.5)
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Iterations"); axes[1].set_ylabel("J  (log)")
    axes[1].set_title("Log scale — see early progress")

    _show(fig)


# ════════════════════════════════════════════════════════════════════════════
# [S8]  FEATURE SCALING
# ════════════════════════════════════════════════════════════════════════════

def plot_feature_scaling_effect():
    """
    Visualise WHY feature scaling speeds up gradient descent:
    elongated ellipses → circular contours.

    Example:
        plot_feature_scaling_effect()
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Feature Scaling: Effect on Cost Contours & GD Path",
                 fontsize=13, fontweight="bold")

    # Before — very elongated
    W1, B1 = np.meshgrid(np.linspace(-3, 3, 200),
                          np.linspace(-300, 300, 200))
    J1 = W1**2 + 0.0001 * B1**2
    axes[0].contour(W1, B1, J1, levels=20, cmap="viridis")
    gw = np.linspace(-2.8, 0, 40)
    gb = np.linspace(280, 0, 40) + 60*np.sin(np.linspace(0, 3, 40))
    axes[0].plot(gw, gb, "r.-", lw=1.8, ms=4, label="GD path (zigzag)")
    axes[0].scatter([-2.8], [280], color="yellow", s=100, zorder=5)
    axes[0].scatter([0], [0], color="red", s=100, zorder=5, marker="*")
    axes[0].set_xlabel("w₁"); axes[0].set_ylabel("w₂")
    axes[0].set_title("❌ Before Scaling\n(Slow, zigzag descent)")
    axes[0].legend()

    # After — circular
    W2, B2 = np.meshgrid(np.linspace(-3, 3, 200), np.linspace(-3, 3, 200))
    J2 = W2**2 + B2**2
    axes[1].contour(W2, B2, J2, levels=20, cmap="viridis")
    axes[1].plot(np.linspace(-2.8, 0, 40),
                  np.linspace(-2.5, 0, 40),
                  "r.-", lw=1.8, ms=4, label="GD path (direct)")
    axes[1].scatter([-2.8], [-2.5], color="yellow", s=100, zorder=5)
    axes[1].scatter([0], [0], color="red", s=100, zorder=5, marker="*")
    axes[1].set_xlabel("w₁ (scaled)"); axes[1].set_ylabel("w₂ (scaled)")
    axes[1].set_title("✅ After Scaling\n(Fast, direct descent)")
    axes[1].legend()

    _show(fig)


def plot_normalization_comparison(data=None):
    """
    Show raw vs Z-score vs Min-Max scaling side by side.

    Args:
        data : 2-D array (n_samples × 2)  — first two features used

    Example:
        plot_normalization_comparison()
        plot_normalization_comparison(X_train[:, :2])
    """
    if data is None:
        np.random.seed(0)
        data = np.column_stack([np.random.normal(500, 100, 200),
                                 np.random.normal(3,   1,   200)])

    z  = (data - data.mean(0)) / data.std(0)
    mm = (data - data.min(0))  / (data.max(0) - data.min(0))

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Data Normalisation Comparison", fontsize=13, fontweight="bold")

    for ax, d, name in zip(axes,
                            [data, z, mm],
                            ["Raw Data",
                             "Z-score  (μ=0, σ=1)",
                             "Min-Max  [0, 1]"]):
        ax.scatter(d[:,0], d[:,1], alpha=0.5, s=30, color=CB,
                   edgecolors="k", lw=0.2)
        ax.set_xlabel("Feature 1"); ax.set_ylabel("Feature 2")
        ax.set_title(name)

    _show(fig)


# ════════════════════════════════════════════════════════════════════════════
# [S9]  LOGISTIC REGRESSION
# ════════════════════════════════════════════════════════════════════════════

def plot_sigmoid(z_range=(-10, 10)):
    """
    Plot the sigmoid / logistic function with class regions shaded.

    Args:
        z_range : tuple (min_z, max_z)

    Example:
        plot_sigmoid()
        plot_sigmoid(z_range=(-5, 5))
    """
    z   = np.linspace(*z_range, 400)
    sig = sigmoid(z)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(z, sig, color=CB, lw=3, label="σ(z) = 1/(1+e⁻ᶻ)")
    ax.axhline(0.5, color=CR, ls="--", lw=1.5, label="Threshold = 0.5")
    ax.axhline(1.0, color=CGR, ls=":",  lw=1)
    ax.axhline(0.0, color=CGR, ls=":",  lw=1)
    ax.axvline(0,   color=CG,  ls="--", lw=1.5, alpha=0.7)
    ax.fill_between(z, 0.5, sig, where=(sig > 0.5), alpha=0.15,
                    color=CB, label="Predict class 1")
    ax.fill_between(z, sig, 0.5, where=(sig < 0.5), alpha=0.15,
                    color=CO, label="Predict class 0")
    ax.set_xlabel("z = w·x + b", fontsize=12)
    ax.set_ylabel("σ(z)", fontsize=12)
    ax.set_title("Sigmoid (Logistic) Function", fontsize=13)
    ax.legend()
    ax.set_ylim(-0.05, 1.05)
    _show(fig)


def plot_log_loss():
    """
    Plot binary cross-entropy loss for y=1 and y=0.

    Example:
        plot_log_loss()
    """
    f = np.linspace(0.001, 0.999, 400)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Binary Cross-Entropy (Log Loss)", fontsize=13, fontweight="bold")

    axes[0].plot(f, -np.log(f), color=CB, lw=2.5)
    axes[0].set_title("Loss when y = 1:  −log(f)")
    axes[0].set_xlabel("f(x) — predicted probability")
    axes[0].set_ylabel("Loss")
    axes[0].annotate("Predict 1 → Loss ≈ 0 ✅",
                     xy=(0.97, 0.03), fontsize=10, color=CG)
    axes[0].annotate("Predict 0 → Loss → ∞ ❌",
                     xy=(0.05, 2.7),  fontsize=10, color=CR)

    axes[1].plot(f, -np.log(1-f), color=CR, lw=2.5)
    axes[1].set_title("Loss when y = 0:  −log(1 − f)")
    axes[1].set_xlabel("f(x) — predicted probability")
    axes[1].set_ylabel("Loss")
    axes[1].annotate("Predict 0 → Loss ≈ 0 ✅",
                     xy=(0.03, 0.03), fontsize=10, color=CG)
    axes[1].annotate("Predict 1 → Loss → ∞ ❌",
                     xy=(0.78, 2.7),  fontsize=10, color=CR)
    for ax in axes: ax.set_ylim(0, 4)
    _show(fig)


def plot_decision_boundary(X=None, y=None, model=None,
                            xlabel="Feature 1", ylabel="Feature 2",
                            title="Decision Boundary"):
    """
    Decision boundary with probability heatmap.

    Args:
        X     : 2-D feature array  (n_samples × 2)
        y     : 1-D label array
        model : fitted sklearn classifier with predict_proba()
                (default: LogisticRegression trained on demo data)

    Example:
        plot_decision_boundary()

        # With your own data + model:
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression().fit(X_train, y_train)
        plot_decision_boundary(X_test, y_test, model)
    """
    if X is None and SK_OK:
        X, y = make_classification(n_samples=250, n_features=2,
                                    n_redundant=0, n_clusters_per_class=1,
                                    random_state=42)
    elif X is None:
        raise ValueError("Pass X, y arrays (sklearn not installed for demo).")

    if model is None:
        model = LogisticRegression()
        model.fit(X, y)

    h  = 0.02
    x0 = np.arange(X[:,0].min()-0.5, X[:,0].max()+0.5, h)
    x1 = np.arange(X[:,1].min()-0.5, X[:,1].max()+0.5, h)
    xx, yy = np.meshgrid(x0, x1)
    Z  = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:,1].reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(9, 6))
    cf  = ax.contourf(xx, yy, Z, levels=50, cmap="RdBu", alpha=0.75)
    plt.colorbar(cf, ax=ax, label="P(y=1)")
    ax.contour(xx, yy, Z, levels=[0.5], colors="black", linewidths=2.5)
    ax.scatter(X[:,0], X[:,1], c=y, cmap="RdBu",
               edgecolors="k", s=55, zorder=5)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.set_title(f"{title}\n(black line = 0.5 probability)")
    _show(fig)


# ════════════════════════════════════════════════════════════════════════════
# [S10] ACTIVATION FUNCTIONS
# ════════════════════════════════════════════════════════════════════════════

def plot_activation_functions(z_range=(-5, 5)):
    """
    Plot all major activation functions side by side.

    Args:
        z_range : x-axis range tuple

    Example:
        plot_activation_functions()
        plot_activation_functions(z_range=(-3, 3))
    """
    z = np.linspace(*z_range, 400)
    fns = {
        "Sigmoid σ(z)":   (sigmoid(z),       CB),
        "Tanh":           (tanh_fn(z),        CG),
        "ReLU":           (relu(z),            CR),
        "Leaky ReLU":     (leaky_relu(z),      CO),
        "ELU":            (elu(z),             CP),
        "Swish":          (swish(z),           CT),
        "GELU":           (gelu(z),            "#8B4513"),
        "Linear":         (linear_fn(z),       CGR),
    }

    n   = len(fns)
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    fig.suptitle("Activation Functions — Neural Networks",
                 fontsize=14, fontweight="bold")

    for ax, (name, (vals, col)) in zip(axes.flatten(), fns.items()):
        ax.plot(z, vals, color=col, lw=2.5)
        ax.axhline(0, color=CGR, lw=0.8, ls=":")
        ax.axvline(0, color=CGR, lw=0.8, ls=":")
        ax.set_title(name, fontsize=12)
        ax.set_xlabel("z"); ax.set_ylabel("a")
        ax.set_xlim(*z_range)

    _show(fig)


def plot_activation_derivatives(z_range=(-5, 5)):
    """
    Plot activation functions AND their derivatives — understand vanishing gradients.

    Example:
        plot_activation_derivatives()
    """
    z = np.linspace(*z_range, 500)
    dz = z[1] - z[0]  # for numerical derivative

    fns = {
        "Sigmoid": (sigmoid(z), CB),
        "Tanh":    (tanh_fn(z), CG),
        "ReLU":    (relu(z),    CR),
        "Swish":   (swish(z),   CO),
    }

    fig, axes = plt.subplots(2, len(fns), figsize=(16, 8))
    fig.suptitle("Activation Functions & Their Derivatives\n"
                 "(Derivative → gradient signal in backprop)",
                 fontsize=13, fontweight="bold")

    for col_i, (name, (vals, color)) in enumerate(fns.items()):
        deriv = np.gradient(vals, z)
        axes[0, col_i].plot(z, vals,  color=color, lw=2.5)
        axes[0, col_i].set_title(name)
        axes[0, col_i].set_ylabel("f(z)")

        axes[1, col_i].plot(z, deriv, color=color, lw=2.5, ls="--")
        axes[1, col_i].set_ylabel("f′(z)")
        axes[1, col_i].set_xlabel("z")
        axes[1, col_i].axhline(0, color=CGR, lw=0.7, ls=":")

    _show(fig)


# ════════════════════════════════════════════════════════════════════════════
# [S11] NEURAL NETWORK VISUALIZATIONS
# ════════════════════════════════════════════════════════════════════════════

def plot_layer_activations(x_input=None):
    """
    Show how data transforms layer-by-layer through a small network.

    Args:
        x_input : 1-D input array  (default: linspace -3→3)

    Example:
        plot_layer_activations()
        plot_layer_activations(np.linspace(-5, 5, 300))
    """
    np.random.seed(7)
    if x_input is None: x_input = np.linspace(-3, 3, 300)
    X = x_input.reshape(1, -1)     # shape: (1, 300)

    # random weights: 1 → 6 → 4 → 1
    W1 = np.random.randn(6, 1);  b1 = np.random.randn(6, 1)
    W2 = np.random.randn(4, 6);  b2 = np.random.randn(4, 1)
    W3 = np.random.randn(1, 4);  b3 = np.random.randn(1, 1)

    A1 = relu(W1 @ X + b1)          # (6, 300)
    A2 = tanh_fn(W2 @ A1 + b2)      # (4, 300)
    A3 = sigmoid(W3 @ A2 + b3)      # (1, 300)

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    fig.suptitle("Neural Network — Forward Pass: Layer Activations",
                 fontsize=13, fontweight="bold")

    axes[0].plot(x_input, x_input, color=CB, lw=2)
    axes[0].set_title("Input  x")
    axes[0].set_xlabel("x"); axes[0].set_ylabel("x")

    for i, row in enumerate(A1):
        axes[1].plot(x_input, row, alpha=0.75, lw=1.5, label=f"n{i+1}")
    axes[1].set_title("Layer 1 — ReLU (6 neurons)")
    axes[1].set_xlabel("x"); axes[1].legend(fontsize=7)

    for i, row in enumerate(A2):
        axes[2].plot(x_input, row, alpha=0.75, lw=1.5)
    axes[2].set_title("Layer 2 — Tanh (4 neurons)")
    axes[2].set_xlabel("x")

    axes[3].plot(x_input, A3[0], color=CR, lw=2.5)
    axes[3].set_ylim(-0.05, 1.05)
    axes[3].set_title("Output — Sigmoid (1 neuron)")
    axes[3].set_xlabel("x"); axes[3].set_ylabel("f(x)")

    _show(fig)


def plot_training_history(train_loss=None, val_loss=None,
                           train_acc=None,  val_acc=None):
    """
    Training & validation loss + accuracy curves (2 panels).

    Args:
        train_loss : list/array of training losses per epoch
        val_loss   : list/array of validation losses per epoch
        train_acc  : list/array of training accuracies  (optional)
        val_acc    : list/array of validation accuracies (optional)

    Example:
        plot_training_history()   ← demo with simulated overfitting

        # With keras / pytorch history:
        plot_training_history(history.history['loss'],
                              history.history['val_loss'],
                              history.history['accuracy'],
                              history.history['val_accuracy'])
    """
    if train_loss is None:
        epochs     = 100
        e          = np.arange(1, epochs+1)
        train_loss = 2.5*np.exp(-0.04*e) + 0.08 + 0.02*np.random.randn(epochs)
        val_loss   = 2.5*np.exp(-0.035*e) + 0.18 + 0.05*np.random.randn(epochs)
        val_loss[60:] += np.linspace(0, 0.5, 40)   # simulate overfitting
        train_acc  = 1 - train_loss / 3
        val_acc    = 1 - val_loss   / 3

    epochs = np.arange(1, len(train_loss)+1)
    has_acc = train_acc is not None

    fig, axes = plt.subplots(1, 2 if has_acc else 1, figsize=(14 if has_acc else 8, 5))
    fig.suptitle("Neural Network Training History", fontsize=13, fontweight="bold")
    if not has_acc: axes = [axes]

    ax = axes[0]
    ax.plot(epochs, train_loss, color=CB, lw=2.5, label="Train Loss")
    ax.plot(epochs, val_loss,   color=CR, lw=2.5, label="Val Loss")
    ax.fill_between(epochs, train_loss, val_loss, alpha=0.08, color=CR)
    # Mark where val loss starts increasing (overfitting point)
    diff = np.array(val_loss) - np.array(train_loss)
    if np.any(np.diff(np.array(val_loss)) > 0):
        overfit_epoch = np.argmax(np.diff(np.array(val_loss)) > 0.02) + 1
        ax.axvline(overfit_epoch, color=CO, ls="--", alpha=0.7,
                   label=f"Overfit ≈ ep{overfit_epoch}")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.set_title("Loss Curves"); ax.legend()

    if has_acc:
        axes[1].plot(epochs, train_acc, color=CB, lw=2.5, label="Train Acc")
        axes[1].plot(epochs, val_acc,   color=CR, lw=2.5, label="Val Acc")
        axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy")
        axes[1].set_title("Accuracy Curves"); axes[1].legend()

    _show(fig)


# ════════════════════════════════════════════════════════════════════════════
# [S12] REGULARISATION
# ════════════════════════════════════════════════════════════════════════════

def plot_overfit_underfit(x=None, y=None):
    """
    Show underfitting / good fit / overfitting on the same dataset.

    Args:
        x : 1-D sorted feature array
        y : 1-D noisy target array

    Example:
        plot_overfit_underfit()
        plot_overfit_underfit(x_train, y_train)
    """
    if x is None:
        np.random.seed(3)
        x = np.sort(np.random.uniform(0, 1, 35))
        y = np.sin(2*np.pi*x) + 0.3*np.random.randn(35)

    x_plot = np.linspace(0, 1, 400)
    degs   = [1, 4, 15]
    labels = ["Underfit\n(degree 1)", "Just Right\n(degree 4)",
              "Overfit\n(degree 15)"]
    colors = [CR, CG, CO]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Underfitting  →  Good Fit  →  Overfitting",
                 fontsize=13, fontweight="bold")

    for ax, deg, lbl, col in zip(axes, degs, labels, colors):
        coef  = np.polyfit(x, y, deg)
        y_fit = np.clip(np.poly1d(coef)(x_plot), -2.5, 2.5)
        ax.scatter(x, y, color=CB, s=40, zorder=5, label="Data")
        ax.plot(x_plot, np.sin(2*np.pi*x_plot), "k--",
                alpha=0.45, lw=1.5, label="True f(x)")
        ax.plot(x_plot, y_fit, color=col, lw=2.5, label=f"deg={deg}")
        ax.set_xlim(0, 1); ax.set_ylim(-2.5, 2.5)
        ax.set_title(lbl, fontsize=12)
        ax.legend(fontsize=8)

    _show(fig)


def plot_regularization_lambda(x=None, y=None,
                                lambdas=None, degree=12):
    """
    Show effect of regularisation strength λ on polynomial fit.

    Args:
        x, y     : 1-D training arrays
        lambdas  : list of λ values  (default: [0, 0.001, 0.1, 10])
        degree   : polynomial degree to fit

    Example:
        plot_regularization_lambda()
        plot_regularization_lambda(x_train, y_train,
                                   lambdas=[0, 0.01, 1, 100])
    """
    if x is None:
        np.random.seed(3)
        x = np.sort(np.random.uniform(0, 1, 35))
        y = np.sin(2*np.pi*x) + 0.3*np.random.randn(35)
    if lambdas is None: lambdas = [0, 0.001, 0.05, 1.0]

    if not SK_OK:
        print("sklearn required for Ridge regression in this plot.")
        return

    from sklearn.linear_model import Ridge

    x_plot  = np.linspace(0, 1, 400)
    fig, axes = plt.subplots(1, len(lambdas), figsize=(5*len(lambdas), 5))
    fig.suptitle(f"Regularisation λ — Polynomial degree {degree}",
                 fontsize=13, fontweight="bold")
    if len(lambdas) == 1: axes = [axes]

    for ax, lam in zip(axes, lambdas):
        pipe = make_pipeline(PolynomialFeatures(degree), Ridge(alpha=lam))
        pipe.fit(x.reshape(-1,1), y)
        y_pred = pipe.predict(x_plot.reshape(-1,1))

        ax.scatter(x, y, s=35, zorder=5, color=CB)
        ax.plot(x_plot, np.sin(2*np.pi*x_plot),
                "k--", alpha=0.4, lw=1.5, label="True f")
        ax.plot(x_plot, np.clip(y_pred,-2.5,2.5), color=CR, lw=2.5)
        ax.set_xlim(0,1); ax.set_ylim(-2.5, 2.5)
        ax.set_title(f"λ = {lam}", fontsize=12)
        ax.legend(fontsize=8)

    _show(fig)


# ════════════════════════════════════════════════════════════════════════════
# [S13] BIAS-VARIANCE TRADEOFF + LEARNING CURVES
# ════════════════════════════════════════════════════════════════════════════

def plot_bias_variance_tradeoff():
    """
    Classic bias² / variance / total error vs model complexity curve.

    Example:
        plot_bias_variance_tradeoff()
    """
    complexity = np.linspace(1, 10, 200)
    bias_sq  = 4   * np.exp(-0.55 * complexity) + 0.05
    variance = 0.02 * np.exp(0.42 * complexity)
    total    = bias_sq + variance + 0.1     # 0.1 = irreducible error

    opt_x = complexity[np.argmin(total)]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(complexity, bias_sq,  color=CB, lw=2.5, label="Bias²")
    ax.plot(complexity, variance, color=CR, lw=2.5, label="Variance")
    ax.plot(complexity, total,    color=CG, lw=3,   label="Total Error")
    ax.axhline(0.1, color=CGR, ls=":", lw=1.5, label="Irreducible Error")
    ax.axvline(opt_x, color=CO, ls="--", lw=2,
               label=f"Optimal complexity ≈ {opt_x:.1f}")

    lo, hi = 0, int(opt_x * 20)
    ax.fill_between(complexity[:hi], bias_sq[:hi], alpha=0.12,
                    color=CB, label="← High Bias region")
    ax.fill_between(complexity[hi:], variance[hi:], alpha=0.12,
                    color=CR, label="High Variance region →")

    ax.set_xlabel("Model Complexity", fontsize=12)
    ax.set_ylabel("Error",            fontsize=12)
    ax.set_title("Bias-Variance Tradeoff", fontsize=13, fontweight="bold")
    ax.legend(loc="upper center", fontsize=10)
    _show(fig)


def plot_learning_curves(X=None, y=None, degree=1, cv=5):
    """
    Learning curves — training size vs train/val error.
    Great for diagnosing high-bias vs high-variance.

    Args:
        X      : 2-D feature array
        y      : 1-D target array
        degree : polynomial degree  (1 = linear, high = complex)
        cv     : cross-validation folds

    Example:
        plot_learning_curves()                    ← compare deg 1 vs 10
        plot_learning_curves(X_train, y_train, degree=3)
    """
    if X is None:
        np.random.seed(3)
        x_raw = np.sort(np.random.uniform(0, 1, 80))
        X     = x_raw.reshape(-1, 1)
        y     = np.sin(2*np.pi*x_raw) + 0.3*np.random.randn(80)

    if not SK_OK:
        print("sklearn required."); return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Learning Curves — Diagnosing Bias vs Variance",
                 fontsize=13, fontweight="bold")

    for ax, deg, label in zip(axes,
                               [1, 10],
                               ["Linear Model (High Bias)",
                                "Degree-10 Poly (High Variance)"]):
        pipe = make_pipeline(PolynomialFeatures(deg), LinearRegression())
        sizes, tr, val = learning_curve(
            pipe, X, y, cv=cv,
            train_sizes=np.linspace(0.1, 1.0, 10),
            scoring="neg_mean_squared_error")

        tr_mean  = -tr.mean(1);  tr_std  = tr.std(1)
        val_mean = -val.mean(1); val_std = val.std(1)

        ax.plot(sizes, tr_mean,  color=CB, lw=2.5, label="Train error")
        ax.plot(sizes, val_mean, color=CR, lw=2.5, label="CV error")
        ax.fill_between(sizes, tr_mean-tr_std,  tr_mean+tr_std,  alpha=0.15, color=CB)
        ax.fill_between(sizes, val_mean-val_std, val_mean+val_std, alpha=0.15, color=CR)
        ax.set_xlabel("Training set size")
        ax.set_ylabel("MSE")
        ax.set_title(label, fontsize=11)
        ax.legend()

    _show(fig)


# ════════════════════════════════════════════════════════════════════════════
# [S14] DECISION BOUNDARIES
# ════════════════════════════════════════════════════════════════════════════

def plot_nonlinear_boundary():
    """
    Show circles and moons datasets — why linear boundaries fail.

    Example:
        plot_nonlinear_boundary()
    """
    if not SK_OK: print("sklearn required."); return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Why Non-Linear Decision Boundaries Are Needed",
                 fontsize=13, fontweight="bold")

    for ax, (X, y), title in zip(axes,
        [make_circles(200, noise=0.08, random_state=0),
         make_moons(200,   noise=0.12, random_state=0)],
        ["Circles", "Moons"]):

        model = LogisticRegression().fit(X, y)
        h  = 0.03
        xx, yy = np.meshgrid(
            np.arange(X[:,0].min()-0.3, X[:,0].max()+0.3, h),
            np.arange(X[:,1].min()-0.3, X[:,1].max()+0.3, h))
        Z  = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
        ax.contourf(xx, yy, Z, alpha=0.35, cmap="RdBu")
        ax.scatter(X[:,0], X[:,1], c=y, cmap="RdBu",
                   edgecolors="k", s=45, zorder=5)
        ax.set_title(f"{title}\n(Linear boundary — clearly poor fit)", fontsize=11)

    _show(fig)


def plot_multiclass_boundary(X=None, y=None, model=None):
    """
    Multiclass decision regions — coloured areas for each class.

    Args:
        X     : 2-D feature array (2 features)
        y     : 1-D integer label array
        model : fitted sklearn classifier  (default: KNN)

    Example:
        plot_multiclass_boundary()
        plot_multiclass_boundary(X_train, y_train, SVC().fit(X_train, y_train))
    """
    if not SK_OK: print("sklearn required."); return
    if X is None:
        X, y = make_classification(n_samples=300, n_features=2, n_classes=3,
                                    n_clusters_per_class=1, n_redundant=0,
                                    random_state=5)
    if model is None:
        from sklearn.neighbors import KNeighborsClassifier
        model = KNeighborsClassifier(5).fit(X, y)

    h  = 0.05
    xx, yy = np.meshgrid(
        np.linspace(X[:,0].min()-0.4, X[:,0].max()+0.4, 300),
        np.linspace(X[:,1].min()-0.4, X[:,1].max()+0.4, 300))
    Z  = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.contourf(xx, yy, Z, alpha=0.35, cmap="Set1")
    ax.contour(xx, yy, Z, colors="black", linewidths=0.6, alpha=0.5)
    ax.scatter(X[:,0], X[:,1], c=y, cmap="Set1",
               edgecolors="k", s=50, zorder=5)
    ax.set_title("Multiclass Decision Regions", fontsize=13)
    ax.set_xlabel("Feature 1"); ax.set_ylabel("Feature 2")
    _show(fig)


# ════════════════════════════════════════════════════════════════════════════
# [S15] CLUSTERING
# ════════════════════════════════════════════════════════════════════════════

def plot_kmeans_steps(X=None, k=3, max_steps=6):
    """
    K-Means clustering — one subplot per iteration to see centroids move.

    Args:
        X         : 2-D array  (default: 3-blob demo)
        k         : number of clusters
        max_steps : how many GD steps to show (subplots)

    Example:
        plot_kmeans_steps()
        plot_kmeans_steps(X_train, k=4, max_steps=5)
    """
    if not SK_OK: print("sklearn required."); return
    if X is None:
        X, _ = make_blobs(n_samples=250, centers=k,
                           cluster_std=0.9, random_state=42)

    steps = list(range(1, max_steps+1))
    ncols = min(3, len(steps))
    nrows = (len(steps) + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(6*ncols, 5*nrows))
    fig.suptitle(f"K-Means Clustering — k={k}  (Watching Centroids Move)",
                 fontsize=13, fontweight="bold")
    axes_flat = np.array(axes).flatten()

    for ax, step in zip(axes_flat, steps):
        km = KMeans(k, max_iter=step, n_init=1, init="random", random_state=42)
        km.fit(X)
        ax.scatter(X[:,0], X[:,1], c=km.labels_,
                   cmap="tab10", alpha=0.65, s=35, edgecolors="k", lw=0.2)
        ax.scatter(km.cluster_centers_[:,0],
                   km.cluster_centers_[:,1],
                   color="red", s=220, marker="X", zorder=6, label="Centroids")
        ax.set_title(f"After {step} iteration(s)")
        ax.legend(fontsize=8)

    # hide unused axes
    for ax in axes_flat[len(steps):]: ax.axis("off")
    _show(fig)


def plot_elbow_method(X=None, k_max=10):
    """
    Elbow method to choose optimal K for K-Means.

    Args:
        X     : 2-D array
        k_max : maximum K to try

    Example:
        plot_elbow_method()
        plot_elbow_method(X_train, k_max=12)
    """
    if not SK_OK: print("sklearn required."); return
    if X is None:
        X, _ = make_blobs(n_samples=300, centers=3, random_state=42)

    inertias = [KMeans(k, random_state=42, n_init=10).fit(X).inertia_
                for k in range(1, k_max+1)]

    # Find elbow via maximum second derivative
    d2   = np.diff(np.diff(inertias))
    elbow = int(np.argmax(d2)) + 2

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(1, k_max+1), inertias, "b-o", lw=2.5, ms=8)
    ax.axvline(elbow, color=CR, ls="--", lw=2,
               label=f"Elbow at K = {elbow}")
    ax.set_xlabel("Number of Clusters K")
    ax.set_ylabel("Inertia (WCSS)")
    ax.set_title("Elbow Method — Optimal K", fontsize=13)
    ax.legend()
    _show(fig)


# ════════════════════════════════════════════════════════════════════════════
# [S16] ANOMALY DETECTION
# ════════════════════════════════════════════════════════════════════════════

def plot_anomaly_detection(X_normal=None, X_anomaly=None, epsilon=0.05):
    """
    Gaussian density model + threshold contour for anomaly detection.

    Args:
        X_normal  : (n×2) normal training points
        X_anomaly : (m×2) suspected anomaly points
        epsilon   : probability threshold — points below = anomaly

    Example:
        plot_anomaly_detection()
        plot_anomaly_detection(X_normal, X_anomaly, epsilon=0.02)
    """
    if X_normal is None:
        np.random.seed(0)
        X_normal  = np.random.multivariate_normal([0,0], [[1,.5],[.5,1]], 120)
        X_anomaly = np.array([[-3.5, 3],[3.5,-3],[-3,-3.5],[0,3.8],[3.2,3]])

    mu  = X_normal.mean(0)
    cov = np.cov(X_normal.T)

    x1 = np.linspace(X_normal[:,0].min()-1, X_normal[:,0].max()+1, 200)
    x2 = np.linspace(X_normal[:,1].min()-1, X_normal[:,1].max()+1, 200)
    XX, YY = np.meshgrid(x1, x2)
    pos    = np.dstack((XX, YY))

    if SP_OK:
        rv = multivariate_normal(mu, cov)
        Z  = rv.pdf(pos)
    else:
        # manual Gaussian — no scipy
        diff = pos - mu
        inv  = np.linalg.inv(cov)
        Z = np.exp(-0.5 * np.einsum("...i,ij,...j->...", diff, inv, diff))
        Z /= (2*np.pi * np.sqrt(np.linalg.det(cov)))

    fig, ax = plt.subplots(figsize=(9, 7))
    cf = ax.contourf(XX, YY, Z, levels=15, cmap="Blues", alpha=0.75)
    ax.contour(XX, YY, Z, levels=[epsilon], colors="red", linewidths=2.5)
    plt.colorbar(cf, ax=ax, label="p(x)")

    ax.scatter(X_normal[:,0],  X_normal[:,1],  color=CB,
               s=30, alpha=0.6, label="Normal data")
    ax.scatter(X_anomaly[:,0], X_anomaly[:,1], color=CR,
               s=120, zorder=6, marker="X", label=f"Anomaly  p(x) < ε={epsilon}")

    ax.set_title(f"Anomaly Detection — Gaussian Density\n"
                 f"Red contour = threshold ε = {epsilon}", fontsize=12)
    ax.legend()
    ax.set_xlabel("Feature 1"); ax.set_ylabel("Feature 2")
    _show(fig)


# ════════════════════════════════════════════════════════════════════════════
# [S17] PCA
# ════════════════════════════════════════════════════════════════════════════

def plot_pca_scree(X=None):
    """
    Scree plot + cumulative variance — choose number of components.

    Args:
        X : 2-D array (default: Wine dataset)

    Example:
        plot_pca_scree()
        plot_pca_scree(X_train)
    """
    if not SK_OK: print("sklearn required."); return
    if X is None:
        wine = load_wine()
        X    = StandardScaler().fit_transform(wine.data)

    pca    = PCA().fit(X)
    evr    = pca.explained_variance_ratio_
    cumvar = np.cumsum(evr)
    n90    = np.searchsorted(cumvar, 0.90) + 1

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("PCA — Explained Variance", fontsize=13, fontweight="bold")

    idx = np.arange(1, len(evr)+1)
    axes[0].bar(idx, evr, color=CB, alpha=0.8, edgecolor="k", linewidth=0.5)
    axes[0].plot(idx, cumvar, "r-o", lw=2, ms=5, label="Cumulative")
    axes[0].axhline(0.90, color=CG, ls="--", lw=1.5, label="90% threshold")
    axes[0].axvline(n90,  color=CO, ls="--", lw=1.5,
                    label=f"{n90} components → 90%")
    axes[0].set_xlabel("Principal Component")
    axes[0].set_ylabel("Explained Variance Ratio")
    axes[0].set_title("Scree Plot")
    axes[0].legend()

    # 2-D projection
    X2   = PCA(2).fit_transform(X)
    axes[1].scatter(X2[:,0], X2[:,1], alpha=0.7, s=40, color=CB,
                    edgecolors="k", lw=0.3)
    axes[1].set_xlabel("PC 1"); axes[1].set_ylabel("PC 2")
    axes[1].set_title("2-D PCA Projection")

    _show(fig)


def plot_pca_2d(X=None, y=None, labels=None):
    """
    2-D PCA projection coloured by class label.

    Args:
        X      : 2-D feature array
        y      : 1-D integer label array
        labels : list of class name strings

    Example:
        plot_pca_2d()
        plot_pca_2d(X_train, y_train, labels=["Cat","Dog","Bird"])
    """
    if not SK_OK: print("sklearn required."); return
    if X is None:
        iris   = load_iris()
        X, y   = iris.data, iris.target
        labels = iris.target_names

    Xs    = StandardScaler().fit_transform(X)
    pca2  = PCA(2)
    X2    = pca2.fit_transform(Xs)
    evr   = pca2.explained_variance_ratio_

    fig, ax = plt.subplots(figsize=(8, 6))
    classes = np.unique(y)
    colors  = plt.cm.Set1(np.linspace(0, 1, len(classes)))
    for cls, col in zip(classes, colors):
        mask = y == cls
        lbl  = labels[cls] if labels is not None else f"Class {cls}"
        ax.scatter(X2[mask, 0], X2[mask, 1],
                   color=col, s=60, alpha=0.8,
                   edgecolors="k", lw=0.3, label=lbl)
    ax.set_xlabel(f"PC 1  ({evr[0]*100:.1f}%)")
    ax.set_ylabel(f"PC 2  ({evr[1]*100:.1f}%)")
    ax.set_title("PCA — 2-D Projection by Class", fontsize=13)
    ax.legend()
    _show(fig)


# ════════════════════════════════════════════════════════════════════════════
# [S18] EVALUATION METRICS
# ════════════════════════════════════════════════════════════════════════════

def plot_confusion_matrix(y_true=None, y_pred=None, class_names=None):
    """
    Seaborn confusion matrix heatmap.

    Args:
        y_true      : 1-D array of ground-truth labels
        y_pred      : 1-D array of predicted labels
        class_names : list of class name strings

    Example:
        plot_confusion_matrix()
        plot_confusion_matrix(y_test, model.predict(X_test),
                              class_names=["Cat","Dog","Bird"])
    """
    if y_true is None and SK_OK:
        iris   = load_iris()
        X, y   = iris.data, iris.target
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0)
        model  = SVC(kernel="rbf").fit(Xtr, ytr)
        y_true = yte
        y_pred = model.predict(Xte)
        class_names = iris.target_names
    elif y_true is None:
        raise ValueError("Pass y_true and y_pred arrays.")

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names,
                linewidths=0.5, ax=ax, cbar_kws={"shrink": 0.8})
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("Confusion Matrix", fontsize=13)
    _show(fig)


def plot_roc_curve(y_true=None, y_prob=None, label="Model"):
    """
    ROC curve with AUC shaded.

    Args:
        y_true : 1-D binary ground-truth labels
        y_prob : 1-D predicted probabilities for class 1
        label  : legend label

    Example:
        plot_roc_curve()
        y_prob = model.predict_proba(X_test)[:, 1]
        plot_roc_curve(y_test, y_prob, label="Logistic Reg")
    """
    if y_true is None and SK_OK:
        X, y = make_classification(500, random_state=0)
        Xtr,Xte,ytr,yte = train_test_split(X, y, test_size=0.3, random_state=0)
        y_true = yte
        y_prob = LogisticRegression().fit(Xtr, ytr).predict_proba(Xte)[:,1]
    elif y_true is None:
        raise ValueError("Pass y_true and y_prob.")

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc     = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color=CB, lw=2.5,
            label=f"{label}  (AUC = {roc_auc:.3f})")
    ax.plot([0,1],[0,1], color=CGR, lw=1.5, ls="--",
            label="Random Classifier")
    ax.fill_between(fpr, tpr, alpha=0.12, color=CB)
    ax.set_xlabel("False Positive Rate (FPR)")
    ax.set_ylabel("True Positive Rate (TPR)")
    ax.set_title("ROC Curve", fontsize=13)
    ax.legend(loc="lower right")
    _show(fig)


def plot_precision_recall(y_true=None, y_prob=None, label="Model"):
    """
    Precision-Recall curve — more informative than ROC for imbalanced data.

    Args:
        y_true : 1-D binary ground-truth labels
        y_prob : 1-D predicted probabilities for class 1

    Example:
        plot_precision_recall()
        plot_precision_recall(y_test, model.predict_proba(X_test)[:,1])
    """
    if y_true is None and SK_OK:
        X, y = make_classification(500, weights=[0.8,0.2], random_state=1)
        Xtr,Xte,ytr,yte = train_test_split(X, y, test_size=0.3, random_state=1)
        y_true = yte
        y_prob = LogisticRegression().fit(Xtr, ytr).predict_proba(Xte)[:,1]

    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    ap           = average_precision_score(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(rec, prec, color=CP, lw=2.5, label=f"{label}  (AP = {ap:.3f})")
    ax.fill_between(rec, prec, alpha=0.12, color=CP)
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve", fontsize=13)
    ax.legend()
    _show(fig)


def plot_feature_importance(model=None, feature_names=None, X=None, y=None):
    """
    Horizontal feature importance bar chart.

    Args:
        model         : fitted sklearn estimator with .feature_importances_
        feature_names : list of strings
        X, y          : used to fit demo model if model=None

    Example:
        plot_feature_importance()
        rf = RandomForestClassifier().fit(X_train, y_train)
        plot_feature_importance(rf, feature_names=["age","income","score"])
    """
    if not SK_OK: print("sklearn required."); return

    if model is None:
        iris   = load_iris()
        X, y   = iris.data, iris.target
        model  = RandomForestClassifier(100, random_state=0).fit(X, y)
        feature_names = iris.feature_names

    imps  = model.feature_importances_
    idx   = np.argsort(imps)          # ascending → horizontal bar looks nice
    names = [feature_names[i] for i in idx] if feature_names else [f"F{i}" for i in idx]

    fig, ax = plt.subplots(figsize=(8, max(4, len(imps)*0.5)))
    bars = ax.barh(names, imps[idx], color=CB, alpha=0.85, edgecolor="k")
    for bar, v in zip(bars, imps[idx]):
        ax.text(v + 0.002, bar.get_y() + bar.get_height()/2,
                f"{v:.3f}", va="center", fontsize=9)
    ax.set_xlabel("Importance Score")
    ax.set_title("Feature Importance — Random Forest", fontsize=13)
    _show(fig)


# ════════════════════════════════════════════════════════════════════════════
# [S19] PLOTLY INTERACTIVE CHARTS
#   Call these when you want zoom/pan/hover in browser.
#   Each mirrors a matplotlib plot above but is interactive.
# ════════════════════════════════════════════════════════════════════════════

def iplot_scatter(x=None, y=None, color=None,
                  xlabel="x", ylabel="y", title="Scatter Plot"):
    """
    Interactive scatter plot via plotly.express.

    Args:
        x, y  : 1-D arrays
        color : 1-D array for colour coding (labels / continuous values)

    Example:
        iplot_scatter()
        iplot_scatter(X[:,0], X[:,1], color=y_labels, title="Classes")
    """
    if not PLOTLY_OK: print("pip install plotly"); return
    import pandas as pd
    if x is None:
        np.random.seed(0)
        x = np.random.randn(150)
        y = 2*x + np.random.randn(150)*0.5
    df = pd.DataFrame({"x": x, "y": y})
    if color is not None: df["color"] = color.astype(str)
    fig = px.scatter(df, x="x", y="y",
                     color="color" if color is not None else None,
                     labels={"x": xlabel, "y": ylabel},
                     title=title, template="plotly_dark")
    fig.show()


def iplot_line(x=None, y=None, xlabel="x", ylabel="y", title="Line Plot"):
    """
    Interactive line plot.

    Example:
        iplot_line()
        iplot_line(epochs, loss, xlabel="Epoch", ylabel="Loss",
                   title="Training Loss")
    """
    if not PLOTLY_OK: print("pip install plotly"); return
    import pandas as pd
    if x is None: x = np.linspace(0, 10, 300)
    if y is None: y = np.sin(x)
    fig = px.line(pd.DataFrame({"x":x,"y":y}), x="x", y="y",
                  labels={"x": xlabel, "y": ylabel},
                  title=title, template="plotly_dark")
    fig.show()


def iplot_cost_3d(x=None, y=None, n=60):
    """
    Interactive 3-D cost surface — rotate/zoom in browser.

    Example:
        iplot_cost_3d()
        iplot_cost_3d(x_train, y_train)
    """
    if not PLOTLY_OK: print("pip install plotly"); return
    if x is None:
        x = np.array([1.,2.,3.,4.,5.])
        y = np.array([200.,400.,600.,800.,1000.])

    W, B, J = _make_grid(x, y, n=n)
    fig = go.Figure(go.Surface(x=W, y=B, z=J,
                                colorscale="Viridis", opacity=0.9))
    fig.update_layout(title="Cost Function J(w,b) — Interactive 3-D Bowl",
                      scene=dict(xaxis_title="w",
                                 yaxis_title="b",
                                 zaxis_title="J(w,b)"),
                      template="plotly_dark")
    fig.show()


def iplot_decision_boundary(X=None, y=None, model=None):
    """
    Interactive decision boundary probability map.

    Example:
        iplot_decision_boundary()
        iplot_decision_boundary(X_test, y_test, model)
    """
    if not PLOTLY_OK: print("pip install plotly"); return
    if not SK_OK:     print("sklearn required.");  return
    if X is None:
        X, y = make_classification(n_samples=300, n_features=2,
                                    n_redundant=0, n_clusters_per_class=1,
                                    random_state=0)
    if model is None:
        model = LogisticRegression().fit(X, y)

    h  = 0.05
    xx, yy = np.meshgrid(
        np.arange(X[:,0].min()-0.5, X[:,0].max()+0.5, h),
        np.arange(X[:,1].min()-0.5, X[:,1].max()+0.5, h))
    Z = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:,1].reshape(xx.shape)

    fig = go.Figure()
    fig.add_trace(go.Contour(x=xx[0], y=yy[:,0], z=Z,
                              colorscale="RdBu", opacity=0.7,
                              contours=dict(coloring="heatmap"),
                              name="P(y=1)"))
    fig.add_trace(go.Scatter(x=X[:,0], y=X[:,1],
                              mode="markers",
                              marker=dict(color=y, colorscale="RdBu",
                                          size=7, line=dict(width=0.5, color="black")),
                              name="Data"))
    fig.update_layout(title="Interactive Decision Boundary",
                      template="plotly_dark")
    fig.show()


def iplot_training_history(train_loss=None, val_loss=None,
                            train_acc=None,  val_acc=None):
    """
    Interactive training history — hover to see exact values.

    Example:
        iplot_training_history()
        iplot_training_history(h['loss'], h['val_loss'],
                               h['accuracy'], h['val_accuracy'])
    """
    if not PLOTLY_OK: print("pip install plotly"); return
    import pandas as pd
    if train_loss is None:
        e = np.arange(1, 101)
        train_loss = 2.5*np.exp(-0.04*e)+0.08+0.02*np.random.randn(100)
        val_loss   = 2.5*np.exp(-0.035*e)+0.18+0.05*np.random.randn(100)
        val_loss[60:] += np.linspace(0, 0.5, 40)
        train_acc  = 1 - train_loss/3
        val_acc    = 1 - val_loss/3

    rows    = 2 if train_acc is not None else 1
    subplot_titles = (["Train Loss", "Val Loss"] if rows==1
                      else ["Loss", "Accuracy"])
    fig = make_subplots(rows=rows, cols=1, subplot_titles=subplot_titles)

    e = list(range(1, len(train_loss)+1))
    fig.add_trace(go.Scatter(x=e, y=train_loss, name="Train Loss",
                              line=dict(color="royalblue")), row=1, col=1)
    fig.add_trace(go.Scatter(x=e, y=val_loss,   name="Val Loss",
                              line=dict(color="tomato")),   row=1, col=1)

    if train_acc is not None:
        fig.add_trace(go.Scatter(x=e, y=train_acc, name="Train Acc",
                                  line=dict(color="royalblue", dash="dot")),
                      row=2, col=1)
        fig.add_trace(go.Scatter(x=e, y=val_acc,   name="Val Acc",
                                  line=dict(color="tomato", dash="dot")),
                      row=2, col=1)

    fig.update_layout(title="Training History — Interactive",
                      template="plotly_dark", height=600)
    fig.show()


def iplot_confusion_matrix(y_true=None, y_pred=None, class_names=None):
    """
    Interactive confusion matrix — hover to see counts.

    Example:
        iplot_confusion_matrix()
        iplot_confusion_matrix(y_test, model.predict(X_test))
    """
    if not PLOTLY_OK: print("pip install plotly"); return
    if y_true is None and SK_OK:
        iris   = load_iris()
        X, y   = iris.data, iris.target
        Xtr,Xte,ytr,yte = train_test_split(X, y, test_size=0.3, random_state=0)
        y_true = yte
        y_pred = SVC().fit(Xtr,ytr).predict(Xte)
        class_names = list(iris.target_names)
    cm = confusion_matrix(y_true, y_pred)
    fig = px.imshow(cm, text_auto=True,
                    x=class_names, y=class_names,
                    color_continuous_scale="Blues",
                    labels=dict(x="Predicted", y="True"),
                    title="Interactive Confusion Matrix")
    fig.update_layout(template="plotly_dark")
    fig.show()


def iplot_roc_curve(y_true=None, y_prob=None):
    """
    Interactive ROC curve.

    Example:
        iplot_roc_curve()
        iplot_roc_curve(y_test, model.predict_proba(X_test)[:,1])
    """
    if not PLOTLY_OK: print("pip install plotly"); return
    if y_true is None and SK_OK:
        X,y = make_classification(500, random_state=0)
        Xtr,Xte,ytr,yte = train_test_split(X,y,test_size=0.3,random_state=0)
        y_true = yte
        y_prob = LogisticRegression().fit(Xtr,ytr).predict_proba(Xte)[:,1]
    fpr,tpr,_ = roc_curve(y_true, y_prob)
    roc_auc   = auc(fpr, tpr)
    import pandas as pd
    fig = px.line(pd.DataFrame({"FPR":fpr,"TPR":tpr}), x="FPR", y="TPR",
                  title=f"ROC Curve  (AUC = {roc_auc:.3f})",
                  template="plotly_dark")
    fig.add_shape(type="line", line=dict(dash="dash", color="gray"),
                  x0=0, x1=1, y0=0, y1=1)
    fig.show()


def iplot_pca_3d(X=None, y=None, labels=None):
    """
    Interactive 3-D PCA projection — rotate in browser.

    Example:
        iplot_pca_3d()
        iplot_pca_3d(X_train, y_train, labels=["Cat","Dog","Bird"])
    """
    if not PLOTLY_OK: print("pip install plotly"); return
    if not SK_OK:     print("sklearn required.");  return
    import pandas as pd
    if X is None:
        iris   = load_iris()
        X, y   = iris.data, iris.target
        labels = list(iris.target_names)
    Xs  = StandardScaler().fit_transform(X)
    X3d = PCA(3).fit_transform(Xs)
    df  = pd.DataFrame(X3d, columns=["PC1","PC2","PC3"])
    df["label"] = [str(labels[i]) if labels else str(i) for i in y]
    fig = px.scatter_3d(df, x="PC1", y="PC2", z="PC3",
                         color="label", opacity=0.8,
                         title="3-D PCA Projection — Interactive",
                         template="plotly_dark")
    fig.show()


def iplot_kmeans(X=None, k=3):
    """
    Interactive K-Means cluster scatter.

    Example:
        iplot_kmeans()
        iplot_kmeans(X_train, k=4)
    """
    if not PLOTLY_OK: print("pip install plotly"); return
    if not SK_OK:     print("sklearn required.");  return
    import pandas as pd
    if X is None:
        X,_ = make_blobs(n_samples=300, centers=k, random_state=42)
    km = KMeans(k, random_state=42, n_init=10).fit(X)
    df = pd.DataFrame(X, columns=["x1","x2"])
    df["cluster"] = km.labels_.astype(str)
    centers = pd.DataFrame(km.cluster_centers_, columns=["x1","x2"])
    centers["cluster"] = "centroid"
    fig = px.scatter(df, x="x1", y="x2", color="cluster",
                     title=f"K-Means Clustering  (k={k}) — Interactive",
                     template="plotly_dark")
    fig.add_trace(go.Scatter(x=centers["x1"], y=centers["x2"],
                              mode="markers",
                              marker=dict(symbol="x", size=16,
                                          color="white", line=dict(width=2)),
                              name="Centroids"))
    fig.show()


# ════════════════════════════════════════════════════════════════════════════
# [S20] DEMO RUNNER
#   run_all_demos() calls every function once with default (demo) data.
#   Great for a "gallery" tour of everything available.
# ════════════════════════════════════════════════════════════════════════════

# ════════════════════════════════════════════════════════════════════════════
# [S21] ANDREW NG SPECIFIC — Softmax, Polynomial, Optimizers, NN Diagram
# ════════════════════════════════════════════════════════════════════════════

def plot_softmax(z=None):
    """
    Softmax function — multiclass output layer visualization.
    Shows how raw logits get converted to probabilities that sum to 1.

    Args:
        z : 1-D array of raw logits (default: demo with 5 classes)

    Example:
        plot_softmax()
        plot_softmax(np.array([2.0, 1.0, 0.1, -1.0, 3.5]))
    """
    if z is None:
        z = np.array([2.0, 1.0, 0.1, -1.5, 3.5])

    exp_z   = np.exp(z - np.max(z))          # numerically stable
    softmax = exp_z / exp_z.sum()
    classes = [f"Class {i}" for i in range(len(z))]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Softmax Function — Multiclass Output Layer",
                 fontsize=13, fontweight="bold")

    # Raw logits
    bars1 = axes[0].bar(classes, z, color=CB, alpha=0.8, edgecolor="k")
    axes[0].set_title("Raw Logits  z")
    axes[0].set_ylabel("z value")
    for bar, v in zip(bars1, z):
        axes[0].text(bar.get_x() + bar.get_width()/2,
                     v + 0.05, f"{v:.1f}", ha="center", fontsize=10)

    # Softmax probabilities
    colors_prob = [CG if s == softmax.max() else CB for s in softmax]
    bars2 = axes[1].bar(classes, softmax, color=colors_prob, alpha=0.8, edgecolor="k")
    axes[1].set_title("Softmax Probabilities  σ(z)   [sum = 1.0]")
    axes[1].set_ylabel("Probability")
    axes[1].set_ylim(0, 1.1)
    axes[1].axhline(1/len(z), color=CGR, ls="--", lw=1.5,
                    label=f"Uniform = {1/len(z):.2f}")
    axes[1].legend()
    for bar, v in zip(bars2, softmax):
        axes[1].text(bar.get_x() + bar.get_width()/2,
                     v + 0.02, f"{v:.3f}", ha="center", fontsize=10)

    _show(fig)


def plot_polynomial_regression(x=None, y=None, degrees=None):
    """
    Overlay multiple polynomial fits on same plot — see complexity effect.

    Args:
        x, y    : 1-D training arrays
        degrees : list of polynomial degrees to fit and overlay

    Example:
        plot_polynomial_regression()
        plot_polynomial_regression(x_train, y_train, degrees=[1, 2, 5, 10])
    """
    if x is None:
        np.random.seed(3)
        x = np.sort(np.random.uniform(0, 1, 40))
        y = np.sin(2 * np.pi * x) + 0.25 * np.random.randn(40)
    if degrees is None:
        degrees = [1, 2, 4, 9]

    x_plot  = np.linspace(x.min(), x.max(), 400)
    colors  = plt.cm.plasma(np.linspace(0.1, 0.9, len(degrees)))

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(x, y, color="black", s=45, zorder=5, label="Training data", alpha=0.8)
    ax.plot(x_plot, np.sin(2*np.pi*x_plot), "k--",
            lw=1.5, alpha=0.4, label="True f(x)")

    for deg, col in zip(degrees, colors):
        coef  = np.polyfit(x, y, deg)
        y_fit = np.clip(np.poly1d(coef)(x_plot), -2.5, 2.5)
        ax.plot(x_plot, y_fit, color=col, lw=2.2, label=f"degree = {deg}")

    ax.set_xlim(x.min(), x.max()); ax.set_ylim(-2.2, 2.2)
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_title("Polynomial Regression — Degree Comparison", fontsize=13)
    ax.legend(fontsize=9)
    _show(fig)


def plot_optimizer_comparison(x=None, y=None, n_iter=150):
    """
    Compare SGD / Momentum / RMSprop / Adam paths on the cost contour.
    Shows WHY adaptive optimizers converge faster & smoother.

    Args:
        x, y   : 1-D training arrays
        n_iter : optimization steps

    Example:
        plot_optimizer_comparison()
        plot_optimizer_comparison(x_train, y_train, n_iter=200)
    """
    if x is None:
        x = np.array([1., 2., 3., 4., 5.])
        y = np.array([200., 400., 600., 800., 1000.])

    opt_w, opt_b = np.polyfit(x, y, 1)
    W, B, J      = _make_grid(x, y)

    def run_optimizer(name, w0, b0, lr):
        """Simulate optimizer paths with momentum/adaptive logic."""
        m  = len(x)
        w, b = float(w0), float(b0)
        ws, bs = [w], [b]
        # state variables
        vw = vb = 0.0       # momentum
        sw = sb = 1e-8      # RMSprop
        mw = mb = 0.0       # Adam first moment
        vw2= vb2= 0.0       # Adam second moment
        beta1, beta2, eps = 0.9, 0.999, 1e-8

        for t in range(1, n_iter + 1):
            err  = w * x + b - y
            gw   = np.dot(err, x) / m
            gb   = np.sum(err)    / m

            if name == "SGD":
                w -= lr * gw; b -= lr * gb
            elif name == "Momentum":
                vw = beta1*vw + (1-beta1)*gw
                vb = beta1*vb + (1-beta1)*gb
                w -= lr * vw; b -= lr * vb
            elif name == "RMSprop":
                sw = beta2*sw + (1-beta2)*gw**2
                sb = beta2*sb + (1-beta2)*gb**2
                w -= lr * gw / (np.sqrt(sw) + eps)
                b -= lr * gb / (np.sqrt(sb) + eps)
            elif name == "Adam":
                mw = beta1*mw + (1-beta1)*gw
                mb = beta1*mb + (1-beta1)*gb
                vw2= beta2*vw2+ (1-beta2)*gw**2
                vb2= beta2*vb2+ (1-beta2)*gb**2
                mw_c = mw/(1-beta1**t); mb_c = mb/(1-beta1**t)
                vw_c = vw2/(1-beta2**t); vb_c = vb2/(1-beta2**t)
                w -= lr * mw_c/(np.sqrt(vw_c)+eps)
                b -= lr * mb_c/(np.sqrt(vb_c)+eps)
            ws.append(w); bs.append(b)
        return ws, bs

    w0, b0 = opt_w - 200, opt_b - 150
    optimizers = [
        ("SGD",       0.008, CR),
        ("Momentum",  0.02,  CG),
        ("RMSprop",   10.0,  CO),
        ("Adam",      10.0,  CP),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Optimizer Comparison — SGD vs Momentum vs RMSprop vs Adam",
                 fontsize=13, fontweight="bold")

    cf = axes[0].contourf(W, B, J, levels=50, cmap="viridis", alpha=0.8)
    plt.colorbar(cf, ax=axes[0], label="J(w,b)")

    for name, lr, col in optimizers:
        ws, bs = run_optimizer(name, w0, b0, lr)
        J_hist = [compute_cost(x, y, w_, b_) for w_, b_ in zip(ws, bs)]
        axes[0].plot(ws, bs, ".-", color=col, lw=1.8, ms=3, label=name, zorder=4)
        axes[1].plot(J_hist, color=col, lw=2.5, label=name)

    axes[0].scatter([w0], [b0], color="yellow", s=120, zorder=6, label="Start")
    axes[0].set_xlabel("w"); axes[0].set_ylabel("b")
    axes[0].set_title("Path on Contour"); axes[0].legend(fontsize=9)

    axes[1].set_xlabel("Iterations"); axes[1].set_ylabel("Cost J")
    axes[1].set_title("Convergence Speed"); axes[1].legend()
    _show(fig)


def plot_minibatch_comparison(x=None, y=None, n_iter=80):
    """
    Compare Batch GD vs Mini-batch vs SGD convergence noise.

    Args:
        x, y   : 1-D training arrays (ideally 50+ points)
        n_iter : epochs

    Example:
        plot_minibatch_comparison()
    """
    np.random.seed(0)
    if x is None:
        x = np.linspace(1, 10, 80)
        y = 2.5 * x + 5 + np.random.randn(80) * 4

    m = len(x)

    def run(batch_size, alpha=0.002):
        w, b = 0.0, 0.0
        J_hist = []
        for epoch in range(n_iter):
            idx = np.random.permutation(m)
            for start in range(0, m, batch_size):
                xi  = x[idx[start:start+batch_size]]
                yi  = y[idx[start:start+batch_size]]
                bm  = len(xi)
                err = w*xi + b - yi
                w  -= alpha * np.dot(err, xi) / bm
                b  -= alpha * np.sum(err)      / bm
            J_hist.append(compute_cost(x, y, w, b))
        return J_hist

    J_batch = run(m,   alpha=0.002)
    J_mini  = run(16,  alpha=0.002)
    J_sgd   = run(1,   alpha=0.0005)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(J_batch, color=CB,  lw=2.5, label="Batch GD (full dataset)")
    ax.plot(J_mini,  color=CG,  lw=2.0, label="Mini-batch (size=16)")
    ax.plot(J_sgd,   color=CR,  lw=1.5, alpha=0.8, label="SGD (size=1) — noisy")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Cost J")
    ax.set_title("Batch GD  vs  Mini-batch  vs  SGD\n"
                 "Trade-off: smoothness vs speed per epoch", fontsize=12)
    ax.legend()
    _show(fig)


def plot_nn_architecture(layer_sizes=None, activation_labels=None):
    """
    Draw a neural network architecture diagram — nodes and edges.

    Args:
        layer_sizes       : list of ints — neurons per layer
                            e.g. [2, 4, 4, 1] = input(2) → hidden(4) → hidden(4) → output(1)
        activation_labels : list of strings, one per layer after input
                            e.g. ['ReLU', 'ReLU', 'Sigmoid']

    Example:
        plot_nn_architecture()
        plot_nn_architecture([3, 5, 4, 2], ['ReLU', 'ReLU', 'Softmax'])
        plot_nn_architecture([1, 4, 4, 4, 1], ['ReLU','ReLU','ReLU','Linear'])
    """
    if layer_sizes is None:
        layer_sizes = [2, 4, 3, 1]
    if activation_labels is None:
        activation_labels = (["ReLU"] * (len(layer_sizes) - 2)) + ["Sigmoid"]

    n_layers  = len(layer_sizes)
    max_nodes = max(layer_sizes)
    fig_h     = max(5, max_nodes * 0.9)
    fig, ax   = plt.subplots(figsize=(n_layers * 2.5, fig_h))
    ax.axis("off")
    ax.set_xlim(-0.5, n_layers - 0.5)
    ax.set_ylim(-0.5, max_nodes + 0.5)
    fig.suptitle("Neural Network Architecture", fontsize=13, fontweight="bold")

    node_pos = {}
    layer_names = (["Input"] +
                   [f"Hidden {i+1}" for i in range(n_layers - 2)] +
                   ["Output"])

    for l_idx, n_nodes in enumerate(layer_sizes):
        offset = (max_nodes - n_nodes) / 2
        for n_idx in range(n_nodes):
            y = offset + n_idx
            node_pos[(l_idx, n_idx)] = (l_idx, y)

            col = (CB if l_idx == 0 else
                   CO if l_idx == n_layers - 1 else CG)
            circle = plt.Circle((l_idx, y), 0.22, color=col,
                                  zorder=3, ec="black", lw=1.2)
            ax.add_patch(circle)

        # Draw edges to next layer
        if l_idx < n_layers - 1:
            for n_i in range(n_nodes):
                for n_j in range(layer_sizes[l_idx + 1]):
                    x0, y0 = node_pos[(l_idx,   n_i)]
                    x1, y1 = node_pos[(l_idx+1, n_j)]
                    ax.plot([x0, x1], [y0, y1], color=CGR,
                            lw=0.5, alpha=0.5, zorder=1)

        # Layer label
        ax.text(l_idx, -0.3, layer_names[l_idx],
                ha="center", fontsize=9, fontweight="bold")
        ax.text(l_idx, max_nodes + 0.2, f"({n_nodes})",
                ha="center", fontsize=8, color=CGR)

        # Activation label
        if l_idx > 0 and l_idx - 1 < len(activation_labels):
            ax.text(l_idx, max_nodes + 0.0,
                    activation_labels[l_idx - 1],
                    ha="center", fontsize=8, color=CR,
                    style="italic")

    # Legend patches
    import matplotlib.patches as mpatches
    legend_elements = [
        mpatches.Patch(color=CB, label="Input layer"),
        mpatches.Patch(color=CG, label="Hidden layer"),
        mpatches.Patch(color=CO, label="Output layer"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=9)
    _show(fig)


def plot_dropout_effect(layer_size=10, drop_rate=0.5):
    """
    Visualize dropout — show which neurons are active vs dropped.

    Args:
        layer_size : number of neurons in the layer
        drop_rate  : fraction of neurons dropped (0 to 1)

    Example:
        plot_dropout_effect()
        plot_dropout_effect(layer_size=20, drop_rate=0.3)
    """
    np.random.seed(42)
    mask   = np.random.rand(layer_size) > drop_rate
    scaled = mask / (1 - drop_rate)   # inverted dropout scaling

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f"Dropout Regularisation  (drop_rate = {drop_rate})",
                 fontsize=13, fontweight="bold")

    # Before dropout
    axes[0].bar(range(layer_size), np.ones(layer_size),
                color=CB, edgecolor="k", alpha=0.85)
    axes[0].set_title("Before Dropout\n(All neurons active)")
    axes[0].set_xlabel("Neuron index"); axes[0].set_ylabel("Active")
    axes[0].set_ylim(0, 1.5)
    axes[0].set_yticks([0, 1]); axes[0].set_yticklabels(["Off", "On"])

    # After dropout
    colors = [CG if m else CR for m in mask]
    bars   = axes[1].bar(range(layer_size), mask.astype(float),
                          color=colors, edgecolor="k", alpha=0.85)
    # Inverted dropout scale annotation
    for i, (bar, m, s) in enumerate(zip(bars, mask, scaled)):
        if m:
            axes[1].text(bar.get_x() + bar.get_width()/2, 1.05,
                          f"×{s:.1f}", ha="center", fontsize=7, color=CG)
    axes[1].set_title(f"After Dropout\n"
                      f"({int(mask.sum())} active / {layer_size} total — "
                      f"scaled ×{1/(1-drop_rate):.1f})")
    axes[1].set_xlabel("Neuron index"); axes[1].set_ylabel("Active")
    axes[1].set_ylim(0, 1.5)
    axes[1].set_yticks([0, 1]); axes[1].set_yticklabels(["Dropped ❌", "Active ✅"])

    import matplotlib.patches as mpatches
    axes[1].legend(handles=[
        mpatches.Patch(color=CG, label="Active neuron"),
        mpatches.Patch(color=CR, label="Dropped neuron"),
    ], fontsize=9)
    _show(fig)


def plot_batch_normalization(data=None):
    """
    Show how batch normalization shifts and scales layer distributions.
    Illustrates internal covariate shift problem + fix.

    Args:
        data : 2-D array (n_samples × n_features) — raw pre-activations

    Example:
        plot_batch_normalization()
        plot_batch_normalization(layer_outputs)
    """
    np.random.seed(5)
    if data is None:
        # Simulate pre-activations with different means/variances per feature
        data = np.column_stack([
            np.random.normal(8,   3,   300),    # very shifted
            np.random.normal(-5,  1,   300),    # negative shifted
            np.random.normal(0,   10,  300),    # large variance
            np.random.normal(3,   0.2, 300),    # tiny variance
        ])

    eps     = 1e-8
    mu      = data.mean(0); sigma = data.std(0)
    normed  = (data - mu) / (sigma + eps)
    gamma   = np.array([1.5, 2.0, 0.5, 1.0])   # learned scale
    beta    = np.array([0.5, -1.0, 0.0, 2.0])  # learned shift
    scaled  = gamma * normed + beta

    n_feat = data.shape[1]
    fig, axes = plt.subplots(3, n_feat, figsize=(4*n_feat, 10))
    fig.suptitle("Batch Normalisation — Distribution at Each Step",
                 fontsize=13, fontweight="bold")

    row_labels = ["Raw Input", "After BN (μ=0, σ=1)", "After γ,β (learned scale+shift)"]
    datasets   = [data, normed, scaled]

    for row, (dset, row_lbl) in enumerate(zip(datasets, row_labels)):
        for col in range(n_feat):
            ax = axes[row, col]
            sns.histplot(dset[:, col], kde=True, ax=ax,
                         color=[CB, CG, CO][row], alpha=0.65, bins=25)
            if row == 0:
                ax.set_title(f"Feature {col+1}\n"
                              f"μ={dset[:,col].mean():.1f}, σ={dset[:,col].std():.1f}",
                              fontsize=9)
            else:
                ax.set_title(f"μ={dset[:,col].mean():.2f}, σ={dset[:,col].std():.2f}",
                              fontsize=9)
            if col == 0:
                ax.set_ylabel(row_lbl, fontsize=9)
            ax.set_xlabel("")

    _show(fig)


def plot_collaborative_filtering(R=None, user_names=None, item_names=None):
    """
    Visualize a user-item rating matrix — core of collaborative filtering.

    Args:
        R          : 2-D array (n_users × n_items), NaN = unrated
        user_names : list of user name strings
        item_names : list of item name strings

    Example:
        plot_collaborative_filtering()
        plot_collaborative_filtering(R=my_rating_matrix,
                                     user_names=["Alice","Bob","Carol"],
                                     item_names=["Movie1","Movie2","Movie3"])
    """
    if R is None:
        np.random.seed(7)
        R = np.array([
            [5, 4, np.nan, 1, np.nan],
            [np.nan, 3, 4, np.nan, 2],
            [1, np.nan, np.nan, 5, 4],
            [np.nan, 2, 5, np.nan, 3],
            [4, np.nan, 3, 2, np.nan],
        ], dtype=float)
        user_names = [f"User {i+1}" for i in range(R.shape[0])]
        item_names = [f"Item {j+1}" for j in range(R.shape[1])]

    import pandas as pd
    df  = pd.DataFrame(R, index=user_names, columns=item_names)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Collaborative Filtering — User-Item Rating Matrix",
                 fontsize=13, fontweight="bold")

    # Raw ratings with missing values shown
    mask_nan = np.isnan(R)
    R_disp   = np.where(mask_nan, -1, R)
    cmap_cf  = plt.cm.get_cmap("RdYlGn", 7)
    im = axes[0].imshow(R_disp, cmap=cmap_cf, vmin=-1, vmax=5, aspect="auto")
    axes[0].set_xticks(range(len(item_names))); axes[0].set_xticklabels(item_names, rotation=30)
    axes[0].set_yticks(range(len(user_names))); axes[0].set_yticklabels(user_names)
    axes[0].set_title("Rating Matrix  (grey = unrated)")
    for i in range(R.shape[0]):
        for j in range(R.shape[1]):
            txt = "?" if np.isnan(R[i,j]) else str(int(R[i,j]))
            axes[0].text(j, i, txt, ha="center", va="center",
                          fontsize=12, color="black",
                          fontweight="bold" if not np.isnan(R[i,j]) else "normal")
    plt.colorbar(im, ax=axes[0], label="Rating (−1 = missing)")

    # Sparsity analysis
    rated  = (~mask_nan).sum()
    total  = R.size
    sparse = mask_nan.mean() * 100
    user_counts = (~mask_nan).sum(axis=1)
    axes[1].barh(user_names, user_counts, color=CB, edgecolor="k", alpha=0.8)
    axes[1].axvline(user_counts.mean(), color=CR, ls="--",
                    label=f"Mean = {user_counts.mean():.1f}")
    axes[1].set_xlabel("Number of ratings")
    axes[1].set_title(f"Ratings per User\n"
                       f"Sparsity = {sparse:.1f}%  "
                       f"({rated}/{total} rated)")
    axes[1].legend()
    _show(fig)


# ════════════════════════════════════════════════════════════════════════════
# [S22] EVALUATION — F1, Calibration, Lift, CV Folds
# ════════════════════════════════════════════════════════════════════════════

def plot_f1_vs_threshold(y_true=None, y_prob=None):
    """
    Plot Precision, Recall, and F1 score vs classification threshold.
    Helps you pick the best threshold beyond the default 0.5.

    Args:
        y_true : 1-D binary ground-truth labels
        y_prob : 1-D predicted probabilities for class 1

    Example:
        plot_f1_vs_threshold()
        y_prob = model.predict_proba(X_test)[:, 1]
        plot_f1_vs_threshold(y_test, y_prob)
    """
    if y_true is None and SK_OK:
        X, y = make_classification(500, random_state=2)
        Xtr,Xte,ytr,yte = train_test_split(X, y, test_size=0.3, random_state=2)
        y_true = yte
        y_prob = LogisticRegression().fit(Xtr,ytr).predict_proba(Xte)[:,1]
    elif y_true is None:
        raise ValueError("Pass y_true and y_prob.")

    from sklearn.metrics import f1_score, precision_score, recall_score
    thresholds = np.linspace(0.01, 0.99, 200)
    precs, recs, f1s = [], [], []
    for t in thresholds:
        preds = (np.array(y_prob) >= t).astype(int)
        precs.append(precision_score(y_true, preds, zero_division=0))
        recs.append(recall_score(y_true, preds, zero_division=0))
        f1s.append(f1_score(y_true, preds, zero_division=0))

    best_t = thresholds[np.argmax(f1s)]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(thresholds, precs, color=CB,  lw=2.0, label="Precision")
    ax.plot(thresholds, recs,  color=CR,  lw=2.0, label="Recall")
    ax.plot(thresholds, f1s,   color=CG,  lw=3.0, label="F1 Score")
    ax.axvline(best_t, color=CO, ls="--", lw=2,
               label=f"Best threshold = {best_t:.2f}  (F1={max(f1s):.3f})")
    ax.axvline(0.5, color=CGR, ls=":", lw=1.5, label="Default = 0.5")
    ax.set_xlabel("Classification Threshold", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Precision / Recall / F1  vs  Threshold", fontsize=13)
    ax.legend(); ax.set_ylim(0, 1.05)
    _show(fig)


def plot_calibration_curve(y_true=None, y_prob=None, model_name="Model"):
    """
    Reliability diagram — is your model's probability output trustworthy?
    A perfectly calibrated model lies on the diagonal.

    Args:
        y_true      : 1-D binary ground-truth labels
        y_prob      : 1-D predicted probabilities
        model_name  : label for legend

    Example:
        plot_calibration_curve()
        plot_calibration_curve(y_test, model.predict_proba(X_test)[:,1],
                               model_name="Random Forest")
    """
    if not SK_OK: print("sklearn required."); return
    from sklearn.calibration import calibration_curve

    if y_true is None:
        X, y = make_classification(600, random_state=3)
        Xtr,Xte,ytr,yte = train_test_split(X,y,test_size=0.3,random_state=3)
        y_true = yte
        y_prob = RandomForestClassifier(50,random_state=3).fit(Xtr,ytr).predict_proba(Xte)[:,1]

    frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=10)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Calibration / Reliability Diagram", fontsize=13, fontweight="bold")

    axes[0].plot([0,1],[0,1], "k--", lw=1.5, label="Perfect calibration")
    axes[0].plot(mean_pred, frac_pos, "s-", color=CB, lw=2.5,
                 ms=8, label=model_name)
    axes[0].fill_between(mean_pred, mean_pred, frac_pos,
                          alpha=0.15, color=CR, label="Calibration gap")
    axes[0].set_xlabel("Mean Predicted Probability")
    axes[0].set_ylabel("Fraction of Positives")
    axes[0].set_title("Calibration Curve")
    axes[0].legend(); axes[0].set_xlim(0,1); axes[0].set_ylim(0,1)

    axes[1].hist(y_prob, bins=30, color=CB, alpha=0.75, edgecolor="k")
    axes[1].set_xlabel("Predicted Probability")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Predicted Probability Distribution")
    _show(fig)


def plot_cv_folds(X=None, y=None, n_folds=5):
    """
    Visualize cross-validation fold splits — see what train/val looks like.

    Args:
        X       : 2-D feature array (uses sample indices)
        y       : 1-D label array
        n_folds : number of CV folds

    Example:
        plot_cv_folds()
        plot_cv_folds(X_train, y_train, n_folds=10)
    """
    if not SK_OK: print("sklearn required."); return
    from sklearn.model_selection import KFold

    n_samples = 80 if X is None else len(X)
    if y is None: y = np.zeros(n_samples)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=0)

    fig, ax = plt.subplots(figsize=(12, 4))
    fig.suptitle(f"K-Fold Cross-Validation  (k = {n_folds})",
                 fontsize=13, fontweight="bold")

    indices_dummy = np.arange(n_samples)
    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(indices_dummy)):
        ax.barh(fold_idx, len(train_idx), left=0,
                height=0.6, color=CB, alpha=0.7)
        # Overlay val fold
        val_start = val_idx.min()
        ax.barh(fold_idx, len(val_idx), left=val_start,
                height=0.6, color=CR, alpha=0.9)
        ax.text(n_samples + 1, fold_idx,
                f"Train={len(train_idx)} | Val={len(val_idx)}",
                va="center", fontsize=9)

    ax.set_xlabel("Sample index"); ax.set_ylabel("Fold")
    ax.set_yticks(range(n_folds))
    ax.set_yticklabels([f"Fold {i+1}" for i in range(n_folds)])
    ax.set_title(f"{n_samples} samples split into {n_folds} folds")

    import matplotlib.patches as mpatches
    ax.legend(handles=[
        mpatches.Patch(color=CB, label="Training set"),
        mpatches.Patch(color=CR, label="Validation set"),
    ], loc="lower right")
    _show(fig)


# ════════════════════════════════════════════════════════════════════════════
# [S23] ADVANCED CLUSTERING — DBSCAN, Dendrogram, GMM
# ════════════════════════════════════════════════════════════════════════════

def plot_dbscan(X=None, eps=0.3, min_samples=5):
    """
    DBSCAN clustering — handles non-convex shapes, detects outliers.
    Compare with K-Means on the same data.

    Args:
        X          : 2-D array  (default: moons dataset)
        eps        : neighbourhood radius
        min_samples: min points to form a core point

    Example:
        plot_dbscan()
        plot_dbscan(X_train, eps=0.5, min_samples=10)
    """
    if not SK_OK: print("sklearn required."); return
    from sklearn.cluster import DBSCAN

    if X is None:
        X, _ = make_moons(n_samples=300, noise=0.08, random_state=0)

    db_labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X)
    km_labels = KMeans(n_clusters=2, random_state=0, n_init=10).fit_predict(X)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("DBSCAN vs K-Means on Non-Convex Data",
                 fontsize=13, fontweight="bold")

    for ax, labels, title in zip(axes,
                                   [km_labels, db_labels],
                                   ["K-Means  (fails on moons)",
                                    f"DBSCAN  ε={eps}  (handles shapes)"]):
        unique = np.unique(labels)
        cmap   = plt.cm.tab10(np.linspace(0, 0.5, max(len(unique), 2)))
        for lbl, col in zip(unique, cmap):
            mask = labels == lbl
            name = f"Noise ({mask.sum()})" if lbl == -1 else f"Cluster {lbl+1} ({mask.sum()})"
            ax.scatter(X[mask, 0], X[mask, 1],
                       color="red" if lbl == -1 else col,
                       s=35, alpha=0.8,
                       marker="x" if lbl == -1 else "o",
                       edgecolors="k" if lbl != -1 else None,
                       lw=0.3, label=name)
        ax.set_title(title); ax.legend(fontsize=8)
        ax.set_xlabel("Feature 1"); ax.set_ylabel("Feature 2")

    _show(fig)


def plot_dendrogram(X=None, method="ward", n_labels=15):
    """
    Hierarchical clustering dendrogram.

    Args:
        X        : 2-D array  (default: iris dataset)
        method   : linkage method — 'ward', 'complete', 'average', 'single'
        n_labels : max samples to label on x-axis

    Example:
        plot_dendrogram()
        plot_dendrogram(X_train, method="complete")
    """
    try:
        from scipy.cluster.hierarchy import dendrogram, linkage
        from scipy.spatial.distance   import pdist
    except ImportError:
        print("scipy required: pip install scipy"); return

    if X is None and SK_OK:
        iris = load_iris()
        X    = StandardScaler().fit_transform(iris.data[:50])
    elif X is None:
        X = np.random.randn(40, 3)

    Z   = linkage(X, method=method)
    fig, ax = plt.subplots(figsize=(12, 5))
    dendrogram(Z, ax=ax, truncate_mode="lastp", p=n_labels,
               leaf_rotation=45, leaf_font_size=9,
               color_threshold=0.7 * max(Z[:, 2]))
    ax.set_xlabel("Sample index / Cluster size")
    ax.set_ylabel("Distance")
    ax.set_title(f"Hierarchical Clustering Dendrogram  (linkage={method})",
                 fontsize=13)
    _show(fig)


def plot_gmm(X=None, n_components=3):
    """
    Gaussian Mixture Model — soft cluster boundaries with probability ellipses.

    Args:
        X            : 2-D array  (default: blobs)
        n_components : number of Gaussian components

    Example:
        plot_gmm()
        plot_gmm(X_train, n_components=4)
    """
    if not SK_OK: print("sklearn required."); return
    from sklearn.mixture import GaussianMixture

    if X is None:
        X, _ = make_blobs(n_samples=300, centers=n_components,
                           cluster_std=0.8, random_state=0)

    gmm    = GaussianMixture(n_components, covariance_type="full",
                              random_state=0)
    labels = gmm.fit_predict(X)
    probs  = gmm.predict_proba(X).max(axis=1)   # confidence per point

    # Probability grid
    h  = 0.05
    xx, yy = np.meshgrid(
        np.linspace(X[:,0].min()-1, X[:,0].max()+1, 300),
        np.linspace(X[:,1].min()-1, X[:,1].max()+1, 300))
    Z  = -gmm.score_samples(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f"Gaussian Mixture Model  (k={n_components})",
                 fontsize=13, fontweight="bold")

    # Hard assignments coloured by confidence
    sc = axes[0].scatter(X[:,0], X[:,1], c=labels, cmap="tab10",
                          s=40, alpha=0.85, edgecolors="k", lw=0.2)
    # Draw ellipses for each component
    from matplotlib.patches import Ellipse
    for k in range(n_components):
        mean_ = gmm.means_[k]
        cov_  = gmm.covariances_[k]
        vals, vecs = np.linalg.eigh(cov_)
        order = vals.argsort()[::-1]
        vals, vecs = vals[order], vecs[:, order]
        theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))
        for nsig, alpha_e in [(1, 0.4), (2, 0.2)]:
            w, h_e = 2 * nsig * np.sqrt(vals)
            ell = Ellipse(xy=mean_, width=w, height=h_e, angle=theta,
                           color="red", alpha=alpha_e, lw=1.5, fill=True)
            axes[0].add_patch(ell)
    axes[0].set_title("Cluster Assignments + Confidence Ellipses")
    plt.colorbar(sc, ax=axes[0], label="Cluster ID")

    # Density surface
    cf = axes[1].contourf(xx, yy, Z, levels=40, cmap="viridis")
    axes[1].scatter(X[:,0], X[:,1], c="white", s=10, alpha=0.4, zorder=5)
    plt.colorbar(cf, ax=axes[1], label="−log likelihood")
    axes[1].set_title("Negative Log-Likelihood Density")

    _show(fig)


# ════════════════════════════════════════════════════════════════════════════
# [S24] DEEP LEARNING — t-SNE, Vanishing Gradients, Weight Dist, LR Finder
# ════════════════════════════════════════════════════════════════════════════

def plot_tsne(X=None, y=None, labels=None, perplexity=30):
    """
    t-SNE 2-D embedding — visualise how a model 'sees' high-dimensional data.
    Great for seeing cluster structure in embeddings / layer outputs.

    Args:
        X          : 2-D array (high-dimensional features or embeddings)
        y          : 1-D integer label array
        labels     : list of class name strings
        perplexity : t-SNE perplexity parameter (try 5–50)

    Example:
        plot_tsne()
        plot_tsne(layer_outputs, y_test, perplexity=20)
    """
    if not SK_OK: print("sklearn required."); return
    from sklearn.manifold import TSNE

    if X is None:
        iris   = load_iris()
        X, y   = iris.data, iris.target
        labels = list(iris.target_names)

    print(f"  Running t-SNE on {X.shape} … (may take a few seconds)")
    X2  = TSNE(n_components=2, perplexity=perplexity,
               random_state=0, n_iter=1000).fit_transform(
               StandardScaler().fit_transform(X))

    fig, ax = plt.subplots(figsize=(8, 7))
    classes = np.unique(y)
    colors  = plt.cm.tab10(np.linspace(0, 0.9, len(classes)))
    for cls, col in zip(classes, colors):
        mask = y == cls
        lbl  = labels[cls] if labels is not None else f"Class {cls}"
        ax.scatter(X2[mask, 0], X2[mask, 1],
                   color=col, s=40, alpha=0.8,
                   edgecolors="k", lw=0.2, label=lbl)
    ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")
    ax.set_title(f"t-SNE Embedding  (perplexity={perplexity})\n"
                 f"Data shape: {X.shape}", fontsize=12)
    ax.legend(fontsize=9)
    _show(fig)


def plot_vanishing_gradient(n_layers=8, activation="sigmoid"):
    """
    Show how gradients shrink (or explode) layer-by-layer during backprop.
    Classic vanishing gradient problem with sigmoid vs ReLU.

    Args:
        n_layers   : number of layers to simulate
        activation : 'sigmoid' or 'relu' or 'tanh'

    Example:
        plot_vanishing_gradient()
        plot_vanishing_gradient(n_layers=12, activation='relu')
    """
    np.random.seed(1)

    def simulate_gradient_norms(act_fn, n_layers, n_neurons=64):
        """Simulate gradient norms flowing backward through random weights."""
        norms = [1.0]
        g = np.ones(n_neurons)
        for _ in range(n_layers):
            W  = np.random.randn(n_neurons, n_neurons) * (0.5 if act_fn == "sigmoid"
                                                           else np.sqrt(2/n_neurons))
            z  = np.random.randn(n_neurons)
            if act_fn == "sigmoid":
                s      = sigmoid(z)
                d_act  = s * (1 - s)          # max = 0.25 → shrinks gradient
            elif act_fn == "relu":
                d_act  = (z > 0).astype(float)
            else:
                d_act  = 1 - np.tanh(z)**2
            g = (W.T @ (d_act * g))
            norms.append(np.linalg.norm(g))
        return norms

    layers  = list(range(n_layers + 1))
    fns     = {"sigmoid": CR, "relu": CG, "tanh": CB}

    fig, ax = plt.subplots(figsize=(10, 6))
    for fn, col in fns.items():
        norms = simulate_gradient_norms(fn, n_layers)
        ax.plot(layers, norms, "o-", color=col, lw=2.5,
                ms=7, label=fn.capitalize())

    ax.axhline(1e-4, color=CGR, ls=":", lw=1.5, label="Near-zero (vanished)")
    ax.set_yscale("log")
    ax.set_xlabel("Layer (counting from output backward)", fontsize=12)
    ax.set_ylabel("Gradient Norm  (log scale)", fontsize=12)
    ax.set_title("Vanishing Gradient Problem\n"
                 "Sigmoid/Tanh crush gradients; ReLU preserves them",
                 fontsize=12)
    ax.legend()
    _show(fig)


def plot_weight_distributions(model=None, X=None, y=None):
    """
    Histogram of weight values per layer — detect dead neurons or exploding weights.

    Args:
        model : fitted sklearn estimator (uses MLPClassifier by default)
        X, y  : training data

    Example:
        plot_weight_distributions()

        # With your keras/pytorch model, pass weights as list of arrays:
        weights_list = [layer.get_weights()[0] for layer in model.layers if len(layer.get_weights()) > 0]
        plot_weight_distributions(weights_list)
    """
    if not SK_OK: print("sklearn required."); return
    from sklearn.neural_network import MLPClassifier

    # Accept list of weight arrays directly
    if isinstance(model, list):
        weight_arrays = model
        layer_names   = [f"Layer {i+1}" for i in range(len(weight_arrays))]
    else:
        if X is None:
            iris = load_iris()
            X, y = iris.data, iris.target
        if model is None:
            model = MLPClassifier(hidden_layer_sizes=(16, 16),
                                   max_iter=200, random_state=0).fit(X, y)
        weight_arrays = model.coefs_
        layer_names   = [f"Layer {i+1} ({w.shape[0]}→{w.shape[1]})"
                          for i, w in enumerate(weight_arrays)]

    n = len(weight_arrays)
    fig, axes = plt.subplots(1, n, figsize=(5*n, 5))
    fig.suptitle("Weight Distributions per Layer\n"
                 "(Look for: very small → vanishing | very large → exploding)",
                 fontsize=12, fontweight="bold")
    if n == 1: axes = [axes]

    colors = plt.cm.viridis(np.linspace(0.2, 0.8, n))
    for ax, W, name, col in zip(axes, weight_arrays, layer_names, colors):
        flat = W.flatten()
        ax.hist(flat, bins=50, color=col, alpha=0.8, edgecolor="k", lw=0.3)
        ax.axvline(0, color="red", lw=1.5, ls="--")
        ax.set_title(f"{name}\nμ={flat.mean():.3f}  σ={flat.std():.3f}", fontsize=9)
        ax.set_xlabel("Weight value")
        if ax == axes[0]: ax.set_ylabel("Count")

    _show(fig)


def plot_lr_finder(losses=None, lrs=None):
    """
    Learning rate finder — plot loss vs LR to find the optimal LR range.
    The ideal LR is just before the loss starts exploding.

    Args:
        losses : list of loss values at increasing LRs
        lrs    : list of corresponding learning rates (log-spaced)

    Example:
        plot_lr_finder()   # simulated curve

        # With real training:
        # (Run 1 epoch while exponentially increasing LR, record losses)
        plot_lr_finder(my_losses, my_lrs)
    """
    if lrs is None:
        lrs  = np.logspace(-5, 0, 200)
        # Simulate: loss decreases then explodes
        noise  = np.random.randn(200) * 0.15
        losses = (3.0 * np.exp(-300*(lrs - 0.0005))
                  + 0.5 + noise
                  + 20 * np.maximum(0, lrs - 0.05)**0.5)
        losses = np.clip(losses, 0, 8)

    lrs    = np.array(lrs)
    losses = np.array(losses)
    # Smooth
    from scipy.ndimage import uniform_filter1d
    smooth = uniform_filter1d(losses, size=10)

    # Find steepest descent zone
    grad    = np.gradient(smooth)
    best_lr = lrs[np.argmin(grad)]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(lrs, losses, color=CGR, lw=1.2, alpha=0.5, label="Raw loss")
    ax.plot(lrs, smooth, color=CB,  lw=2.5, label="Smoothed loss")
    ax.axvline(best_lr, color=CR, ls="--", lw=2,
               label=f"Steepest descent ≈ {best_lr:.1e}  ← suggested LR")
    ax.axvspan(best_lr*0.1, best_lr, alpha=0.1, color=CG,
               label="Good LR range")
    ax.set_xscale("log")
    ax.set_xlabel("Learning Rate (log scale)", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title("Learning Rate Finder\n"
                 "Use LR just before the loss starts to rise", fontsize=12)
    ax.legend()
    _show(fig)


# ════════════════════════════════════════════════════════════════════════════
# [S25] REGRESSION DIAGNOSTICS
# ════════════════════════════════════════════════════════════════════════════

def plot_predicted_vs_actual(y_true=None, y_pred=None, xlabel="Actual y",
                              ylabel="Predicted ŷ"):
    """
    Scatter ŷ vs y — a perfect model lies on the diagonal.
    Also shows residuals as a secondary panel.

    Args:
        y_true : 1-D ground-truth values
        y_pred : 1-D predicted values

    Example:
        plot_predicted_vs_actual()
        y_pred = model.predict(X_test)
        plot_predicted_vs_actual(y_test, y_pred)
    """
    if y_true is None and SK_OK:
        X, y = make_regression(200, n_features=5, noise=20, random_state=0)
        Xtr,Xte,ytr,yte = train_test_split(X, y, test_size=0.3, random_state=0)
        y_true = yte
        y_pred = LinearRegression().fit(Xtr,ytr).predict(Xte)
    elif y_true is None:
        raise ValueError("Pass y_true and y_pred.")

    from sklearn.metrics import mean_squared_error, r2_score
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)

    mn = min(y_true.min(), y_pred.min())
    mx = max(y_true.max(), y_pred.max())

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Regression Diagnostics   RMSE={rmse:.2f}   R²={r2:.3f}",
                 fontsize=13, fontweight="bold")

    # Predicted vs actual
    axes[0].scatter(y_true, y_pred, color=CB, alpha=0.6,
                    s=40, edgecolors="k", lw=0.2)
    axes[0].plot([mn,mx],[mn,mx], "r--", lw=2, label="Perfect fit")
    axes[0].set_xlabel(xlabel); axes[0].set_ylabel(ylabel)
    axes[0].set_title("Predicted vs Actual")
    axes[0].legend()

    # Residuals
    resid = np.array(y_pred) - np.array(y_true)
    axes[1].scatter(y_pred, resid, color=CO, alpha=0.6,
                    s=40, edgecolors="k", lw=0.2)
    axes[1].axhline(0, color="red", lw=2, ls="--")
    axes[1].set_xlabel("Predicted ŷ"); axes[1].set_ylabel("Residual  (ŷ − y)")
    axes[1].set_title("Residuals vs Predicted\n(should be random around 0)")
    _show(fig)


def plot_qq_residuals(y_true=None, y_pred=None):
    """
    Q-Q plot of residuals — check normality assumption of linear regression.
    Points should lie on the diagonal for normally distributed residuals.

    Args:
        y_true : 1-D ground-truth values
        y_pred : 1-D predicted values

    Example:
        plot_qq_residuals()
        plot_qq_residuals(y_test, model.predict(X_test))
    """
    if y_true is None and SK_OK:
        X, y = make_regression(200, n_features=3, noise=15, random_state=1)
        Xtr,Xte,ytr,yte = train_test_split(X, y, test_size=0.3, random_state=1)
        y_true = yte
        y_pred = LinearRegression().fit(Xtr,ytr).predict(Xte)
    elif y_true is None:
        raise ValueError("Pass y_true and y_pred.")

    resid = np.array(y_pred) - np.array(y_true)
    resid_sorted = np.sort((resid - resid.mean()) / resid.std())
    n     = len(resid_sorted)
    theor = np.array([np.percentile(np.random.randn(10000), 100*(i-0.5)/n)
                       for i in range(1, n+1)])

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Residual Normality Check", fontsize=13, fontweight="bold")

    # Q-Q plot
    axes[0].scatter(theor, resid_sorted, color=CB, s=40, alpha=0.7,
                    edgecolors="k", lw=0.2)
    mn_ = min(theor.min(), resid_sorted.min())
    mx_ = max(theor.max(), resid_sorted.max())
    axes[0].plot([mn_, mx_], [mn_, mx_], "r--", lw=2, label="Normal line")
    axes[0].set_xlabel("Theoretical Quantiles")
    axes[0].set_ylabel("Sample Quantiles (residuals)")
    axes[0].set_title("Q-Q Plot\n(deviations = non-normal residuals)")
    axes[0].legend()

    # Residual histogram
    sns.histplot(resid, kde=True, ax=axes[1], color=CB, alpha=0.7, bins=30)
    axes[1].set_xlabel("Residual"); axes[1].set_title("Residual Distribution")
    _show(fig)


# ════════════════════════════════════════════════════════════════════════════
# [S26] EDA — Missing Values, Class Imbalance, Outliers, Correlation Rank
# ════════════════════════════════════════════════════════════════════════════

def plot_missing_values(df=None):
    """
    Heatmap of missing values + bar chart of missing % per feature.
    Essential first step in any EDA.

    Args:
        df : pandas DataFrame  (default: generated with injected NaNs)

    Example:
        plot_missing_values()
        plot_missing_values(my_df)
    """
    import pandas as pd
    if df is None:
        np.random.seed(0)
        data = np.random.randn(100, 8)
        # Inject missing values
        for col in range(8):
            miss_rate = np.random.uniform(0, 0.4)
            mask = np.random.rand(100) < miss_rate
            data[mask, col] = np.nan
        df = pd.DataFrame(data, columns=[f"Feature {i+1}" for i in range(8)])

    missing_pct = df.isnull().mean() * 100
    missing_pct = missing_pct.sort_values(ascending=False)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Missing Value Analysis  "
                 f"({df.isnull().sum().sum()} missing / {df.size} total = "
                 f"{df.isnull().mean().mean()*100:.1f}%)",
                 fontsize=13, fontweight="bold")

    # Heatmap
    sns.heatmap(df.isnull(), cbar=True, yticklabels=False,
                cmap="viridis", ax=axes[0])
    axes[0].set_title("Missing Value Map\n(yellow = missing)")
    axes[0].set_xlabel("Feature")

    # Bar chart
    colors = [CR if p > 20 else CO if p > 10 else CG for p in missing_pct]
    axes[1].barh(missing_pct.index, missing_pct.values,
                  color=colors, edgecolor="k", alpha=0.85)
    axes[1].axvline(20, color=CR, ls="--", lw=1.5, label=">20% = high risk")
    axes[1].axvline(10, color=CO, ls="--", lw=1.5, label=">10% = moderate")
    axes[1].set_xlabel("Missing %")
    axes[1].set_title("% Missing per Feature")
    axes[1].legend(fontsize=9)
    _show(fig)


def plot_class_imbalance(y=None, class_names=None):
    """
    Visualise class distribution — detect imbalance before training.

    Args:
        y           : 1-D integer label array
        class_names : list of class name strings

    Example:
        plot_class_imbalance()
        plot_class_imbalance(y_train, class_names=["Benign","Malignant"])
    """
    if y is None:
        y = np.concatenate([np.zeros(300), np.ones(80), np.full(20, 2)])
        class_names = ["Class 0 (majority)", "Class 1", "Class 2 (minority)"]

    classes, counts = np.unique(y, return_counts=True)
    total = counts.sum()
    pcts  = counts / total * 100
    if class_names is None:
        class_names = [f"Class {int(c)}" for c in classes]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Class Imbalance Analysis", fontsize=13, fontweight="bold")

    bar_colors = [CR if p < 10 else CO if p < 20 else CG for p in pcts]
    bars = axes[0].bar(class_names, counts, color=bar_colors,
                        edgecolor="k", alpha=0.85)
    axes[0].axhline(total / len(classes), color="blue",
                    ls="--", lw=2, label="Balanced baseline")
    for bar, cnt, pct in zip(bars, counts, pcts):
        axes[0].text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + total*0.01,
                     f"{cnt}\n({pct:.1f}%)", ha="center", fontsize=10)
    axes[0].set_ylabel("Count"); axes[0].set_title("Class Distribution")
    axes[0].legend()

    wedge_colors = [CR if p < 10 else CO if p < 20 else CG for p in pcts]
    axes[1].pie(counts, labels=class_names, autopct="%1.1f%%",
                colors=wedge_colors, startangle=90,
                wedgeprops=dict(edgecolor="white", linewidth=2))
    axes[1].set_title("Class Proportion\n"
                       f"Imbalance ratio = {counts.max()/counts.min():.1f}:1")
    _show(fig)


def plot_feature_target_correlation(X=None, y=None, feature_names=None):
    """
    Ranked bar chart of each feature's correlation with the target.
    Useful for feature selection / understanding data.

    Args:
        X            : 2-D array (n_samples × n_features)
        y            : 1-D target array
        feature_names: list of strings

    Example:
        plot_feature_target_correlation()
        plot_feature_target_correlation(X_train, y_train,
                                        feature_names=["age","income","score"])
    """
    import pandas as pd
    if X is None and SK_OK:
        iris   = load_iris()
        X, y   = iris.data, iris.target.astype(float)
        feature_names = iris.feature_names
    elif X is None:
        X = np.random.randn(100, 5)
        y = X[:,0]*2 + X[:,2] + np.random.randn(100)
        feature_names = [f"F{i+1}" for i in range(5)]

    if feature_names is None:
        feature_names = [f"F{i+1}" for i in range(X.shape[1])]

    corrs = [np.corrcoef(X[:,i], y)[0,1] for i in range(X.shape[1])]
    df    = pd.DataFrame({"feature": feature_names, "correlation": corrs})
    df    = df.reindex(df["correlation"].abs().sort_values(ascending=True).index)

    colors = [CR if c < 0 else CG for c in df["correlation"]]

    fig, ax = plt.subplots(figsize=(9, max(4, len(feature_names)*0.6)))
    bars = ax.barh(df["feature"], df["correlation"],
                   color=colors, edgecolor="k", alpha=0.85)
    for bar, v in zip(bars, df["correlation"]):
        ax.text(v + (0.01 if v >= 0 else -0.01), bar.get_y() + bar.get_height()/2,
                f"{v:.3f}", va="center",
                ha="left" if v >= 0 else "right", fontsize=9)
    ax.axvline(0, color="black", lw=1.2)
    ax.set_xlabel("Pearson Correlation with Target")
    ax.set_title("Feature-Target Correlation (ranked)", fontsize=13)
    import matplotlib.patches as mpatches
    ax.legend(handles=[mpatches.Patch(color=CG, label="Positive correlation"),
                        mpatches.Patch(color=CR, label="Negative correlation")])
    _show(fig)


def plot_outlier_detection(X=None, feature_names=None, method="iqr"):
    """
    Scatter + outlier highlights using IQR or Z-score method.

    Args:
        X            : 2-D array (uses first 2 features for scatter)
        feature_names: list of strings
        method       : 'iqr' or 'zscore'

    Example:
        plot_outlier_detection()
        plot_outlier_detection(X_train, method='zscore')
    """
    import pandas as pd
    if X is None:
        np.random.seed(0)
        X = np.random.randn(150, 3)
        # Inject outliers
        outlier_rows = np.random.choice(150, 10, replace=False)
        X[outlier_rows] += np.random.choice([-4, 4], size=(10, 3))
        feature_names = ["Feature 1", "Feature 2", "Feature 3"]

    if feature_names is None:
        feature_names = [f"F{i+1}" for i in range(X.shape[1])]

    # Detect outliers
    if method == "iqr":
        Q1, Q3  = np.percentile(X, 25, axis=0), np.percentile(X, 75, axis=0)
        IQR     = Q3 - Q1
        outlier_mask = np.any((X < Q1 - 1.5*IQR) | (X > Q3 + 1.5*IQR), axis=1)
        title_method = "IQR Method  (|x − median| > 1.5 × IQR)"
    else:
        z_scores     = np.abs((X - X.mean(0)) / X.std(0))
        outlier_mask = np.any(z_scores > 3, axis=1)
        title_method = "Z-score Method  (|z| > 3)"

    n_out = outlier_mask.sum()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Outlier Detection — {title_method}\n"
                 f"Found {n_out} outliers out of {len(X)} samples",
                 fontsize=12, fontweight="bold")

    # Scatter (first 2 features)
    axes[0].scatter(X[~outlier_mask, 0], X[~outlier_mask, 1],
                    color=CB, s=40, alpha=0.7, label=f"Normal ({(~outlier_mask).sum()})")
    axes[0].scatter(X[outlier_mask, 0],  X[outlier_mask, 1],
                    color=CR, s=100, marker="X", zorder=5,
                    label=f"Outlier ({n_out})")
    axes[0].set_xlabel(feature_names[0]); axes[0].set_ylabel(feature_names[1])
    axes[0].set_title("Scatter: Outliers Highlighted"); axes[0].legend()

    # Box plot per feature coloured by outlier status
    df_plot = pd.DataFrame(X, columns=feature_names)
    df_plot["outlier"] = outlier_mask
    df_melt = df_plot.melt(id_vars="outlier",
                            value_vars=feature_names,
                            var_name="Feature", value_name="Value")
    sns.boxplot(data=df_melt, x="Feature", y="Value", ax=axes[1],
                color=CB, flierprops=dict(marker="x", color=CR,
                                           markeredgewidth=1.5, markersize=6))
    axes[1].set_title("Box Plots per Feature\n(× markers = outliers)")
    _show(fig)


# ════════════════════════════════════════════════════════════════════════════
# [S27]  SVM — MARGIN, KERNELS, SUPPORT VECTORS
# ════════════════════════════════════════════════════════════════════════════

def plot_svm_margin(X=None, y=None, C=1.0):
    """
    SVM decision boundary with margin bands + support vectors highlighted.
    Andrew Ng ML Specialization — Week 7 topic.

    Args:
        X : 2-D array (n_samples × 2 features)
        y : 1-D binary labels {0, 1}
        C : regularisation parameter (larger C = smaller margin)

    Example:
        plot_svm_margin()
        plot_svm_margin(X_train, y_train, C=0.1)   # wide margin
        plot_svm_margin(X_train, y_train, C=100)   # tight margin
    """
    if not SK_OK: print("sklearn required."); return
    if X is None:
        X, y = make_classification(n_samples=80, n_features=2, n_redundant=0,
                                    n_clusters_per_class=1, class_sep=1.2,
                                    random_state=1)

    clf = SVC(kernel="linear", C=C)
    clf.fit(X, y)

    h  = 0.02
    xx, yy = np.meshgrid(np.arange(X[:,0].min()-0.5, X[:,0].max()+0.5, h),
                          np.arange(X[:,1].min()-0.5, X[:,1].max()+0.5, h))
    Z  = clf.decision_function(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(f"Support Vector Machine  (C={C})", fontsize=13, fontweight="bold")

    for ax, show_margin in zip(axes, [False, True]):
        ax.contourf(xx, yy, Z, levels=[-1e5, 0, 1e5],
                    colors=[CR, CB], alpha=0.2)
        ax.contour(xx, yy, Z, levels=[0], colors="black", linewidths=2.5)
        if show_margin:
            ax.contour(xx, yy, Z, levels=[-1, 1],
                       colors=["red","blue"], linewidths=1.5, linestyles="--")
            ax.scatter(clf.support_vectors_[:,0], clf.support_vectors_[:,1],
                       s=200, facecolors="none", edgecolors="black",
                       linewidths=2.5, zorder=6, label=f"Support vectors ({len(clf.support_vectors_)})")
            ax.legend()

        ax.scatter(X[y==0,0], X[y==0,1], color=CR, s=50, edgecolors="k",
                   lw=0.4, alpha=0.85, label="Class 0")
        ax.scatter(X[y==1,0], X[y==1,1], color=CB, s=50, edgecolors="k",
                   lw=0.4, alpha=0.85, label="Class 1")
        ax.set_xlabel("Feature 1"); ax.set_ylabel("Feature 2")
        ax.set_title("Decision Boundary" if not show_margin
                     else "Margin + Support Vectors\n(dashed = margin, circled = SVs)")
        ax.legend(fontsize=9)

    _show(fig)


def plot_kernel_comparison(X=None, y=None):
    """
    Compare SVM kernel decision boundaries: Linear / RBF / Polynomial / Sigmoid.
    Shows WHY non-linear kernels are needed for non-separable data.

    Args:
        X : 2-D array
        y : 1-D binary labels

    Example:
        plot_kernel_comparison()
        plot_kernel_comparison(X_train, y_train)
    """
    if not SK_OK: print("sklearn required."); return
    if X is None:
        X, y = make_circles(n_samples=200, noise=0.1, factor=0.4, random_state=1)

    kernels = ["linear", "rbf", "poly", "sigmoid"]
    titles  = ["Linear kernel\n(fails on circles)",
               "RBF kernel ✅\n(handles circles)",
               "Polynomial kernel\n(deg=3)",
               "Sigmoid kernel"]
    colors  = [CR, CG, CO, CP]

    h = 0.03
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle("SVM Kernel Comparison", fontsize=13, fontweight="bold")

    for ax, kernel, title in zip(axes, kernels, titles):
        clf = SVC(kernel=kernel, C=1.0, gamma="auto")
        clf.fit(X, y)
        xx, yy = np.meshgrid(
            np.arange(X[:,0].min()-0.3, X[:,0].max()+0.3, h),
            np.arange(X[:,1].min()-0.3, X[:,1].max()+0.3, h))
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
        ax.contourf(xx, yy, Z, alpha=0.35, cmap="RdBu")
        ax.scatter(X[:,0], X[:,1], c=y, cmap="RdBu",
                   edgecolors="k", s=40, zorder=5)
        ax.set_title(title, fontsize=11)

    _show(fig)


# ════════════════════════════════════════════════════════════════════════════
# [S28]  GRADIENT CHECKING  (Andrew Ng DL Specialization Week 1)
# ════════════════════════════════════════════════════════════════════════════

def plot_gradient_checking(x=None, y=None, w_check=None):
    """
    Numerical gradient vs analytical gradient — Andrew Ng DL Spec debugging tool.
    Shows that analytical gradient is correct by comparing to finite differences.

    Args:
        x, y    : 1-D training arrays
        w_check : weight value to check around (default: near optimum)

    Example:
        plot_gradient_checking()
        plot_gradient_checking(x_train, y_train)
    """
    if x is None:
        x = np.array([1., 2., 3., 4., 5.])
        y = np.array([200., 400., 600., 800., 1000.])

    opt_w, opt_b = np.polyfit(x, y, 1)
    if w_check is None: w_check = opt_w - 50

    # Analytical gradient at w_check, b=opt_b
    m          = len(x)
    analytical = np.dot(w_check * x + opt_b - y, x) / m

    # Numerical gradient via finite differences at multiple epsilon
    epsilons   = np.logspace(-6, 0, 60)
    numerical  = [(compute_cost(x, y, w_check+e, opt_b) -
                   compute_cost(x, y, w_check-e, opt_b)) / (2*e)
                  for e in epsilons]
    rel_error  = [abs(n - analytical) / (abs(n) + abs(analytical) + 1e-12)
                  for n in numerical]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Gradient Checking — Numerical vs Analytical",
                 fontsize=13, fontweight="bold")

    axes[0].axhline(analytical, color=CB, lw=2.5, ls="--",
                    label=f"Analytical: {analytical:.4f}")
    axes[0].plot(epsilons, numerical, color=CR, lw=2.0,
                 label="Numerical ≈ [J(w+ε)−J(w−ε)] / 2ε")
    axes[0].set_xscale("log")
    axes[0].set_xlabel("ε (epsilon)"); axes[0].set_ylabel("Gradient value")
    axes[0].set_title("Gradient Values vs ε")
    axes[0].legend()

    axes[1].loglog(epsilons, rel_error, color=CG, lw=2.5)
    axes[1].axhline(1e-5, color=CR, ls="--", lw=1.5,
                    label="Threshold: 1e-5 → OK ✅")
    axes[1].axhline(1e-3, color=CO, ls="--", lw=1.5,
                    label="Threshold: 1e-3 → Suspicious ⚠️")
    best_eps = epsilons[np.argmin(rel_error)]
    axes[1].axvline(best_eps, color=CB, ls=":", lw=1.5,
                    label=f"Best ε ≈ {best_eps:.1e}")
    axes[1].set_xlabel("ε"); axes[1].set_ylabel("Relative Error")
    axes[1].set_title("Relative Error\n(should be < 1e-5 for correct gradient)")
    axes[1].legend(fontsize=9)
    _show(fig)


# ════════════════════════════════════════════════════════════════════════════
# [S29]  DECISION TREE VISUALIZATION
# ════════════════════════════════════════════════════════════════════════════

def plot_decision_tree(X=None, y=None, max_depth=3, feature_names=None,
                        class_names=None):
    """
    Visualize a decision tree structure — splits, Gini/entropy, samples.

    Args:
        X            : 2-D feature array
        y            : 1-D label array
        max_depth    : max tree depth to display
        feature_names: list of strings
        class_names  : list of class name strings

    Example:
        plot_decision_tree()
        plot_decision_tree(X_train, y_train, max_depth=4,
                           feature_names=["Age","Income"],
                           class_names=["No","Yes"])
    """
    if not SK_OK: print("sklearn required."); return
    from sklearn.tree import DecisionTreeClassifier, plot_tree

    if X is None:
        iris = load_iris()
        X, y = iris.data, iris.target
        feature_names = iris.feature_names
        class_names   = list(iris.target_names)

    dt = DecisionTreeClassifier(max_depth=max_depth, random_state=0)
    dt.fit(X, y)

    n_nodes = dt.tree_.node_count
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    fig.suptitle(f"Decision Tree  (max_depth={max_depth}, nodes={n_nodes})",
                 fontsize=13, fontweight="bold")

    # Tree diagram
    plot_tree(dt, ax=axes[0], feature_names=feature_names,
              class_names=class_names, filled=True, rounded=True,
              fontsize=8, impurity=True, proportion=False)
    axes[0].set_title("Tree Structure")

    # Decision boundary (first 2 features)
    h  = 0.05
    xx, yy = np.meshgrid(
        np.linspace(X[:,0].min()-0.3, X[:,0].max()+0.3, 300),
        np.linspace(X[:,1].min()-0.3, X[:,1].max()+0.3, 300))
    Z  = dt.predict(np.column_stack([
        xx.ravel(), yy.ravel(),
        np.full(xx.size, X[:,2:].mean(0)[0]) if X.shape[1]>2 else np.zeros(xx.size),
        np.full(xx.size, X[:,3:].mean(0)[0]) if X.shape[1]>3 else np.zeros(xx.size),
    ])[:, :X.shape[1]]).reshape(xx.shape)
    axes[1].contourf(xx, yy, Z, alpha=0.35, cmap="Set1")
    axes[1].scatter(X[:,0], X[:,1], c=y, cmap="Set1",
                    edgecolors="k", s=40, zorder=5)
    fn = feature_names if feature_names else ["F1","F2"]
    axes[1].set_xlabel(fn[0]); axes[1].set_ylabel(fn[1])
    axes[1].set_title("Decision Boundary (first 2 features)")
    _show(fig)


# ════════════════════════════════════════════════════════════════════════════
# [S30]  HYPERPARAMETER TUNING
# ════════════════════════════════════════════════════════════════════════════

def plot_validation_curve(X=None, y=None, param_name="C",
                           param_range=None, model_cls=None):
    """
    Validation curve — model hyperparameter vs train & val score.
    DIFFERENT from learning curve (this varies complexity, not data size).

    Args:
        X           : 2-D feature array
        y           : 1-D label array
        param_name  : hyperparameter name to sweep (string)
        param_range : array of values to try
        model_cls   : sklearn estimator class (default: SVC)

    Example:
        plot_validation_curve()
        # SVM with different C values:
        plot_validation_curve(X_train, y_train, param_name='C',
                              param_range=np.logspace(-3,3,20),
                              model_cls=SVC)
        # Random Forest with different depths:
        from sklearn.ensemble import RandomForestClassifier
        plot_validation_curve(X_train, y_train, param_name='max_depth',
                              param_range=range(1,16),
                              model_cls=RandomForestClassifier)
    """
    if not SK_OK: print("sklearn required."); return
    from sklearn.model_selection import validation_curve as vc

    if X is None:
        X, y = make_classification(200, n_features=5, random_state=0)
    if param_range is None:
        param_range = np.logspace(-3, 3, 15)
    if model_cls is None:
        model_cls = SVC

    tr_sc, val_sc = vc(model_cls(), X, y,
                        param_name=param_name,
                        param_range=param_range,
                        cv=5, scoring="accuracy", n_jobs=-1)

    tr_mean  = tr_sc.mean(1);  tr_std  = tr_sc.std(1)
    val_mean = val_sc.mean(1); val_std = val_sc.std(1)
    best_idx = val_mean.argmax()

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(param_range, tr_mean,  color=CB, lw=2.5, label="Train score")
    ax.plot(param_range, val_mean, color=CR, lw=2.5, label="CV score")
    ax.fill_between(param_range, tr_mean-tr_std,  tr_mean+tr_std,  alpha=0.15, color=CB)
    ax.fill_between(param_range, val_mean-val_std, val_mean+val_std, alpha=0.15, color=CR)
    ax.axvline(param_range[best_idx], color=CO, ls="--", lw=2,
               label=f"Best {param_name} = {param_range[best_idx]:.3g}")
    try: ax.set_xscale("log")
    except Exception: pass
    ax.set_xlabel(f"Parameter: {param_name}")
    ax.set_ylabel("Score (Accuracy)")
    ax.set_title(f"Validation Curve — {model_cls.__name__}\n"
                 f"Hyperparameter: {param_name}", fontsize=12)
    ax.legend(); ax.set_ylim(0, 1.05)
    _show(fig)


def plot_hyperparameter_heatmap(X=None, y=None, C_range=None, gamma_range=None):
    """
    2-D grid search heatmap — CV accuracy for every (C, gamma) pair.
    Classic SVM hyperparameter tuning visualization.

    Args:
        X, y        : training data
        C_range     : array of C values  (x-axis)
        gamma_range : array of gamma values  (y-axis)

    Example:
        plot_hyperparameter_heatmap()
        plot_hyperparameter_heatmap(X_train, y_train,
                                     C_range=np.logspace(-2,3,6),
                                     gamma_range=np.logspace(-4,1,6))
    """
    if not SK_OK: print("sklearn required."); return
    from sklearn.model_selection import cross_val_score

    if X is None:
        X, y = make_classification(200, n_features=5, random_state=0)
    if C_range     is None: C_range     = np.logspace(-2, 3, 8)
    if gamma_range is None: gamma_range = np.logspace(-4, 1, 8)

    scores = np.zeros((len(gamma_range), len(C_range)))
    print(f"  Grid search: {len(C_range)}×{len(gamma_range)} = {scores.size} fits …")
    for i, g in enumerate(gamma_range):
        for j, c in enumerate(C_range):
            sc = cross_val_score(SVC(C=c, gamma=g), X, y, cv=3, scoring="accuracy")
            scores[i, j] = sc.mean()

    best_i, best_j = np.unravel_index(scores.argmax(), scores.shape)

    fig, ax = plt.subplots(figsize=(10, 7))
    im = ax.imshow(scores, cmap="RdYlGn", aspect="auto",
                   vmin=scores.min(), vmax=scores.max())
    plt.colorbar(im, ax=ax, label="CV Accuracy")

    ax.set_xticks(range(len(C_range)))
    ax.set_xticklabels([f"{c:.2g}" for c in C_range], rotation=30)
    ax.set_yticks(range(len(gamma_range)))
    ax.set_yticklabels([f"{g:.2g}" for g in gamma_range])
    ax.set_xlabel("C"); ax.set_ylabel("gamma")
    ax.set_title("SVM Grid Search — Cross-Validation Accuracy Heatmap\n"
                 f"Best: C={C_range[best_j]:.2g}, gamma={gamma_range[best_i]:.2g}  "
                 f"→ {scores[best_i,best_j]:.3f}")
    ax.scatter(best_j, best_i, s=200, color="gold",
               marker="*", zorder=5, label="Best params")
    ax.legend()

    for i in range(len(gamma_range)):
        for j in range(len(C_range)):
            ax.text(j, i, f"{scores[i,j]:.2f}", ha="center",
                    va="center", fontsize=7,
                    color="black" if 0.3 < scores[i,j] < 0.9 else "white")
    _show(fig)


# ════════════════════════════════════════════════════════════════════════════
# [S31]  CLUSTERING QUALITY — SILHOUETTE
# ════════════════════════════════════════════════════════════════════════════

def plot_silhouette_score(X=None, k_range=None):
    """
    Silhouette plot — cluster quality for each sample + mean score per K.
    Better than elbow for comparing K values.

    Args:
        X       : 2-D array
        k_range : list of K values to evaluate  (default: 2–6)

    Example:
        plot_silhouette_score()
        plot_silhouette_score(X_train, k_range=[2,3,4,5])
    """
    if not SK_OK: print("sklearn required."); return
    from sklearn.metrics import silhouette_samples, silhouette_score

    if X is None:
        X, _ = make_blobs(n_samples=300, centers=3, random_state=42)
    if k_range is None: k_range = [2, 3, 4, 5]

    n = len(k_range)
    fig, axes = plt.subplots(1, n, figsize=(5*n, 5))
    fig.suptitle("Silhouette Analysis — Cluster Quality per K",
                 fontsize=13, fontweight="bold")
    if n == 1: axes = [axes]

    cmap = plt.cm.tab10

    for ax, k in zip(axes, k_range):
        km      = KMeans(k, random_state=42, n_init=10).fit(X)
        labels  = km.labels_
        sil_avg = silhouette_score(X, labels)
        sil_sam = silhouette_samples(X, labels)

        y_lower = 10
        for c in range(k):
            c_vals = np.sort(sil_sam[labels == c])
            size   = c_vals.shape[0]
            y_upper = y_lower + size
            ax.fill_betweenx(np.arange(y_lower, y_upper), 0, c_vals,
                              facecolor=cmap(c/k), alpha=0.8)
            ax.text(-0.05, y_lower + size/2, str(c), fontsize=8)
            y_lower = y_upper + 5

        ax.axvline(sil_avg, color="red", ls="--", lw=2,
                   label=f"Mean = {sil_avg:.3f}")
        ax.set_xlabel("Silhouette coefficient")
        ax.set_title(f"K = {k}\nMean silhouette = {sil_avg:.3f}")
        ax.set_yticks([])
        ax.set_xlim(-0.2, 1.0)
        ax.legend(fontsize=8)

    _show(fig)


# ════════════════════════════════════════════════════════════════════════════
# [S32]  DEEP LEARNING EXTRAS
# ════════════════════════════════════════════════════════════════════════════

def plot_lr_schedule(n_epochs=100, warmup_epochs=10):
    """
    Visualize common learning rate schedules:
    Constant / Step Decay / Cosine Annealing / Warmup + Cosine.

    Args:
        n_epochs      : total training epochs
        warmup_epochs : warmup period for warmup+cosine schedule

    Example:
        plot_lr_schedule()
        plot_lr_schedule(n_epochs=200, warmup_epochs=20)
    """
    e    = np.arange(1, n_epochs + 1)
    lr0  = 0.1

    def step_decay(epoch, drop=0.5, every=20):
        return lr0 * (drop ** (epoch // every))

    def cosine_decay(epoch, n):
        return lr0 * 0.5 * (1 + np.cos(np.pi * epoch / n))

    def warmup_cosine(epoch, warmup, n):
        if epoch < warmup:
            return lr0 * epoch / warmup
        return lr0 * 0.5 * (1 + np.cos(np.pi * (epoch-warmup) / (n-warmup)))

    schedules = {
        "Constant":                  np.full(n_epochs, lr0),
        "Step Decay (×0.5 / 20 ep)": np.array([step_decay(i) for i in e]),
        "Cosine Annealing":           np.array([cosine_decay(i, n_epochs) for i in e]),
        f"Warmup ({warmup_epochs}ep) + Cosine": np.array(
            [warmup_cosine(i, warmup_epochs, n_epochs) for i in e]),
        "Exponential Decay":          lr0 * np.exp(-0.03 * e),
    }
    colors = [CGR, CO, CB, CG, CP]

    fig, ax = plt.subplots(figsize=(11, 6))
    for (name, lrs), col in zip(schedules.items(), colors):
        ax.plot(e, lrs, color=col, lw=2.5, label=name)

    ax.set_xlabel("Epoch"); ax.set_ylabel("Learning Rate")
    ax.set_title("Learning Rate Schedules Comparison", fontsize=13)
    ax.legend(); ax.set_ylim(bottom=0)
    _show(fig)


def plot_weight_initialization(n=500):
    """
    Compare weight initialization strategies — distributions they produce
    and their effect on signal propagation through layers.

    Args:
        n : number of weights to sample per strategy

    Example:
        plot_weight_initialization()
    """
    np.random.seed(0)
    fan_in = 256   # typical layer width

    inits = {
        "Zero Init":          np.zeros(n),
        "Random Normal\n(σ=1)": np.random.randn(n),
        "Too-small Normal\n(σ=0.01)": np.random.randn(n) * 0.01,
        "Xavier / Glorot\n(σ=1/√fan_in)": np.random.randn(n) * np.sqrt(1/fan_in),
        "He Init\n(σ=√2/fan_in)": np.random.randn(n) * np.sqrt(2/fan_in),
        "LeCun Init\n(σ=1/√fan_in)": np.random.randn(n) * np.sqrt(1/fan_in),
    }

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle("Weight Initialization Strategies\n"
                 "(Xavier/He avoid vanishing & exploding signals)",
                 fontsize=13, fontweight="bold")

    rec_colors = {
        "Zero Init": CR,
        "Too-small Normal\n(σ=0.01)": CO,
    }

    for ax, (name, weights) in zip(axes.flatten(), inits.items()):
        col = rec_colors.get(name, CG)
        sns.histplot(weights, bins=50, kde=True, ax=ax,
                     color=col, alpha=0.7)
        ax.set_title(f"{name}\nμ={weights.mean():.4f}  σ={weights.std():.4f}",
                     fontsize=10)
        ax.set_xlabel("Weight value")
        if name == "Zero Init":
            ax.set_title(f"{name} ❌ Dead neurons\nAll gradients = 0", fontsize=10)
        elif "small" in name:
            ax.set_title(f"{name} ⚠️ Vanishing signal", fontsize=10)
        elif "He" in name:
            ax.set_title(f"{name} ✅ Best for ReLU\n"
                          f"μ={weights.mean():.4f}  σ={weights.std():.4f}", fontsize=10)

    _show(fig)


def plot_loss_landscape(model_fn=None, n=80):
    """
    2-D slice through a neural network loss surface (Li et al. 2018 style).
    Shows flat regions, sharp minima, and why optimizers matter.

    Args:
        model_fn : callable(w1, w2) → scalar loss (default: simulated surface)
        n        : grid resolution

    Example:
        plot_loss_landscape()

        # With your own loss surface:
        def my_loss(w1, w2):
            # perturb model weights in two random directions and return loss
            ...
        plot_loss_landscape(my_loss)
    """
    w1 = np.linspace(-3, 3, n)
    w2 = np.linspace(-3, 3, n)
    W1, W2 = np.meshgrid(w1, w2)

    if model_fn is None:
        # Simulate: one sharp minimum + one flat wide basin
        Z = (0.3 * np.sin(2*W1) * np.cos(2*W2)
             + 0.5 * W1**2 + 0.5 * W2**2
             + 0.8 * np.exp(-5*((W1-0.8)**2 + (W2+0.5)**2))
             + 0.2 * np.exp(-20*((W1+0.5)**2 + (W2-0.5)**2)))
    else:
        Z = np.vectorize(model_fn)(W1, W2)

    fig = plt.figure(figsize=(16, 6))
    fig.suptitle("Neural Network Loss Landscape\n"
                 "(Sharp minima → poor generalisation | Flat minima → better generalisation)",
                 fontsize=12, fontweight="bold")

    from mpl_toolkits.mplot3d import Axes3D  # noqa

    ax1 = fig.add_subplot(121, projection="3d")
    surf = ax1.plot_surface(W1, W2, Z, cmap="viridis", alpha=0.85, edgecolor="none")
    fig.colorbar(surf, ax=ax1, shrink=0.4, label="Loss")
    ax1.set_xlabel("Direction 1"); ax1.set_ylabel("Direction 2")
    ax1.set_zlabel("Loss"); ax1.set_title("3-D Loss Surface")
    ax1.view_init(elev=30, azim=225)

    ax2 = fig.add_subplot(122)
    cf = ax2.contourf(W1, W2, Z, levels=40, cmap="viridis")
    ax2.contour(W1, W2, Z, levels=15, colors="white", alpha=0.25, linewidths=0.5)
    plt.colorbar(cf, ax=ax2, label="Loss")
    ax2.set_xlabel("Direction 1"); ax2.set_ylabel("Direction 2")
    ax2.set_title("Contour Map")
    _show(fig)


def plot_attention_heatmap(attention_matrix=None, tokens=None, title="Attention Weights"):
    """
    Transformer self-attention heatmap — visualise which tokens attend to which.

    Args:
        attention_matrix : 2-D array (seq_len × seq_len) of attention weights
        tokens           : list of token strings
        title            : plot title

    Example:
        plot_attention_heatmap()

        # With real transformer attention:
        # attention = model.get_attention_weights(input_ids)  # shape: (seq_len, seq_len)
        plot_attention_heatmap(attention[0], tokens=["The","cat","sat","on","mat"])
    """
    if attention_matrix is None:
        np.random.seed(2)
        tokens = ["The", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog"]
        n = len(tokens)
        # Simulate attention with some structure
        raw = np.random.rand(n, n) * 0.3
        for i in range(n):
            raw[i, i] += 1.0        # diagonal self-attention
            if i > 0: raw[i, i-1] += 0.5  # attend to prev token
        # Softmax row-wise
        attention_matrix = np.exp(raw) / np.exp(raw).sum(1, keepdims=True)

    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(attention_matrix, cmap="Blues", aspect="auto",
                   vmin=0, vmax=attention_matrix.max())
    plt.colorbar(im, ax=ax, label="Attention weight")

    if tokens is not None:
        ax.set_xticks(range(len(tokens)))
        ax.set_yticks(range(len(tokens)))
        ax.set_xticklabels(tokens, rotation=45, ha="right")
        ax.set_yticklabels(tokens)

    # Annotate cells
    for i in range(attention_matrix.shape[0]):
        for j in range(attention_matrix.shape[1]):
            ax.text(j, i, f"{attention_matrix[i,j]:.2f}",
                    ha="center", va="center", fontsize=7,
                    color="white" if attention_matrix[i,j] > 0.4 else "black")

    ax.set_xlabel("Key (attending TO)")
    ax.set_ylabel("Query (attending FROM)")
    ax.set_title(f"{title}\n(each row sums to 1 — softmax)", fontsize=12)
    _show(fig)


# ════════════════════════════════════════════════════════════════════════════
# [S33]  DISTRIBUTION SHIFT & UMAP
# ════════════════════════════════════════════════════════════════════════════

def plot_train_test_distribution(X_train=None, X_test=None, feature_names=None):
    """
    Compare train vs test feature distributions — detect covariate shift.
    Covariate shift = train and test come from different distributions → model fails.

    Args:
        X_train      : 2-D training array
        X_test       : 2-D test array
        feature_names: list of strings

    Example:
        plot_train_test_distribution()
        plot_train_test_distribution(X_train, X_test,
                                      feature_names=["age","income","score"])
    """
    if X_train is None:
        np.random.seed(0)
        # Simulate shift: test has different mean
        X_train = np.random.randn(200, 4)
        X_test  = np.random.randn(80, 4) + np.array([1.5, 0, -1, 0.5])
        feature_names = [f"Feature {i+1}" for i in range(4)]

    if feature_names is None:
        feature_names = [f"F{i+1}" for i in range(X_train.shape[1])]

    n_feat = X_train.shape[1]
    fig, axes = plt.subplots(1, n_feat, figsize=(5*n_feat, 5))
    fig.suptitle("Train vs Test Distribution  — Covariate Shift Detection\n"
                 "(large gap = distribution shift = model may fail on test)",
                 fontsize=12, fontweight="bold")
    if n_feat == 1: axes = [axes]

    for ax, i, name in zip(axes, range(n_feat), feature_names):
        sns.kdeplot(X_train[:,i], ax=ax, color=CB, fill=True,
                    alpha=0.5, label=f"Train (n={len(X_train)})", lw=2)
        sns.kdeplot(X_test[:,i],  ax=ax, color=CR, fill=True,
                    alpha=0.5, label=f"Test  (n={len(X_test)})",  lw=2)
        # KS statistic as a simple drift measure
        from scipy.stats import ks_2samp
        ks_stat, p_val = ks_2samp(X_train[:,i], X_test[:,i])
        color_warn = CR if ks_stat > 0.2 else CO if ks_stat > 0.1 else CG
        ax.set_title(f"{name}\nKS={ks_stat:.3f}  p={p_val:.3f}",
                      color=color_warn, fontsize=10)
        ax.set_xlabel(name); ax.legend(fontsize=8)

    _show(fig)


def plot_umap(X=None, y=None, labels=None, n_neighbors=15, min_dist=0.1):
    """
    UMAP 2-D embedding — faster and often better structure than t-SNE.

    Args:
        X          : 2-D feature array or embeddings
        y          : 1-D integer label array
        labels     : list of class name strings
        n_neighbors: UMAP n_neighbors parameter
        min_dist   : UMAP min_dist parameter

    Example:
        plot_umap()
        plot_umap(layer_outputs, y_test, n_neighbors=30)
    """
    try:
        import umap
        UMAP_OK = True
    except ImportError:
        UMAP_OK = False

    if X is None and SK_OK:
        iris   = load_iris()
        X, y   = iris.data, iris.target
        labels = list(iris.target_names)

    if UMAP_OK:
        print("  Running UMAP …")
        X2 = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist,
                        random_state=0).fit_transform(
                        StandardScaler().fit_transform(X))
        method = f"UMAP  (n_neighbors={n_neighbors}, min_dist={min_dist})"
    else:
        print("  umap-learn not installed — falling back to t-SNE.")
        print("  Install: pip install umap-learn")
        from sklearn.manifold import TSNE
        X2 = TSNE(n_components=2, random_state=0, n_iter=1000).fit_transform(
                  StandardScaler().fit_transform(X))
        method = "t-SNE (install umap-learn for UMAP)"

    fig, ax = plt.subplots(figsize=(8, 7))
    classes = np.unique(y)
    colors  = plt.cm.tab10(np.linspace(0, 0.9, len(classes)))
    for cls, col in zip(classes, colors):
        mask = y == cls
        lbl  = labels[cls] if labels is not None else f"Class {cls}"
        ax.scatter(X2[mask,0], X2[mask,1], color=col, s=40, alpha=0.8,
                   edgecolors="k", lw=0.2, label=lbl)
    ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
    ax.set_title(f"{method}\nData shape: {X.shape}", fontsize=12)
    ax.legend(fontsize=9)
    _show(fig)


# ════════════════════════════════════════════════════════════════════════════
# [S34]  EVALUATION EXTRAS
# ════════════════════════════════════════════════════════════════════════════

def plot_multiclass_roc(X=None, y=None, model=None, class_names=None):
    """
    One-vs-Rest ROC curves for K classes on one plot.

    Args:
        X, y        : test features and labels
        model       : fitted sklearn classifier with predict_proba()
        class_names : list of class name strings

    Example:
        plot_multiclass_roc()
        model = RandomForestClassifier().fit(X_train, y_train)
        plot_multiclass_roc(X_test, y_test, model)
    """
    if not SK_OK: print("sklearn required."); return
    from sklearn.preprocessing import label_binarize
    from sklearn.metrics import roc_curve, auc

    if X is None:
        iris = load_iris()
        X, y = iris.data, iris.target
        class_names = list(iris.target_names)
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0)
        model = RandomForestClassifier(50, random_state=0).fit(Xtr, ytr)
        X, y  = Xte, yte

    classes  = np.unique(y)
    n_cls    = len(classes)
    if class_names is None: class_names = [f"Class {c}" for c in classes]
    y_bin    = label_binarize(y, classes=classes)
    y_prob   = model.predict_proba(X)
    colors   = plt.cm.tab10(np.linspace(0, 0.9, n_cls))

    fig, ax  = plt.subplots(figsize=(8, 7))
    for i, (cls, col) in enumerate(zip(classes, colors)):
        fpr, tpr, _ = roc_curve(y_bin[:,i], y_prob[:,i])
        roc_auc     = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=col, lw=2.2,
                label=f"{class_names[i]}  (AUC={roc_auc:.2f})")

    ax.plot([0,1],[0,1], "k--", lw=1.5, label="Random")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("One-vs-Rest ROC Curves — Multiclass", fontsize=13)
    ax.legend(loc="lower right")
    _show(fig)


def plot_partial_dependence(model=None, X=None, feature_idx=None, feature_names=None):
    """
    Partial Dependence Plots (PDP) — how does changing one feature affect the prediction?
    Great for model interpretability.

    Args:
        model        : fitted sklearn estimator
        X            : 2-D feature array (used for marginalising other features)
        feature_idx  : list of feature indices to plot  (default: first 4)
        feature_names: list of feature name strings

    Example:
        plot_partial_dependence()
        rf = RandomForestClassifier().fit(X_train, y_train)
        plot_partial_dependence(rf, X_test, feature_idx=[0,1,2])
    """
    if not SK_OK: print("sklearn required."); return
    try:
        from sklearn.inspection import PartialDependenceDisplay
    except ImportError:
        print("sklearn >= 0.24 required for PartialDependenceDisplay"); return

    if X is None and model is None:
        iris  = load_iris()
        X, y  = iris.data, iris.target
        model = RandomForestClassifier(50, random_state=0).fit(X, y)
        feature_names = iris.feature_names
        feature_idx   = [0, 1, 2, 3]

    if feature_idx is None: feature_idx = list(range(min(4, X.shape[1])))

    fig, ax = plt.subplots(figsize=(5*len(feature_idx), 5))
    PartialDependenceDisplay.from_estimator(
        model, X, features=feature_idx,
        feature_names=feature_names,
        ax=ax, grid_resolution=50,
        kind="average")
    ax.set_title("Partial Dependence Plots\n"
                  "(y-axis = marginal effect on prediction)")
    _show(fig)


def plot_error_analysis(y_true=None, y_pred=None, X=None,
                         feature_names=None, class_names=None):
    """
    Error analysis — break down WHERE the model gets it wrong.
    Shows misclassification counts per class + feature distributions of errors.

    Args:
        y_true       : 1-D ground-truth labels
        y_pred       : 1-D predicted labels
        X            : 2-D feature array (for feature-level error breakdown)
        feature_names: list of strings
        class_names  : list of class name strings

    Example:
        plot_error_analysis()
        plot_error_analysis(y_test, model.predict(X_test), X_test)
    """
    if not SK_OK: print("sklearn required."); return
    import pandas as pd

    if y_true is None:
        iris  = load_iris()
        X     = iris.data
        Xtr,Xte,ytr,yte = train_test_split(X, iris.target, test_size=0.3, random_state=0)
        model  = SVC().fit(Xtr, ytr)
        y_true = yte; y_pred = model.predict(Xte); X = Xte
        feature_names = iris.feature_names
        class_names   = list(iris.target_names)

    y_true = np.array(y_true); y_pred = np.array(y_pred)
    wrong  = y_true != y_pred
    classes = np.unique(y_true)
    if class_names is None: class_names = [f"Class {c}" for c in classes]

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    fig.suptitle(f"Error Analysis — {wrong.sum()} errors / {len(y_true)} samples "
                 f"({wrong.mean()*100:.1f}% error rate)",
                 fontsize=13, fontweight="bold")

    # Errors per class (true label)
    err_counts = [((y_true==c) & wrong).sum() for c in classes]
    tot_counts = [(y_true==c).sum()           for c in classes]
    err_rates  = [e/t*100 if t>0 else 0 for e,t in zip(err_counts, tot_counts)]

    axes[0].barh(class_names, err_rates, color=CR, alpha=0.8, edgecolor="k")
    axes[0].axvline(wrong.mean()*100, color=CB, ls="--", lw=2,
                    label=f"Overall {wrong.mean()*100:.1f}%")
    axes[0].set_xlabel("Error Rate (%)"); axes[0].set_title("Error Rate per Class")
    axes[0].legend()
    for i, (rate, cnt) in enumerate(zip(err_rates, err_counts)):
        axes[0].text(rate+0.3, i, f"{cnt} wrong", va="center", fontsize=9)

    # Confusion: what were errors predicted as?
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Reds",
                xticklabels=class_names, yticklabels=class_names,
                ax=axes[1], cbar=False)
    axes[1].set_xlabel("Predicted"); axes[1].set_ylabel("True")
    axes[1].set_title("Confusion Matrix")

    # Feature distribution: correct vs wrong
    if X is not None and X.shape[1] >= 2:
        fname = feature_names[0] if feature_names else "Feature 1"
        sns.kdeplot(X[~wrong, 0], ax=axes[2], label="Correct ✅",
                    color=CG, fill=True, alpha=0.5)
        sns.kdeplot(X[wrong,  0], ax=axes[2], label="Wrong ❌",
                    color=CR, fill=True, alpha=0.5)
        axes[2].set_xlabel(fname)
        axes[2].set_title(f"Feature Distribution\nCorrect vs Wrong predictions")
        axes[2].legend()

    _show(fig)


# ════════════════════════════════════════════════════════════════════════════
# [S35]  TIME SERIES
# ════════════════════════════════════════════════════════════════════════════

def plot_time_series(data=None, timestamps=None, title="Time Series",
                     window=20, xlabel="Time", ylabel="Value"):
    """
    Time series plot with rolling mean + rolling std bands.

    Args:
        data       : 1-D array of values
        timestamps : 1-D array of x-axis values (dates or integers)
        window     : rolling window size for mean/std
        title      : plot title

    Example:
        plot_time_series()
        plot_time_series(stock_prices, dates, title="Stock Price", window=30)
        plot_time_series(loss_history, title="Training Loss over Time")
    """
    if data is None:
        np.random.seed(0)
        n          = 200
        timestamps = np.arange(n)
        trend      = 0.05 * timestamps
        season     = 5 * np.sin(2*np.pi*timestamps/50)
        noise      = np.random.randn(n) * 2
        data       = trend + season + noise

    data = np.array(data)
    t    = np.array(timestamps) if timestamps is not None else np.arange(len(data))

    import pandas as pd
    s       = pd.Series(data)
    roll_m  = s.rolling(window, center=True).mean()
    roll_s  = s.rolling(window, center=True).std()

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    fig.suptitle(title, fontsize=13, fontweight="bold")

    axes[0].plot(t, data, color=CGR, lw=1.0, alpha=0.6, label="Raw")
    axes[0].plot(t, roll_m, color=CB, lw=2.5, label=f"Rolling mean (w={window})")
    axes[0].fill_between(t,
                          roll_m - roll_s,
                          roll_m + roll_s,
                          alpha=0.25, color=CB, label="±1 std")
    axes[0].set_ylabel(ylabel); axes[0].legend()
    axes[0].set_title("Signal + Rolling Statistics")

    # Residual / detrended
    residual = data - roll_m.fillna(data.mean())
    axes[1].plot(t, residual, color=CR, lw=1.2, alpha=0.8)
    axes[1].axhline(0, color="black", lw=1, ls="--")
    axes[1].fill_between(t, residual, 0,
                          where=(residual > 0), alpha=0.3, color=CG, label="Above mean")
    axes[1].fill_between(t, residual, 0,
                          where=(residual < 0), alpha=0.3, color=CR, label="Below mean")
    axes[1].set_xlabel(xlabel); axes[1].set_ylabel("Residual")
    axes[1].set_title("Detrended Residual")
    axes[1].legend()

    _show(fig)


def plot_autocorrelation(data=None, max_lag=40, title="Autocorrelation Analysis"):
    """
    ACF (Autocorrelation Function) + PACF (Partial ACF) — diagnose time series structure.
    Used to choose AR/MA orders for ARIMA models.

    Args:
        data    : 1-D time series array
        max_lag : maximum lag to compute
        title   : plot title

    Example:
        plot_autocorrelation()
        plot_autocorrelation(stock_returns, max_lag=60)
    """
    if data is None:
        np.random.seed(0)
        # AR(2) process: x_t = 0.6*x_{t-1} + 0.3*x_{t-2} + noise
        n    = 300
        x    = np.zeros(n)
        eps  = np.random.randn(n) * 0.5
        for t in range(2, n):
            x[t] = 0.6*x[t-1] + 0.3*x[t-2] + eps[t]
        data = x

    data = np.array(data)
    n    = len(data)

    # ACF
    acf_vals = [1.0]
    for lag in range(1, max_lag+1):
        c = np.corrcoef(data[lag:], data[:-lag])[0,1]
        acf_vals.append(c)

    # PACF via Yule-Walker (simplified)
    pacf_vals = [1.0]
    for lag in range(1, max_lag+1):
        if lag == 1:
            pacf_vals.append(acf_vals[1])
        else:
            # Levinson-Durbin (simple approximation)
            A = np.array([[acf_vals[abs(i-j)] for j in range(lag)]
                           for i in range(lag)])
            b = np.array([acf_vals[k+1] for k in range(lag)])
            try:
                phi = np.linalg.solve(A, b)
                pacf_vals.append(phi[-1])
            except Exception:
                pacf_vals.append(0)

    conf = 1.96 / np.sqrt(n)  # 95% confidence interval
    lags = np.arange(max_lag+1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(title, fontsize=13, fontweight="bold")

    for ax, vals, name in zip(axes,
                               [acf_vals, pacf_vals],
                               ["ACF — Autocorrelation Function",
                                "PACF — Partial Autocorrelation"]):
        ax.bar(lags, vals, color=CB, alpha=0.7, width=0.4)
        ax.axhline(conf,  color=CR, ls="--", lw=1.5, label=f"95% CI (±{conf:.3f})")
        ax.axhline(-conf, color=CR, ls="--", lw=1.5)
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xlabel("Lag"); ax.set_ylabel("Correlation")
        ax.set_title(name); ax.legend(fontsize=9)

    _show(fig)


# ════════════════════════════════════════════════════════════════════════════
# [S36]  ENSEMBLE METHODS
# ════════════════════════════════════════════════════════════════════════════

def plot_ensemble_boundaries(X=None, y=None, n_trees=5):
    """
    Show individual decision tree boundaries vs ensemble (Random Forest) boundary.
    Visualises WHY ensembles generalise better than single trees.

    Args:
        X       : 2-D feature array
        y       : 1-D label array
        n_trees : number of individual trees to show

    Example:
        plot_ensemble_boundaries()
        plot_ensemble_boundaries(X_train, y_train, n_trees=4)
    """
    if not SK_OK: print("sklearn required."); return
    from sklearn.tree import DecisionTreeClassifier

    if X is None:
        X, y = make_moons(n_samples=200, noise=0.2, random_state=0)

    h  = 0.05
    xx, yy = np.meshgrid(
        np.arange(X[:,0].min()-0.3, X[:,0].max()+0.3, h),
        np.arange(X[:,1].min()-0.3, X[:,1].max()+0.3, h))

    # Train individual trees with different subsets
    ncols = n_trees + 1
    fig, axes = plt.subplots(1, ncols, figsize=(4.5*ncols, 5))
    fig.suptitle("Ensemble: Individual Trees vs Random Forest\n"
                 "(Each tree overfits; ensemble averages out the noise)",
                 fontsize=12, fontweight="bold")

    for i in range(n_trees):
        np.random.seed(i)
        idx  = np.random.choice(len(X), len(X), replace=True)
        dt   = DecisionTreeClassifier(max_depth=None, random_state=i)
        dt.fit(X[idx], y[idx])
        Z = dt.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
        axes[i].contourf(xx, yy, Z, alpha=0.35, cmap="RdBu")
        axes[i].scatter(X[:,0], X[:,1], c=y, cmap="RdBu",
                         edgecolors="k", s=25, alpha=0.7)
        axes[i].set_title(f"Tree {i+1}\n(bootstrapped subset)", fontsize=10)

    # Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=0)
    rf.fit(X, y)
    Z = rf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:,1].reshape(xx.shape)
    axes[-1].contourf(xx, yy, Z, levels=50, cmap="RdBu", alpha=0.75)
    axes[-1].contour(xx, yy, Z, levels=[0.5], colors="black", linewidths=2)
    axes[-1].scatter(X[:,0], X[:,1], c=y, cmap="RdBu",
                      edgecolors="k", s=25, alpha=0.7)
    axes[-1].set_title("Random Forest (100 trees)\n✅ Smooth, generalised", fontsize=10)

    _show(fig)


_ALL_STATIC_DEMOS = [
    # ── Basic ──────────────────────────────────────────────────────────────
    (plot_line,                       "Basic: Line"),
    (plot_scatter,                    "Basic: Scatter"),
    (plot_histogram,                  "Basic: Histogram"),
    (plot_bar,                        "Basic: Bar"),
    (plot_boxplot,                    "Basic: Boxplot"),
    (plot_heatmap,                    "Basic: Heatmap"),
    (plot_pairplot,                   "Basic: Pairplot"),
    # ── Linear Regression ─────────────────────────────────────────────────
    (plot_linear_regression,          "LinReg: Fit + Residuals"),
    (plot_multi_feature_regression,   "LinReg: Multi-Feature"),
    (plot_polynomial_regression,      "LinReg: Polynomial Degree Sweep"),
    (plot_predicted_vs_actual,        "LinReg: Predicted vs Actual"),
    (plot_qq_residuals,               "LinReg: Q-Q Residuals"),
    # ── Cost Function ─────────────────────────────────────────────────────
    (plot_cost_vs_w,                  "Cost: J vs w"),
    (plot_cost_vs_wb,                 "Cost: J vs w and b"),
    (plot_cost_3d,                    "Cost: 3-D Bowl 🥣"),
    # ── Gradient Descent & Optimizers ─────────────────────────────────────
    (plot_gradient_descent_path,      "GD: Path on Contour"),
    (plot_learning_rate_comparison,   "GD: Learning Rate Compare"),
    (plot_optimizer_comparison,       "GD: SGD vs Momentum vs Adam"),
    (plot_minibatch_comparison,       "GD: Batch vs Mini-batch vs SGD"),
    (plot_lr_finder,                  "GD: Learning Rate Finder"),
    (plot_lr_schedule,                "GD: LR Schedules (cosine/step/warmup)"),
    (plot_gradient_checking,          "GD: Gradient Checking"),
    # ── Feature Scaling ───────────────────────────────────────────────────
    (plot_feature_scaling_effect,     "Scaling: Before vs After"),
    (plot_normalization_comparison,   "Scaling: Raw/Z/MinMax"),
    # ── Logistic Regression ───────────────────────────────────────────────
    (plot_sigmoid,                    "LogReg: Sigmoid"),
    (plot_softmax,                    "LogReg: Softmax"),
    (plot_log_loss,                   "LogReg: Log Loss"),
    (plot_decision_boundary,          "LogReg: Decision Boundary"),
    # ── SVM ───────────────────────────────────────────────────────────────
    (plot_svm_margin,                 "SVM: Margin + Support Vectors"),
    (plot_kernel_comparison,          "SVM: Kernel Comparison"),
    # ── Neural Network ────────────────────────────────────────────────────
    (plot_activation_functions,       "NN: Activation Functions"),
    (plot_activation_derivatives,     "NN: Derivatives"),
    (plot_nn_architecture,            "NN: Architecture Diagram"),
    (plot_layer_activations,          "NN: Layer Activations"),
    (plot_training_history,           "NN: Training History"),
    (plot_dropout_effect,             "NN: Dropout Effect"),
    (plot_batch_normalization,        "NN: Batch Normalisation"),
    (plot_vanishing_gradient,         "NN: Vanishing Gradient"),
    (plot_weight_distributions,       "NN: Weight Distributions"),
    (plot_weight_initialization,      "NN: Weight Initialization"),
    (plot_loss_landscape,             "NN: Loss Landscape"),
    (plot_attention_heatmap,          "NN: Attention Heatmap"),
    # ── Regularisation ────────────────────────────────────────────────────
    (plot_overfit_underfit,           "Reg: Overfit / Underfit"),
    (plot_regularization_lambda,      "Reg: Lambda Sweep"),
    # ── Bias-Variance ─────────────────────────────────────────────────────
    (plot_bias_variance_tradeoff,     "BiasVar: Tradeoff Curve"),
    (plot_learning_curves,            "BiasVar: Learning Curves"),
    # ── Decision Boundaries & Trees ───────────────────────────────────────
    (plot_nonlinear_boundary,         "DB: Non-linear"),
    (plot_multiclass_boundary,        "DB: Multiclass"),
    (plot_decision_tree,              "DB: Decision Tree Structure"),
    (plot_ensemble_boundaries,        "DB: Ensemble vs Single Tree"),
    # ── Hyperparameter Tuning ─────────────────────────────────────────────
    (plot_validation_curve,           "HPT: Validation Curve"),
    (plot_hyperparameter_heatmap,     "HPT: Grid Search Heatmap"),
    # ── Clustering ────────────────────────────────────────────────────────
    (plot_kmeans_steps,               "Cluster: K-Means Steps"),
    (plot_elbow_method,               "Cluster: Elbow"),
    (plot_silhouette_score,           "Cluster: Silhouette"),
    (plot_dbscan,                     "Cluster: DBSCAN"),
    (plot_dendrogram,                 "Cluster: Dendrogram"),
    (plot_gmm,                        "Cluster: GMM"),
    # ── Anomaly Detection ─────────────────────────────────────────────────
    (plot_anomaly_detection,          "Anomaly: Gaussian"),
    # ── PCA & Embeddings ──────────────────────────────────────────────────
    (plot_pca_scree,                  "Embed: PCA Scree"),
    (plot_pca_2d,                     "Embed: PCA 2-D Projection"),
    (plot_tsne,                       "Embed: t-SNE"),
    (plot_umap,                       "Embed: UMAP"),
    # ── Distribution & Shift ──────────────────────────────────────────────
    (plot_train_test_distribution,    "Shift: Train vs Test Distribution"),
    # ── Evaluation ────────────────────────────────────────────────────────
    (plot_confusion_matrix,           "Eval: Confusion Matrix"),
    (plot_roc_curve,                  "Eval: ROC Curve"),
    (plot_multiclass_roc,             "Eval: Multiclass ROC"),
    (plot_precision_recall,           "Eval: Precision-Recall"),
    (plot_f1_vs_threshold,            "Eval: F1 vs Threshold"),
    (plot_calibration_curve,          "Eval: Calibration Curve"),
    (plot_cv_folds,                   "Eval: CV Fold Splits"),
    (plot_feature_importance,         "Eval: Feature Importance"),
    (plot_partial_dependence,         "Eval: Partial Dependence"),
    (plot_error_analysis,             "Eval: Error Analysis"),
    # ── Collaborative Filtering ───────────────────────────────────────────
    (plot_collaborative_filtering,    "RecSys: Collaborative Filtering"),
    # ── Time Series ───────────────────────────────────────────────────────
    (plot_time_series,                "TS: Time Series + Rolling Stats"),
    (plot_autocorrelation,            "TS: ACF + PACF"),
    # ── EDA ───────────────────────────────────────────────────────────────
    (plot_missing_values,             "EDA: Missing Values"),
    (plot_class_imbalance,            "EDA: Class Imbalance"),
    (plot_feature_target_correlation, "EDA: Feature-Target Correlation"),
    (plot_outlier_detection,          "EDA: Outlier Detection"),
]

_ALL_INTERACTIVE_DEMOS = [
    (iplot_scatter,            "iPlot: Scatter"),
    (iplot_line,               "iPlot: Line"),
    (iplot_cost_3d,            "iPlot: 3-D Cost"),
    (iplot_decision_boundary,  "iPlot: Decision Boundary"),
    (iplot_training_history,   "iPlot: Training History"),
    (iplot_confusion_matrix,   "iPlot: Confusion Matrix"),
    (iplot_roc_curve,          "iPlot: ROC"),
    (iplot_pca_3d,             "iPlot: 3-D PCA"),
    (iplot_kmeans,             "iPlot: K-Means"),
]


def list_all_functions():
    """
    Print every available plot function with a short description.

    Example:
        list_all_functions()
    """
    print("\n" + "="*70)
    print("  📊  ML VISUALIZER — Available Functions")
    print("="*70)
    print("\n  🖼️  STATIC (matplotlib / seaborn)\n")
    for fn, lbl in _ALL_STATIC_DEMOS:
        print(f"    {fn.__name__:<42}  {lbl}")
    print("\n  🌐  INTERACTIVE (plotly.express)  — opens in browser\n")
    for fn, lbl in _ALL_INTERACTIVE_DEMOS:
        print(f"    {fn.__name__:<42}  {lbl}")
    print("\n  🛠️  HELPERS (math / data utils)\n")
    for name in ["sigmoid","relu","leaky_relu","elu","tanh_fn","swish","gelu",
                 "compute_cost","gradient_descent"]:
        print(f"    {name}")
    print("\n  🚀  run_all_demos(interactive=False) — see ALL plots in sequence")
    print("="*70 + "\n")


def run_all_demos(interactive=False, skip=None):
    """
    Run EVERY demo plot in sequence — a full gallery tour.

    Args:
        interactive : also run plotly charts (opens browser tabs)
        skip        : list of function names to skip
                      e.g. skip=['plot_pairplot','plot_training_history']

    Example:
        run_all_demos()                      ← all static plots
        run_all_demos(interactive=True)      ← static + plotly
        run_all_demos(skip=['plot_pairplot'])
    """
    skip = set(skip or [])
    total = len(_ALL_STATIC_DEMOS) + (len(_ALL_INTERACTIVE_DEMOS) if interactive else 0)
    print(f"\n🚀  Running {total} demos  (close each window to advance)\n")

    for i, (fn, lbl) in enumerate(_ALL_STATIC_DEMOS, 1):
        if fn.__name__ in skip:
            print(f"  ⏭  Skipping [{i}] {lbl}")
            continue
        print(f"  ▶  [{i}/{total}]  {lbl}")
        try:
            fn()
        except Exception as e:
            print(f"      ⚠  {fn.__name__} failed: {e}")

    if interactive:
        for i, (fn, lbl) in enumerate(_ALL_INTERACTIVE_DEMOS,
                                       len(_ALL_STATIC_DEMOS)+1):
            if fn.__name__ in skip:
                print(f"  ⏭  Skipping [{i}] {lbl}")
                continue
            print(f"  ▶  [{i}/{total}]  {lbl}")
            try:
                fn()
            except Exception as e:
                print(f"      ⚠  {fn.__name__} failed: {e}")

    print("\n✅  All demos complete!\n")