# utils.py
import numpy as np
import time
import pandas as pd
from scipy.stats import t as t_distv

##############################
# Data‑splitting helper      #
##############################

def train_test_split(X, y, test_size=0.2, random_state=None):
    """Custom implementation of train‑test split (shuffle + split)."""
    assert X.shape[0] == y.shape[0], "X and y must contain the same number of samples."
    n_samples = X.shape[0]
    if random_state is not None:
        np.random.seed(random_state)
    indices = np.random.permutation(n_samples)
    test_size = int(n_samples * test_size)
    test_idx  = indices[:test_size]
    train_idx = indices[test_size:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

##############################
# Linear‑regression solvers  #
##############################

def _add_bias(X: np.ndarray) -> np.ndarray:
    """Add a column of ones (bias term) to the left side of X."""
    return np.hstack([np.ones((X.shape[0], 1)), X])


def linear_regression_normal(X: np.ndarray, y: np.ndarray):
    """Solve linear regression using the Normal Equation.
    Returns (weights, time_elapsed)."""
    start = time.time()
    Xb    = _add_bias(X)
    XtX   = Xb.T @ Xb
    Xty   = Xb.T @ y

    try:
        # Try regular inverse
        w = np.linalg.inv(XtX) @ Xty
    except np.linalg.LinAlgError:
        # If singular, use pseudoinverse
        print("Warning: XtX is singular; using pseudoinverse instead.")
        w = np.linalg.pinv(Xb) @ y

    return w, time.time() - start



def linear_regression_qr(X: np.ndarray, y: np.ndarray):
    """Solve linear regression using QR decomposition.
    Returns (weights, time_elapsed)."""
    start = time.time()
    Xb    = _add_bias(X)
    Q, R  = np.linalg.qr(Xb)
    w     = np.linalg.solve(R, Q.T @ y)
    return w, time.time() - start


def train_interaction_model(X1: np.ndarray, X2: np.ndarray, y: np.ndarray):
    """
    Train linear regression model with interaction term using normal equation.
    """
    assert X1.shape == X2.shape == y.shape, "Inputs must have the same shape"

    # Create interaction term
    interaction = X1 * X2

    # Construct design matrix with intercept, X1, X2, and interaction
    X_design = np.column_stack([np.ones_like(X1), X1, X2, interaction])

    # Normal equation: w = (X^T X)^(-1) X^T y
    XtX = X_design.T @ X_design
    Xty = X_design.T @ y
    w = np.linalg.inv(XtX) @ Xty

    return w


##############################
# Prediction helper          #
##############################


def predict_linear_regression(weights: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Generate predictions given learned weights and feature matrix."""
    Xb = _add_bias(X)
    return Xb @ weights


##############################
# Metrics for inference      #
##############################


def rss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Residual Sum of Squares (RSS)."""
    return float(np.sum((y_true - y_pred) ** 2))


def tss(y_true: np.ndarray) -> float:
    """Total Sum of Squares (TSS)."""
    return float(np.sum((y_true - y_true.mean()) ** 2))


def rse(rss_value: float, n: int, p: int) -> float:
    """Residual Standard Error (RSE).
    n : number of observations
    p : number of predictors (not counting the intercept)"""
    return np.sqrt(rss_value / (n - p - 1))


def f_statistic(rss_value: float, tss_value: float, n: int, p: int) -> float:
    """F‑statistic to test the null hypothesis that all slope coefficients are 0.
    Formula:  ((TSS - RSS)/p) / (RSS / (n - p - 1))"""
    numerator   = (tss_value - rss_value) / p
    denominator = rss_value / (n - p - 1)
    return numerator / denominator


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient of determination (R²)."""
    return 1.0 - rss(y_true, y_pred) / tss(y_true)


def t_statistics_and_p_values(X, y, weights):
    """
    Computes t-statistics and p-values for each coefficient in a linear regression model.
    """
    n, p = X.shape
    X_bias = np.hstack([np.ones((n, 1)), X])  # Add intercept
    y_pred = X_bias @ weights
    residuals = y - y_pred

    RSS = np.sum(residuals ** 2)
    sigma_squared = RSS / (n - p - 1)

    XtX_inv = np.linalg.inv(X_bias.T @ X_bias)
    var_beta = sigma_squared * XtX_inv
    se = np.sqrt(np.diag(var_beta))  # Standard errors

    t_values = weights / se
    p_values = 2 * (1 - t_dist.cdf(np.abs(t_values), df=n - p - 1))

    return t_values, p_values, se


def regression_summary(X, y, weights, feature_names=None):
    """
    Returns a summary DataFrame with coefficients, standard errors,
    t-statistics, and p-values for a regression model.
    """
    t_vals, p_vals, se_vals = t_statistics_and_p_values(X, y, weights)
    
    if feature_names is None:
        feature_names = [f"x{i}" for i in range(X.shape[1])]
    
    feature_names = ["intercept"] + feature_names

    df_summary = pd.DataFrame({
        "Coefficient": weights,
        "Std. Error": se_vals,
        "t-Statistic": t_vals,
        "p-Value": p_vals
    }, index=feature_names)

    return df_summary


def calculate_vif_manual(X: np.ndarray, feature_names=None):
    """
    Compute Variance Inflation Factor (VIF) for each column of X.
    """
    n_samples, n_features = X.shape
    vif_values = []

    for j in range(n_features):
        # Dependent variable = current feature
        y_j = X[:, j]

        # Independent variables = all other features
        X_others = np.delete(X, j, axis=1)

        # Fit linear regression with our own normal-equation helper
        w_j, _ = linear_regression_normal(X_others, y_j)

        # Predict and compute R² manually
        y_pred_j = predict_linear_regression(w_j, X_others)
        rss_j = rss(y_j, y_pred_j)
        tss_j = tss(y_j)
        r2_j = 1.0 - rss_j / tss_j

        # Guard against division by zero / perfect collinearity
        vif_j = np.inf if np.isclose(1.0 - r2_j, 0.0) else 1.0 / (1.0 - r2_j)
        vif_values.append(vif_j)

    if feature_names is None:
        feature_names = [f"x{i}" for i in range(n_features)]

    return pd.DataFrame({"Feature": feature_names, "VIF": vif_values})


__all__ = [
    "confidence_intervals",
    "calculate_vif_manual",
    "train_interaction_model",
    "regression_summary",
    "t_statistics_and_p_values",
    "train_test_split",
    "linear_regression_normal",
    "linear_regression_qr",
    "predict_linear_regression",
    "rss",
    "tss",
    "rse",
    "f_statistic",
    "r2_score",
]  # explicit export list
