import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge, LinearRegression
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def train_models(df: pd.DataFrame, target_col: str | None, task: str) -> dict:
    """
    Train and evaluate models based on detected task type.
    Returns a dict with scores, best_model, feature_importance, predictions.
    """
    if task == "clustering":
        return _cluster(df)

    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Keep only numeric features
    X = X.select_dtypes(include=[np.number])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    if task == "classification":
        return _classify(X_train_s, X_test_s, y_train, y_test, X.columns)
    else:
        return _regress(X_train_s, X_test_s, y_train, y_test, X.columns)


# ── Classification ─────────────────────────────────────────────────────────────
def _classify(X_train, X_test, y_train, y_test, feature_names):
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
    }

    scores = {}
    trained = {}
    for name, model in models.items():
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="f1_weighted")
        model.fit(X_train, y_train)
        test_score = model.score(X_test, y_test)
        scores[name] = {
            "CV F1 (mean)": round(cv_scores.mean(), 4),
            "CV F1 (std)": round(cv_scores.std(), 4),
            "Test Accuracy": round(test_score, 4),
        }
        trained[name] = model

    best_name = max(scores, key=lambda k: scores[k]["Test Accuracy"])
    best_model = trained[best_name]

    fi = None
    if hasattr(best_model, "feature_importances_"):
        fi = pd.DataFrame({
            "feature": feature_names,
            "importance": best_model.feature_importances_,
        }).sort_values("importance", ascending=False).round(4)

    return {
        "task": "classification",
        "scores": scores,
        "best_model": best_name,
        "feature_importance": fi,
    }


# ── Regression ─────────────────────────────────────────────────────────────────
def _regress(X_train, X_test, y_train, y_test, feature_names):
    from sklearn.metrics import mean_squared_error, r2_score

    models = {
        "Linear Regression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
    }

    scores = {}
    trained = {}
    for name, model in models.items():
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="r2")
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        scores[name] = {
            "CV R² (mean)": round(cv_scores.mean(), 4),
            "CV R² (std)": round(cv_scores.std(), 4),
            "Test R²": round(r2_score(y_test, preds), 4),
            "Test RMSE": round(np.sqrt(mean_squared_error(y_test, preds)), 4),
        }
        trained[name] = model

    best_name = max(scores, key=lambda k: scores[k]["Test R²"])
    best_model = trained[best_name]

    fi = None
    if hasattr(best_model, "feature_importances_"):
        fi = pd.DataFrame({
            "feature": feature_names,
            "importance": best_model.feature_importances_,
        }).sort_values("importance", ascending=False).round(4)

    return {
        "task": "regression",
        "scores": scores,
        "best_model": best_name,
        "feature_importance": fi,
    }


# ── Clustering ─────────────────────────────────────────────────────────────────
def _cluster(df: pd.DataFrame):
    X = df.select_dtypes(include=[np.number]).dropna()
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)

    best_k, best_score, best_labels = 2, -1, None
    scores = {}
    for k in range(2, min(9, len(X))):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_s)
        sil = silhouette_score(X_s, labels)
        scores[f"K={k}"] = {"Silhouette Score": round(sil, 4)}
        if sil > best_score:
            best_score, best_k, best_labels = sil, k, labels

    return {
        "task": "clustering",
        "scores": scores,
        "best_model": f"KMeans (k={best_k})",
        "silhouette": best_score,
        "feature_importance": None,
    }
