"""Cluster and model plants using graph-derived features exported from Neo4j."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    mean_squared_error,
    r2_score,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FEATURES = ROOT / "data" / "plant_graph_features.csv"
DEFAULT_CHARTS = ROOT / "outputs" / "charts"

CLUSTER_FEATURES = [
    "ailment_count",
    "category_count",
    "shop_count",
    "location_count",
    "degree_centrality",
    "betweenness",
    "community_id",
    "toxicity_numeric",
    "famine_food",
    "safe_multi_use",
]

ID_COLS = ["plant_id", "plant_name", "irish_name", "toxicity_class"]


def prepare_ml_frame(features: pd.DataFrame) -> pd.DataFrame:
    ml_df = features.copy()
    tox_map = {"Low": 0, "Medium": 1, "High": 2}
    ml_df["toxicity_numeric"] = ml_df["toxicity_class"].map(tox_map)
    ml_df["famine_food"] = ml_df["famine_food"].astype(int)
    ml_df["safe_multi_use"] = ml_df["safe_multi_use"].astype(int)
    return pd.get_dummies(ml_df, columns=["family", "access_pattern"], drop_first=True)


def run_kmeans(ml_df: pd.DataFrame, chart_dir: Path) -> pd.DataFrame:
    X = ml_df[CLUSTER_FEATURES].fillna(0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = KMeans(n_clusters=4, random_state=42, n_init=10)
    ml_df = ml_df.copy()
    ml_df["k_cluster"] = model.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, ml_df["k_cluster"])
    print(f"K-means silhouette: {score:.3f}")
    print(ml_df.groupby("k_cluster")[CLUSTER_FEATURES].mean().round(2))

    chart_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 6))
    plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=ml_df["k_cluster"])
    plt.xlabel(CLUSTER_FEATURES[0])
    plt.ylabel(CLUSTER_FEATURES[1])
    plt.title("K-means clusters of medicinal plants")
    plt.tight_layout()
    plt.savefig(chart_dir / "kmeans_clusters.png", dpi=150)
    plt.close()
    return ml_df


def run_lasso(ml_df: pd.DataFrame, chart_dir: Path) -> None:
    target = "ailment_count"
    predictors = [col for col in ml_df.columns if col not in ID_COLS + [target, "k_cluster"]]
    X = ml_df[predictors].fillna(0)
    y = ml_df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    model = Lasso(alpha=0.1, random_state=42)
    model.fit(X_train_scaled, y_train)
    pred = model.predict(X_test_scaled)
    print(f"Lasso R2: {r2_score(y_test, pred):.3f}")
    print(f"Lasso RMSE: {np.sqrt(mean_squared_error(y_test, pred)):.3f}")

    coef_df = (
        pd.DataFrame({"feature": X.columns, "coefficient": model.coef_})
        .sort_values("coefficient", key=abs, ascending=False)
        .head(15)
    )
    print(coef_df)
    plt.figure(figsize=(10, 6))
    plt.bar(coef_df["feature"], coef_df["coefficient"])
    plt.xticks(rotation=75, ha="right")
    plt.title("Top Lasso coefficients for plant versatility")
    plt.tight_layout()
    plt.savefig(chart_dir / "lasso_coefficients.png", dpi=150)
    plt.close()


def run_logistic(ml_df: pd.DataFrame) -> None:
    target = "safe_multi_use"
    predictors = [col for col in ml_df.columns if col not in ID_COLS + [target, "k_cluster"]]
    X = ml_df[predictors].fillna(0)
    y = ml_df[target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    model = LogisticRegression(max_iter=1000)
    model.fit(scaler.fit_transform(X_train), y_train)
    pred = model.predict(scaler.transform(X_test))
    print(f"Logistic accuracy: {accuracy_score(y_test, pred):.3f}")
    print(classification_report(y_test, pred))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--features",
        type=Path,
        default=DEFAULT_FEATURES,
        help="CSV exported from neo4j/5_Export_And_ML_Bridge/24_plant_level_feature_table.txt",
    )
    parser.add_argument("--charts", type=Path, default=DEFAULT_CHARTS)
    args = parser.parse_args()

    if not args.features.exists():
        raise SystemExit(
            f"Missing {args.features}. Export the Neo4j plant feature table first "
            "(see neo4j/5_Export_And_ML_Bridge/24_plant_level_feature_table.txt)."
        )

    features = pd.read_csv(args.features)
    print(f"loaded {features.shape} from {args.features}")
    ml_df = prepare_ml_frame(features)
    ml_df = run_kmeans(ml_df, args.charts)
    run_lasso(ml_df, args.charts)
    run_logistic(ml_df)


if __name__ == "__main__":
    main()
