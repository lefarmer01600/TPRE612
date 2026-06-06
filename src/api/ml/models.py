import os
import glob
import logging
import numpy as np
import pandas as pd
import joblib

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
_HERE         = os.path.dirname(__file__)
_PROJECT_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))

DATA_FOLDER   = os.getenv(
    "DATA_FOLDER",
    os.path.join(_PROJECT_ROOT, "data", "output"),
)
MODEL_FOLDER  = os.getenv(
    "MODEL_FOLDER",
    os.path.join(_PROJECT_ROOT, "src", "training", "model"),
)

CLASSIFIER_PATH = os.path.join(MODEL_FOLDER, "classifier.joblib")
CLUSTERER_PATH  = os.path.join(MODEL_FOLDER, "clusterer.joblib")
META_PATH       = os.path.join(MODEL_FOLDER, "meta.joblib")

RANDOM_STATE = 42
N_CLUSTERS   = 3

# ── In-memory state ───────────────────────────────────────────────────────────
_classifier:        RandomForestClassifier | None = None
_clusterer:         KMeans | None                 = None
_scaler_clf:        StandardScaler | None         = None
_scaler_clu:        StandardScaler | None         = None
_le_source:         LabelEncoder | None           = None
_le_route:          LabelEncoder | None           = None
_le_origin:         LabelEncoder | None           = None
_le_dest:           LabelEncoder | None           = None
_le_target:         LabelEncoder | None           = None
_cluster_profiles:  dict                          = {}


# ── Data loading ──────────────────────────────────────────────────────────────
def _load_raw_data() -> pd.DataFrame:
    csv_files = glob.glob(os.path.join(DATA_FOLDER, "*.csv"))
    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found in {DATA_FOLDER!r}. "
            "Check the DATA_FOLDER env variable."
        )
    df = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)
    logger.info("Loaded %d rows from %d file(s) in %s",
                len(df), len(csv_files), DATA_FOLDER)
    return df


# ── Encoding helpers ──────────────────────────────────────────────────────────
def _safe_transform(le: LabelEncoder, series: pd.Series) -> np.ndarray:
    """LabelEncoder.transform that maps unseen labels to the first known class."""
    known = set(le.classes_)
    values = [v if v in known else le.classes_[0] for v in series.astype(str)]
    return np.asarray(le.transform(values))


def _encode_df(df: pd.DataFrame) -> np.ndarray:
    """Encode a dataframe into the feature matrix used by both models."""
    assert _le_source is not None
    assert _le_route is not None
    assert _le_origin is not None
    assert _le_dest is not None

    cols = [
        _safe_transform(_le_source, df["data_source"]),
        _safe_transform(_le_route,  df["route_id"]),
        _safe_transform(_le_origin, df["id_origin_city"]),
        _safe_transform(_le_dest,   df["id_destination_city"]),
        np.asarray(df["weekly_train"].to_numpy()),
    ]
    return np.column_stack(cols)


def _fit_encoders(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Fit all LabelEncoders on the full dataset; return (X_raw, y)."""
    global _le_source, _le_route, _le_origin, _le_dest, _le_target

    _le_source = LabelEncoder().fit(df["data_source"].astype(str))
    _le_route  = LabelEncoder().fit(df["route_id"].astype(str))
    _le_origin = LabelEncoder().fit(df["id_origin_city"].astype(str))
    _le_dest   = LabelEncoder().fit(df["id_destination_city"].astype(str))
    _le_target = LabelEncoder().fit(df["desserte_type"])

    y = np.asarray(_le_target.transform(df["desserte_type"]))
    return _encode_df(df), y


# ── Cluster profiling ─────────────────────────────────────────────────────────
def _build_cluster_profiles(df: pd.DataFrame, labels: np.ndarray) -> dict:
    df_p = df.copy()
    df_p["cluster"] = labels
    profiles: dict = {}

    for cluster_id, group in df_p.groupby("cluster"):
        cid = int(cluster_id)  # type: ignore[arg-type]
        if cid == -1:
            continue
        profiles[cid] = {
            "size":                int(len(group)),
            "weekly_train_mean":   round(float(group["weekly_train"].mean()),   2),
            "weekly_train_median": round(float(group["weekly_train"].median()), 2),
            "weekly_train_std":    round(float(group["weekly_train"].std()),    2),
            "desserte_type_dist": (
                group["desserte_type"]
                .value_counts(normalize=True)
                .round(3)
                .to_dict()
            ),
        }
    return profiles


# ── Training ──────────────────────────────────────────────────────────────────
def train_and_save() -> None:
    global _classifier, _clusterer, _scaler_clf, _scaler_clu, _cluster_profiles

    os.makedirs(MODEL_FOLDER, exist_ok=True)

    df       = _load_raw_data()
    X_raw, y = _fit_encoders(df)

    # Random Forest
    X_train, _, y_train, _ = train_test_split(
        X_raw, y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    _scaler_clf = StandardScaler().fit(X_train)
    _classifier = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )
    _classifier.fit(_scaler_clf.transform(X_train), y_train)
    logger.info("Classifier trained.")

    # KMeans
    _scaler_clu    = StandardScaler().fit(X_raw)
    _clusterer     = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE, n_init=15)
    cluster_labels = _clusterer.fit_predict(_scaler_clu.transform(X_raw))
    _cluster_profiles = _build_cluster_profiles(df, cluster_labels)
    logger.info("Clusterer trained — %d clusters.", N_CLUSTERS)

    # Persist
    joblib.dump(_classifier, CLASSIFIER_PATH)
    joblib.dump(_clusterer,  CLUSTERER_PATH)
    joblib.dump(
        {
            "scaler_clf":       _scaler_clf,
            "scaler_clu":       _scaler_clu,
            "le_source":        _le_source,
            "le_route":         _le_route,
            "le_origin":        _le_origin,
            "le_dest":          _le_dest,
            "le_target":        _le_target,
            "cluster_profiles": _cluster_profiles,
        },
        META_PATH,
    )
    logger.info("Models saved to %s", MODEL_FOLDER)


# ── Loading ───────────────────────────────────────────────────────────────────
def load_models() -> None:
    """Load persisted models at startup; train from scratch if any file is missing."""
    global _classifier, _clusterer
    global _scaler_clf, _scaler_clu
    global _le_source, _le_route, _le_origin, _le_dest, _le_target
    global _cluster_profiles

    all_exist = all(
        os.path.exists(p) for p in [CLASSIFIER_PATH, CLUSTERER_PATH, META_PATH]
    )

    if all_exist:
        logger.info("Pre-trained models found — loading from %s", MODEL_FOLDER)
        _classifier = joblib.load(CLASSIFIER_PATH)
        _clusterer  = joblib.load(CLUSTERER_PATH)
        meta = joblib.load(META_PATH)

        _scaler_clf       = meta["scaler_clf"]
        _scaler_clu       = meta["scaler_clu"]
        _le_source        = meta["le_source"]
        _le_route         = meta["le_route"]
        _le_origin        = meta["le_origin"]
        _le_dest          = meta["le_dest"]
        _le_target        = meta["le_target"]
        _cluster_profiles = meta["cluster_profiles"]
        logger.info("Models loaded successfully.")
    else:
        logger.info("No saved models found — training from scratch…")
        train_and_save()


# ── Inference ─────────────────────────────────────────────────────────────────
def predict_desserte(payload: dict) -> dict:
    if _classifier is None or _scaler_clf is None or _le_target is None:
        raise RuntimeError("Models not initialised — call load_models() first.")

    row      = pd.DataFrame([payload])
    X_scaled = _scaler_clf.transform(_encode_df(row))

    label_idx = int(_classifier.predict(X_scaled)[0])
    proba     = _classifier.predict_proba(X_scaled)[0]

    return {
        "desserte_type": _le_target.inverse_transform([label_idx])[0],
        "probabilities": {
            cls: round(float(p), 4)
            for cls, p in zip(_le_target.classes_, proba)
        },
        "model_used": "RandomForest",
    }


def predict_cluster(payload: dict) -> dict:
    if _clusterer is None or _scaler_clu is None:
        raise RuntimeError("Models not initialised — call load_models() first.")

    row      = pd.DataFrame([payload])
    X_scaled = _scaler_clu.transform(_encode_df(row))

    cluster_id = int(_clusterer.predict(X_scaled)[0])

    return {
        "cluster_id":      cluster_id,
        "cluster_profile": _cluster_profiles.get(cluster_id),
        "model_used":      "KMeans",
    }
