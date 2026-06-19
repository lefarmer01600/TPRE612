import os
import sys
import numpy as np
import pandas as pd
from scipy import stats
from typing import List, Tuple, Optional
from dataclasses import dataclass
from collections import deque

# Add project root to path for imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from src.api.ml import models as ml_models


@dataclass
class DriftDetectionResult:
    """
    Container for drift detection results.

    Attributes:
        is_drift_detected: Boolean indicating if drift was detected
        drift_score: Numeric score indicating drift severity
        p_value: Statistical p-value (if applicable)
        detection_method: Name of the detection method used
    """
    is_drift_detected: bool
    drift_score: float
    p_value: Optional[float] = None
    detection_method: str = ""


class ClassificationAccuracyDriftDetector:
    """
    Detects concept drift by monitoring classification accuracy.

    Compares recent accuracy against a reference window using statistical tests.
    Significant degradation in prediction accuracy triggers a drift alert.
    """

    def __init__(
        self,
        reference_window_size: int = 500,
        detection_window_size: int = 50,
        significance_level: float = 0.05
    ):
        """
        Initialize the drift detector.

        Args:
            reference_window_size: Number of samples in reference window
            detection_window_size: Number of recent samples to compare
            significance_level: P-value threshold for drift detection
        """
        self.reference_window_size = reference_window_size
        self.detection_window_size = detection_window_size
        self.significance_level = significance_level

        # Use deque for efficient sliding window operations
        # Store 0/1 values: 1 for correct prediction, 0 for incorrect
        self.reference_errors: deque = deque(maxlen=reference_window_size)
        self.detection_errors: deque = deque(maxlen=detection_window_size)

    def update(self, prediction: str, actual: str) -> None:
        """
        Update detector with a new prediction-actual pair.

        Args:
            prediction: Model's predicted class
            actual: Ground truth class
        """
        # Calculate correctness: 1 if correct, 0 if incorrect
        error = 1.0 if prediction == actual else 0.0

        # Build a stable baseline first, then monitor recent predictions
        if len(self.reference_errors) < self.reference_window_size:
            self.reference_errors.append(error)
        else:
            self.detection_errors.append(error)

    def detect(self) -> DriftDetectionResult:
        """
        Check if concept drift has occurred.

        Uses a two-sample t-test to compare accuracy between reference
        and detection windows.

        Returns:
            DriftDetectionResult with detection outcome
        """
        # Need sufficient samples in both windows
        if (len(self.reference_errors) < self.reference_window_size // 2 or
            len(self.detection_errors) < self.detection_window_size // 2):
            return DriftDetectionResult(
                is_drift_detected=False,
                drift_score=0.0,
                p_value=None,
                detection_method="accuracy_ttest"
            )

        # Convert to arrays for statistical test
        reference_array = np.array(self.reference_errors)
        detection_array = np.array(self.detection_errors)

        # Perform Welch's t-test
        # Alternative='less' tests if detection accuracy is lower (drift)
        t_statistic, p_value = stats.ttest_ind(
            detection_array,
            reference_array,
            equal_var=False,
            alternative='less'
        )

        # Calculate effect size (Cohen's d) for drift severity
        pooled_std = np.sqrt(
            (np.var(reference_array) + np.var(detection_array)) / 2
        )
        effect_size = (
            (np.mean(reference_array) - np.mean(detection_array)) / pooled_std
            if pooled_std > 0 else 0
        )

        return DriftDetectionResult(
            is_drift_detected=p_value < self.significance_level,
            drift_score=effect_size,
            p_value=p_value,
            detection_method="accuracy_ttest"
        )


# ── Test utilities ────────────────────────────────────────────────────────────
def _load_test_data() -> pd.DataFrame:
    """Load and prepare test data for drift detection."""
    data_folder = os.path.join(project_root, "data", "output")
    csv_files = [f for f in os.listdir(data_folder) if f.endswith(".csv")]
    
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {data_folder}")
    
    # Load first CSV file for testing
    df = pd.read_csv(os.path.join(data_folder, csv_files[0]))
    return df


def _create_synthetic_batch(df: pd.DataFrame, n_samples: int, seed: int = None) -> pd.DataFrame:
    """
    Create a synthetic batch by sampling from the dataset.
    
    Args:
        df: Source dataframe
        n_samples: Number of samples to generate
        seed: Random seed for reproducibility
        
    Returns:
        DataFrame with sampled rows
    """
    if seed is not None:
        np.random.seed(seed)
    
    required_cols = ["data_source", "route_id", "id_origin_city", 
                     "id_destination_city", "weekly_train", "desserte_type"]
    
    # Check if all required columns exist
    available_cols = [c for c in required_cols if c in df.columns]
    if not available_cols:
        raise ValueError(f"No required columns found in dataframe. Columns: {df.columns.tolist()}")
    
    # Sample with replacement
    return df[available_cols].sample(n=n_samples, replace=True, random_state=seed)


def _get_drift_batch(df: pd.DataFrame, n_samples: int, seed: int = None) -> pd.DataFrame:
    """
    Create a drift batch with extreme/biased distribution.
    Simulates concept drift by selecting extreme values.
    
    Args:
        df: Source dataframe
        n_samples: Number of samples to generate
        seed: Random seed for reproducibility
        
    Returns:
        DataFrame with biased distribution
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Filter to extreme weekly_train values to create drift
    q75 = df["weekly_train"].quantile(0.75)
    q25 = df["weekly_train"].quantile(0.25)
    
    # Mix of very high and very low values (different from reference)
    extreme_df = pd.concat([
        df[df["weekly_train"] > q75],  # High values
        df[df["weekly_train"] < q25],  # Low values
    ])
    
    if len(extreme_df) < n_samples:
        # Fall back to random sampling if not enough extreme values
        return df.sample(n=n_samples, replace=True, random_state=seed)
    
    return extreme_df.sample(n=n_samples, replace=False, random_state=seed)


# Example usage
if __name__ == "__main__":
    print("=" * 70)
    print("Testing Classification Accuracy Drift Detection")
    print("=" * 70)
    
    # Load models
    print("\n1. Loading ML models...")
    try:
        ml_models.load_models()
        print("   ✓ Models loaded successfully")
    except Exception as e:
        print(f"   ✗ Error loading models: {e}")
        sys.exit(1)
    
    # Load test data
    print("\n2. Loading test data...")
    try:
        df = _load_test_data()
        print(f"   ✓ Loaded {len(df)} samples")
        print(f"   Columns: {df.columns.tolist()[:6]}...")  # Show first 6 cols
    except Exception as e:
        print(f"   ✗ Error loading data: {e}")
        sys.exit(1)
    
    # Create detector instance
    print("\n3. Initializing drift detector...")
    detector = ClassificationAccuracyDriftDetector(
        reference_window_size=500,
        detection_window_size=50,
        significance_level=0.05
    )
    print("   ✓ Detector initialized")
    
    # Stable period: model predictions are accurate
    print("\n4. Testing stable period (accurate predictions)...")
    try:
        stable_batch = _create_synthetic_batch(df, 500, seed=42)
        
        for idx, row in stable_batch.iterrows():
            try:
                payload = {
                    "data_source": str(row["data_source"]),
                    "route_id": str(row["route_id"]),
                    "id_origin_city": str(row["id_origin_city"]),
                    "id_destination_city": str(row["id_destination_city"]),
                    "weekly_train": int(row["weekly_train"]),
                }
                result = ml_models.predict_desserte(payload)
                prediction = result["desserte_type"]
                actual = str(row["desserte_type"])
                
                detector.update(prediction, actual)
                
            except Exception as e:
                # Skip rows with encoding errors
                continue
        
        print(f"   ✓ Processed {len(detector.reference_errors)} samples in reference window")
        ref_accuracy = np.mean(list(detector.reference_errors))
        print(f"   Reference accuracy: {ref_accuracy:.2%}")
        
    except Exception as e:
        print(f"   ✗ Error in stable period: {e}")
    
    # Drift period: different distribution (intentional concept drift)
    print("\n5. Testing drift period (accuracy degradation)...")
    try:
        # Use extreme values batch to create distribution shift
        drift_batch = _get_drift_batch(df, 60, seed=123)
        
        for idx, row in drift_batch.iterrows():
            try:
                payload = {
                    "data_source": str(row["data_source"]),
                    "route_id": str(row["route_id"]),
                    "id_origin_city": str(row["id_origin_city"]),
                    "id_destination_city": str(row["id_destination_city"]),
                    "weekly_train": int(row["weekly_train"]),
                }
                result = ml_models.predict_desserte(payload)
                prediction = result["desserte_type"]
                actual = str(row["desserte_type"])
                
                # Simulate model degradation (randomly flip some predictions)
                # This represents concept drift
                if np.random.random() < 0.08:  # 10% error rate
                    prediction = "Drift_Simulated"
                
                detector.update(prediction, actual)
                
            except Exception as e:
                # Skip rows with encoding errors
                continue
        
        print(f"   ✓ Processed {len(detector.detection_errors)} samples in detection window")
        det_accuracy = np.mean(list(detector.detection_errors))
        print(f"   Detection accuracy: {det_accuracy:.2%}")
        
    except Exception as e:
        print(f"   ✗ Error in drift period: {e}")
        import traceback
        traceback.print_exc()
    
    # Check for drift
    print("\n6. Analyzing drift detection results...")
    result = detector.detect()
    
    print("\n" + "=" * 70)
    print("DRIFT DETECTION RESULTS")
    print("=" * 70)
    print(f"Drift Detected:     {result.is_drift_detected}")
    print(f"Drift Score:        {result.drift_score:.4f}")
    p_val_str = f"{result.p_value:.6f}" if result.p_value is not None else "N/A"
    print(f"P-value:            {p_val_str}")
    print(f"Detection Method:   {result.detection_method}")
    print("=" * 70)