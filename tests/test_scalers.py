import os
import pickle
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')

SCALER_X_PATH = os.path.join(DATA_DIR, 'scaler_X.pkl')
SCALER_Y_PATH = os.path.join(DATA_DIR, 'scaler_y.pkl')
SAMPLE_DATA_PATH = os.path.join(DATA_DIR, 'df_full_sem_outliers_lof_preprocess.csv')

SENSOR_COLS = ['S1_1','S1_2','S1_3','S1_4','S1_5','S1_6','S2_1','S2_2','S3_1','S3_2','S4_1','S4_2']
TARGET_COLS = ['Nivel_agua','Nivel_total']

def test_scalers_exist():
    assert os.path.exists(SCALER_X_PATH), f"Missing scaler file: {SCALER_X_PATH}"
    assert os.path.exists(SCALER_Y_PATH), f"Missing scaler file: {SCALER_Y_PATH}"


def test_scalers_can_transform():
    # Load scalers
    with open(SCALER_X_PATH, 'rb') as f:
        scaler_X = pickle.load(f)
    with open(SCALER_Y_PATH, 'rb') as f:
        scaler_Y = pickle.load(f)

    # Load sample data
    assert os.path.exists(SAMPLE_DATA_PATH), f"Sample dataset not found: {SAMPLE_DATA_PATH}"
    df = pd.read_csv(SAMPLE_DATA_PATH)

    # Basic column presence check
    for col in SENSOR_COLS + TARGET_COLS:
        assert col in df.columns, f"Expected column {col} not found in sample dataset"

    X = df[SENSOR_COLS].head(10)
    y = df[TARGET_COLS].head(10)

    X_scaled = scaler_X.transform(X)
    y_scaled = scaler_Y.transform(y)

    # Value ranges should be within [0,1] with MinMaxScaler (allow small floating tolerance)
    assert (X_scaled >= -1e-9).all() and (X_scaled <= 1 + 1e-9).all(), "Scaled X out of [0,1] range"
    assert (y_scaled >= -1e-9).all() and (y_scaled <= 1 + 1e-9).all(), "Scaled y out of [0,1] range"


def test_inverse_transform_round_trip():
    with open(SCALER_X_PATH, 'rb') as f:
        scaler_X = pickle.load(f)

    df = pd.read_csv(SAMPLE_DATA_PATH)
    X = df[SENSOR_COLS].head(5)
    X_scaled = scaler_X.transform(X)
    X_back = scaler_X.inverse_transform(X_scaled)

    # Ensure the round trip preserves original values (within small tolerance)
    import numpy as np
    assert np.allclose(X.values, X_back, atol=1e-6), "Inverse transform did not reproduce original X within tolerance"