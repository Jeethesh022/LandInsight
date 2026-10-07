import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# FILES
# ============================================================

DATASET_PATH = "Land_Datasets_2.csv"
MODEL_PATH = "land_price_model_final.pkl"
METADATA_PATH = "land_metadata_final.pkl"


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("LANDINSIGHT - MODEL TRAINING")
print("=" * 70)

df = pd.read_csv(DATASET_PATH)

print("\nDataset loaded successfully")
print("Rows:", len(df))
print("Columns:", len(df.columns))


# ============================================================
# TARGET
# ============================================================

TARGET = "Market Price (Rs per sq.ft)"

if TARGET not in df.columns:
    raise ValueError(f"Target column not found: {TARGET}")


# ============================================================
# FEATURES
# ============================================================

FEATURE_COLUMNS = [
    "Location",
    "Area",
    "District",
    "Latitude",
    "Longitude",
    "Land Area (sq.ft)",
    "Land Type",
    "Road Width (ft)",
    "Road Type",
    "Corner Plot",
    "Facing",
    "Distance to Main Road (km)",
    "Distance to Highway (km)",
    "Distance to Bus Stand (km)",
    "Distance to Railway Station (km)",
    "Distance to School (km)",
    "Distance to Hospital (km)",
    "Distance to Market (km)",
    "Water Facility",
    "Electricity",
    "Sewerage Facility",
    "Guideline Value (Rs per sq.ft)",
    "Registration Year",
]


missing = [c for c in FEATURE_COLUMNS if c not in df.columns]

if missing:
    raise ValueError(
        "Missing feature columns:\n" +
        "\n".join(missing)
    )


# ============================================================
# CLEAN TARGET
# ============================================================

df[TARGET] = pd.to_numeric(
    df[TARGET],
    errors="coerce"
)

df = df.dropna(subset=[TARGET]).copy()

print("\nTarget:", TARGET)
print("Training rows:", len(df))

print("\nTarget statistics:")
print(df[TARGET].describe())


# ============================================================
# X / y
# ============================================================

X = df[FEATURE_COLUMNS].copy()
y = df[TARGET].copy()


# ============================================================
# FEATURE TYPES
# ============================================================

NUMERIC_FEATURES = [
    "Latitude",
    "Longitude",
    "Land Area (sq.ft)",
    "Road Width (ft)",
    "Distance to Main Road (km)",
    "Distance to Highway (km)",
    "Distance to Bus Stand (km)",
    "Distance to Railway Station (km)",
    "Distance to School (km)",
    "Distance to Hospital (km)",
    "Distance to Market (km)",
    "Guideline Value (Rs per sq.ft)",
    "Registration Year",
]


CATEGORICAL_FEATURES = [
    "Location",
    "Area",
    "District",
    "Land Type",
    "Road Type",
    "Corner Plot",
    "Facing",
    "Water Facility",
    "Electricity",
    "Sewerage Facility",
]


# ============================================================
# PREPROCESSING
# ============================================================

numeric_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        )
    ]
)


categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            )
        )
    ]
)


preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            numeric_pipeline,
            NUMERIC_FEATURES
        ),
        (
            "cat",
            categorical_pipeline,
            CATEGORICAL_FEATURES
        )
    ]
)


# ============================================================
# MODEL
# ============================================================

model = RandomForestRegressor(
    n_estimators=500,
    max_depth=15,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)


pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "model",
            model
        )
    ]
)


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


print("\nTrain rows:", len(X_train))
print("Test rows:", len(X_test))


# ============================================================
# TRAIN
# ============================================================

print("\nTraining model...")

pipeline.fit(
    X_train,
    y_train
)

print("Training completed.")


# ============================================================
# EVALUATION
# ============================================================

predictions = pipeline.predict(X_test)

mae = mean_absolute_error(
    y_test,
    predictions
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        predictions
    )
)

r2 = r2_score(
    y_test,
    predictions
)


non_zero = y_test != 0

mape = (
    np.mean(
        np.abs(
            (
                y_test[non_zero]
                - predictions[non_zero]
            )
            / y_test[non_zero]
        )
    )
    * 100
)


print("\n" + "=" * 70)
print("MODEL PERFORMANCE")
print("=" * 70)

print(f"MAE  : ₹{mae:,.2f} / sq.ft")
print(f"RMSE : ₹{rmse:,.2f} / sq.ft")
print(f"R²   : {r2:.4f}")
print(f"MAPE : {mape:.2f}%")


# ============================================================
# SAVE MODEL
# ============================================================

with open(
    MODEL_PATH,
    "wb"
) as f:
    pickle.dump(
        pipeline,
        f
    )


# ============================================================
# SAVE METADATA
# ============================================================

metadata = {
    "target_column": TARGET,

    "prediction_type": "direct_market_price_per_sqft",

    "feature_columns": FEATURE_COLUMNS,

    "numeric_features": NUMERIC_FEATURES,

    "categorical_features": CATEGORICAL_FEATURES,

    "model_type": "RandomForestRegressor",

    "n_estimators": 500,

    "max_depth": 15,

    "min_samples_leaf": 2,

    "random_state": 42,

    "training_rows": len(df),

    "test_rows": len(X_test),

    "mae": float(mae),

    "rmse": float(rmse),

    "r2": float(r2),

    "mape": float(mape),
}


with open(
    METADATA_PATH,
    "wb"
) as f:
    pickle.dump(
        metadata,
        f
    )


# ============================================================
# ADYAR SANITY TEST
# ============================================================

print("\n" + "=" * 70)
print("ADYAR SANITY TEST")
print("=" * 70)

adyar = df[
    (df["Location"] == "Chennai") &
    (df["Area"] == "Adyar")
].copy()


if len(adyar) > 0:

    adyar_predictions = pipeline.predict(
        adyar[FEATURE_COLUMNS]
    )

    result = adyar[
        [
            "Land Area (sq.ft)",
            "Guideline Value (Rs per sq.ft)",
            TARGET
        ]
    ].copy()

    result["Predicted Price (Rs per sq.ft)"] = (
        adyar_predictions
    )

    result["Prediction Error %"] = (
        np.abs(
            result[TARGET]
            - result["Predicted Price (Rs per sq.ft)"]
        )
        / result[TARGET]
        * 100
    )

    print(
        result.to_string(
            index=False
        )
    )


# ============================================================
# FINAL CHECK
# ============================================================

test_model = pipeline.predict(
    X_test.iloc[:5]
)

print("\nSample predictions:")

for value in test_model:
    print(
        f"₹{value:,.2f} / sq.ft"
    )


if np.any(test_model < 0):
    raise ValueError(
        "ERROR: Model produced negative price."
    )


print("\n" + "=" * 70)
print("MODEL SAVED SUCCESSFULLY")
print("=" * 70)

print("Model file   :", MODEL_PATH)
print("Metadata file:", METADATA_PATH)
print("\nIMPORTANT:")
print("This model directly predicts Market Price (Rs per sq.ft).")
print("It is NOT a premium-ratio model.")