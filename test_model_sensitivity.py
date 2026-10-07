import os
import pickle
import pandas as pd
import numpy as np


# ============================================================
# FILE PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CSV_FILE = os.path.join(BASE_DIR, "Land_Datasets_2.csv")
MODEL_FILE = os.path.join(BASE_DIR, "land_price_model_final.pkl")


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("LANDINSIGHT - MODEL FEATURE SENSITIVITY TEST")
print("=" * 70)

print("\nLoading dataset...")

df = pd.read_csv(CSV_FILE)

print("Dataset shape:", df.shape)


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading trained model...")

with open(MODEL_FILE, "rb") as f:
    model = pickle.load(f)

print("Model loaded successfully!")
print("Model type:", type(model))


# ============================================================
# FEATURES USED BY MODEL
# ============================================================

FEATURES = [
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
    "Registration Year"
]


# ============================================================
# CHECK FEATURES
# ============================================================

missing_features = [
    col for col in FEATURES
    if col not in df.columns
]

if missing_features:
    print("\nERROR!")
    print("Missing columns:")
    for col in missing_features:
        print(" -", col)
    raise SystemExit


# ============================================================
# DATASET RANGES
# ============================================================

print("\n" + "=" * 70)
print("DATASET RANGES")
print("=" * 70)

NUMERIC_COLUMNS = [
    "Road Width (ft)",
    "Distance to Main Road (km)",
    "Distance to Highway (km)",
    "Distance to Bus Stand (km)",
    "Distance to Railway Station (km)",
    "Distance to School (km)",
    "Distance to Hospital (km)",
    "Distance to Market (km)"
]

for col in NUMERIC_COLUMNS:
    print(
        f"{col:<40} "
        f"Min = {df[col].min():.2f} | "
        f"Max = {df[col].max():.2f}"
    )


# ============================================================
# SELECT A REAL PROPERTY FROM DATASET
# ============================================================

# Use a real row from the dataset.
# This prevents us from creating a completely artificial property.

BASE_ROW = df.iloc[0][FEATURES].copy()


print("\n" + "=" * 70)
print("BASE PROPERTY")
print("=" * 70)

for col in FEATURES:
    print(f"{col:<42}: {BASE_ROW[col]}")


# ============================================================
# CREATE GOOD PROPERTY
# ============================================================

good = BASE_ROW.copy()

# Road
good["Road Width (ft)"] = df["Road Width (ft)"].max()
good["Road Type"] = "Tar"

# Facilities
good["Water Facility"] = "Yes"
good["Electricity"] = "Yes"
good["Sewerage Facility"] = "Yes"

# Accessibility - smaller distance = better accessibility
good["Distance to Main Road (km)"] = df[
    "Distance to Main Road (km)"
].min()

good["Distance to Highway (km)"] = df[
    "Distance to Highway (km)"
].min()

good["Distance to Bus Stand (km)"] = df[
    "Distance to Bus Stand (km)"
].min()

good["Distance to Railway Station (km)"] = df[
    "Distance to Railway Station (km)"
].min()

good["Distance to School (km)"] = df[
    "Distance to School (km)"
].min()

good["Distance to Hospital (km)"] = df[
    "Distance to Hospital (km)"
].min()

good["Distance to Market (km)"] = df[
    "Distance to Market (km)"
].min()


# ============================================================
# CREATE BAD PROPERTY
# ============================================================

bad = BASE_ROW.copy()

# Road
bad["Road Width (ft)"] = df["Road Width (ft)"].min()
bad["Road Type"] = "Mud"

# Facilities
bad["Water Facility"] = "No"
bad["Electricity"] = "No"
bad["Sewerage Facility"] = "No"

# Accessibility - larger distance = worse accessibility
bad["Distance to Main Road (km)"] = df[
    "Distance to Main Road (km)"
].max()

bad["Distance to Highway (km)"] = df[
    "Distance to Highway (km)"
].max()

bad["Distance to Bus Stand (km)"] = df[
    "Distance to Bus Stand (km)"
].max()

bad["Distance to Railway Station (km)"] = df[
    "Distance to Railway Station (km)"
].max()

bad["Distance to School (km)"] = df[
    "Distance to School (km)"
].max()

bad["Distance to Hospital (km)"] = df[
    "Distance to Hospital (km)"
].max()

bad["Distance to Market (km)"] = df[
    "Distance to Market (km)"
].max()


# ============================================================
# CONVERT TO DATAFRAME
# ============================================================

base_df = pd.DataFrame([BASE_ROW])
good_df = pd.DataFrame([good])
bad_df = pd.DataFrame([bad])


# ============================================================
# PREDICTIONS
# ============================================================

print("\n" + "=" * 70)
print("PREDICTIONS")
print("=" * 70)

base_prediction = float(model.predict(base_df)[0])
good_prediction = float(model.predict(good_df)[0])
bad_prediction = float(model.predict(bad_df)[0])


print(f"\nBASE PROPERTY : ₹ {base_prediction:,.2f} / sq.ft")
print(f"GOOD PROPERTY : ₹ {good_prediction:,.2f} / sq.ft")
print(f"BAD PROPERTY  : ₹ {bad_prediction:,.2f} / sq.ft")


# ============================================================
# COMPARISON
# ============================================================

print("\n" + "=" * 70)
print("GOOD vs BAD ANALYSIS")
print("=" * 70)

difference = good_prediction - bad_prediction

if bad_prediction != 0:
    percentage_difference = (
        difference / bad_prediction
    ) * 100
else:
    percentage_difference = 0


print(f"\nGood Property : ₹ {good_prediction:,.2f}")
print(f"Bad Property  : ₹ {bad_prediction:,.2f}")
print(f"Difference    : ₹ {difference:,.2f}")
print(f"Difference %  : {percentage_difference:.2f}%")


# ============================================================
# RESULT INTERPRETATION
# ============================================================

print("\n" + "=" * 70)
print("INTERPRETATION")
print("=" * 70)

if good_prediction > bad_prediction:
    print("""
GOOD RESULT

The model predicts a higher price for the property
with better road/accessibility/facilities conditions.

This indicates that these features are contributing
to the prediction for this test case.
""")

elif good_prediction < bad_prediction:
    print("""
WARNING

The model predicted the BAD property at a higher price
than the GOOD property.

This indicates that the current Random Forest model is
not learning the expected relationship reliably for this
test case.

We should investigate and potentially retrain the model.
""")

else:
    print("""
WARNING

Good and Bad properties received exactly the same prediction.

The model may be relying heavily on other dominant features
such as Guideline Value and Land Area.
""")


# ============================================================
# INDIVIDUAL FEATURE SENSITIVITY
# ============================================================

print("\n" + "=" * 70)
print("INDIVIDUAL FEATURE SENSITIVITY")
print("=" * 70)

# Start from the real base property.
reference = BASE_ROW.copy()

base_pred = float(
    model.predict(
        pd.DataFrame([reference])
    )[0]
)

tests = []


# ------------------------------------------------------------
# Road Width
# ------------------------------------------------------------

test = reference.copy()
test["Road Width (ft)"] = df["Road Width (ft)"].min()

low_pred = float(
    model.predict(pd.DataFrame([test]))[0]
)

test["Road Width (ft)"] = df["Road Width (ft)"].max()

high_pred = float(
    model.predict(pd.DataFrame([test]))[0]
)

tests.append(
    (
        "Road Width",
        low_pred,
        high_pred
    )
)


# ------------------------------------------------------------
# Road Type
# ------------------------------------------------------------

test = reference.copy()

test["Road Type"] = "Mud"

mud_pred = float(
    model.predict(pd.DataFrame([test]))[0]
)

test["Road Type"] = "Concrete"

concrete_pred = float(
    model.predict(pd.DataFrame([test]))[0]
)

test["Road Type"] = "Tar"

tar_pred = float(
    model.predict(pd.DataFrame([test]))[0]
)

print("\nRoad Type:")
print(f"  Mud      : ₹ {mud_pred:,.2f}")
print(f"  Concrete : ₹ {concrete_pred:,.2f}")
print(f"  Tar      : ₹ {tar_pred:,.2f}")


# ------------------------------------------------------------
# Water
# ------------------------------------------------------------

test = reference.copy()

test["Water Facility"] = "No"

water_no_pred = float(
    model.predict(pd.DataFrame([test]))[0]
)

test["Water Facility"] = "Yes"

water_yes_pred = float(
    model.predict(pd.DataFrame([test]))[0]
)

print("\nWater Facility:")
print(f"  No  : ₹ {water_no_pred:,.2f}")
print(f"  Yes : ₹ {water_yes_pred:,.2f}")


# ------------------------------------------------------------
# Electricity
# ------------------------------------------------------------

test = reference.copy()

test["Electricity"] = "No"

electricity_no_pred = float(
    model.predict(pd.DataFrame([test]))[0]
)

test["Electricity"] = "Yes"

electricity_yes_pred = float(
    model.predict(pd.DataFrame([test]))[0]
)

print("\nElectricity:")
print(f"  No  : ₹ {electricity_no_pred:,.2f}")
print(f"  Yes : ₹ {electricity_yes_pred:,.2f}")


# ------------------------------------------------------------
# Sewerage
# ------------------------------------------------------------

test = reference.copy()

test["Sewerage Facility"] = "No"

sewerage_no_pred = float(
    model.predict(pd.DataFrame([test]))[0]
)

test["Sewerage Facility"] = "Yes"

sewerage_yes_pred = float(
    model.predict(pd.DataFrame([test]))[0]
)

print("\nSewerage:")
print(f"  No  : ₹ {sewerage_no_pred:,.2f}")
print(f"  Yes : ₹ {sewerage_yes_pred:,.2f}")


# ------------------------------------------------------------
# DISTANCE FEATURES
# ------------------------------------------------------------

DISTANCE_TESTS = [
    "Distance to Main Road (km)",
    "Distance to Highway (km)",
    "Distance to Bus Stand (km)",
    "Distance to Railway Station (km)",
    "Distance to School (km)",
    "Distance to Hospital (km)",
    "Distance to Market (km)"
]


for col in DISTANCE_TESTS:

    test = reference.copy()

    test[col] = df[col].min()

    near_pred = float(
        model.predict(pd.DataFrame([test]))[0]
    )

    test[col] = df[col].max()

    far_pred = float(
        model.predict(pd.DataFrame([test]))[0]
    )

    print(f"\n{col}:")
    print(f"  Near : ₹ {near_pred:,.2f}")
    print(f"  Far  : ₹ {far_pred:,.2f}")


# ============================================================
# FINISH
# ============================================================

print("\n" + "=" * 70)
print("TEST COMPLETED")
print("=" * 70)

print("""
Next step:

1. Check GOOD vs BAD prediction.
2. Check individual feature sensitivity.
3. If the relationships are sensible, keep the model.
4. If many features behave incorrectly, retrain the model.
""")