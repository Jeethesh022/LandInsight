import os
import re
import json
import logging
import sqlite3
import pickle
import warnings
import math
from datetime import datetime
from functools import wraps

import numpy as np
import pandas as pd

from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    session,
    jsonify,
    flash,
    send_file,
)

try:
    from explainer import LandExplainer
except Exception as _explainer_import_error:
    LandExplainer = None
    print("Explainer import failed:", _explainer_import_error)

from werkzeug.security import (
    generate_password_hash,
    check_password_hash,
)

from io import BytesIO

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    REPORTLAB_AVAILABLE = True
except Exception as _reportlab_error:
    REPORTLAB_AVAILABLE = False
    _reportlab_error_message = str(_reportlab_error)



# ============================================================
# APP CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

app = Flask(__name__)

app.secret_key = os.environ.get(
    "LANDINSIGHT_SECRET_KEY",
    "landinsight-development-secret-key-change-this"
)

app.config["DATABASE"] = os.path.join(
    BASE_DIR,
    "landinsight.db"
)

app.config["DATASET"] = os.path.join(
    BASE_DIR,
    "Land_Datasets_2.csv"
)


# ============================================================
# MODEL FILE DETECTION
# ============================================================

MODEL_CANDIDATES = [
    "land_price_premium_model_final.pkl",
    "land_price_model_final.pkl",
]

METADATA_CANDIDATES = [
    "land_price_model_metadata.pkl",
    "land_metadata_final.pkl",
]


def find_existing_file(candidates):

    for filename in candidates:

        path = os.path.join(
            BASE_DIR,
            filename
        )

        if os.path.exists(path):
            return path

    return None


MODEL_PATH = find_existing_file(
    MODEL_CANDIDATES
)

METADATA_PATH = find_existing_file(
    METADATA_CANDIDATES
)


# ============================================================
# GLOBAL DATA
# ============================================================

df = None
model = None
metadata = None

DATASET_ERROR = None
MODEL_ERROR = None

explainer = None
EXPLAINER_ERROR = None


# ============================================================
# FEATURE CONFIGURATION
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
# DATASET LOADING
# ============================================================

def load_dataset():

    global df
    global DATASET_ERROR

    try:

        dataset_path = app.config["DATASET"]

        if not os.path.exists(dataset_path):

            raise FileNotFoundError(
                f"Dataset not found: {dataset_path}"
            )

        df = pd.read_csv(
            dataset_path
        )

        # Remove accidental unnamed columns
        df = df.loc[
            :,
            ~df.columns.astype(str)
            .str.startswith("Unnamed")
        ]

        # Clean column names
        df.columns = [
            str(column).strip()
            for column in df.columns
        ]

        # Required feature check
        missing = [
            column
            for column in FEATURE_COLUMNS
            if column not in df.columns
        ]

        if missing:

            raise ValueError(
                "Dataset is missing required columns: "
                + ", ".join(missing)
            )

        DATASET_ERROR = None

        print("=" * 70)
        print("DATASET LOADED")
        print("=" * 70)
        print("Rows:", len(df))
        print("Columns:", len(df.columns))
        print("Dataset:", dataset_path)
        print("=" * 70)

        return True

    except Exception as e:

        DATASET_ERROR = str(e)

        df = pd.DataFrame()

        print("=" * 70)
        print("DATASET LOAD ERROR")
        print("=" * 70)
        print(e)
        print("=" * 70)

        return False


# ============================================================
# MODEL LOADING
# ============================================================

def load_model():

    global model
    global metadata
    global MODEL_ERROR

    model = None
    metadata = None
    MODEL_ERROR = None

    if MODEL_PATH is None:

        MODEL_ERROR = (
            "No model file found. Expected: "
            + ", ".join(MODEL_CANDIDATES)
        )

        print(
            "MODEL ERROR:",
            MODEL_ERROR
        )

        return False

    try:

        warnings.filterwarnings(
            "ignore"
        )

        # ----------------------------------------------------
        # LOAD MODEL
        # ----------------------------------------------------

        with open(
            MODEL_PATH,
            "rb"
        ) as file:

            model = pickle.load(
                file
            )

        # ----------------------------------------------------
        # LOAD METADATA
        # ----------------------------------------------------

        if (
            METADATA_PATH
            and os.path.exists(
                METADATA_PATH
            )
        ):

            with open(
                METADATA_PATH,
                "rb"
            ) as file:

                metadata = pickle.load(
                    file
                )

        # ----------------------------------------------------
        # VALIDATE MODEL
        # ----------------------------------------------------

        if not hasattr(
            model,
            "predict"
        ):

            raise ValueError(
                "Loaded model does not have predict()."
            )

        if metadata is None:

            raise ValueError(
                "Model metadata file was not found."
            )

        prediction_type = metadata.get(
            "prediction_type",
            "direct_market_price_per_sqft"
        )

        if prediction_type not in (
            "premium_ratio",
            "direct_market_price_per_sqft",
        ):

            raise ValueError(
                "Unsupported prediction_type: "
                + str(prediction_type)
            )

        print("=" * 70)
        print("MODEL LOADED")
        print("=" * 70)
        print("Model:", MODEL_PATH)
        print("Metadata:", METADATA_PATH)
        print("Model type:", type(model))
        print("Has predict:", hasattr(model, "predict"))
        print(
            "Target:",
            metadata.get("target_column")
        )
        print(
            "Prediction type:",
            prediction_type
        )
        print("=" * 70)

        return True

    except Exception as e:

        MODEL_ERROR = str(e)

        model = None
        metadata = None

        print("=" * 70)
        print("MODEL LOAD ERROR")
        print("=" * 70)
        print(e)
        print("=" * 70)

        return False


# ============================================================
# LOAD DATASET + MODEL
# ============================================================

load_dataset()
load_model()


# ============================================================
# SHAP EXPLAINER (created once, reused for every request)
# ============================================================

def init_explainer():

    global explainer
    global EXPLAINER_ERROR

    explainer = None
    EXPLAINER_ERROR = None

    if model is None or metadata is None:
        EXPLAINER_ERROR = "Model not loaded."
        return False

    if LandExplainer is None:
        EXPLAINER_ERROR = "explainer.py could not be imported (is shap installed?)."
        return False

    try:
        explainer = LandExplainer(
    model,
    metadata.get(
        "prediction_type",
        "direct_market_price_per_sqft"
    ),
    FEATURE_COLUMNS,
    background_df=df,
)
        print("Explanation engine: READY")
        return True

    except Exception as e:
        EXPLAINER_ERROR = str(e)
        print("SHAP explainer: NOT READY ->", e)
        return False


init_explainer()


# ============================================================
# DATASET MAPPINGS
# ============================================================

LOCATION_AREAS = {}
AREA_DETAILS = {}
AREA_COORDINATES = {}
AREA_DISTRICTS = {}
LOCATION_COORDINATES = {}
GUIDELINE_VALUES = {}


# ============================================================
# SAFE CLEAN
# ============================================================

def safe_clean(value):

    if value is None:
        return ""

    try:

        if pd.isna(value):
            return ""

    except Exception:
        pass

    return str(value).strip()


# ============================================================
# SAFE FLOAT
# ============================================================

def safe_float(
    value,
    default=0.0
):

    try:

        if value is None:
            return default

        value = str(value).strip()

        if not value:
            return default

        number = float(value)

        if not np.isfinite(number):
            return default

        return number

    except Exception:

        return default


# ============================================================
# SAFE INT
# ============================================================

def safe_int(
    value,
    default=0
):

    try:

        if value is None:
            return default

        value = str(value).strip()

        if not value:
            return default

        return int(float(value))

    except Exception:

        return default


# ============================================================
# EMAIL VALIDATION
# ============================================================

def validate_email(email):

    return bool(
        re.match(
            r"^[^@\s]+@[^@\s]+\.[^@\s]+$",
            email
        )
    )


# ============================================================
# UNIQUE VALUES
# ============================================================

def unique_values(column):

    if (
        df is None
        or df.empty
        or column not in df.columns
    ):

        return []

    values = []

    for value in df[column].dropna().tolist():

        value = safe_clean(value)

        if (
            value
            and value not in values
        ):

            values.append(value)

    return sorted(values)


# ============================================================
# DATASET MAPPINGS
# ============================================================

def build_dataset_mappings():

    global LOCATION_AREAS
    global AREA_DETAILS
    global AREA_COORDINATES
    global AREA_DISTRICTS
    global LOCATION_COORDINATES
    global GUIDELINE_VALUES

    LOCATION_AREAS = {}
    AREA_DETAILS = {}
    AREA_COORDINATES = {}
    AREA_DISTRICTS = {}
    LOCATION_COORDINATES = {}
    GUIDELINE_VALUES = {}

    if (
        df is None
        or df.empty
    ):

        return

    required = [

        "Location",
        "Area",
        "District",
        "Latitude",
        "Longitude",
        "Guideline Value (Rs per sq.ft)",

    ]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:

        print(
            "Missing dataset columns:",
            missing
        )

        return

    temp = df.copy()

    for column in required:

        temp[column] = (
            temp[column]
            .apply(safe_clean)
        )

    # ========================================================
    # LOCATION -> AREAS
    # ========================================================

    for location, group in temp.groupby(
        "Location"
    ):

        location = safe_clean(
            location
        )

        if not location:
            continue

        areas = sorted({

            safe_clean(value)

            for value
            in group["Area"].tolist()

            if safe_clean(value)

        })

        LOCATION_AREAS[
            location
        ] = areas

    # ========================================================
    # LOCATION + AREA
    # ========================================================

    grouped = temp.groupby(
        [
            "Location",
            "Area"
        ],
        dropna=False
    )

    for (
        location,
        area
    ), group in grouped:

        location = safe_clean(
            location
        )

        area = safe_clean(
            area
        )

        if (
            not location
            or not area
        ):

            continue

        # ----------------------------------------------------
        # DISTRICT
        # ----------------------------------------------------

        district = get_mode_value(
            group["District"]
        )

        # ----------------------------------------------------
        # LATITUDE
        # ----------------------------------------------------

        try:

            lat = pd.to_numeric(
                group["Latitude"],
                errors="coerce"
            ).median()

        except Exception:

            lat = None

        # ----------------------------------------------------
        # LONGITUDE
        # ----------------------------------------------------

        try:

            lon = pd.to_numeric(
                group["Longitude"],
                errors="coerce"
            ).median()

        except Exception:

            lon = None

        # ----------------------------------------------------
        # GUIDELINE
        # ----------------------------------------------------

        try:

            guideline = pd.to_numeric(
                group[
                    "Guideline Value (Rs per sq.ft)"
                ],
                errors="coerce"
            ).median()

        except Exception:

            guideline = None

        # ----------------------------------------------------
        # DETAILS
        # ----------------------------------------------------

        AREA_DETAILS[
            (
                location,
                area
            )
        ] = {

            "district":
                district,

            "latitude":
                (
                    None
                    if (
                        lat is None
                        or pd.isna(lat)
                    )
                    else float(lat)
                ),

            "longitude":
                (
                    None
                    if (
                        lon is None
                        or pd.isna(lon)
                    )
                    else float(lon)
                ),

            "guideline_value":
                (
                    None
                    if (
                        guideline is None
                        or pd.isna(guideline)
                    )
                    else float(guideline)
                ),
        }

        # ----------------------------------------------------
        # DISTRICT MAP
        # ----------------------------------------------------

        AREA_DISTRICTS[
            (
                location,
                area
            )
        ] = district

        # ----------------------------------------------------
        # AREA COORDINATES
        # ----------------------------------------------------

        if (
            lat is not None
            and lon is not None
            and not pd.isna(lat)
            and not pd.isna(lon)
        ):

            if location not in AREA_COORDINATES:

                AREA_COORDINATES[
                    location
                ] = {}

            AREA_COORDINATES[
                location
            ][area] = {

                "lat":
                    float(lat),

                "lon":
                    float(lon),

            }

    # ========================================================
    # LOCATION COORDINATES
    # ========================================================

    for location, group in temp.groupby(
        "Location"
    ):

        location = safe_clean(
            location
        )

        if not location:
            continue

        try:

            lat = pd.to_numeric(
                group["Latitude"],
                errors="coerce"
            ).median()

            lon = pd.to_numeric(
                group["Longitude"],
                errors="coerce"
            ).median()

            if (
                not pd.isna(lat)
                and not pd.isna(lon)
            ):

                LOCATION_COORDINATES[
                    location
                ] = {

                    "latitude":
                        float(lat),

                    "longitude":
                        float(lon),

                }

        except Exception:

            pass

    # ========================================================
    # LOCATION GUIDELINE
    # ========================================================

    for location, group in temp.groupby(
        "Location"
    ):

        location = safe_clean(
            location
        )

        if not location:
            continue

        try:

            value = pd.to_numeric(
                group[
                    "Guideline Value (Rs per sq.ft)"
                ],
                errors="coerce"
            ).median()

            if not pd.isna(value):

                GUIDELINE_VALUES[
                    location
                ] = float(value)

        except Exception:

            pass

    print("=" * 70)
    print("DATASET MAPPINGS CREATED")
    print("=" * 70)
    print(
        "Locations:",
        len(LOCATION_AREAS)
    )
    print(
        "Total areas:",
        sum(
            len(value)
            for value in LOCATION_AREAS.values()
        )
    )
    print(
        "Area profiles:",
        len(AREA_DETAILS)
    )
    print("=" * 70)


# ============================================================
# MODE VALUE
# ============================================================

def get_mode_value(
    series,
    default=""
):

    try:

        values = [

            safe_clean(value)

            for value
            in series.tolist()

            if safe_clean(value)

        ]

        if not values:
            return default

        counts = (
            pd.Series(values)
            .value_counts()
        )

        return safe_clean(
            counts.index[0]
        )

    except Exception:

        return default


build_dataset_mappings()


# ============================================================
# CATEGORICAL OPTIONS
# ============================================================

LOCATIONS = unique_values(
    "Location"
)

AREAS = unique_values(
    "Area"
)

DISTRICTS = unique_values(
    "District"
)

LAND_TYPES = unique_values(
    "Land Type"
)

ROAD_TYPES = unique_values(
    "Road Type"
)

CORNER_PLOTS = unique_values(
    "Corner Plot"
)

FACINGS = unique_values(
    "Facing"
)

WATER_FACILITIES = unique_values(
    "Water Facility"
)

ELECTRICITY_OPTIONS = unique_values(
    "Electricity"
)

SEWERAGE_OPTIONS = unique_values(
    "Sewerage Facility"
)


# ============================================================
# ROAD TYPE NORMALIZATION
# ============================================================

ROAD_TYPE_MAP = {

    "tar":
        "Tar",

    "tar road":
        "Tar",

    "concrete":
        "Concrete",

    "concrete road":
        "Concrete",

    "mud":
        "Mud",

    "mud road":
        "Mud",

}


def normalize_road_type(value):

    value = safe_clean(value)

    if not value:
        return value

    key = value.lower().strip()

    return ROAD_TYPE_MAP.get(
        key,
        value
    )


# ============================================================
# LOCATION / AREA RESOLUTION
# ============================================================

def resolve_location(
    location
):

    location = safe_clean(
        location
    )

    if not location:
        return None

    for existing_location in LOCATION_AREAS:

        if (
            existing_location.casefold()
            == location.casefold()
        ):

            return existing_location

    return None


def resolve_area(
    location,
    area
):

    location = resolve_location(
        location
    )

    area = safe_clean(
        area
    )

    if (
        not location
        or not area
    ):

        return None

    for existing_area in LOCATION_AREAS.get(
        location,
        []
    ):

        if (
            existing_area.casefold()
            == area.casefold()
        ):

            return existing_area

    return None


# ============================================================
# DATABASE
# ============================================================

def get_db():

    conn = sqlite3.connect(
        app.config["DATABASE"]
    )

    conn.row_factory = sqlite3.Row

    return conn


# ============================================================
# DATABASE MIGRATION
# ============================================================

def migrate_predictions_table(
    conn
):

    columns = [

        row["name"]

        for row
        in conn.execute(
            "PRAGMA table_info(predictions)"
        ).fetchall()

    ]

    migrations = {

        "prediction_type":
            "TEXT",

        "location":
            "TEXT",

        "area":
            "TEXT",

        "district":
            "TEXT",

        "land_area":
            "REAL",

        "land_type":
            "TEXT",

        "road_width":
            "REAL",

        "road_type":
            "TEXT",

        "corner_plot":
            "TEXT",

        "facing":
            "TEXT",

        "guideline_value":
            "REAL",

        "predicted_price_per_sqft":
            "REAL",

        "estimated_total_price":
            "REAL",

        "created_at":
            "TEXT",

        "explanation_json":
            "TEXT",

    }

    for column, datatype in migrations.items():

        if column not in columns:

            try:

                conn.execute(
                    f"""
                    ALTER TABLE predictions
                    ADD COLUMN {column} {datatype}
                    """
                )

                print(
                    f"Database migration: added {column}"
                )

            except sqlite3.OperationalError as e:

                print(
                    f"Migration skipped for "
                    f"{column}: {e}"
                )

    conn.commit()


# ============================================================
# INITIALIZE DATABASE
# ============================================================

def init_db():

    conn = get_db()

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS users (

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            name TEXT NOT NULL,

            email TEXT NOT NULL UNIQUE,

            password TEXT NOT NULL,

            created_at TEXT NOT NULL

        )
        """
    )

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS predictions (

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            user_id INTEGER NOT NULL,

            prediction_type TEXT,

            location TEXT,

            area TEXT,

            district TEXT,

            land_area REAL,

            land_type TEXT,

            road_width REAL,

            road_type TEXT,

            corner_plot TEXT,

            facing TEXT,

            guideline_value REAL,

            predicted_price_per_sqft REAL,

            estimated_total_price REAL,

            created_at TEXT NOT NULL,

            FOREIGN KEY(user_id)
            REFERENCES users(id)

        )
        """
    )

    conn.commit()

    migrate_predictions_table(
        conn
    )

    conn.close()

    print(
        "Database initialized successfully."
    )


init_db()


# ============================================================
# AUTH
# ============================================================

def get_current_user():

    user_id = session.get(
        "user_id"
    )

    if not user_id:
        return None

    conn = get_db()

    user = conn.execute(
        """
        SELECT
            id,
            name,
            email,
            created_at
        FROM users
        WHERE id = ?
        """,
        (user_id,)
    ).fetchone()

    conn.close()

    return user


def login_required(view):

    @wraps(view)
    def wrapped_view(
        *args,
        **kwargs
    ):

        if not session.get(
            "user_id"
        ):

            next_url = request.path

            if request.query_string:

                next_url += (
                    "?"
                    + request.query_string.decode()
                )

            return redirect(
                url_for(
                    "login",
                    next=next_url
                )
            )

        return view(
            *args,
            **kwargs
        )

    return wrapped_view


# ============================================================
# MODEL VALIDATION
# ============================================================

def validate_prediction_model():

    if model is None:

        return (
            False,
            MODEL_ERROR
            or "Model not loaded."
        )

    if not hasattr(
        model,
        "predict"
    ):

        return (
            False,
            "Loaded model does not have predict()."
        )

    if metadata is None:

        return (
            False,
            "Model metadata not loaded."
        )

    prediction_type = metadata.get(
        "prediction_type"
    )

    if prediction_type not in (

        "premium_ratio",

        "direct_market_price_per_sqft",

    ):

        return (
            False,
            "Unsupported prediction type: "
            + str(prediction_type)
        )

    return (
        True,
        "Model ready."
    )


# ============================================================
# GET AREA GROUP
# ============================================================

def get_area_group(
    location,
    area
):

    if (
        df is None
        or df.empty
    ):

        return pd.DataFrame()

    canonical_location = resolve_location(
        location
    )

    if not canonical_location:
        return pd.DataFrame()

    canonical_area = resolve_area(
        canonical_location,
        area
    )

    if not canonical_area:
        return pd.DataFrame()

    location_mask = (
        df["Location"]
        .astype(str)
        .str.strip()
        .str.casefold()
        == canonical_location.casefold()
    )

    area_mask = (
        df["Area"]
        .astype(str)
        .str.strip()
        .str.casefold()
        == canonical_area.casefold()
    )

    return df[
        location_mask
        & area_mask
    ].copy()


# ============================================================
# AUTO DEFAULTS
# ============================================================

def get_auto_defaults(
    location,
    area
):

    canonical_location = resolve_location(
        location
    )

    canonical_area = (
        resolve_area(
            canonical_location,
            area
        )
        if canonical_location
        else None
    )

    if not canonical_location or not canonical_area:
        return {}

    # --------------------------------------------------------
    # GET DATASET GROUP
    # --------------------------------------------------------

    group = df[
        (
            df["Location"]
            .astype(str)
            .str.strip()
            .str.lower()
            == canonical_location.lower()
        )
        &
        (
            df["Area"]
            .astype(str)
            .str.strip()
            .str.lower()
            == canonical_area.lower()
        )
    ].copy()

    defaults = {}

    # --------------------------------------------------------
    # BASIC VALUES
    # --------------------------------------------------------

    defaults["Location"] = canonical_location
    defaults["Area"] = canonical_area

    # --------------------------------------------------------
    # DISTRICT
    # --------------------------------------------------------

    district = AREA_DISTRICTS.get(
        (
            canonical_location,
            canonical_area
        ),
        ""
    )

    if not district and not group.empty:
        district = get_mode_value(
            group["District"]
        )

    defaults["District"] = district

    # --------------------------------------------------------
    # NUMERIC DEFAULTS
    # --------------------------------------------------------

    numeric_default_columns = [

        "Latitude",

        "Longitude",

        "Road Width (ft)",

        "Distance to Main Road (km)",

        "Distance to Highway (km)",

        "Distance to Bus Stand (km)",

        "Distance to Railway Station (km)",

        "Distance to School (km)",

        "Distance to Hospital (km)",

        "Distance to Market (km)",

        "Guideline Value (Rs per sq.ft)",

    ]

    for column in numeric_default_columns:

        value = None

        # ----------------------------------------------------
        # LATITUDE
        # ----------------------------------------------------

        if column == "Latitude":

            details = AREA_DETAILS.get(
                (
                    canonical_location,
                    canonical_area
                ),
                {}
            )

            value = details.get(
                "latitude"
            )

        # ----------------------------------------------------
        # LONGITUDE
        # ----------------------------------------------------

        elif column == "Longitude":

            details = AREA_DETAILS.get(
                (
                    canonical_location,
                    canonical_area
                ),
                {}
            )

            value = details.get(
                "longitude"
            )

        # ----------------------------------------------------
        # GUIDELINE VALUE
        # ----------------------------------------------------

        elif column == "Guideline Value (Rs per sq.ft)":

            details = AREA_DETAILS.get(
                (
                    canonical_location,
                    canonical_area
                ),
                {}
            )

            value = details.get(
                "guideline_value"
            )

        # ----------------------------------------------------
        # DATASET MEDIAN FALLBACK
        # ----------------------------------------------------

        if (
            value is None
            or safe_float(
                value,
                0
            ) == 0
        ):

            if (
                not group.empty
                and column in group.columns
            ):

                try:

                    value = pd.to_numeric(
                        group[column],
                        errors="coerce"
                    ).median()

                except Exception:

                    value = 0

        numeric_value = safe_float(
            value,
            0
        )

        # Latitude / longitude need more precision.
        if column in [
            "Latitude",
            "Longitude"
        ]:

            numeric_value = round(
                numeric_value,
                6
            )

        else:

            numeric_value = round(
                numeric_value,
                2
            )

        defaults[column] = numeric_value

    # --------------------------------------------------------
    # CATEGORICAL DEFAULTS
    # --------------------------------------------------------

    categorical_default_columns = [

        "Land Type",

        "Road Type",

        "Corner Plot",

        "Facing",

        "Water Facility",

        "Electricity",

        "Sewerage Facility",

    ]

    for column in categorical_default_columns:

        value = ""

        if (
            not group.empty
            and column in group.columns
        ):

            value = get_mode_value(
                group[column]
            )

        defaults[column] = value

    # --------------------------------------------------------
    # REGISTRATION YEAR
    # --------------------------------------------------------

    registration_year = 0

    if (
        not group.empty
        and "Registration Year" in group.columns
    ):

        try:

            years = pd.to_numeric(
                group["Registration Year"],
                errors="coerce"
            ).dropna()

            if not years.empty:

                registration_year = int(
                    years.max()
                )

        except Exception:

            registration_year = 0

    if registration_year <= 0:

        registration_year = 2025

    defaults["Registration Year"] = (
        registration_year
    )

    return defaults


# ============================================================
# BUILD MODEL INPUT
# ============================================================

def build_prediction_data(
    form
):

    location = safe_clean(
        form.get("location")
    )

    if not location:

        raise ValueError(
            "Please select a location."
        )

    canonical_location = resolve_location(
        location
    )

    if not canonical_location:

        raise ValueError(
            "Selected location was not found in the dataset."
        )

    area = safe_clean(
        form.get("area")
    )

    if not area:

        raise ValueError(
            "Please select an area."
        )

    canonical_area = resolve_area(
        canonical_location,
        area
    )

    if not canonical_area:

        raise ValueError(
            "Selected area was not found for this location."
        )

    defaults = get_auto_defaults(
        canonical_location,
        canonical_area
    )

    # --------------------------------------------------------
    # TEXT HELPER
    # --------------------------------------------------------

    def text_value(
        field_name,
        default_key
    ):

        value = safe_clean(
            form.get(field_name)
        )

        if value:

            return value

        return safe_clean(
            defaults.get(
                default_key,
                ""
            )
        )

    # --------------------------------------------------------
    # NUMERIC HELPER
    # --------------------------------------------------------

    def numeric_value(
        field_name,
        default_key
    ):

        raw = form.get(
            field_name
        )

        # ----------------------------------------------------
        # USER ENTERED VALUE
        # ----------------------------------------------------

        if (
            raw is not None
            and str(raw).strip() != ""
        ):

            value = safe_float(
                raw,
                0
            )

            return round(
                value,
                6
            )

        # ----------------------------------------------------
        # AUTO-FILLED DEFAULT
        # ----------------------------------------------------

        value = safe_float(
            defaults.get(
                default_key,
                0
            ),
            0
        )

        if default_key in [
            "Latitude",
            "Longitude"
        ]:

            return round(
                value,
                6
            )

        return round(
            value,
            2
        )


    # --------------------------------------------------------
    # DISTRICT
    # --------------------------------------------------------

    district = text_value(
        "district",
        "District"
    )

    # --------------------------------------------------------
    # LATITUDE
    # --------------------------------------------------------

    latitude = numeric_value(
        "latitude",
        "Latitude"
    )

    # --------------------------------------------------------
    # LONGITUDE
    # --------------------------------------------------------

    longitude = numeric_value(
        "longitude",
        "Longitude"
    )

    # --------------------------------------------------------
    # LAND AREA
    # --------------------------------------------------------

    land_area = numeric_value(
        "land_area",
        "Land Area (sq.ft)"
    )

    # --------------------------------------------------------
    # LAND TYPE
    # --------------------------------------------------------

    land_type = text_value(
        "land_type",
        "Land Type"
    )

    # --------------------------------------------------------
    # ROAD WIDTH
    # --------------------------------------------------------

    road_width = numeric_value(
        "road_width",
        "Road Width (ft)"
    )

    # --------------------------------------------------------
    # ROAD TYPE
    # --------------------------------------------------------

    road_type = normalize_road_type(
        text_value(
            "road_type",
            "Road Type"
        )
    )

    # --------------------------------------------------------
    # CORNER PLOT
    # --------------------------------------------------------

    corner_plot = text_value(
        "corner_plot",
        "Corner Plot"
    )

    # --------------------------------------------------------
    # FACING
    # --------------------------------------------------------

    facing = text_value(
        "facing",
        "Facing"
    )

    # --------------------------------------------------------
    # DISTANCES
    # --------------------------------------------------------

    distance_main_road = numeric_value(
        "distance_main_road",
        "Distance to Main Road (km)"
    )

    distance_highway = numeric_value(
        "distance_highway",
        "Distance to Highway (km)"
    )

    distance_bus_stand = numeric_value(
        "distance_bus_stand",
        "Distance to Bus Stand (km)"
    )

    distance_railway = numeric_value(
        "distance_railway",
        "Distance to Railway Station (km)"
    )

    distance_school = numeric_value(
        "distance_school",
        "Distance to School (km)"
    )

    distance_hospital = numeric_value(
        "distance_hospital",
        "Distance to Hospital (km)"
    )

    distance_market = numeric_value(
        "distance_market",
        "Distance to Market (km)"
    )

    # --------------------------------------------------------
    # FACILITIES
    # --------------------------------------------------------

    water_facility = text_value(
        "water_facility",
        "Water Facility"
    )

    electricity = text_value(
        "electricity",
        "Electricity"
    )

    sewerage = text_value(
        "sewerage",
        "Sewerage Facility"
    )

    # --------------------------------------------------------
    # GUIDELINE
    # --------------------------------------------------------

    guideline = numeric_value(
        "guideline_value",
        "Guideline Value (Rs per sq.ft)"
    )

    # --------------------------------------------------------
    # REGISTRATION YEAR
    # --------------------------------------------------------

    registration_year = safe_int(
        form.get(
            "registration_year"
        ),
        safe_int(
            defaults.get(
                "Registration Year",
                2025
            ),
            2025
        )
    )

    # --------------------------------------------------------
    # FINAL DATA
    # --------------------------------------------------------

    data = {

        "Location":
            canonical_location,

        "Area":
            canonical_area,

        "District":
            district,

        "Latitude":
            latitude,

        "Longitude":
            longitude,

        "Land Area (sq.ft)":
            land_area,

        "Land Type":
            land_type,

        "Road Width (ft)":
            road_width,

        "Road Type":
            road_type,

        "Corner Plot":
            corner_plot,

        "Facing":
            facing,

        "Distance to Main Road (km)":
            distance_main_road,

        "Distance to Highway (km)":
            distance_highway,

        "Distance to Bus Stand (km)":
            distance_bus_stand,

        "Distance to Railway Station (km)":
            distance_railway,

        "Distance to School (km)":
            distance_school,

        "Distance to Hospital (km)":
            distance_hospital,

        "Distance to Market (km)":
            distance_market,

        "Water Facility":
            water_facility,

        "Electricity":
            electricity,

        "Sewerage Facility":
            sewerage,

        "Guideline Value (Rs per sq.ft)":
            guideline,

        "Registration Year":
            registration_year,

    }

    return pd.DataFrame(
        [data],
        columns=FEATURE_COLUMNS
    )


# ============================================================
# PREDICT PRICE
# ============================================================

def predict_price(
    input_df
):

    ready, message = (
        validate_prediction_model()
    )

    if not ready:

        raise RuntimeError(
            message
        )

    prediction = model.predict(
        input_df
    )

    if isinstance(
        prediction,
        (
            list,
            tuple,
            np.ndarray
        )
    ):

        raw_prediction = float(
            prediction[0]
        )

    else:

        raw_prediction = float(
            prediction
        )

    if not np.isfinite(
        raw_prediction
    ):

        raise ValueError(
            "Model returned an invalid prediction."
        )

    prediction_type = metadata.get(
        "prediction_type",
        "direct_market_price_per_sqft"
    )

    # --------------------------------------------------------
    # DIRECT MODEL
    # --------------------------------------------------------

    if (
        prediction_type
        == "direct_market_price_per_sqft"
    ):

        final_price = raw_prediction

    # --------------------------------------------------------
    # PREMIUM RATIO MODEL
    # --------------------------------------------------------

    elif (
        prediction_type
        == "premium_ratio"
    ):

        guideline = safe_float(
            input_df.iloc[0][
                "Guideline Value (Rs per sq.ft)"
            ]
        )

        if guideline <= 0:

            raise ValueError(
                "Guideline value is required "
                "for premium-ratio prediction."
            )

        final_price = (
            raw_prediction
            * guideline
        )

    else:

        raise ValueError(
            "Unsupported prediction type: "
            + str(prediction_type)
        )

    if not np.isfinite(
        final_price
    ):

        raise ValueError(
            "Final predicted price is invalid."
        )

    if final_price <= 0:

        raise ValueError(
            "Model returned an invalid "
            "non-positive price."
        )

    # --------------------------------------------------------
    # DEBUG
    # --------------------------------------------------------

    print("=" * 70)
    print("PREDICTION DEBUG")
    print("=" * 70)

    print(
        "Location:",
        input_df.iloc[0]["Location"]
    )

    print(
        "Area:",
        input_df.iloc[0]["Area"]
    )

    print(
        "District:",
        input_df.iloc[0]["District"]
    )

    print(
        "Guideline:",
        input_df.iloc[0][
            "Guideline Value (Rs per sq.ft)"
        ]
    )

    print(
        "Prediction type:",
        prediction_type
    )

    print(
        "Raw model output:",
        raw_prediction
    )

    print(
        "Final price / sq.ft:",
        final_price
    )

    print("=" * 70)

    return float(
        final_price
    )


# ============================================================
# SAVE PREDICTION
# ============================================================

def save_prediction(
    user_id,
    prediction_type,
    data,
    predicted_price,
    total_price,
    explanation_json=None
):

    row = data.iloc[0]

    conn = get_db()

    try:

        conn.execute(
            """
            INSERT INTO predictions (

                user_id,
                prediction_type,
                location,
                area,
                district,
                land_area,
                land_type,
                road_width,
                road_type,
                corner_plot,
                facing,
                guideline_value,
                predicted_price_per_sqft,
                estimated_total_price,
                created_at,
                explanation_json

            )
            VALUES (
                ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?, ?
            )
            """,
            (

                user_id,

                prediction_type,

                str(
                    row["Location"]
                ),

                str(
                    row["Area"]
                ),

                str(
                    row["District"]
                ),

                float(
                    row["Land Area (sq.ft)"]
                ),

                str(
                    row["Land Type"]
                ),

                float(
                    row["Road Width (ft)"]
                ),

                str(
                    row["Road Type"]
                ),

                str(
                    row["Corner Plot"]
                ),

                str(
                    row["Facing"]
                ),

                float(
                    row[
                        "Guideline Value (Rs per sq.ft)"
                    ]
                ),

                float(
                    predicted_price
                ),

                float(
                    total_price
                ),

                datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),

                explanation_json,

            )
        )

        conn.commit()

    finally:

        conn.close()


# ============================================================
# EXPLANATION HELPERS (SHAP)
# ============================================================

def get_assumed_features(form, input_df, mode):
    """
    Which features were NOT really typed by the user?

    Quick    : the page auto-fills area averages and sends them, so a value
               that still equals the area default counts as "assumed".
    Advanced : only blank fields (filled by the server) are "assumed".
    """

    assumed = set()

    try:

        row = input_df.iloc[0]

        if mode == "Quick":

            defaults = get_auto_defaults(
                row["Location"],
                row["Area"]
            )

            for feature in FEATURE_COLUMNS:

                if feature in ("Location", "Area", "Land Area (sq.ft)"):
                    continue

                default = defaults.get(feature)

                if default in (None, ""):
                    continue

                value = row[feature]

                if feature in NUMERIC_FEATURES:

                    if abs(
                        safe_float(value) - safe_float(default)
                    ) < 1e-6:

                        assumed.add(feature)

                elif (
                    str(value).strip().casefold()
                    == str(default).strip().casefold()
                ):

                    assumed.add(feature)

        else:

            field_map = {
                "District": "district",
                "Latitude": "latitude",
                "Longitude": "longitude",
                "Land Type": "land_type",
                "Road Width (ft)": "road_width",
                "Road Type": "road_type",
                "Corner Plot": "corner_plot",
                "Facing": "facing",
                "Distance to Main Road (km)": "distance_main_road",
                "Distance to Highway (km)": "distance_highway",
                "Distance to Bus Stand (km)": "distance_bus_stand",
                "Distance to Railway Station (km)": "distance_railway",
                "Distance to School (km)": "distance_school",
                "Distance to Hospital (km)": "distance_hospital",
                "Distance to Market (km)": "distance_market",
                "Water Facility": "water_facility",
                "Electricity": "electricity",
                "Sewerage Facility": "sewerage",
                "Guideline Value (Rs per sq.ft)": "guideline_value",
                "Registration Year": "registration_year",
            }

            for feature, field in field_map.items():

                raw = form.get(field)

                if raw is None or str(raw).strip() == "":
                    assumed.add(feature)

    except Exception:

        app.logger.exception("Could not work out assumed features")

    return assumed


def build_explanation(input_df, form, mode):
    """Never raises: a SHAP problem must not break the prediction."""

    if explainer is None:

        return {
            "available": False,
            "message": EXPLAINER_ERROR or "Explainer not ready."
        }

    try:

        return explainer.explain(
            input_df,
            assumed=get_assumed_features(
                form,
                input_df,
                mode
            )
        )

    except Exception as e:

        app.logger.exception("SHAP explanation failed")

        return {
            "available": False,
            "message": str(e)
        }


def explanation_to_json(explanation):

    if (
        explainer is None
        or not explanation
        or not explanation.get("available")
    ):
        return None

    try:

        return json.dumps(
            explainer.compact(explanation)
        )

    except Exception:

        return None


# ============================================================
# CONTEXT PROCESSOR
# ============================================================

@app.context_processor
def inject_globals():

    return {

        "current_user":
            get_current_user(),

        "locations":
            LOCATIONS,

        "areas":
            AREAS,

        "districts":
            DISTRICTS,

        "land_types":
            LAND_TYPES,

        "road_types":
            ROAD_TYPES,

        "corner_plots":
            CORNER_PLOTS,

        "facings":
            FACINGS,

        "water_facilities":
            WATER_FACILITIES,

        "electricity_options":
            ELECTRICITY_OPTIONS,

        "sewerage_options":
            SEWERAGE_OPTIONS,

    }


# ============================================================
# INDEX
# ============================================================

@app.route("/")
def index():

    return render_template(
        "index.html"
    )


# ============================================================
# HOME
# ============================================================

@app.route("/home")
def home():

    return redirect(
        url_for("index")
    )


# ============================================================
# REGISTER
# ============================================================

@app.route(
    "/register",
    methods=["GET", "POST"]
)
def register():

    if request.method == "POST":

        name = request.form.get(
            "name",
            ""
        ).strip()

        email = request.form.get(
            "email",
            ""
        ).strip().lower()

        password = request.form.get(
            "password",
            ""
        )

        confirm_password = request.form.get(
            "confirm_password",
            ""
        )

        if not name:

            flash(
                "Please enter your name.",
                "error"
            )

            return render_template(
                "register.html"
            )

        if not validate_email(
            email
        ):

            flash(
                "Please enter a valid email address.",
                "error"
            )

            return render_template(
                "register.html"
            )

        if len(password) < 6:

            flash(
                "Password must contain at least 6 characters.",
                "error"
            )

            return render_template(
                "register.html"
            )

        if (
            password
            != confirm_password
        ):

            flash(
                "Passwords do not match.",
                "error"
            )

            return render_template(
                "register.html"
            )

        conn = get_db()

        existing = conn.execute(
            """
            SELECT id
            FROM users
            WHERE LOWER(email) = ?
            """,
            (email,)
        ).fetchone()

        if existing:

            conn.close()

            flash(
                "An account with this email already exists.",
                "error"
            )

            return render_template(
                "register.html"
            )

        password_hash = (
            generate_password_hash(
                password
            )
        )

        conn.execute(
            """
            INSERT INTO users (
                name,
                email,
                password,
                created_at
            )
            VALUES (?, ?, ?, ?)
            """,
            (
                name,
                email,
                password_hash,
                datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),
            )
        )

        conn.commit()
        conn.close()

        flash(
            "Registration successful. Please login.",
            "success"
        )

        return redirect(
            url_for("login")
        )

    return render_template(
        "register.html"
    )


# ============================================================
# LOGIN
# ============================================================

@app.route(
    "/login",
    methods=["GET", "POST"]
)
def login():

    if request.method == "POST":

        email = request.form.get(
            "email",
            ""
        ).strip().lower()

        password = request.form.get(
            "password",
            ""
        )

        if (
            not email
            or not password
        ):

            flash(
                "Please enter both email and password.",
                "error"
            )

            return render_template(
                "login.html"
            )

        conn = get_db()

        user = conn.execute(
            """
            SELECT
                id,
                name,
                email,
                password
            FROM users
            WHERE LOWER(email) = ?
            """,
            (email,)
        ).fetchone()

        conn.close()

        if user is None:

            flash(
                "No account found with this email.",
                "error"
            )

            return render_template(
                "login.html"
            )

        stored_password = user["password"]

        password_ok = False

        try:

            password_ok = (
                check_password_hash(
                    stored_password,
                    password
                )
            )

        except Exception:

            password_ok = False

        # Old plaintext compatibility
        if (
            not password_ok
            and stored_password == password
        ):

            password_ok = True

        if not password_ok:

            flash(
                "Incorrect password.",
                "error"
            )

            return render_template(
                "login.html"
            )

        session.clear()

        session["user_id"] = user["id"]

        session["user_name"] = user["name"]

        session["user_email"] = user["email"]

        next_page = request.args.get(
            "next"
        )

        if (
            next_page
            and next_page.startswith("/")
        ):

            return redirect(
                next_page
            )

        return redirect(
            url_for("dashboard")
        )

    return render_template(
        "login.html"
    )


# ============================================================
# LOGOUT
# ============================================================

@app.route("/logout")
def logout():

    session.clear()

    flash(
        "You have been logged out.",
        "success"
    )

    return redirect(
        url_for("index")
    )


# ============================================================
# VALUATION RANGE + RELIABILITY
# ============================================================

def _normalised_text(value):
    return re.sub(r"\s+", " ", str(value or "").strip().lower())


def valuation_range(input_df, predicted_price):
    """Return an indicative range and transparent dataset-support rating."""
    target = "Market Price (Rs per sq.ft)"
    empty = {
        "price_low": None,
        "price_high": None,
        "reliability": "Unavailable",
        "comparables_count": 0,
        "range_basis": "No comparable market-price records were available.",
    }
    if df is None or target not in df.columns:
        return empty

    row = input_df.iloc[0]
    loc = _normalised_text(row.get("Location"))
    area = _normalised_text(row.get("Area"))
    work = df.copy()
    loc_col = work["Location"].map(_normalised_text) if "Location" in work.columns else pd.Series("", index=work.index)
    area_col = work["Area"].map(_normalised_text) if "Area" in work.columns else pd.Series("", index=work.index)

    exact = work[(loc_col == loc) & (area_col == area)]
    basis = "same area"
    candidates = exact[target]

    if len(candidates) < 5:
        candidates = work.loc[loc_col == loc, target]
        basis = "same location"

    if len(candidates) < 5:
        candidates = work[target]
        basis = "full dataset"

    values = pd.to_numeric(candidates, errors="coerce").dropna()
    if values.empty:
        return empty

    low = float(values.quantile(0.10))
    high = float(values.quantile(0.90))
    prediction = float(predicted_price)
    low = min(low, prediction)
    high = max(high, prediction)

    if high <= low:
        spread = max(abs(prediction) * 0.05, 1.0)
        low, high = prediction - spread, prediction + spread

    count = int(len(values))
    mean = float(values.mean())
    std = float(values.std(ddof=1)) if count > 1 else 0.0
    cv = abs(std / mean) if mean else 1.0

    if basis == "same area" and count >= 15 and cv <= 0.30:
        reliability = "High"
    elif count >= 5 and cv <= 0.60:
        reliability = "Medium"
    else:
        reliability = "Low"

    return {
        "price_low": round(low, 2),
        "price_high": round(high, 2),
        "reliability": reliability,
        "comparables_count": count,
        "range_basis": f"Indicative 10th–90th percentile of {basis} records.",
    }


def _report_value(value):
    if value is None or value == "":
        return "—"
    if isinstance(value, (int, float, np.integer, np.floating)):
        return f"{float(value):,.2f}"
    return str(value)


def _report_explanation_lines(explanation):
    if not isinstance(explanation, dict) or not explanation.get("available"):
        return []
    items = explanation.get("waterfall") or explanation.get("features") or []
    lines = []
    for item in items[:6]:
        label = item.get("label") or item.get("feature") or "Feature"
        display = item.get("display_value") or ""
        impact = item.get("impact")
        if impact is None:
            continue
        sign = "+" if float(impact) >= 0 else ""
        lines.append(f"{label}{(' (' + display + ')') if display else ''}: {sign}₹{float(impact):,.0f}/sq.ft")
    return lines


def build_valuation_report(data):
    if not REPORTLAB_AVAILABLE:
        raise RuntimeError("PDF reporting requires reportlab. Run: pip install reportlab")

    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=16*mm, leftMargin=16*mm, topMargin=16*mm, bottomMargin=16*mm)
    styles = getSampleStyleSheet()
    title = ParagraphStyle("ReportTitle", parent=styles["Title"], fontSize=20, leading=24, spaceAfter=8)
    sub = ParagraphStyle("Sub", parent=styles["Normal"], fontSize=9, textColor=colors.HexColor("#64748b"), spaceAfter=12)
    section = ParagraphStyle("Section", parent=styles["Heading2"], fontSize=13, leading=16, spaceBefore=12, spaceAfter=7)
    body = ParagraphStyle("Body", parent=styles["Normal"], fontSize=9.5, leading=13)

    story = [
        Paragraph("LandInsight Valuation Report", title),
        Paragraph(datetime.now().strftime("Generated on %d-%m-%Y %H:%M"), sub),
        Paragraph("Property Details", section),
    ]

    details = [
        ["Location", _report_value(data.get("location"))],
        ["Area", _report_value(data.get("area"))],
        ["District", _report_value(data.get("district"))],
        ["Land Area", f"{_report_value(data.get('land_area'))} sq.ft"],
        ["Guideline Value", f"₹{_report_value(data.get('guideline_value'))}/sq.ft"],
    ]
    table = Table(details, colWidths=[48*mm, 125*mm])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (0,-1), colors.HexColor("#eff6ff")),
        ("GRID", (0,0), (-1,-1), 0.4, colors.HexColor("#dbeafe")),
        ("FONTNAME", (0,0), (0,-1), "Helvetica-Bold"),
        ("FONTNAME", (1,0), (1,-1), "Helvetica"),
        ("FONTSIZE", (0,0), (-1,-1), 9),
        ("VALIGN", (0,0), (-1,-1), "TOP"),
        ("PADDING", (0,0), (-1,-1), 7),
    ]))
    story += [table, Paragraph("Valuation", section)]

    price = float(data.get("predicted_price_per_sqft") or 0)
    total = float(data.get("estimated_total_price") or 0)
    low = data.get("price_low")
    high = data.get("price_high")
    range_text = f"₹{float(low):,.0f} – ₹{float(high):,.0f}/sq.ft" if low is not None and high is not None else "Unavailable"

    valuation = [
        ["Estimated Price / sq.ft", f"₹{price:,.2f}"],
        ["Estimated Total Price", f"₹{total:,.2f}"],
        ["Indicative Price Range", range_text],
        ["Estimate Reliability", _report_value(data.get("reliability"))],
        ["Comparable Records", _report_value(data.get("comparables_count"))],
    ]
    table2 = Table(valuation, colWidths=[60*mm, 113*mm])
    table2.setStyle(TableStyle([
        ("GRID", (0,0), (-1,-1), 0.4, colors.HexColor("#dbeafe")),
        ("BACKGROUND", (0,0), (0,-1), colors.HexColor("#f8fafc")),
        ("FONTNAME", (0,0), (0,-1), "Helvetica-Bold"),
        ("FONTSIZE", (0,0), (-1,-1), 9),
        ("PADDING", (0,0), (-1,-1), 7),
    ]))
    story += [table2]

    explanation = data.get("explanation")
    if isinstance(explanation, str):
        try:
            explanation = json.loads(explanation)
        except Exception:
            explanation = None

    lines = _report_explanation_lines(explanation)
    if lines:
        story.append(Paragraph("Why this price?", section))
        story.append(Paragraph("Top model-based feature contributions:", body))
        for line in lines:
            story.append(Paragraph("• " + line, body))
        method = explanation.get("explanation_method", "permutation") if isinstance(explanation, dict) else "permutation"
        story.append(Spacer(1, 5))
        story.append(Paragraph(f"Explanation method: {method}.", sub))

    story.append(Paragraph("Important Note", section))
    story.append(Paragraph(
        "This report contains an ML-generated indicative valuation. The price range and reliability are dataset-support indicators, not guaranteed market prices or statistical confidence intervals. Feature contributions describe model behavior and are not guaranteed causal effects.",
        body,
    ))

    doc.build(story)
    buffer.seek(0)
    return buffer


# ============================================================
# DASHBOARD
# ============================================================

@app.route("/dashboard")
@login_required
def dashboard():

    user_id = session["user_id"]

    conn = get_db()

    total_predictions = conn.execute(
        """
        SELECT COUNT(*) AS count
        FROM predictions
        WHERE user_id = ?
        """,
        (user_id,)
    ).fetchone()["count"]

    average_price = conn.execute(
        """
        SELECT
            AVG(predicted_price_per_sqft)
            AS avg_price
        FROM predictions
        WHERE user_id = ?
        AND predicted_price_per_sqft IS NOT NULL
        """,
        (user_id,)
    ).fetchone()["avg_price"]

    latest_predictions = conn.execute(
        """
        SELECT *
        FROM predictions
        WHERE user_id = ?
        ORDER BY id DESC
        LIMIT 5
        """,
        (user_id,)
    ).fetchall()

    conn.close()

    if average_price is None:

        average_price = 0

    return render_template(
        "dashboard.html",
        total_predictions=total_predictions,
        average_price=average_price,
        latest_predictions=latest_predictions
    )


# ============================================================
# QUICK PAGE
# ============================================================

@app.route(
    "/quick",
    methods=["GET", "POST"]
)
@login_required
def quick():

    result = None
    error = None

    if request.method == "POST":

        try:

            input_df = build_prediction_data(
                request.form
            )

            land_area = float(
                input_df.iloc[0][
                    "Land Area (sq.ft)"
                ]
            )

            if land_area <= 0:

                raise ValueError(
                    "Land area must be greater than 0."
                )

            predicted_price = predict_price(
                input_df
            )

            total_price = (
                predicted_price
                * land_area
            )

            explanation = build_explanation(
                input_df,
                request.form,
                "Quick"
            )

            save_prediction(
                session["user_id"],
                "Quick",
                input_df,
                predicted_price,
                total_price,
                explanation_to_json(explanation)
            )

            range_info = valuation_range(input_df, predicted_price)

            result = {

                "price_per_sqft":
                    predicted_price,

                "total_price":
                    total_price,

                "land_area":
                    land_area,

                "location":
                    input_df.iloc[0]["Location"],

                "area":
                    input_df.iloc[0]["Area"],

                "district":
                    input_df.iloc[0]["District"],

                "guideline_value":
                    input_df.iloc[0][
                        "Guideline Value (Rs per sq.ft)"
                    ],

                "prediction_type":
                    (
                        metadata.get(
                            "prediction_type"
                        )
                        if metadata
                        else None
                    ),

                "price_low": range_info["price_low"],
                "price_high": range_info["price_high"],
                "reliability": range_info["reliability"],
                "comparables_count": range_info["comparables_count"],
                "range_basis": range_info["range_basis"],
                "explanation": explanation,

            }

        except Exception as e:

            app.logger.exception(
                "Quick prediction error"
            )

            error = str(e)

    return render_template(
        "quick.html",
        result=result,
        error=error,
        location_areas=LOCATION_AREAS,
        area_coordinates=AREA_COORDINATES,
    )


# ============================================================
# ADVANCED PAGE
# ============================================================

@app.route(
    "/advanced",
    methods=["GET", "POST"]
)
@login_required
def advanced():

    result = None
    error = None

    if request.method == "POST":

        try:

            input_df = build_prediction_data(
                request.form
            )

            land_area = float(
                input_df.iloc[0][
                    "Land Area (sq.ft)"
                ]
            )

            if land_area <= 0:

                raise ValueError(
                    "Land area must be greater than 0."
                )

            predicted_price = predict_price(
                input_df
            )

            total_price = (
                predicted_price
                * land_area
            )

            explanation = build_explanation(
                input_df,
                request.form,
                "Advanced"
            )

            range_info = valuation_range(input_df, predicted_price)

            save_prediction(
                session["user_id"],
                "Advanced",
                input_df,
                predicted_price,
                total_price,
                explanation_to_json(explanation)
            )

            result = {

                "price_per_sqft":
                    predicted_price,

                "total_price":
                    total_price,

                "land_area":
                    land_area,

                "location":
                    input_df.iloc[0]["Location"],

                "area":
                    input_df.iloc[0]["Area"],

                "district":
                    input_df.iloc[0]["District"],

                "guideline_value":
                    input_df.iloc[0][
                        "Guideline Value (Rs per sq.ft)"
                    ],

                "prediction_type":
                    (
                        metadata.get(
                            "prediction_type"
                        )
                        if metadata
                        else None
                    ),

                "price_low": range_info["price_low"],
                "price_high": range_info["price_high"],
                "reliability": range_info["reliability"],
                "comparables_count": range_info["comparables_count"],
                "range_basis": range_info["range_basis"],

                "explanation":
                    explanation,

            }

        except Exception as e:

            app.logger.exception(
                "Advanced prediction error"
            )

            error = str(e)

    return render_template(
        "advanced.html",
        result=result,
        error=error,
        location_areas=LOCATION_AREAS,
        area_coordinates=AREA_COORDINATES,
    )


# ============================================================
# DOWNLOAD VALUATION REPORT
# ============================================================

@app.route("/download-report", methods=["POST"])
@login_required
def download_report():
    try:
        data = request.get_json(silent=True)
        if not data:
            data = request.form.to_dict()

        explanation = data.get("explanation")
        if isinstance(explanation, str):
            try:
                explanation = json.loads(explanation)
            except Exception:
                pass
        data["explanation"] = explanation

        pdf = build_valuation_report(data)
        return send_file(
            pdf,
            mimetype="application/pdf",
            as_attachment=True,
            download_name="LandInsight_Valuation_Report.pdf",
        )
    except Exception as e:
        app.logger.exception("Valuation report generation failed")
        return jsonify({"success": False, "error": str(e)}), 500


# ============================================================
# HISTORY
# ============================================================

@app.route("/history")
@login_required
def history():

    user_id = session["user_id"]

    conn = get_db()

    # history.html expects the names  predicted_price  and  total_price,
    # but the table columns are  predicted_price_per_sqft  and
    # estimated_total_price, so they are renamed in the query.
    rows = conn.execute(
        """
        SELECT
            id,
            prediction_type,
            location,
            area,
            district,
            land_area,
            land_type,
            road_width,
            road_type,
            corner_plot,
            facing,
            guideline_value,
            predicted_price_per_sqft AS predicted_price,
            estimated_total_price AS total_price,
            created_at,
            explanation_json
        FROM predictions
        WHERE user_id = ?
        ORDER BY id DESC
        """,
        (user_id,)
    ).fetchall()

    conn.close()

    history_data = []

    for row in rows:

        item = dict(row)

        # Saved SHAP explanation (top contributions) for this prediction.
        try:

            item["explanation"] = (
                json.loads(item["explanation_json"])
                if item.get("explanation_json")
                else None
            )

        except Exception:

            item["explanation"] = None

        item.pop("explanation_json", None)

        history_data.append(item)

    return render_template(
        "history.html",
        history_data=history_data,
        predictions=history_data,
        username=(
            session.get("user_name")
            or session.get("user_email")
            or "you"
        )
    )


# ============================================================
# DELETE HISTORY
# ============================================================

@app.route(
    "/history/delete/<int:prediction_id>",
    methods=["POST", "GET"]
)
@login_required
def delete_history(
    prediction_id
):

    user_id = session["user_id"]

    conn = get_db()

    conn.execute(
        """
        DELETE FROM predictions
        WHERE id = ?
        AND user_id = ?
        """,
        (
            prediction_id,
            user_id
        )
    )

    conn.commit()

    conn.close()

    flash(
        "Prediction deleted.",
        "success"
    )

    return redirect(
        url_for("history")
    )


# ============================================================
# CLEAR HISTORY
# ============================================================

@app.route(
    "/history/clear",
    methods=["POST", "GET"]
)
@login_required
def clear_history():

    user_id = session["user_id"]

    conn = get_db()

    conn.execute(
        """
        DELETE FROM predictions
        WHERE user_id = ?
        """,
        (user_id,)
    )

    conn.commit()

    conn.close()

    flash(
        "Prediction history cleared.",
        "success"
    )

    return redirect(
        url_for("history")
    )


# ============================================================
# API - LOCATION AREAS
#
# THIS IS THE IMPORTANT FIX FOR:
# "Unable to load areas"
# ============================================================

@app.route(
    "/api/location-areas"
)
@login_required
def api_location_areas():

    try:

        location = safe_clean(
            request.args.get(
                "location",
                ""
            )
        )

        if not location:

            return jsonify({

                "success":
                    False,

                "areas":
                    [],

                "message":
                    "Location is required."

            }), 400

        canonical_location = resolve_location(
            location
        )

        if not canonical_location:

            return jsonify({

                "success":
                    False,

                "areas":
                    [],

                "message":
                    "Location not found."

            }), 404

        areas = LOCATION_AREAS.get(
            canonical_location,
            []
        )

        return jsonify({

            "success":
                True,

            "location":
                canonical_location,

            "areas":
                areas

        })

    except Exception as e:

        app.logger.exception(
            "Location areas API error"
        )

        return jsonify({

            "success":
                False,

            "areas":
                [],

            "message":
                str(e)

        }), 500


# ============================================================
# API - AREAS/<LOCATION>
# ============================================================

@app.route(
    "/api/areas/<path:location>"
)
@login_required
def api_areas(
    location
):

    canonical_location = resolve_location(
        location
    )

    if not canonical_location:

        return jsonify([])

    return jsonify(
        LOCATION_AREAS.get(
            canonical_location,
            []
        )
    )


# ============================================================
# API - AREA DETAILS
# ============================================================

@app.route(
    "/api/area-details"
)
@login_required
def api_area_details():

    location = safe_clean(
        request.args.get(
            "location",
            ""
        )
    )

    area = safe_clean(
        request.args.get(
            "area",
            ""
        )
    )

    canonical_location = resolve_location(
        location
    )

    canonical_area = resolve_area(
        location,
        area
    )

    if (
        not canonical_location
        or not canonical_area
    ):

        return jsonify({

            "success":
                False,

            "message":
                "Area details not found."

        }), 404

    details = AREA_DETAILS.get(
        (
            canonical_location,
            canonical_area
        )
    )

    if not details:

        return jsonify({

            "success":
                False,

            "message":
                "Area details not found."

        }), 404

    return jsonify({

        "success":
            True,

        "location":
            canonical_location,

        "area":
            canonical_area,

        **details

    })


# ============================================================
# API - AREA DEFAULTS
# ============================================================

@app.route(
    "/api/area-defaults"
)
@login_required
def api_area_defaults():

    location = safe_clean(
        request.args.get(
            "location",
            ""
        )
    )

    area = safe_clean(
        request.args.get(
            "area",
            ""
        )
    )

    canonical_location = resolve_location(
        location
    )

    canonical_area = resolve_area(
        location,
        area
    )

    if (
        not canonical_location
        or not canonical_area
    ):

        return jsonify({

            "success":
                False,

            "message":
                "Location and area are required."

        }), 400

    defaults = get_auto_defaults(
        canonical_location,
        canonical_area
    )

    return jsonify({

        "success":
            True,

        "location":
            canonical_location,

        "area":
            canonical_area,

        "defaults":
            defaults

    })


# ============================================================
# API - LOCATIONS
# ============================================================

@app.route(
    "/api/locations"
)
@login_required
def api_locations():

    return jsonify(
        LOCATIONS
    )


# ============================================================
# API - LOCATION DETAILS
# ============================================================

@app.route(
    "/api/location/<path:location>"
)
@login_required
def api_location(
    location
):

    canonical_location = resolve_location(
        location
    )

    if not canonical_location:

        return jsonify({

            "success":
                False,

            "message":
                "Location not found."

        }), 404

    coordinates = (
        LOCATION_COORDINATES.get(
            canonical_location
        )
    )

    return jsonify({

        "success":
            True,

        "location":
            canonical_location,

        "areas":
            LOCATION_AREAS.get(
                canonical_location,
                []
            ),

        "coordinates":
            coordinates,

        "guideline_value":
            GUIDELINE_VALUES.get(
                canonical_location
            )

    })


# ============================================================
# API - QUICK PREDICTION
#
# QUICK MODE:
#
# Location
# Area
# Land Area
#
# Everything else comes from defaults,
# BUT if quick.html sends edited values,
# those edited values are used.
# ============================================================

@app.route(
    "/api/quick-predict",
    methods=["POST"]
)
@login_required
def quick_predict():

    try:

        data = request.get_json(
            silent=True
        ) or {}

        # ----------------------------------------------------
        # BUILD FORM-LIKE DATA
        # ----------------------------------------------------

        location = safe_clean(
            data.get("location")
        )

        area = safe_clean(
            data.get("area")
        )

        land_area = data.get(
            "land_area"
        )

        if not location:

            return jsonify({

                "success":
                    False,

                "error":
                    "Please select a location."

            }), 400

        if not area:

            return jsonify({

                "success":
                    False,

                "error":
                    "Please select an area."

            }), 400

        if (
            land_area is None
            or str(land_area).strip() == ""
        ):

            return jsonify({

                "success":
                    False,

                "error":
                    "Please enter land area."

            }), 400

        # ----------------------------------------------------
        # VALIDATION
        # ----------------------------------------------------

        land_area_value = safe_float(
            land_area
        )

        if land_area_value <= 0:

            return jsonify({

                "success":
                    False,

                "error":
                    "Please enter a valid land area."

            }), 400

        # ----------------------------------------------------
        # CREATE FORM DATA
        #
        # This allows quick.html to send
        # edited defaults too.
        # ----------------------------------------------------

        form_data = {

            "location":
                location,

            "area":
                area,

            "land_area":
                land_area_value,

        }

        # ----------------------------------------------------
        # COPY OPTIONAL EDITED VALUES
        # ----------------------------------------------------

        optional_fields = [

            "district",

            "latitude",

            "longitude",

            "land_type",

            "road_width",

            "road_type",

            "corner_plot",

            "facing",

            "distance_main_road",

            "distance_highway",

            "distance_bus_stand",

            "distance_railway",

            "distance_school",

            "distance_hospital",

            "distance_market",

            "water_facility",

            "electricity",

            "sewerage",

            "guideline_value",

            "registration_year",

        ]

        for field in optional_fields:

            if field in data:

                form_data[field] = data[field]

        # ----------------------------------------------------
        # BUILD FINAL 23 FEATURES
        # ----------------------------------------------------

        input_df = build_prediction_data(
            form_data
        )

        # ----------------------------------------------------
        # MODEL CHECK
        # ----------------------------------------------------

        ready, message = (
            validate_prediction_model()
        )

        if not ready:

            return jsonify({

                "success":
                    False,

                "error":
                    message

            }), 500

        # ----------------------------------------------------
        # PREDICTION
        # ----------------------------------------------------

        predicted_price = predict_price(
            input_df
        )

        total_price = (
            predicted_price
            * land_area_value
        )

        # ----------------------------------------------------
        # EXPLANATION + RANGE
        # ----------------------------------------------------

        explanation = build_explanation(
            input_df,
            form_data,
            "Quick"
        )

        range_info = valuation_range(input_df, predicted_price)

        # ----------------------------------------------------
        # SAVE HISTORY
        # ----------------------------------------------------

        try:

            save_prediction(

                session["user_id"],

                "Quick",

                input_df,

                predicted_price,

                total_price,

                explanation_to_json(explanation)

            )

        except Exception as history_error:

            app.logger.warning(

                "Could not save quick prediction: %s",

                history_error

            )

        # ----------------------------------------------------
        # RESPONSE
        # ----------------------------------------------------

        row = input_df.iloc[0]

        auto_filled = {}

        for column in FEATURE_COLUMNS:

            value = row[column]

            if isinstance(
                value,
                (
                    np.integer,
                    np.floating
                )
            ):

                value = float(value)

            auto_filled[column] = value

        return jsonify({

            "success":
                True,

            "location":
                row["Location"],

            "area":
                row["Area"],

            "district":
                row["District"],

            "land_area":
                round(
                    land_area_value,
                    2
                ),

            "predicted_price":
                round(
                    predicted_price,
                    2
                ),

            "predicted_price_per_sqft":
                round(
                    predicted_price,
                    2
                ),

            "total_price":
                round(
                    total_price,
                    2
                ),

            "estimated_total_price":
                round(
                    total_price,
                    2
                ),

            "guideline_value":
                round(
                    safe_float(
                        row[
                            "Guideline Value (Rs per sq.ft)"
                        ]
                    ),
                    2
                ),

            "prediction_type":
                (
                    metadata.get(
                        "prediction_type"
                    )
                    if metadata
                    else None
                ),

            "price_low": range_info["price_low"],
            "price_high": range_info["price_high"],
            "reliability": range_info["reliability"],
            "comparables_count": range_info["comparables_count"],
            "range_basis": range_info["range_basis"],

            "auto_filled":
                auto_filled,

            "explanation":
                explanation,

        })

    except Exception as e:

        app.logger.exception(
            "Quick prediction failed"
        )

        return jsonify({

            "success":
                False,

            "error":
                str(e)

        }), 500


# ============================================================
# API - EXPLAIN
#
# Accepts the same JSON / form fields as /api/quick-predict.
# Body may contain  "mode": "Quick" | "Advanced"
# ============================================================

@app.route(
    "/api/explain",
    methods=["POST"]
)
@login_required
def api_explain():

    try:

        data = (
            request.get_json(silent=True)
            or request.form.to_dict()
        )

        mode = (
            "Advanced"
            if str(data.get("mode", "Quick")).lower() == "advanced"
            else "Quick"
        )

        land_area = safe_float(
            data.get("land_area")
        )

        if land_area <= 0:

            return jsonify({
                "success": False,
                "error": "Please enter a valid land area."
            }), 400

        input_df = build_prediction_data(
            data
        )

        ready, message = validate_prediction_model()

        if not ready:

            return jsonify({
                "success": False,
                "error": message
            }), 500

        predicted_price = predict_price(
            input_df
        )

        explanation = build_explanation(
            input_df,
            data,
            mode
        )

        range_info = valuation_range(input_df, predicted_price)

        return jsonify({
            "success": True,
            "predicted_price_per_sqft": round(predicted_price, 2),
            "estimated_total_price": round(predicted_price * land_area, 2),
            "price_low": range_info["price_low"],
            "price_high": range_info["price_high"],
            "reliability": range_info["reliability"],
            "comparables_count": range_info["comparables_count"],
            "range_basis": range_info["range_basis"],
            "explanation": explanation,
        })

    except ValueError as e:

        return jsonify({
            "success": False,
            "error": str(e)
        }), 400

    except Exception as e:

        app.logger.exception("Explain API failed")

        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# ============================================================
# API - SAVED EXPLANATION FOR A HISTORY ROW
# ============================================================

@app.route(
    "/api/history/<int:prediction_id>/explanation"
)
@login_required
def api_history_explanation(prediction_id):

    conn = get_db()

    row = conn.execute(
        """
        SELECT explanation_json
        FROM predictions
        WHERE id = ?
        AND user_id = ?
        """,
        (
            prediction_id,
            session["user_id"]
        )
    ).fetchone()

    conn.close()

    if row is None:

        return jsonify({
            "success": False,
            "message": "Prediction not found."
        }), 404

    if not row["explanation_json"]:

        return jsonify({
            "success": False,
            "message": "No explanation was saved for this prediction."
        }), 404

    return jsonify({
        "success": True,
        "explanation": json.loads(row["explanation_json"])
    })


# ============================================================
# API - MAP LOCATION
# ============================================================

@app.route(
    "/api/map-location"
)
@login_required
def api_map_location():

    try:

        lat = safe_float(
            request.args.get("lat"),
            None
        )

        lon = safe_float(
            request.args.get("lon"),
            None
        )

        if lat is None or lon is None:

            return jsonify({
                "success": False,
                "message": "Latitude and longitude are required."
            }), 400

        if df is None or df.empty:

            return jsonify({
                "success": False,
                "message": "Dataset is not loaded."
            }), 500

        # ----------------------------------------------------
        # Build unique area coordinate list
        # ----------------------------------------------------

        candidates = []

        for (
            location,
            area
        ), details in AREA_DETAILS.items():

            area_lat = details.get(
                "latitude"
            )

            area_lon = details.get(
                "longitude"
            )

            if (
                area_lat is None
                or area_lon is None
            ):
                continue

            try:

                area_lat = float(
                    area_lat
                )

                area_lon = float(
                    area_lon
                )

            except Exception:

                continue

            # Haversine distance in km.
            lat1 = math.radians(
                float(lat)
            )

            lon1 = math.radians(
                float(lon)
            )

            lat2 = math.radians(
                area_lat
            )

            lon2 = math.radians(
                area_lon
            )

            dlat = lat2 - lat1
            dlon = lon2 - lon1

            a = (
                math.sin(dlat / 2) ** 2
                +
                math.cos(lat1)
                * math.cos(lat2)
                * math.sin(dlon / 2) ** 2
            )

            distance_km = (
                6371.0
                * 2
                * math.asin(
                    math.sqrt(a)
                )
            )

            candidates.append({
                "location": location,
                "area": area,
                "district": details.get(
                    "district",
                    ""
                ),
                "latitude": round(
                    area_lat,
                    6
                ),
                "longitude": round(
                    area_lon,
                    6
                ),
                "guideline_value": round(
                    safe_float(
                        details.get(
                            "guideline_value",
                            0
                        ),
                        0
                    ),
                    2
                ),
                "distance_km": round(
                    distance_km,
                    2
                )
            })

        if not candidates:

            return jsonify({
                "success": False,
                "message": "No mapped areas are available."
            }), 404

        nearest = min(
            candidates,
            key=lambda item: item["distance_km"]
        )

        # ----------------------------------------------------
        # Safety warning if the click is far away.
        # We still return the nearest dataset area so the UI
        # can show the user what was matched.
        # ----------------------------------------------------

        nearest["clicked_latitude"] = round(
            float(lat),
            6
        )

        nearest["clicked_longitude"] = round(
            float(lon),
            6
        )

        nearest["warning"] = (
            "Selected point is far from the nearest "
            "dataset area. Please verify the area."
            if nearest["distance_km"] > 25
            else ""
        )

        nearest["success"] = True

        return jsonify(
            nearest
        )

    except Exception as e:

        app.logger.exception(
            "Map location lookup failed"
        )

        return jsonify({
            "success": False,
            "message": str(e)
        }), 500


# ============================================================
# API - MODEL STATUS
# ============================================================

@app.route(
    "/api/model-status"
)
def api_model_status():

    ready, message = (
        validate_prediction_model()
    )

    prediction_type = None

    target_column = None

    if metadata:

        prediction_type = metadata.get(
            "prediction_type"
        )

        target_column = metadata.get(
            "target_column"
        )

    return jsonify({

        "ready":
            ready,

        "message":
            message,

        "model_path":
            MODEL_PATH,

        "metadata_path":
            METADATA_PATH,

        "prediction_type":
            prediction_type,

        "target_column":
            target_column,

        "dataset_path":
            app.config["DATASET"],

        "dataset_rows":
            (
                len(df)
                if df is not None
                else 0
            ),

        "dataset_columns":
            (
                len(df.columns)
                if df is not None
                else 0
            ),

        "feature_count":
            len(FEATURE_COLUMNS),

        "explainer_ready":
            explainer is not None,

        "explainer_error":
            EXPLAINER_ERROR,

    })


# ============================================================
# HEALTH
# ============================================================

@app.route(
    "/health"
)
def health():

    model_ready, model_message = (
        validate_prediction_model()
    )

    return jsonify({

        "status":
            "ok",

        "model_ready":
            model_ready,

        "model_message":
            model_message,

        "dataset_loaded":
            (
                df is not None
                and not df.empty
            ),

        "dataset_rows":
            (
                len(df)
                if df is not None
                else 0
            ),

        "timestamp":
            datetime.now().isoformat()

    })


# ============================================================
# FAVICON
# ============================================================

@app.route(
    "/favicon.ico"
)
def favicon():

    return "", 204


# ============================================================
# ERROR 404
# ============================================================

@app.errorhandler(404)
def page_not_found(error):

    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Page Not Found</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                background: #f8fafc;
                text-align: center;
                padding: 80px 20px;
            }

            h1 {
                font-size: 64px;
                margin-bottom: 10px;
                color: #0f172a;
            }

            p {
                color: #64748b;
                font-size: 18px;
            }

            a {
                display: inline-block;
                margin-top: 20px;
                padding: 12px 22px;
                background: #2563eb;
                color: white;
                text-decoration: none;
                border-radius: 8px;
            }
        </style>
    </head>

    <body>

        <h1>404</h1>

        <p>
            The page you are looking for does not exist.
        </p>

        <a href="/">
            Go Home
        </a>

    </body>
    </html>
    """, 404


# ============================================================
# ERROR 500
# ============================================================

@app.errorhandler(500)
def internal_server_error(error):

    return """
    <!DOCTYPE html>
    <html>
    <head>

        <title>LandInsight Error</title>

        <style>

            body {
                font-family: Arial, sans-serif;
                background: #f8fafc;
                text-align: center;
                padding: 80px 20px;
            }

            h1 {
                color: #dc2626;
            }

            p {
                color: #64748b;
            }

            a {
                display: inline-block;
                margin-top: 20px;
                padding: 12px 22px;
                background: #2563eb;
                color: white;
                text-decoration: none;
                border-radius: 8px;
            }

        </style>

    </head>

    <body>

        <h1>
            Something went wrong
        </h1>

        <p>
            LandInsight encountered an unexpected error.
        </p>

        <a href="/">
            Go Home
        </a>

    </body>
    </html>
    """, 500


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":

    print()

    print("=" * 70)
    print("LANDINSIGHT")
    print("=" * 70)

    print(
        "Dataset:",
        (
            "READY"
            if (
                df is not None
                and not df.empty
            )
            else "NOT READY"
        )
    )

    print(
        "Model:",
        (
            "READY"
            if model is not None
            else "NOT READY"
        )
    )

    print(
        "Model file:",
        MODEL_PATH
    )

    print(
        "Metadata:",
        METADATA_PATH
    )

    if metadata:

        print(
            "Prediction type:",
            metadata.get(
                "prediction_type"
            )
        )

        print(
            "Target:",
            metadata.get(
                "target_column"
            )
        )

    print(
        "Database:",
        app.config["DATABASE"]
    )

    print("=" * 70)

    print(
        "Quick Mode:",
        "Location + Area + Sq.ft required"
    )

    print(
        "Quick defaults:",
        "ALL ML FEATURES AUTO-FILLED + EDITABLE"
    )

    print(
        "ML Features:",
        len(FEATURE_COLUMNS)
    )

    print(
        "Locations:",
        len(LOCATIONS)
    )

    print(
        "Areas:",
        len(AREAS)
    )

    print("=" * 70)

    print(
        "Open: http://127.0.0.1:5000"
    )

    print("=" * 70)

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )