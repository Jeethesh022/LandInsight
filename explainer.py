import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# SHAP is optional. If the SHAP/Numba native DLL cannot be loaded,
# LandInsight automatically uses the permutation explanation fallback.
try:
    import shap
    SHAP_AVAILABLE = True
except Exception as exc:
    shap = None
    SHAP_AVAILABLE = False
    log.warning(
        "SHAP could not be imported. Using permutation fallback: %s",
        exc,
    )


FEATURE_LABELS = {
    "Location": "Location",
    "Area": "Area",
    "District": "District",
    "Latitude": "Latitude",
    "Longitude": "Longitude",
    "Land Area (sq.ft)": "Land Area",
    "Land Type": "Land Type",
    "Road Width (ft)": "Road Width",
    "Road Type": "Road Type",
    "Corner Plot": "Corner Plot",
    "Facing": "Facing",
    "Distance to Main Road (km)": "Distance to Main Road",
    "Distance to Highway (km)": "Distance to Highway",
    "Distance to Bus Stand (km)": "Distance to Bus Stand",
    "Distance to Railway Station (km)": "Distance to Railway Station",
    "Distance to School (km)": "Distance to School",
    "Distance to Hospital (km)": "Distance to Hospital",
    "Distance to Market (km)": "Distance to Market",
    "Water Facility": "Water Facility",
    "Electricity": "Electricity",
    "Sewerage Facility": "Sewerage Facility",
    "Guideline Value (Rs per sq.ft)": "Guideline Value",
    "Registration Year": "Registration Year",
}

CATEGORY_OF = {
    "Location": "Location",
    "Area": "Location",
    "District": "Location",
    "Latitude": "Location",
    "Longitude": "Location",
    "Land Area (sq.ft)": "Property",
    "Land Type": "Property",
    "Road Width (ft)": "Road & Access",
    "Road Type": "Road & Access",
    "Corner Plot": "Property",
    "Facing": "Property",
    "Distance to Main Road (km)": "Accessibility",
    "Distance to Highway (km)": "Accessibility",
    "Distance to Bus Stand (km)": "Accessibility",
    "Distance to Railway Station (km)": "Accessibility",
    "Distance to School (km)": "Accessibility",
    "Distance to Hospital (km)": "Accessibility",
    "Distance to Market (km)": "Accessibility",
    "Water Facility": "Utilities",
    "Electricity": "Utilities",
    "Sewerage Facility": "Utilities",
    "Guideline Value (Rs per sq.ft)": "Market",
    "Registration Year": "Property",
}


def _to_python(value):
    if value is None:
        return None

    try:
        if pd.isna(value):
            return None
    except Exception:
        pass

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        return float(value)

    if isinstance(value, np.bool_):
        return bool(value)

    return value


def fmt_inr(value, signed=False):
    try:
        number = float(value)
    except Exception:
        return "₹0"

    if signed:
        if number > 0:
            return f"+₹{number:,.0f}"
        if number < 0:
            return f"-₹{abs(number):,.0f}"

    return f"₹{abs(number):,.0f}"


def display_value(feature, value):
    value = _to_python(value)

    if value is None:
        return ""

    if feature == "Land Area (sq.ft)":
        try:
            return f"{float(value):,.0f} sq.ft"
        except Exception:
            return str(value)

    if feature == "Road Width (ft)":
        try:
            return f"{float(value):,.1f} ft"
        except Exception:
            return str(value)

    if feature == "Guideline Value (Rs per sq.ft)":
        try:
            return f"₹{float(value):,.0f}/sq.ft"
        except Exception:
            return str(value)

    if feature in {
        "Distance to Main Road (km)",
        "Distance to Highway (km)",
        "Distance to Bus Stand (km)",
        "Distance to Railway Station (km)",
        "Distance to School (km)",
        "Distance to Hospital (km)",
        "Distance to Market (km)",
    }:
        try:
            return f"{float(value):,.2f} km"
        except Exception:
            return str(value)

    if feature in {"Latitude", "Longitude"}:
        try:
            return f"{float(value):,.6f}"
        except Exception:
            return str(value)

    if feature == "Registration Year":
        try:
            return str(int(float(value)))
        except Exception:
            return str(value)

    if isinstance(value, float):
        return f"{value:,.2f}"

    return str(value)


class LandExplainer:
    """
    LandInsight explanation engine.

    Preferred path:
        SHAP TreeExplainer

    Fallback path:
        permutation-based model contribution.

    The fallback is NOT SHAP. It estimates how much each feature
    contributes to the difference between a typical reference property
    and the user's property.
    """

    def __init__(
        self,
        model,
        prediction_type,
        feature_columns,
        background_df=None,
    ):
        if not hasattr(model, "named_steps"):
            raise RuntimeError(
                "Model is not an sklearn Pipeline; cannot split "
                "preprocessor and estimator."
            )

        self.model = model
        self.prediction_type = prediction_type
        self.feature_columns = list(feature_columns)

        self.pre = model.named_steps["preprocessor"]
        self.estimator = model.named_steps["model"]

        # --------------------------------------------------------------
        # Try real SHAP first.
        #
        # On this Windows machine SHAP currently fails because Numba's
        # native DLL is blocked by Code Integrity / Smart App Control.
        # Therefore failure here must NOT disable explanations entirely.
        # --------------------------------------------------------------
        self.explainer = None
        self.base_raw = None
        self.explanation_method = "permutation"

        if shap is not None:
            try:
                self.explainer = shap.TreeExplainer(self.estimator)
                self.base_raw = float(
                    np.ravel(self.explainer.expected_value)[0]
                )
                self.explanation_method = "shap"

                log.info(
                    "SHAP TreeExplainer initialized successfully."
                )

            except Exception as exc:
                log.warning(
                    "SHAP initialization failed. "
                    "Using permutation fallback: %s",
                    exc,
                )

        else:
            log.warning(
                "SHAP is unavailable. "
                "Using permutation-based explanation fallback."
            )

        # --------------------------------------------------------------
        # Map transformed one-hot columns back to original features.
        # This is needed by the real SHAP path.
        # --------------------------------------------------------------
        transformed_names = [
            str(n) for n in self.pre.get_feature_names_out()
        ]

        by_length = sorted(
            self.feature_columns,
            key=len,
            reverse=True,
        )

        self.column_to_feature = []

        for name in transformed_names:
            body = (
                name.split("__", 1)[1]
                if "__" in name
                else name
            )

            owner = None

            for feature in by_length:
                if (
                    body == feature
                    or body.startswith(feature + "_")
                ):
                    owner = feature
                    break

            if owner is None:
                raise RuntimeError(
                    f"Could not map transformed column "
                    f"'{name}' to a feature."
                )

            self.column_to_feature.append(owner)

        self.feature_index = {
            feature: np.array(
                [
                    i
                    for i, f
                    in enumerate(self.column_to_feature)
                    if f == feature
                ]
            )
            for feature in self.feature_columns
        }

        # --------------------------------------------------------------
        # Prepare a typical/reference property for fallback mode.
        # --------------------------------------------------------------
        self.background_row = self._build_background_row(
            background_df
        )

        if self.explanation_method == "shap":
            log.info(
                "LandExplainer ready: SHAP | %s | "
                "%d transformed columns -> %d features",
                type(self.estimator).__name__,
                len(transformed_names),
                len(self.feature_columns),
            )
        else:
            log.info(
                "LandExplainer ready: PERMUTATION FALLBACK | %s | "
                "%d features",
                type(self.estimator).__name__,
                len(self.feature_columns),
            )

    # ------------------------------------------------------------------
    # Background/reference row
    # ------------------------------------------------------------------

    def _build_background_row(self, background_df):
        """
        Build a typical property from the training dataset.

        Numeric features:
            median

        Categorical/text features:
            mode

        This row is used only for the fallback explanation.
        """

        if (
            background_df is None
            or not isinstance(background_df, pd.DataFrame)
            or background_df.empty
        ):
            return None

        missing = [
            feature
            for feature in self.feature_columns
            if feature not in background_df.columns
        ]

        if missing:
            log.warning(
                "Cannot build fallback background row. "
                "Missing columns: %s",
                missing,
            )
            return None

        result = {}

        for feature in self.feature_columns:
            series = background_df[feature]

            # Remove missing values when possible.
            clean = series.dropna()

            if clean.empty:
                continue

            # Numeric columns -> median.
            if pd.api.types.is_numeric_dtype(clean):
                result[feature] = clean.median()

            # Categorical/text columns -> most common value.
            else:
                mode = clean.mode()

                if not mode.empty:
                    result[feature] = mode.iloc[0]
                else:
                    result[feature] = clean.iloc[0]

        if len(result) != len(self.feature_columns):
            log.warning(
                "Background row contains only %d/%d features.",
                len(result),
                len(self.feature_columns),
            )
            return None

        return result

    # ------------------------------------------------------------------

    def _scale(self, row):
        """
        Multiplier that turns raw model output into Rs / sq.ft.
        """

        if self.prediction_type == "premium_ratio":
            guideline = float(
                row["Guideline Value (Rs per sq.ft)"]
            )

            if guideline <= 0:
                raise ValueError(
                    "Guideline value must be greater than 0."
                )

            return guideline

        return 1.0

    # ------------------------------------------------------------------

    def _raw_predict(self, row_df):
        """
        Get the raw model output.

        The model itself is unchanged.
        """

        prediction = self.model.predict(
            row_df[self.feature_columns]
        )

        return float(np.asarray(prediction).reshape(-1)[0])

    # ------------------------------------------------------------------

    def _price_predict(self, row_df, output_guideline):
        """
        Convert a model prediction to Rs / sq.ft.

        For premium_ratio models we deliberately use the ORIGINAL
        property's guideline value during fallback calculations.

        This keeps the explanation additive in the final displayed
        price unit.
        """

        raw = self._raw_predict(row_df)

        if self.prediction_type == "premium_ratio":
            return raw * float(output_guideline)

        return raw

    # ------------------------------------------------------------------

    def _make_reference_dataframe(self, input_df):
        """
        Create one typical/reference property.
        """

        if self.background_row is None:
            # No dataset available.
            # Use current property as reference rather than inventing
            # values.
            return input_df[self.feature_columns].copy()

        reference = pd.DataFrame(
            [
                {
                    feature: self.background_row[feature]
                    for feature in self.feature_columns
                }
            ]
        )

        return reference

    # ------------------------------------------------------------------

    def _permutation_explain(
        self,
        input_df,
        assumed=None,
        top_n=8,
        permutations=8,
    ):
        """
        SHAP-free fallback.

        We start from a typical/reference property and gradually replace
        its features with the user's actual values.

        Different feature orders produce different marginal contributions.
        We average several random orders to reduce order dependence.

        IMPORTANT:
        These are model-based permutation contributions, not SHAP values.
        """

        assumed = set(assumed or [])

        row = input_df[self.feature_columns].iloc[0]

        # Original property's guideline is kept fixed when converting
        # premium-ratio predictions into Rs/sq.ft.
        output_guideline = float(
            row["Guideline Value (Rs per sq.ft)"]
        )

        if output_guideline <= 0:
            raise ValueError(
                "Guideline value must be greater than 0."
            )

        reference_df = self._make_reference_dataframe(input_df)

        # --------------------------------------------------------------
        # Baseline and final prediction
        # --------------------------------------------------------------

        baseline_price = self._price_predict(
            reference_df,
            output_guideline,
        )

        predicted_price = self._price_predict(
            input_df,
            output_guideline,
        )

        # --------------------------------------------------------------
        # Average marginal contribution over several feature orders.
        # --------------------------------------------------------------

        contributions = {
            feature: 0.0
            for feature in self.feature_columns
        }

        rng = np.random.default_rng(42)

        for _ in range(permutations):
            order = list(self.feature_columns)
            rng.shuffle(order)

            current = reference_df.copy()

            previous_price = baseline_price

            for feature in order:
                current.loc[0, feature] = row[feature]

                current_price = self._price_predict(
                    current,
                    output_guideline,
                )

                delta = current_price - previous_price

                contributions[feature] += delta

                previous_price = current_price

        # Average.
        for feature in contributions:
            contributions[feature] /= float(permutations)

        # --------------------------------------------------------------
        # Numerical correction.
        #
        # Averaged marginal contributions should already add to:
        #
        # predicted_price - baseline_price
        #
        # but tiny floating point differences can occur.
        # --------------------------------------------------------------

        target_gap = predicted_price - baseline_price

        contribution_sum = sum(contributions.values())

        correction = target_gap - contribution_sum

        if contributions:
            largest_feature = max(
                contributions,
                key=lambda f: abs(contributions[f]),
            )
            contributions[largest_feature] += correction

        # --------------------------------------------------------------
        # Build normal explanation feature objects.
        # --------------------------------------------------------------

        land_area = float(
            row["Land Area (sq.ft)"]
        )

        features = []

        for feature in self.feature_columns:
            impact = float(contributions[feature])

            raw_value = _to_python(row[feature])

            features.append(
                {
                    "feature": feature,
                    "label": FEATURE_LABELS.get(
                        feature,
                        feature,
                    ),
                    "category": CATEGORY_OF.get(
                        feature,
                        "Other",
                    ),
                    "value": raw_value,
                    "display_value": display_value(
                        feature,
                        raw_value,
                    ),
                    "impact": round(impact, 2),
                    "impact_total": round(
                        impact * land_area,
                        2,
                    ),
                    "direction": (
                        "increase"
                        if impact >= 0
                        else "decrease"
                    ),
                    "assumed": feature in assumed,
                }
            )

        features.sort(
            key=lambda item: abs(item["impact"]),
            reverse=True,
        )

        # --------------------------------------------------------------
        # Category totals
        # --------------------------------------------------------------

        totals = {}

        for item in features:
            category = item["category"]

            totals[category] = (
                totals.get(category, 0.0)
                + item["impact"]
            )

        categories = sorted(
            (
                {
                    "category": name,
                    "impact": round(value, 2),
                    "direction": (
                        "increase"
                        if value >= 0
                        else "decrease"
                    ),
                }
                for name, value in totals.items()
            ),
            key=lambda item: abs(item["impact"]),
            reverse=True,
        )

        # --------------------------------------------------------------
        # Waterfall
        # --------------------------------------------------------------

        top = features[:top_n]
        rest = features[top_n:]

        waterfall = [
            {
                "label": item["label"],
                "display_value": item["display_value"],
                "impact": item["impact"],
                "assumed": item["assumed"],
            }
            for item in top
        ]

        if rest:
            waterfall.append(
                {
                    "label": (
                        f"{len(rest)} other features"
                    ),
                    "display_value": "",
                    "impact": round(
                        sum(
                            item["impact"]
                            for item in rest
                        ),
                        2,
                    ),
                    "assumed": False,
                }
            )

        return {
            "available": True,
            "unit": "Rs per sq.ft",
            "model_type": type(
                self.estimator
            ).__name__,
            "explanation_method": (
                "permutation"
            ),
            "base_value": round(
                baseline_price,
                2,
            ),
            "predicted_price": round(
                predicted_price,
                2,
            ),
            "land_area": land_area,
            "features": features,
            "categories": categories,
            "waterfall": waterfall,
            "summary": self._summary(
                features,
                baseline_price,
                predicted_price,
            ),
            "note": (
                "Model-based feature contribution "
                "using permutation averaging. "
                "This is an explanation of model behavior, "
                "not a guaranteed causal effect."
            ),
        }

    # ------------------------------------------------------------------

    def explain(
        self,
        input_df,
        assumed=None,
        top_n=8,
    ):
        """
        Explain one prediction.

        Uses real SHAP when available.
        Automatically falls back to permutation contribution
        when SHAP cannot initialize.
        """

        # --------------------------------------------------------------
        # FALLBACK
        # --------------------------------------------------------------

        if self.explanation_method != "shap":
            return self._permutation_explain(
                input_df,
                assumed=assumed,
                top_n=top_n,
                permutations=8,
            )

        # --------------------------------------------------------------
        # REAL SHAP
        # --------------------------------------------------------------

        assumed = set(assumed or [])

        row = input_df.iloc[0]

        scale = self._scale(row)

        land_area = float(
            row["Land Area (sq.ft)"]
        )

        transformed = np.asarray(
            self.pre.transform(input_df),
            dtype=float,
        )

        values = np.asarray(
            self.explainer.shap_values(
                transformed,
                check_additivity=False,
            )
        ).reshape(-1)

        base_price = self.base_raw * scale

        features = []

        for feature in self.feature_columns:
            impact = float(
                values[
                    self.feature_index[feature]
                ].sum()
                * scale
            )

            raw_value = _to_python(
                row[feature]
            )

            features.append(
                {
                    "feature": feature,
                    "label": FEATURE_LABELS.get(
                        feature,
                        feature,
                    ),
                    "category": CATEGORY_OF.get(
                        feature,
                        "Other",
                    ),
                    "value": raw_value,
                    "display_value": display_value(
                        feature,
                        raw_value,
                    ),
                    "impact": round(
                        impact,
                        2,
                    ),
                    "impact_total": round(
                        impact * land_area,
                        2,
                    ),
                    "direction": (
                        "increase"
                        if impact >= 0
                        else "decrease"
                    ),
                    "assumed": feature in assumed,
                }
            )

        features.sort(
            key=lambda item: abs(
                item["impact"]
            ),
            reverse=True,
        )

        # --------------------------------------------------------------
        # Category totals
        # --------------------------------------------------------------

        totals = {}

        for item in features:
            totals[item["category"]] = (
                totals.get(
                    item["category"],
                    0.0,
                )
                + item["impact"]
            )

        categories = sorted(
            (
                {
                    "category": name,
                    "impact": round(
                        value,
                        2,
                    ),
                    "direction": (
                        "increase"
                        if value >= 0
                        else "decrease"
                    ),
                }
                for name, value
                in totals.items()
            ),
            key=lambda item: abs(
                item["impact"]
            ),
            reverse=True,
        )

        # --------------------------------------------------------------
        # Waterfall
        # --------------------------------------------------------------

        top = features[:top_n]

        rest = features[top_n:]

        waterfall = [
            {
                "label": item["label"],
                "display_value": item[
                    "display_value"
                ],
                "impact": item["impact"],
                "assumed": item["assumed"],
            }
            for item in top
        ]

        if rest:
            waterfall.append(
                {
                    "label": (
                        f"{len(rest)} other features"
                    ),
                    "display_value": "",
                    "impact": round(
                        sum(
                            item["impact"]
                            for item in rest
                        ),
                        2,
                    ),
                    "assumed": False,
                }
            )

        predicted_price = (
            base_price
            + sum(
                item["impact"]
                for item in features
            )
        )

        return {
            "available": True,
            "unit": "Rs per sq.ft",
            "model_type": type(
                self.estimator
            ).__name__,
            "explanation_method": "shap",
            "base_value": round(
                base_price,
                2,
            ),
            "predicted_price": round(
                predicted_price,
                2,
            ),
            "land_area": land_area,
            "features": features,
            "categories": categories,
            "waterfall": waterfall,
            "summary": self._summary(
                features,
                base_price,
                predicted_price,
            ),
            "note": (
                "Based on patterns the model learned "
                "from past land records. It shows what "
                "influenced this estimate, not a guaranteed cause."
            ),
        }

    # ------------------------------------------------------------------

    @staticmethod
    def _phrase(item):
        text = item["label"]

        if item["display_value"]:
            text += (
                f" ({item['display_value']})"
            )

        return (
            f"{text} "
            f"{fmt_inr(item['impact'], signed=True)}"
        )

    # ------------------------------------------------------------------

    def _summary(
        self,
        features,
        base_price,
        predicted_price,
    ):
        gap = (
            predicted_price
            - base_price
        )

        direction = (
            "above"
            if gap >= 0
            else "below"
        )

        text = (
            f"The estimate of "
            f"{fmt_inr(predicted_price)}/sq.ft "
            f"is {fmt_inr(abs(gap))} "
            f"{direction} the reference property's "
            f"model estimate of "
            f"{fmt_inr(base_price)}/sq.ft."
        )

        up = [
            f
            for f in features
            if f["impact"] > 0
        ][:3]

        down = [
            f
            for f in features
            if f["impact"] < 0
        ][:2]

        if up:
            text += (
                " Pushing the price up: "
                + ", ".join(
                    self._phrase(f)
                    for f in up
                )
                + "."
            )

        if down:
            text += (
                " Pulling it down: "
                + ", ".join(
                    self._phrase(f)
                    for f in down
                )
                + "."
            )

        return text

    # ------------------------------------------------------------------

    @staticmethod
    def compact(
        explanation,
        n=5,
    ):
        """
        Small version stored in history.
        """

        if (
            not explanation
            or not explanation.get(
                "available"
            )
        ):
            return None

        return {
            "base_value": explanation[
                "base_value"
            ],
            "predicted_price": explanation[
                "predicted_price"
            ],
            "summary": explanation[
                "summary"
            ],
            "explanation_method": explanation.get(
                "explanation_method",
                "unknown",
            ),
            "top": [
                {
                    "label": f["label"],
                    "display_value": f[
                        "display_value"
                    ],
                    "impact": f["impact"],
                }
                for f
                in explanation["features"][:n]
            ],
        }