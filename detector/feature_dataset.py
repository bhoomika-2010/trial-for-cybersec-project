import numpy as np
import pandas as pd


class FeatureDatasetBuilder:

    # =====================================================
    # CONSTANT FEATURES TO REMOVE
    # =====================================================

    CONSTANT_FEATURES = [
        "device_changed",
        "mac_changed",
        "os_changed",
        "baseline_type"
    ]

    # =====================================================
    # METADATA
    # These are NOT model features
    # =====================================================

    METADATA_COLUMNS = [
        "entity_id",
        "timestamp",
        "label"
    ]

    # =====================================================
    # INIT
    # =====================================================

    def __init__(self, extractor):

        self.extractor = extractor

        self.feature_df = None

        self.cleaned_df = None

        self.feature_columns = []

        self.numeric_columns = []

        self.categorical_columns = []

    # =====================================================
    # BUILD RAW FEATURE DATASET
    # =====================================================

    def build(self):

        rows = []

        # -------------------------------------------------
        # Process each entity separately
        # -------------------------------------------------

        for entity_id, entity_events in (
            self.extractor.events_by_entity.items()
        ):

            # -------------------------------------------------
            # Make sure events are chronological
            # -------------------------------------------------

            entity_events = sorted(
                entity_events,
                key=lambda e: e["timestamp"]
            )

            for index, event in enumerate(
                entity_events
            ):

                previous_event = (
                    entity_events[index - 1]
                    if index > 0
                    else None
                )

                features = (
                    self.extractor
                    .extract_event_features(
                        event,
                        previous_event,
                        index
                    )
                )

                row = features.copy()

                # Metadata
                row["entity_id"] = entity_id
                row["timestamp"] = event["timestamp"]
                row["label"] = event["label"]

                rows.append(row)

        self.feature_df = pd.DataFrame(rows)

        return self.feature_df

    # =====================================================
    # CLEAN FEATURE DATASET
    # =====================================================

    def clean(self):

        if self.feature_df is None:

            raise ValueError(
                "Build the feature dataset first."
            )

        df = self.feature_df.copy()

        # -------------------------------------------------
        # Ensure timestamp is datetime
        # -------------------------------------------------

        df["timestamp"] = pd.to_datetime(
            df["timestamp"]
        )

        # -------------------------------------------------
        # Sort chronologically PER USER
        #
        # This is extremely important for the LSTM.
        # -------------------------------------------------

        df = df.sort_values(
            by=[
                "entity_id",
                "timestamp"
            ]
        ).reset_index(drop=True)

        # -------------------------------------------------
        # Remove constant features
        # -------------------------------------------------

        columns_to_remove = [
            column
            for column in self.CONSTANT_FEATURES
            if column in df.columns
        ]

        df = df.drop(
            columns=columns_to_remove
        )

        # -------------------------------------------------
        # Encode auth_method
        #
        # One-hot encoding:
        #
        # MFA
        # SSO
        # password
        # biometric
        #
        # becomes:
        #
        # auth_method_MFA
        # auth_method_SSO
        # auth_method_biometric
        # auth_method_password
        # -------------------------------------------------

        if "auth_method" in df.columns:

            df = pd.get_dummies(
                df,
                columns=["auth_method"],
                prefix="auth_method",
                dtype=int
            )

        # -------------------------------------------------
        # Identify final feature columns
        # -------------------------------------------------

        self.feature_columns = [
            column
            for column in df.columns
            if column not in self.METADATA_COLUMNS
        ]

        # -------------------------------------------------
        # Check for remaining categorical features
        # -------------------------------------------------

        self.categorical_columns = [
            column
            for column in self.feature_columns
            if df[column].dtype == "object"
        ]

        # -------------------------------------------------
        # Numeric columns
        # -------------------------------------------------

        self.numeric_columns = [
            column
            for column in self.feature_columns
            if column not in self.categorical_columns
        ]

        self.cleaned_df = df

        return self.cleaned_df

    # =====================================================
    # CHECK CHRONOLOGICAL ORDER
    # =====================================================

    def check_chronological_order(self):

        if self.cleaned_df is None:

            raise ValueError(
                "Clean the dataset first."
            )

        problems = []

        for entity_id, group in (
            self.cleaned_df.groupby(
                "entity_id"
            )
        ):

            timestamps = (
                group["timestamp"]
                .tolist()
            )

            for i in range(
                1,
                len(timestamps)
            ):

                if (
                    timestamps[i]
                    < timestamps[i - 1]
                ):

                    problems.append(
                        entity_id
                    )

                    break

        return problems

    # =====================================================
    # CHECK MISSING VALUES
    # =====================================================

    def check_missing_values(self):

        return (
            self.cleaned_df[
                self.feature_columns
            ]
            .isnull()
            .sum()
            .loc[
                lambda x: x > 0
            ]
        )

    # =====================================================
    # CHECK NaN / INFINITY
    # =====================================================

    def check_invalid_values(self):

        numeric_df = (
            self.cleaned_df[
                self.numeric_columns
            ]
        )

        nan_counts = (
            numeric_df
            .isna()
            .sum()
        )

        infinite_counts = pd.Series(
            np.isinf(
                numeric_df.to_numpy()
            ).sum(axis=0),
            index=numeric_df.columns
        )

        invalid = pd.DataFrame({
            "NaN": nan_counts,
            "infinite": infinite_counts
        })

        return invalid[
            (invalid["NaN"] > 0)
            | (invalid["infinite"] > 0)
        ]

    # =====================================================
    # CHECK CONSTANT FEATURES
    # =====================================================

    def check_constant_features(self):

        constant = []

        for column in self.feature_columns:

            if (
                self.cleaned_df[column]
                .nunique(dropna=False)
                <= 1
            ):

                constant.append(column)

        return constant

    # =====================================================
    # LABEL DISTRIBUTION
    # =====================================================

    def get_label_distribution(self):

        return (
            self.cleaned_df["label"]
            .value_counts()
        )

    # =====================================================
    # FINAL VALIDATION
    # =====================================================

    def validation_report(self):

        if self.cleaned_df is None:

            raise ValueError(
                "Clean the dataset first."
            )

        print("\n")
        print("=" * 70)
        print("CLEAN FEATURE DATASET VALIDATION")
        print("=" * 70)

        # -------------------------------------------------
        # Dataset
        # -------------------------------------------------

        print("\nDATASET SIZE")
        print("-" * 70)

        print(
            "Rows:",
            len(self.cleaned_df)
        )

        print(
            "Total columns:",
            len(self.cleaned_df.columns)
        )

        print(
            "ML feature columns:",
            len(self.feature_columns)
        )

        # -------------------------------------------------
        # Feature columns
        # -------------------------------------------------

        print("\nFINAL ML FEATURES")
        print("-" * 70)

        for i, column in enumerate(
            self.feature_columns,
            start=1
        ):

            print(
                f"{i:2}. {column}"
            )

        # -------------------------------------------------
        # Data types
        # -------------------------------------------------

        print("\nFEATURE TYPES")
        print("-" * 70)

        print(
            "Numeric:",
            len(self.numeric_columns)
        )

        print(
            "Categorical:",
            len(self.categorical_columns)
        )

        if self.categorical_columns:

            print(
                "\nRemaining categorical:"
            )

            for column in (
                self.categorical_columns
            ):

                print(
                    f"  {column}"
                )

        else:

            print(
                "All ML features are numerical."
            )

        # -------------------------------------------------
        # Missing
        # -------------------------------------------------

        print("\nMISSING VALUES")
        print("-" * 70)

        missing = (
            self.check_missing_values()
        )

        if missing.empty:

            print(
                "No missing values."
            )

        else:

            print(missing)

        # -------------------------------------------------
        # Invalid
        # -------------------------------------------------

        print("\nNaN / INFINITE VALUES")
        print("-" * 70)

        invalid = (
            self.check_invalid_values()
        )

        if invalid.empty:

            print(
                "No NaN or infinite values."
            )

        else:

            print(invalid)

        # -------------------------------------------------
        # Constant
        # -------------------------------------------------

        print("\nCONSTANT FEATURES")
        print("-" * 70)

        constant = (
            self.check_constant_features()
        )

        if constant:

            for column in constant:

                print(column)

        else:

            print(
                "No constant features."
            )

        # -------------------------------------------------
        # Chronological order
        # -------------------------------------------------

        print("\nCHRONOLOGICAL ORDER")
        print("-" * 70)

        chronological_problems = (
            self.check_chronological_order()
        )

        if chronological_problems:

            print(
                "ERROR: chronological ordering "
                "problem for:"
            )

            for entity_id in (
                chronological_problems
            ):

                print(
                    f"  {entity_id}"
                )

        else:

            print(
                "All entities are chronologically ordered."
            )

        # -------------------------------------------------
        # Labels
        # -------------------------------------------------

        print("\nLABEL DISTRIBUTION")
        print("-" * 70)

        print(
            self.get_label_distribution()
        )

        # -------------------------------------------------
        # Dtypes
        # -------------------------------------------------

        print("\nFINAL DATA TYPES")
        print("-" * 70)

        print(
            self.cleaned_df[
                self.feature_columns
            ].dtypes
        )

        print("\n")
        print("=" * 70)
        print("CLEANING + VALIDATION COMPLETE")
        print("=" * 70)