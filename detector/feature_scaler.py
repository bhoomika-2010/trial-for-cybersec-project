import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


class FeatureScaler:

    def __init__(self, feature_columns):

        self.feature_columns = feature_columns

        self.scaler = StandardScaler()

        self.is_fitted = False

    # =====================================================
    # FIT ONLY ON TRAINING DATA
    # =====================================================

    def fit(self, train_df):

        train_features = train_df[
            self.feature_columns
        ]

        self.scaler.fit(
            train_features
        )

        self.is_fitted = True

        return self

    # =====================================================
    # TRANSFORM DATA
    # =====================================================

    def transform(self, df):

        if not self.is_fitted:

            raise ValueError(
                "Scaler must be fitted on training "
                "data before transformation."
            )

        transformed = df.copy()

        transformed[
            self.feature_columns
        ] = self.scaler.transform(
            df[self.feature_columns]
        )

        return transformed

    # =====================================================
    # FIT TRAIN + TRANSFORM
    # =====================================================

    def fit_transform_train(
        self,
        train_df
    ):

        self.fit(train_df)

        return self.transform(
            train_df
        )

    # =====================================================
    # VALIDATE SCALING
    # =====================================================

    def validation_report(
        self,
        train_scaled,
        test_scaled
    ):

        print("\n")
        print("=" * 70)
        print("FEATURE SCALING VALIDATION")
        print("=" * 70)

        # -------------------------------------------------
        # TRAIN statistics
        # -------------------------------------------------

        train_features = train_scaled[
            self.feature_columns
        ]

        test_features = test_scaled[
            self.feature_columns
        ]

        print("\nTRAINING DATA")
        print("-" * 70)

        print(
            "Mean of feature means:",
            train_features.mean().mean()
        )

        print(
            "Mean of feature std:",
            train_features.std().mean()
        )

        # -------------------------------------------------
        # Test statistics
        # -------------------------------------------------

        print("\nTEST DATA")
        print("-" * 70)

        print(
            "Mean of feature means:",
            test_features.mean().mean()
        )

        print(
            "Mean of feature std:",
            test_features.std().mean()
        )

        # -------------------------------------------------
        # Check invalid values
        # -------------------------------------------------

        print("\nINVALID VALUES")
        print("-" * 70)

        train_invalid = np.isinf(
            train_features.to_numpy()
        ).sum()

        test_invalid = np.isinf(
            test_features.to_numpy()
        ).sum()

        train_nan = (
            train_features
            .isna()
            .sum()
            .sum()
        )

        test_nan = (
            test_features
            .isna()
            .sum()
            .sum()
        )

        print(
            "Train NaN:",
            train_nan
        )

        print(
            "Train infinite:",
            train_invalid
        )

        print(
            "Test NaN:",
            test_nan
        )

        print(
            "Test infinite:",
            test_invalid
        )

        # -------------------------------------------------
        # Show sample statistics
        # -------------------------------------------------

        print("\nSCALED FEATURE SAMPLE")
        print("-" * 70)

        print(
            train_features[
                self.feature_columns[:10]
            ].describe().T[
                ["mean", "std", "min", "max"]
            ]
        )

        print("\n")
        print("=" * 70)
        print("SCALING VALIDATION COMPLETE")
        print("=" * 70)