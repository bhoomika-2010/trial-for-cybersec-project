import numpy as np
import pandas as pd


class SequenceGenerator:

    def __init__(
        self,
        feature_columns,
        small_window=10,
        large_window=50
    ):

        self.feature_columns = feature_columns

        self.small_window = small_window
        self.large_window = large_window

    # =====================================================
    # CREATE SEQUENCES
    # =====================================================

    def create_sequences(
        self,
        df,
        window_size
    ):

        X = []
        y = []

        metadata = []

        # -------------------------------------------------
        # Process each entity independently
        # -------------------------------------------------

        for entity_id, entity_df in (
            df.groupby(
                "entity_id",
                sort=False
            )
        ):

            entity_df = (
                entity_df
                .sort_values("timestamp")
                .reset_index(drop=True)
            )

            # -------------------------------------------------
            # Build a classification window around the current
            # event instead of forecasting the next one.
            #
            # This is required for anomaly detection: the LSTM
            # must see the suspicious event inside the window it
            # is trying to classify.
            # -------------------------------------------------

            if len(entity_df) <= window_size:
                continue

            for i in range(
                window_size - 1,
                len(entity_df)
            ):

                # ---------------------------------------------
                # Input events include the current event and the
                # previous window_size - 1 events.
                # ---------------------------------------------

                window = entity_df.iloc[
                    i - window_size + 1:i + 1
                ]

                # ---------------------------------------------
                # Target event
                # ---------------------------------------------

                target = entity_df.iloc[i]

                # ---------------------------------------------
                # Feature matrix
                # ---------------------------------------------

                sequence = (
                    window[
                        self.feature_columns
                    ]
                    .to_numpy(
                        dtype=np.float32
                    )
                )

                X.append(sequence)

                # ---------------------------------------------
                # Target label
                # ---------------------------------------------

                y.append(
                    target["label"]
                )

                # ---------------------------------------------
                # Metadata
                # ---------------------------------------------

                metadata.append({

                    "entity_id":
                        entity_id,

                    "target_timestamp":
                        target["timestamp"],

                    "target_label":
                        target["label"]
                })

        X = np.asarray(
            X,
            dtype=np.float32
        )

        y = np.asarray(
            y
        )

        metadata = pd.DataFrame(
            metadata
        )

        return X, y, metadata

    # =====================================================
    # CREATE SMALL WINDOW
    # =====================================================

    def create_small_sequences(
        self,
        df
    ):

        # First create the sequences.
        X, y, metadata = self.create_sequences(
            df,
            self.small_window
        )

        # THEN inspect the metadata.
        print("\nSEQUENCE TARGET COVERAGE")
        print("=" * 80)

        if len(metadata) > 0:

            print(
                metadata[
                    "target_label"
                ]
                .value_counts()
                .sort_index()
            )

        else:

            print(
                "No metadata generated."
            )

        return X, y, metadata

    # =====================================================
    # CREATE LARGE WINDOW
    # =====================================================

    def create_large_sequences(
        self,
        df
    ):

        X, y, metadata = self.create_sequences(
            df,
            self.large_window
        )

        print("\nLARGE WINDOW TARGET COVERAGE")
        print("=" * 80)

        if len(metadata) > 0:

            print(
                metadata[
                    "target_label"
                ]
                .value_counts()
                .sort_index()
            )

        else:

            print(
                "No metadata generated."
            )

        return X, y, metadata

    # =====================================================
    # VALIDATION
    # =====================================================

    def validation_report(
        self,
        X,
        y,
        metadata,
        window_name
    ):

        print("\n")
        print("=" * 70)

        print(
            f"{window_name.upper()} WINDOW "
            "SEQUENCE VALIDATION"
        )

        print("=" * 70)

        # -------------------------------------------------
        # Shape
        # -------------------------------------------------

        print("\nSHAPE")
        print("-" * 70)

        print(
            "X shape:",
            X.shape
        )

        print(
            "y shape:",
            y.shape
        )

        # -------------------------------------------------
        # Expected dimensions
        # -------------------------------------------------

        if len(X) > 0:

            print(
                "\nSequence length:",
                X.shape[1]
            )

            print(
                "Number of features:",
                X.shape[2]
            )

        # -------------------------------------------------
        # Labels
        # -------------------------------------------------

        print("\nTARGET LABEL DISTRIBUTION")
        print("-" * 70)

        print(
            pd.Series(y)
            .value_counts()
        )

        # -------------------------------------------------
        # Metadata
        # -------------------------------------------------

        print("\nTARGET EVENTS")
        print("-" * 70)

        if len(metadata) == 0:

            print(
                "No metadata available."
            )

            return

        print(
            "Unique entities:",
            metadata[
                "entity_id"
            ].nunique()
        )

        print(
            "Target events:",
            len(metadata)
        )

        # -------------------------------------------------
        # Check X dimensions
        # -------------------------------------------------

        print("\nVALIDATION")
        print("-" * 70)

        if len(X) == 0:

            print(
                "ERROR: No sequences generated."
            )

            return

        if X.ndim != 3:

            print(
                "ERROR: X is not 3-dimensional."
            )

        else:

            print(
                "X is correctly 3-dimensional."
            )

        if len(X) != len(y):

            print(
                "ERROR: X and y lengths differ."
            )

        else:

            print(
                "X and y lengths match."
            )

        if len(X) != len(metadata):

            print(
                "ERROR: X and metadata lengths differ."
            )

        else:

            print(
                "X and metadata lengths match."
            )

        # -------------------------------------------------
        # Invalid values
        # -------------------------------------------------

        print("\nINVALID VALUES")
        print("-" * 70)

        print(
            "NaN:",
            np.isnan(X).sum()
        )

        print(
            "Infinite:",
            np.isinf(X).sum()
        )

        # -------------------------------------------------
        # Example
        # -------------------------------------------------

        print("\nEXAMPLE SEQUENCE")
        print("-" * 70)

        print(
            "Entity:",
            metadata.iloc[0][
                "entity_id"
            ]
        )

        print(
            "Target timestamp:",
            metadata.iloc[0][
                "target_timestamp"
            ]
        )

        print(
            "Target label:",
            metadata.iloc[0][
                "target_label"
            ]
        )

        print(
            "Input shape:",
            X[0].shape
        )

        print("\n")
        print("=" * 70)