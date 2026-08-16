import pandas as pd


class ChronologicalSplitter:

    def __init__(
        self,
        feature_df,
        train_ratio=0.8
    ):

        self.feature_df = feature_df
        self.train_ratio = train_ratio

        self.train_df = None
        self.test_df = None

    # =====================================================
    # SPLIT EACH ENTITY CHRONOLOGICALLY
    # =====================================================

    def split(self):

        train_parts = []
        test_parts = []

        # -------------------------------------------------
        # Process each entity independently
        # -------------------------------------------------

        for entity_id, entity_df in (
            self.feature_df.groupby(
                "entity_id",
                sort=False
            )
        ):

            entity_df = (
                entity_df
                .sort_values("timestamp")
                .reset_index(drop=True)
            )

            total_events = len(entity_df)

            # -------------------------------------------------
            # Initial 80% position
            # -------------------------------------------------

            split_index = int(
                total_events
                * self.train_ratio
            )

            if split_index <= 0:
                split_index = 1

            if split_index >= total_events:
                split_index = total_events - 1

            # -------------------------------------------------
            # Use the timestamp at the proposed boundary
            # -------------------------------------------------

            cutoff_timestamp = entity_df.iloc[
                split_index
            ]["timestamp"]

            # -------------------------------------------------
            # IMPORTANT:
            #
            # Never split events having the same timestamp.
            #
            # Everything at/before cutoff → train
            # Everything after cutoff       → test
            # -------------------------------------------------

            train_entity = entity_df[
                entity_df["timestamp"]
                < cutoff_timestamp
            ].copy()

            test_entity = entity_df[
                entity_df["timestamp"]
                >= cutoff_timestamp
            ].copy()

            # -------------------------------------------------
            # Safety fallback
            #
            # If every event before cutoff was empty,
            # use the original index split.
            # -------------------------------------------------

            if train_entity.empty:

                train_entity = entity_df.iloc[
                    :split_index
                ].copy()

                test_entity = entity_df.iloc[
                    split_index:
                ].copy()

            # -------------------------------------------------
            # Safety: if test became empty
            # -------------------------------------------------

            if test_entity.empty:

                train_entity = entity_df.iloc[
                    :split_index
                ].copy()

                test_entity = entity_df.iloc[
                    split_index:
                ].copy()

            train_parts.append(
                train_entity
            )

            test_parts.append(
                test_entity
            )

        # -------------------------------------------------
        # Combine
        # -------------------------------------------------

        self.train_df = pd.concat(
            train_parts,
            ignore_index=True
        )

        self.test_df = pd.concat(
            test_parts,
            ignore_index=True
        )

        # -------------------------------------------------
        # Final ordering
        # -------------------------------------------------

        self.train_df = (
            self.train_df
            .sort_values(
                ["entity_id", "timestamp"]
            )
            .reset_index(drop=True)
        )

        self.test_df = (
            self.test_df
            .sort_values(
                ["entity_id", "timestamp"]
            )
            .reset_index(drop=True)
        )

        return (
            self.train_df,
            self.test_df
        )

    # =====================================================
    # VALIDATE SPLIT
    # =====================================================

    def validate(self):

        if (
            self.train_df is None
            or self.test_df is None
        ):

            raise ValueError(
                "Run split() first."
            )

        print("\n")
        print("=" * 70)
        print("CHRONOLOGICAL TRAIN / TEST SPLIT")
        print("=" * 70)

        print("\nDATASET SIZE")
        print("-" * 70)

        print(
            "Total events:",
            len(self.feature_df)
        )

        print(
            "Training events:",
            len(self.train_df)
        )

        print(
            "Testing events:",
            len(self.test_df)
        )

        print("\nSPLIT RATIO")
        print("-" * 70)

        train_ratio = (
            len(self.train_df)
            / len(self.feature_df)
        )

        test_ratio = (
            len(self.test_df)
            / len(self.feature_df)
        )

        print(
            f"Training: {train_ratio:.2%}"
        )

        print(
            f"Testing : {test_ratio:.2%}"
        )

        # -------------------------------------------------
        # Verify every entity exists in both sets
        # -------------------------------------------------

        train_entities = set(
            self.train_df["entity_id"]
        )

        test_entities = set(
            self.test_df["entity_id"]
        )

        print("\nENTITY SPLIT")
        print("-" * 70)

        print(
            "Training entities:",
            len(train_entities)
        )

        print(
            "Testing entities:",
            len(test_entities)
        )

        if train_entities != test_entities:

            print(
                "WARNING: Entity sets differ."
            )

        else:

            print(
                "Every entity appears in both "
                "training and testing."
            )

        # -------------------------------------------------
        # Verify chronological separation
        # -------------------------------------------------

        chronology_errors = []

        for entity_id in train_entities:

            train_entity = (
                self.train_df[
                    self.train_df["entity_id"]
                    == entity_id
                ]
            )

            test_entity = (
                self.test_df[
                    self.test_df["entity_id"]
                    == entity_id
                ]
            )

            latest_train_time = (
                train_entity["timestamp"].max()
            )

            earliest_test_time = (
                test_entity["timestamp"].min()
            )

            if (
                latest_train_time
                >= earliest_test_time
            ):

                chronology_errors.append(
                    entity_id
                )

        print("\nCHRONOLOGICAL SEPARATION")
        print("-" * 70)

        if chronology_errors:

            print(
                "ERROR: chronological leakage "
                "detected for:"
            )

            for entity_id in (
                chronology_errors
            ):

                print(entity_id)

        else:

            print(
                "No chronological leakage detected."
            )

        # -------------------------------------------------
        # Label distribution
        # -------------------------------------------------

        print("\nTRAINING LABEL DISTRIBUTION")
        print("-" * 70)

        print(
            self.train_df[
                "label"
            ].value_counts()
        )

        print("\nTEST LABEL DISTRIBUTION")
        print("-" * 70)

        print(
            self.test_df[
                "label"
            ].value_counts()
        )

        print("\n")
        print("=" * 70)
        print("SPLIT VALIDATION COMPLETE")
        print("=" * 70)