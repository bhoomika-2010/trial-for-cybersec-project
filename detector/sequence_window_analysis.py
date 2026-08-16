import pandas as pd
import numpy as np


class SequenceWindowAnalyzer:

    def __init__(
        self,
        feature_df,
        window_sizes=None
    ):

        self.feature_df = feature_df

        if window_sizes is None:
            self.window_sizes = [
                5,
                10,
                15,
                20,
                30,
                40,
                50
            ]
        else:
            self.window_sizes = window_sizes

    # =====================================================
    # ANALYZE ONE WINDOW SIZE
    # =====================================================

    def analyze_window(self, window_size):

        results = []

        for entity_id, entity_df in (
            self.feature_df.groupby(
                "entity_id"
            )
        ):

            entity_df = (
                entity_df
                .sort_values("timestamp")
                .reset_index(drop=True)
            )

            event_count = len(entity_df)

            # ---------------------------------------------
            # Cannot create a sequence if there aren't
            # enough events
            # ---------------------------------------------

            if event_count < window_size:

                continue

            # ---------------------------------------------
            # Sliding windows
            #
            # E1 ... E10
            # E2 ... E11
            # E3 ... E12
            # ...
            # ---------------------------------------------

            for start in range(
                0,
                event_count - window_size + 1
            ):

                end = (
                    start
                    + window_size
                )

                window = entity_df.iloc[
                    start:end
                ]

                start_time = (
                    window["timestamp"].iloc[0]
                )

                end_time = (
                    window["timestamp"].iloc[-1]
                )

                duration_hours = (
                    (
                        end_time
                        - start_time
                    )
                    .total_seconds()
                    / 3600
                )

                duration_days = (
                    duration_hours / 24
                )

                results.append({

                    "entity_id": entity_id,

                    "window_size":
                        window_size,

                    "start_time":
                        start_time,

                    "end_time":
                        end_time,

                    "duration_hours":
                        duration_hours,

                    "duration_days":
                        duration_days
                })

        return pd.DataFrame(results)

    # =====================================================
    # ANALYZE ALL WINDOW SIZES
    # =====================================================

    def analyze(self):

        all_results = []

        for window_size in (
            self.window_sizes
        ):

            result = (
                self.analyze_window(
                    window_size
                )
            )

            if not result.empty:

                all_results.append(
                    result
                )

        if not all_results:

            return pd.DataFrame()

        return pd.concat(
            all_results,
            ignore_index=True
        )

    # =====================================================
    # SUMMARY
    # =====================================================

    def summary(self):

        results = self.analyze()

        if results.empty:

            print(
                "No windows could be created."
            )

            return

        print("\n")
        print("=" * 90)
        print("SEQUENCE WINDOW ANALYSIS")
        print("=" * 90)

        print(
            "\nThe values below show how much real time "
            "each event-based window covers."
        )

        print(
            "\nWINDOW SIZE = number of consecutive events"
        )

        # -------------------------------------------------
        # Summary by window size
        # -------------------------------------------------

        summary = (
            results
            .groupby("window_size")
            .agg(
                windows=(
                    "window_size",
                    "count"
                ),

                min_days=(
                    "duration_days",
                    "min"
                ),

                median_days=(
                    "duration_days",
                    "median"
                ),

                mean_days=(
                    "duration_days",
                    "mean"
                ),

                max_days=(
                    "duration_days",
                    "max"
                )
            )
            .reset_index()
        )

        print("\n")
        print(
            summary.to_string(
                index=False,
                float_format=lambda x:
                    f"{x:.2f}"
            )
        )

        # -------------------------------------------------
        # Events per entity
        # -------------------------------------------------

        print("\n")
        print("=" * 90)
        print("EVENT COUNT PER ENTITY")
        print("=" * 90)

        entity_counts = (
            self.feature_df
            .groupby("entity_id")
            .size()
        )

        print(
            entity_counts.describe()
        )

        # -------------------------------------------------
        # Number of possible sequences per entity
        # -------------------------------------------------

        print("\n")
        print("=" * 90)
        print("POSSIBLE SEQUENCES PER USER")
        print("=" * 90)

        for window_size in (
            self.window_sizes
        ):

            sequence_counts = []

            for entity_id, entity_df in (
                self.feature_df
                .groupby("entity_id")
            ):

                n = len(entity_df)

                possible = max(
                    0,
                    n - window_size + 1
                )

                sequence_counts.append(
                    possible
                )

            print(
                f"\nWindow {window_size}:"
            )

            print(
                f"  Minimum sequences: "
                f"{min(sequence_counts)}"
            )

            print(
                f"  Median sequences: "
                f"{np.median(sequence_counts):.0f}"
            )

            print(
                f"  Maximum sequences: "
                f"{max(sequence_counts)}"
            )

        # -------------------------------------------------
        # Recommended interpretation
        # -------------------------------------------------

        print("\n")
        print("=" * 90)
        print("INTERPRETATION")
        print("=" * 90)

        print(
            """
Small windows:
Useful for short-term behavioral patterns
such as brute force, credential stuffing,
lateral movement and sudden behavior changes.

Large windows:
Useful for gradual behavioral patterns
such as low-and-slow data exfiltration.

The final window sizes should be selected
after looking at the actual time span above.
"""
        )


# =========================================================
# DIRECT TEST
# =========================================================

if __name__ == "__main__":

    print(
        "SequenceWindowAnalyzer loaded."
    )