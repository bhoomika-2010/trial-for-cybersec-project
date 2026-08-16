import random
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from generator.synthetic_generator import Office
from generator.event_generator import EventGenerator
from generator.anomaly_injector import AnomalyInjector


# =========================================================
# CONFIGURATION
# =========================================================

SEED = 42

TRAIN_RATIO = 0.80

# We want the 20% test period to be long enough to contain
# a genuine 15-day low-and-slow attack.
MIN_FULL_TIMELINE_DAYS = 100
MIN_TEST_DAYS = 15

OUTPUT_DIR = Path("data")

TRAIN_FILE = OUTPUT_DIR / "train_events.csv"
TEST_FILE = OUTPUT_DIR / "test_events.csv"
NORMAL_FILE = OUTPUT_DIR / "normal_events.csv"


# More attack episodes in training, fewer in testing.
TRAIN_ATTACK_COUNTS = {
    "brute_force": 25,
    "credential_stuffing": 20,
    "lateral_movement": 18,
    "low_slow_exfiltration": 18,
    "device_spoofing": 15,
    "impossible_travel": 15,
    "privilege_escalation": 15,
}

TEST_ATTACK_COUNTS = {
    "brute_force": 8,
    "credential_stuffing": 7,
    "lateral_movement": 6,
    "low_slow_exfiltration": 5,
    "device_spoofing": 5,
    "impossible_travel": 5,
    "privilege_escalation": 5,
}


# =========================================================
# RANDOM SEED
# =========================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


# =========================================================
# NORMAL TIMELINE GENERATION
# =========================================================

def generate_normal_events():
    """
    Generate normal events for all 100 entities.

    The existing EventGenerator API in the project does not expose
    a timeline-duration argument, so we generate normal chunks and
    place consecutive chunks after one another until the combined
    timeline is at least MIN_FULL_TIMELINE_DAYS.

    This keeps the dataset-builder independent of changes to
    EventGenerator while giving the chronological test split enough
    real time for low-and-slow attacks.
    """

    office = Office()
    entities = office.generate_entities()

    entity_generator = EventGenerator(
        entities,
        office.resources,
    )

    all_events = []

    timeline_start = None
    timeline_end = None

    chunk_number = 0

    while True:

        chunk_events = entity_generator.generate_events()

        if not chunk_events:
            raise RuntimeError(
                "EventGenerator returned no events."
            )

        chunk_times = [
            pd.Timestamp(event["timestamp"])
            for event in chunk_events
        ]

        chunk_start = min(chunk_times)
        chunk_end = max(chunk_times)
        chunk_duration = (
            chunk_end - chunk_start
        )

        if timeline_start is None:
            timeline_start = chunk_start
            shift = timedelta(0)

        else:
            # Place the next generated chunk immediately after the
            # current timeline, with a one-minute separation.
            shift = (
                timeline_end
                + timedelta(minutes=1)
                - chunk_start
            )

        shifted_events = []

        for event in chunk_events:

            event_copy = event.copy()

            event_copy["timestamp"] = (
                pd.Timestamp(event["timestamp"])
                + shift
            ).to_pydatetime()

            shifted_events.append(
                event_copy
            )

        all_events.extend(
            shifted_events
        )

        shifted_times = [
            pd.Timestamp(event["timestamp"])
            for event in shifted_events
        ]

        timeline_end = max(shifted_times)

        chunk_number += 1

        total_days = (
            timeline_end - timeline_start
        ).total_seconds() / 86400

        print(
            f"Generated normal timeline chunk "
            f"{chunk_number}: "
            f"{total_days:.2f} days total"
        )

        if total_days >= MIN_FULL_TIMELINE_DAYS:
            break

        # Safety guard.
        if chunk_number >= 20:
            raise RuntimeError(
                "Could not generate a sufficiently long "
                f"normal timeline after {chunk_number} chunks. "
                f"Current duration: {total_days:.2f} days."
            )

    # Final chronological ordering.
    all_events.sort(
        key=lambda event: (
            event["entity_id"],
            event["timestamp"],
        )
    )

    return office, entities, all_events


# =========================================================
# ENTITY-WISE CHRONOLOGICAL SPLIT
# =========================================================

def chronological_split(events):
    """
    Split every entity's NORMAL timeline independently.

    IMPORTANT:
        The split happens BEFORE anomaly injection.

    This guarantees that attack injection into TRAIN and TEST
    cannot move an attack from one split to the other.
    """

    df = pd.DataFrame(events)

    if df.empty:
        raise ValueError("Cannot split an empty dataset.")

    df["timestamp"] = pd.to_datetime(
        df["timestamp"]
    )

    train_parts = []
    test_parts = []

    for entity_id, entity_df in df.groupby(
        "entity_id",
        sort=False,
    ):

        entity_df = (
            entity_df
            .sort_values(
                "timestamp",
                kind="stable",
            )
            .reset_index(drop=True)
        )

        if len(entity_df) < 2:
            raise ValueError(
                f"{entity_id} has fewer than 2 normal events."
            )

        split_index = int(
            len(entity_df) * TRAIN_RATIO
        )

        split_index = max(
            1,
            min(
                split_index,
                len(entity_df) - 1,
            ),
        )

        train_parts.append(
            entity_df.iloc[:split_index]
        )

        test_parts.append(
            entity_df.iloc[split_index:]
        )

    train_df = pd.concat(
        train_parts,
        ignore_index=True,
    )

    test_df = pd.concat(
        test_parts,
        ignore_index=True,
    )

    train_df = (
        train_df
        .sort_values(
            ["entity_id", "timestamp"],
            kind="stable",
        )
        .reset_index(drop=True)
    )

    test_df = (
        test_df
        .sort_values(
            ["entity_id", "timestamp"],
            kind="stable",
        )
        .reset_index(drop=True)
    )

    return train_df, test_df


# =========================================================
# SPLIT TIME REPORT
# =========================================================

def get_dataset_bounds(df):
    return (
        pd.Timestamp(df["timestamp"].min()).to_pydatetime(),
        pd.Timestamp(df["timestamp"].max()).to_pydatetime(),
    )


def get_entity_bounds(df):
    """
    Return the chronological boundary of every entity inside this
    already-separated dataset.

    This is different from the global dataset min/max. Because TRAIN and
    TEST are split independently per entity, each entity has its own exact
    boundary.
    """
    bounds = {}

    for entity_id, entity_df in df.groupby(
        "entity_id",
        sort=False,
    ):
        bounds[entity_id] = (
            pd.Timestamp(
                entity_df["timestamp"].min()
            ).to_pydatetime(),
            pd.Timestamp(
                entity_df["timestamp"].max()
            ).to_pydatetime(),
        )

    return bounds


def print_split_time_report(train_df, test_df):
    train_start, train_end = get_dataset_bounds(
        train_df
    )

    test_start, test_end = get_dataset_bounds(
        test_df
    )

    train_days = (
        train_end - train_start
    ).total_seconds() / 86400

    test_days = (
        test_end - test_start
    ).total_seconds() / 86400

    print("\n" + "=" * 80)
    print("NORMAL TIMELINE / SPLIT DURATION")
    print("=" * 80)

    print(
        f"\nTRAIN:"
        f"\n  start: {train_start}"
        f"\n  end:   {train_end}"
        f"\n  span:  {train_days:.2f} days"
    )

    print(
        f"\nTEST:"
        f"\n  start: {test_start}"
        f"\n  end:   {test_end}"
        f"\n  span:  {test_days:.2f} days"
    )

    if test_days < MIN_TEST_DAYS:
        raise RuntimeError(
            f"TEST span is only {test_days:.2f} days. "
            f"We require at least {MIN_TEST_DAYS} days "
            "for low-and-slow exfiltration."
        )

    print(
        f"\nTEST duration requirement: "
        f"PASS (>= {MIN_TEST_DAYS} days)"
    )


# =========================================================
# ATTACK INJECTION
# =========================================================

def inject_attacks(
    events_df,
    entities,
    office,
    attack_counts,
    dataset_name,
):
    """
    Inject attacks into one already-separated dataset.

    The injector receives this dataset's own start/end boundaries.
    """

    events = events_df.to_dict(
        orient="records"
    )

    dataset_start, dataset_end = (
        get_dataset_bounds(events_df)
    )

    entity_bounds = get_entity_bounds(
        events_df
    )

    injector = AnomalyInjector(
        events,
        entities,
        office.resources,
        start_time=dataset_start,
        end_time=dataset_end,
        entity_bounds=entity_bounds,
    )

    entity_ids = [
        entity["entity_id"]
        for entity in entities
    ]

    if len(entity_ids) < 2:
        raise RuntimeError(
            "At least two entities are required."
        )

    def select_entities(count):
        if count > len(entity_ids):
            raise ValueError(
                f"{dataset_name}: requested {count} entities, "
                f"but only {len(entity_ids)} exist."
            )

        return random.sample(
            entity_ids,
            count,
        )

    print(
        f"\nInjecting attacks into {dataset_name}..."
    )

    # -----------------------------------------------------
    # Brute force
    # -----------------------------------------------------

    for entity_id in select_entities(
        attack_counts["brute_force"]
    ):
        added = injector.add_brute_force(
            entity_id
        )

        if added == 0:
            raise RuntimeError(
                f"{dataset_name}: brute_force injection "
                f"failed for {entity_id}."
            )

    # -----------------------------------------------------
    # Impossible travel
    # -----------------------------------------------------

    for entity_id in select_entities(
        attack_counts["impossible_travel"]
    ):
        added = injector.add_impossible_travel(
            entity_id
        )

        if added == 0:
            raise RuntimeError(
                f"{dataset_name}: impossible_travel injection "
                f"failed for {entity_id}."
            )

    # -----------------------------------------------------
    # Credential stuffing
    # -----------------------------------------------------

    credential_entities = select_entities(
        attack_counts["credential_stuffing"]
    )

    added = injector.add_credential_stuffing(
        credential_entities
    )

    if added < 2:
        raise RuntimeError(
            f"{dataset_name}: credential_stuffing injection "
            "failed."
        )

    # -----------------------------------------------------
    # Lateral movement
    # -----------------------------------------------------

    for entity_id in select_entities(
        attack_counts["lateral_movement"]
    ):
        added = injector.add_lateral_movement(
            entity_id
        )

        if added == 0:
            raise RuntimeError(
                f"{dataset_name}: lateral_movement injection "
                f"failed for {entity_id}."
            )

    # -----------------------------------------------------
    # Device spoofing
    # -----------------------------------------------------

    for entity_id in select_entities(
        attack_counts["device_spoofing"]
    ):
        added = injector.add_device_spoofing(
            entity_id
        )

        if added == 0:
            raise RuntimeError(
                f"{dataset_name}: device_spoofing injection "
                f"failed for {entity_id}."
            )

    # -----------------------------------------------------
    # Low-and-slow exfiltration
    # -----------------------------------------------------

    # The GLOBAL test span being >= 15 days is not enough.
    # The selected user's own TEST timeline must contain 15 days.
    low_slow_duration = injector.ATTACK_DURATIONS[
        "low_slow_exfiltration"
    ]

    eligible_low_slow_entities = [
        entity_id
        for entity_id in entity_ids
        if injector.can_fit_attack(
            "low_slow_exfiltration",
            entity_id,
        )
    ]

    required_low_slow = attack_counts[
        "low_slow_exfiltration"
    ]

    if len(eligible_low_slow_entities) < required_low_slow:
        raise RuntimeError(
            f"{dataset_name}: only "
            f"{len(eligible_low_slow_entities)} entities "
            f"can fit a 15-day low-and-slow attack; "
            f"{required_low_slow} are required."
        )

    for entity_id in random.sample(
        eligible_low_slow_entities,
        required_low_slow,
    ):
        added = injector.add_low_slow_exfiltration(
            entity_id
        )

        if added == 0:
            raise RuntimeError(
                f"{dataset_name}: low_slow_exfiltration "
                f"injection failed for {entity_id}. "
                "Check that this split spans at least 15 days."
            )

    # -----------------------------------------------------
    # Privilege escalation
    # -----------------------------------------------------

    for entity_id in select_entities(
        attack_counts["privilege_escalation"]
    ):
        added = injector.add_privilege_escalation(
            entity_id
        )

        if added == 0:
            raise RuntimeError(
                f"{dataset_name}: privilege_escalation "
                f"injection failed for {entity_id}."
            )

    result = pd.DataFrame(
        injector.events
    )

    result["timestamp"] = pd.to_datetime(
        result["timestamp"]
    )

    # IMPORTANT:
    # Sort AFTER injection so newly-created attack events
    # are inserted into their correct chronological position.
    result = (
        result
        .sort_values(
            ["entity_id", "timestamp"],
            kind="stable",
        )
        .reset_index(drop=True)
    )

    return result


# =========================================================
# CHRONOLOGICAL LEAKAGE CHECK
# =========================================================

def check_chronology(
    train_df,
    test_df,
):
    problems = []

    train_last = (
        train_df
        .groupby("entity_id")["timestamp"]
        .max()
    )

    test_first = (
        test_df
        .groupby("entity_id")["timestamp"]
        .min()
    )

    for entity_id in train_last.index:

        if entity_id not in test_first.index:
            problems.append(
                (entity_id, "missing_test")
            )
            continue

        if train_last[entity_id] >= test_first[entity_id]:
            problems.append(
                (
                    entity_id,
                    train_last[entity_id],
                    test_first[entity_id],
                )
            )

    return problems


# =========================================================
# ATTACK BOUNDARY CHECK
# =========================================================

def check_attack_boundaries(
    df,
    dataset_name,
):
    """
    Every attack event must remain inside the dataset's own
    min/max timestamp.
    """

    start, end = get_dataset_bounds(df)

    attack_mask = df["label"] != "normal"

    attack_df = df.loc[
        attack_mask
    ]

    invalid = attack_df[
        (attack_df["timestamp"] < start)
        | (attack_df["timestamp"] > end)
    ]

    if not invalid.empty:
        raise RuntimeError(
            f"{dataset_name}: attack timestamp escaped "
            "dataset boundary."
        )


# =========================================================
# LABEL DISTRIBUTION
# =========================================================

def label_distribution(df):
    return (
        df["label"]
        .value_counts()
        .sort_index()
    )


# =========================================================
# VALIDATE ATTACK COVERAGE
# =========================================================

def validate_attack_coverage(
    train_df,
    test_df,
):
    train_counts = train_df["label"].value_counts()
    test_counts = test_df["label"].value_counts()

    print("\n" + "=" * 80)
    print("ATTACK COVERAGE")
    print("=" * 80)

    valid = True

    for attack in TRAIN_ATTACK_COUNTS:

        train_count = int(
            train_counts.get(
                attack,
                0,
            )
        )

        test_count = int(
            test_counts.get(
                attack,
                0,
            )
        )

        print(
            f"{attack:25s}"
            f" train={train_count:4d}"
            f"  test={test_count:4d}"
        )

        if train_count == 0:
            valid = False

        if test_count == 0:
            valid = False

    if not valid:
        raise RuntimeError(
            "At least one attack type is missing from "
            "TRAIN or TEST."
        )


# =========================================================
# FINAL VALIDATION
# =========================================================

def validate_final_dataset(
    train_df,
    test_df,
):
    required_columns = {
        "entity_id",
        "timestamp",
        "label",
    }

    for name, df in (
        ("TRAIN", train_df),
        ("TEST", test_df),
    ):

        missing = (
            required_columns
            - set(df.columns)
        )

        if missing:
            raise RuntimeError(
                f"{name}: missing columns {missing}"
            )

        if df.empty:
            raise RuntimeError(
                f"{name}: dataset is empty."
            )

        if df["timestamp"].isna().any():
            raise RuntimeError(
                f"{name}: missing timestamps."
            )

    # Every original entity must appear in both splits.
    train_entities = set(
        train_df["entity_id"].unique()
    )

    test_entities = set(
        test_df["entity_id"].unique()
    )

    if train_entities != test_entities:
        raise RuntimeError(
            "TRAIN and TEST do not contain the same "
            "entity set."
        )

    # No temporal leakage.
    #
    # This compares the LAST TRAIN event and FIRST TEST event
    # independently for every entity.
    problems = check_chronology(
        train_df,
        test_df,
    )

    if problems:
        raise RuntimeError(
            "Chronological leakage detected:\n"
            + str(problems[:10])
        )

    # Attack events must stay inside each dataset's bounds.
    check_attack_boundaries(
        train_df,
        "TRAIN",
    )

    check_attack_boundaries(
        test_df,
        "TEST",
    )

    # Attack coverage.
    validate_attack_coverage(
        train_df,
        test_df,
    )

    # Both datasets must remain chronologically sorted.
    for name, df in (
        ("TRAIN", train_df),
        ("TEST", test_df),
    ):

        sorted_df = df.sort_values(
            ["entity_id", "timestamp"],
            kind="stable",
        )

        if not df.reset_index(drop=True).equals(
            sorted_df.reset_index(drop=True)
        ):
            raise RuntimeError(
                f"{name}: dataset is not chronologically sorted."
            )

    print("\n" + "=" * 80)
    print("FINAL VALIDATION: PASSED")
    print("=" * 80)


# =========================================================
# REPORT
# =========================================================

def print_report(
    normal_df,
    train_df,
    test_df,
):
    total = len(train_df) + len(test_df)

    print("\n" + "=" * 80)
    print("FINAL DATASET")
    print("=" * 80)

    print("\nNORMAL DATA")
    print("-" * 80)
    print(
        f"Events:   {len(normal_df)}"
    )
    print(
        f"Entities: {normal_df['entity_id'].nunique()}"
    )

    normal_start, normal_end = get_dataset_bounds(
        normal_df
    )

    print(
        f"Start:    {normal_start}"
    )
    print(
        f"End:      {normal_end}"
    )

    print("\nDATASET SIZE AFTER INJECTION")
    print("-" * 80)
    print(
        f"Total events:  {total}"
    )
    print(
        f"Training:      {len(train_df)} "
        f"({len(train_df) / total * 100:.2f}%)"
    )
    print(
        f"Testing:       {len(test_df)} "
        f"({len(test_df) / total * 100:.2f}%)"
    )

    print("\nTRAINING LABEL DISTRIBUTION")
    print("-" * 80)
    print(
        label_distribution(
            train_df
        )
    )

    print("\nTEST LABEL DISTRIBUTION")
    print("-" * 80)
    print(
        label_distribution(
            test_df
        )
    )

    print("\nENTITY COUNTS")
    print("-" * 80)
    print(
        "Training entities:",
        train_df["entity_id"].nunique(),
    )
    print(
        "Testing entities:",
        test_df["entity_id"].nunique(),
    )

    print("\nCHRONOLOGICAL CHECK")
    print("-" * 80)

    problems = check_chronology(
        train_df,
        test_df,
    )

    if problems:
        print(
            "ERROR:",
            problems[:10],
        )
    else:
        print(
            "No chronological leakage detected."
        )


# =========================================================
# MAIN
# =========================================================

def main():

    set_seed(SEED)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 80)
    print("BUILDING SYNTHETIC CYBERSECURITY DATASET")
    print("=" * 80)

    # -----------------------------------------------------
    # 1. Generate LONG normal timelines
    # -----------------------------------------------------

    office, entities, normal_events = (
        generate_normal_events()
    )

    normal_df = pd.DataFrame(
        normal_events
    )

    normal_df["timestamp"] = pd.to_datetime(
        normal_df["timestamp"]
    )

    normal_df = (
        normal_df
        .sort_values(
            ["entity_id", "timestamp"],
            kind="stable",
        )
        .reset_index(drop=True)
    )

    normal_df.to_csv(
        NORMAL_FILE,
        index=False,
    )

    # -----------------------------------------------------
    # 2. Split NORMAL data first
    # -----------------------------------------------------

    train_df, test_df = chronological_split(
        normal_events
    )

    print_split_time_report(
        train_df,
        test_df,
    )

    # -----------------------------------------------------
    # 3. Inject attacks separately
    # -----------------------------------------------------

    train_df = inject_attacks(
        train_df,
        entities,
        office,
        TRAIN_ATTACK_COUNTS,
        "TRAIN",
    )

    test_df = inject_attacks(
        test_df,
        entities,
        office,
        TEST_ATTACK_COUNTS,
        "TEST",
    )

    # -----------------------------------------------------
    # 4. Final validation
    # -----------------------------------------------------

    validate_final_dataset(
        train_df,
        test_df,
    )

    # -----------------------------------------------------
    # 5. Save
    # -----------------------------------------------------

    train_df.to_csv(
        TRAIN_FILE,
        index=False,
    )

    test_df.to_csv(
        TEST_FILE,
        index=False,
    )

    # -----------------------------------------------------
    # 6. Report
    # -----------------------------------------------------

    print_report(
        normal_df,
        train_df,
        test_df,
    )

    print("\nSaved:")
    print(f"  {NORMAL_FILE}")
    print(f"  {TRAIN_FILE}")
    print(f"  {TEST_FILE}")


if __name__ == "__main__":
    main()