import json
import random
import numpy as np
import pandas as pd

from generator.synthetic_generator import Office
from detector.feature_extractor import FeatureExtractor
from detector.feature_dataset import FeatureDatasetBuilder
from detector.feature_scaler import FeatureScaler
from detector.sequence_generator import SequenceGenerator
from detector.lstm_classifier import LSTMClassifier


# =========================================================
# CONFIGURATION
# =========================================================

TRAIN_PATH = "data/train_events.csv"
TEST_PATH = "data/test_events.csv"

USER_BASELINE_PATH = "data/user_baselines.json"
DEPARTMENT_BASELINE_PATH = "data/department_baselines.json"

SMALL_WINDOW = 10
LARGE_WINDOW = 50

RANDOM_SEED = 42

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# =========================================================
# 1. LOAD DATASET
# =========================================================

print("\n")
print("=" * 80)
print("LOADING DATASET")
print("=" * 80)

train_events_df = pd.read_csv(
    TRAIN_PATH,
    parse_dates=["timestamp"]
)

test_events_df = pd.read_csv(
    TEST_PATH,
    parse_dates=["timestamp"]
)

print(
    f"\nTraining events: {len(train_events_df)}"
)

print(
    f"Testing events:  {len(test_events_df)}"
)

print(
    f"Training entities: "
    f"{train_events_df['entity_id'].nunique()}"
)

print(
    f"Testing entities: "
    f"{test_events_df['entity_id'].nunique()}"
)


# =========================================================
# 2. BASIC DATASET VALIDATION
# =========================================================

print("\n")
print("=" * 80)
print("DATASET VALIDATION")
print("=" * 80)


# ---------------------------------------------------------
# Entity coverage
# ---------------------------------------------------------

train_entities = set(
    train_events_df["entity_id"]
)

test_entities = set(
    test_events_df["entity_id"]
)

if train_entities != test_entities:

    print(
        "\nWARNING: Training and testing "
        "entity sets differ."
    )

    print(
        "Missing from TEST:",
        train_entities - test_entities
    )

    print(
        "Missing from TRAIN:",
        test_entities - train_entities
    )

else:

    print(
        "\nAll entities appear in "
        "both TRAIN and TEST."
    )


# ---------------------------------------------------------
# Raw labels
# ---------------------------------------------------------

print("\nTRAIN LABEL DISTRIBUTION")
print("-" * 80)

print(
    train_events_df["label"]
    .value_counts()
)


print("\nTEST LABEL DISTRIBUTION")
print("-" * 80)

print(
    test_events_df["label"]
    .value_counts()
)


# ---------------------------------------------------------
# Required attack labels
# ---------------------------------------------------------

required_attack_labels = {
    "brute_force",
    "credential_stuffing",
    "lateral_movement",
    "low_slow_exfiltration",
    "device_spoofing",
    "impossible_travel",
    "privilege_escalation"
}


train_attack_labels = set(
    train_events_df["label"]
) & required_attack_labels

test_attack_labels = set(
    test_events_df["label"]
) & required_attack_labels


missing_train = (
    required_attack_labels
    - train_attack_labels
)

missing_test = (
    required_attack_labels
    - test_attack_labels
)


if missing_train:

    raise RuntimeError(
        "TRAINING DATA IS MISSING ATTACKS:\n"
        + "\n".join(
            sorted(missing_train)
        )
    )


if missing_test:

    raise RuntimeError(
        "TESTING DATA IS MISSING ATTACKS:\n"
        + "\n".join(
            sorted(missing_test)
        )
    )


print(
    "\nAll attack types exist in "
    "both TRAIN and TEST."
)


# =========================================================
# 3. LOAD THE SAME BASELINES USED DURING DATASET BUILDING
# =========================================================

print("\n")
print("=" * 80)
print("LOADING BASELINES")
print("=" * 80)


with open(
    USER_BASELINE_PATH,
    "r"
) as file:

    user_baselines = json.load(
        file
    )


with open(
    DEPARTMENT_BASELINE_PATH,
    "r"
) as file:

    department_baselines = json.load(
        file
    )


print(
    "User baselines:",
    len(user_baselines)
)

print(
    "Department baselines:",
    len(department_baselines)
)


# =========================================================
# 4. RECREATE ENTITY / RESOURCE INFORMATION
# =========================================================
#
# We are NOT generating a new dataset.
#
# We only need the entity/resource definitions required
# by FeatureExtractor.
# =========================================================

office = Office()

entities = office.generate_entities()

print(
    "Entities:",
    len(entities)
)

print(
    "Resources:",
    len(office.resources)
)


# =========================================================
# 5. COMBINE TRAIN + TEST FOR FEATURE EXTRACTION
# =========================================================
#
# IMPORTANT
#
# We extract features from the complete chronological
# timeline first.
#
# Why?
#
# Suppose:
#
# TRAIN:
# Aug ---------------- Nov
#
# TEST:
#                         Nov -------- Dec
#
# An event near the beginning of TEST can legitimately
# depend on the user's previous TRAIN history.
#
# Therefore:
#
#     TRAIN + TEST
#             ↓
#       FeatureExtractor
#             ↓
#       feature dataset
#             ↓
#       split back into
#       TRAIN / TEST
#
# This prevents the beginning of TEST from losing its
# historical context.
# =========================================================

combined_events_df = pd.concat(
    [
        train_events_df,
        test_events_df
    ],
    ignore_index=True
)

combined_events_df = (
    combined_events_df
    .sort_values(
        ["entity_id", "timestamp"]
    )
    .reset_index(drop=True)
)


print("\n")
print("=" * 80)
print("COMBINED TIMELINE")
print("=" * 80)

print(
    "Total events:",
    len(combined_events_df)
)


# =========================================================
# 6. CREATE FEATURE EXTRACTOR
# =========================================================

events = (
    combined_events_df
    .to_dict("records")
)


extractor = FeatureExtractor(
    events,
    entities,
    user_baselines,
    department_baselines
)


feature_builder = FeatureDatasetBuilder(
    extractor
)


# =========================================================
# 7. BUILD FEATURES
# =========================================================

print("\n")
print("=" * 80)
print("BUILDING FEATURES")
print("=" * 80)


feature_df = (
    feature_builder.build()
)


clean_feature_df = (
    feature_builder.clean()
)


feature_builder.validation_report()


# =========================================================
# 8. IDENTIFY FEATURE COLUMNS
# =========================================================

feature_columns = (
    feature_builder.feature_columns
)


print("\n")
print("=" * 80)
print("FINAL ML FEATURES")
print("=" * 80)

print(
    "Number of features:",
    len(feature_columns)
)

for i, column in enumerate(
    feature_columns,
    start=1
):

    print(
        f"{i:2d}. {column}"
    )


# =========================================================
# 9. SPLIT FEATURES BACK INTO TRAIN / TEST
# =========================================================
#
# We DO NOT run ChronologicalSplitter here.
#
# The dataset was already split by build_dataset.py.
#
# We identify each event using:
#
#     entity_id + timestamp
#
# and use those keys to recover the original split.
# =========================================================

train_keys = set(
    zip(
        train_events_df["entity_id"],
        train_events_df["timestamp"]
    )
)

test_keys = set(
    zip(
        test_events_df["entity_id"],
        test_events_df["timestamp"]
    )
)


feature_keys = list(
    zip(
        clean_feature_df["entity_id"],
        pd.to_datetime(
            clean_feature_df["timestamp"]
        )
    )
)


train_mask = np.array([
    key in train_keys
    for key in feature_keys
])


test_mask = np.array([
    key in test_keys
    for key in feature_keys
])


train_df = (
    clean_feature_df[
        train_mask
    ]
    .copy()
    .reset_index(drop=True)
)


test_df = (
    clean_feature_df[
        test_mask
    ]
    .copy()
    .reset_index(drop=True)
)


print("\n")
print("=" * 80)
print("FEATURE DATASET SPLIT")
print("=" * 80)

print(
    "Training feature events:",
    len(train_df)
)

print(
    "Testing feature events:",
    len(test_df)
)


# ---------------------------------------------------------
# Check that nothing disappeared
# ---------------------------------------------------------

if len(train_df) != len(
    train_events_df
):

    raise RuntimeError(
        "TRAIN feature count does not "
        "match TRAIN event count.\n"
        f"Raw TRAIN: {len(train_events_df)}\n"
        f"Features:  {len(train_df)}"
    )


if len(test_df) != len(
    test_events_df
):

    raise RuntimeError(
        "TEST feature count does not "
        "match TEST event count.\n"
        f"Raw TEST: {len(test_events_df)}\n"
        f"Features:  {len(test_df)}"
    )


print(
    "TRAIN/TEST feature split is correct."
)


# =========================================================
# 10. CHECK LABELS AFTER FEATURE EXTRACTION
# =========================================================

print("\n")
print("=" * 80)
print("LABELS AFTER FEATURE EXTRACTION")
print("=" * 80)


print("\nTRAIN")
print(
    train_df["label"]
    .value_counts()
)


print("\nTEST")
print(
    test_df["label"]
    .value_counts()
)


# =========================================================
# 11. SCALE FEATURES
# =========================================================

print("\n")
print("=" * 80)
print("SCALING FEATURES")
print("=" * 80)


scaler = FeatureScaler(
    feature_columns
)


# ---------------------------------------------------------
# FIT ONLY ON TRAIN
# ---------------------------------------------------------

train_scaled_df = (
    scaler.fit_transform_train(
        train_df
    )
)


# ---------------------------------------------------------
# TRANSFORM TEST USING TRAIN SCALER
# ---------------------------------------------------------

test_scaled_df = (
    scaler.transform(
        test_df
    )
)


scaler.validation_report(
    train_scaled_df,
    test_scaled_df
)


# =========================================================
# 12. CREATE SEQUENCE GENERATOR
# =========================================================

sequence_generator = SequenceGenerator(
    feature_columns=feature_columns,
    small_window=SMALL_WINDOW,
    large_window=LARGE_WINDOW
)


# =========================================================
# 13. CREATE SMALL 10-EVENT SEQUENCES
# =========================================================

print("\n")
print("=" * 80)
print("CREATING 10-EVENT SEQUENCES")
print("=" * 80)


(
    X_train_small,
    y_train_small,
    meta_train_small
) = sequence_generator.create_small_sequences(
    train_scaled_df
)


(
    X_test_small,
    y_test_small,
    meta_test_small
) = sequence_generator.create_small_sequences(
    test_scaled_df
)


# =========================================================
# 14. SEQUENCE VALIDATION
# =========================================================

sequence_generator.validation_report(
    X_train_small,
    y_train_small,
    meta_train_small,
    "small training"
)


sequence_generator.validation_report(
    X_test_small,
    y_test_small,
    meta_test_small,
    "small testing"
)


# =========================================================
# 15. CRITICAL ATTACK COVERAGE CHECK
# =========================================================
#
# This is the diagnostic we just added.
#
# The raw TEST dataset may contain an attack, but if the
# sequence generator does not produce a target sequence
# for that event, the LSTM cannot learn/evaluate it.
# =========================================================

print("\n")
print("=" * 80)
print("FINAL SEQUENCE ATTACK COVERAGE CHECK")
print("=" * 80)


train_sequence_labels = set(
    y_train_small
)

test_sequence_labels = set(
    y_test_small
)


print("\nTRAIN SEQUENCE LABELS:")
print(
    pd.Series(
        y_train_small
    ).value_counts()
)


print("\nTEST SEQUENCE LABELS:")
print(
    pd.Series(
        y_test_small
    ).value_counts()
)


missing_train_sequence_attacks = (
    required_attack_labels
    - train_sequence_labels
)


missing_test_sequence_attacks = (
    required_attack_labels
    - test_sequence_labels
)


if missing_train_sequence_attacks:

    print(
        "\nWARNING:"
    )

    print(
        "Attacks missing from TRAIN sequences:"
    )

    for label in sorted(
        missing_train_sequence_attacks
    ):

        print(
            " -",
            label
        )


if missing_test_sequence_attacks:

    print(
        "\nWARNING:"
    )

    print(
        "Attacks missing from TEST sequences:"
    )

    for label in sorted(
        missing_test_sequence_attacks
    ):

        print(
            " -",
            label
        )


# =========================================================
# STOP BEFORE LSTM IF TEST COVERAGE IS BROKEN
# =========================================================
#
# We deliberately do NOT train the model if an attack
# disappears from the TEST sequence targets.
#
# Otherwise accuracy/reporting would be misleading.
# =========================================================

if missing_test_sequence_attacks:

    raise RuntimeError(
        "\nSome attack classes disappeared "
        "during sequence generation.\n\n"
        "The raw dataset contains these attacks, "
        "but the generated TEST sequences do not.\n"
        "Fix SequenceGenerator before training "
        "the LSTM."
    )


# =========================================================
# 16. BUILD LSTM CLASSIFIER
# =========================================================

from detector.lstm_classifier import (
    LSTMClassifier
)


print("\n")
print("=" * 80)
print("BUILDING LSTM CLASSIFIER")
print("=" * 80)


lstm_classifier = LSTMClassifier(

    input_shape=(
        X_train_small.shape[1],
        X_train_small.shape[2]
    ),

    labels=y_train_small
)


# =========================================================
# 17. ENCODE LABELS
# =========================================================

(
    y_train_encoded,
    y_test_encoded
) = lstm_classifier.encode_labels(
    y_train_small,
    y_test_small
)


# =========================================================
# 18. LABEL MAPPING
# =========================================================

print("\n")
print("LABEL ENCODING")
print("-" * 80)


for i, label in enumerate(
    lstm_classifier.label_encoder.classes_
):

    print(
        f"{label} -> {i}"
    )


# =========================================================
# 19. CLASS WEIGHTS
# =========================================================

class_weights = (
    lstm_classifier
    .calculate_class_weights(
        y_train_encoded
    )
)


print("\n")
print("CLASS WEIGHTS")
print("-" * 80)


for class_id, weight in (
    class_weights.items()
):

    label = (
        lstm_classifier
        .label_encoder
        .inverse_transform(
            [class_id]
        )[0]
    )

    print(
        f"{label}: {weight:.4f}"
    )


# =========================================================
# 20. MODEL SUMMARY
# =========================================================

lstm_classifier.summary()


# =========================================================
# 21. TRAIN LSTM
# =========================================================

print("\n")
print("=" * 80)
print("TRAINING LSTM")
print("=" * 80)


history = (
    lstm_classifier.train(

        X_train_small,

        y_train_encoded,

        class_weights,

        validation_split=0.15,

        epochs=30,

        batch_size=64
    )
)


# =========================================================
# 22. TEST LSTM
# =========================================================

print("\n")
print("=" * 80)
print("TESTING LSTM")
print("=" * 80)


(
    predictions,
    probabilities,
    confusion
) = lstm_classifier.evaluate(

    X_test_small,

    y_test_encoded
)


# =========================================================
# 23. FINAL LABEL CHECK INSIDE EXTRACTOR
# =========================================================

print("\n")
print("=" * 80)
print("LABELS INSIDE FEATURE EXTRACTOR")
print("=" * 80)


label_counts = {}


for entity_id, entity_events in (
    extractor.events_by_entity.items()
):

    for event in entity_events:

        label = event["label"]

        label_counts[label] = (
            label_counts.get(
                label,
                0
            ) + 1
        )


for label, count in (
    sorted(
        label_counts.items()
    )
):

    print(
        f"{label}: {count}"
    )


# =========================================================
# DONE
# =========================================================

print("\n")
print("=" * 80)
print("PIPELINE COMPLETE")
print("=" * 80)