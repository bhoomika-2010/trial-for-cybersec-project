import pandas as pd
import ast
from datetime import datetime


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv("data/anomalous_events.csv")

print("=" * 70)
print("DATASET VALIDATION")
print("=" * 70)

print(f"\nTotal events: {len(df)}")


# ============================================================
# 1. LABEL VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("1. LABEL VALIDATION")
print("=" * 70)

expected_labels = {
    "normal",
    "brute_force",
    "impossible_travel",
    "credential_stuffing",
    "lateral_movement",
    "device_spoofing",
    "low_slow_exfiltration",
    "privilege_escalation"
}

actual_labels = set(df["label"].dropna().unique())

print("Labels found:")
for label in sorted(actual_labels):
    print(f"  {label}")

unexpected_labels = actual_labels - expected_labels

if unexpected_labels:
    print("\n❌ Unexpected labels:", unexpected_labels)
else:
    print("\n✅ All labels are valid")


# ============================================================
# 2. MISSING VALUE CHECK
# ============================================================

print("\n" + "=" * 70)
print("2. MISSING VALUE CHECK")
print("=" * 70)

required_columns = [
    "entity_id",
    "timestamp",
    "source_ip",
    "geo_location",
    "auth_method",
    "auth_result",
    "device_id",
    "mac_address",
    "os_version",
    "label"
]

for column in required_columns:

    missing = df[column].isna().sum()

    if missing == 0:
        print(f"✅ {column}: no missing values")
    else:
        print(f"⚠️ {column}: {missing} missing values")


# ============================================================
# 3. BRUTE FORCE
# ============================================================

print("\n" + "=" * 70)
print("3. BRUTE FORCE VALIDATION")
print("=" * 70)

brute_force = df[
    df["label"] == "brute_force"
]

if len(brute_force) == 0:

    print("⚠️ No brute-force events found")

else:

    failures = (
        brute_force["auth_result"] == "failure"
    ).all()

    no_resources = (
        brute_force["resource_accessed"].isna()
    ).all()

    if failures:
        print("✅ All brute-force events are authentication failures")
    else:
        print("❌ Some brute-force events are successful")

    if no_resources:
        print("✅ Brute-force events have no resource access")
    else:
        print("❌ Some brute-force events access resources")


# ============================================================
# 4. CREDENTIAL STUFFING
# ============================================================

print("\n" + "=" * 70)
print("4. CREDENTIAL STUFFING VALIDATION")
print("=" * 70)

credential = df[
    df["label"] == "credential_stuffing"
]

if len(credential) == 0:

    print("⚠️ No credential-stuffing events found")

else:

    unique_ips = credential["source_ip"].nunique()
    unique_entities = credential["entity_id"].nunique()

    print("Unique attacker IPs:", unique_ips)
    print("Unique targeted entities:", unique_entities)

    if unique_ips == 1:
        print("✅ All credential-stuffing events use the same source IP")
    else:
        print("❌ Credential-stuffing events use multiple source IPs")

    if unique_entities > 1:
        print("✅ Multiple entities are targeted")
    else:
        print("❌ Only one entity is targeted")

    successes = credential[
        credential["auth_result"] == "success"
    ]

    failures = credential[
        credential["auth_result"] == "failure"
    ]

    print("Successful attempts:", len(successes))
    print("Failed attempts:", len(failures))

    if len(successes) > 0:
        print("✅ Successful credential-stuffing attempts exist")

        resources_present = (
            successes["resource_accessed"].notna()
        ).all()

        if resources_present:
            print("✅ Successful attempts access a resource")
        else:
            print("❌ Successful attempts missing resource access")


# ============================================================
# 5. IMPOSSIBLE TRAVEL
# ============================================================

print("\n" + "=" * 70)
print("5. IMPOSSIBLE TRAVEL VALIDATION")
print("=" * 70)

impossible = df[
    df["label"] == "impossible_travel"
].copy()

if len(impossible) == 0:

    print("⚠️ No impossible-travel events found")

else:

    print("Events:", len(impossible))

    success = (
        impossible["auth_result"] == "success"
    ).all()

    if success:
        print("✅ All impossible-travel events are successful logins")
    else:
        print("❌ Some impossible-travel events are not successful")

    unique_locations = impossible[
        "geo_location"
    ].nunique()

    print("Different locations:", unique_locations)

    if unique_locations > 1:
        print("✅ Multiple geographic locations detected")
    else:
        print("⚠️ Only one geographic location detected")


# ============================================================
# 6. DEVICE SPOOFING
# ============================================================

print("\n" + "=" * 70)
print("6. DEVICE SPOOFING VALIDATION")
print("=" * 70)

spoofing = df[
    df["label"] == "device_spoofing"
]

if len(spoofing) == 0:

    print("⚠️ No device-spoofing events found")

else:

    print("Events:", len(spoofing))

    print(
        "Unique devices:",
        spoofing["device_id"].nunique()
    )

    print(
        "Unique MAC addresses:",
        spoofing["mac_address"].nunique()
    )

    print(
        "Unique source IPs:",
        spoofing["source_ip"].nunique()
    )

    print("⚠️ Detailed comparison with each user's baseline should be checked against entities")


# ============================================================
# 7. LOW-SLOW EXFILTRATION
# ============================================================

print("\n" + "=" * 70)
print("7. LOW-SLOW EXFILTRATION VALIDATION")
print("=" * 70)

exfiltration = df[
    df["label"] == "low_slow_exfiltration"
].copy()

if len(exfiltration) == 0:

    print("⚠️ No low-slow exfiltration events found")

else:

    exfiltration["timestamp"] = pd.to_datetime(
        exfiltration["timestamp"]
    )

    for entity_id, group in exfiltration.groupby(
        "entity_id"
    ):

        group = group.sort_values("timestamp")

        first_date = group["timestamp"].min()
        last_date = group["timestamp"].max()

        days = (
            last_date - first_date
        ).days

        print(
            f"{entity_id}: "
            f"{len(group)} events, "
            f"spread across {days} days"
        )

    if exfiltration["data_downloaded"].min() >= 15:
        print("✅ Download amounts are within injected range")
    else:
        print("⚠️ Some download values are below expected range")


# ============================================================
# 8. LATERAL MOVEMENT
# ============================================================

print("\n" + "=" * 70)
print("8. LATERAL MOVEMENT VALIDATION")
print("=" * 70)

lateral = df[
    df["label"] == "lateral_movement"
].copy()

if len(lateral) == 0:

    print("⚠️ No lateral-movement events found")

else:

    print("Events:", len(lateral))

    print(
        "Entities affected:",
        lateral["entity_id"].nunique()
    )

    print(
        "Resources accessed:",
        lateral["resource_accessed"].nunique()
    )

    print(
        "Source IPs:",
        lateral["source_ip"].nunique()
    )

    if lateral["resource_accessed"].notna().all():
        print("✅ All lateral-movement events access resources")
    else:
        print("❌ Some lateral-movement events have no resource")


# ============================================================
# 9. PRIVILEGE ESCALATION
# ============================================================

print("\n" + "=" * 70)
print("9. PRIVILEGE ESCALATION VALIDATION")
print("=" * 70)

privilege = df[
    df["label"] == "privilege_escalation"
]

if len(privilege) == 0:

    print("⚠️ No privilege-escalation events found")

else:

    print("Events:", len(privilege))

    successful = (
        privilege["auth_result"] == "success"
    ).all()

    if successful:
        print("✅ All privilege-escalation events are successful")
    else:
        print("❌ Some privilege-escalation events are failures")

    if privilege["resource_accessed"].notna().all():
        print("✅ All privilege-escalation events access resources")
    else:
        print("❌ Some privilege-escalation events have no resource")


# ============================================================
# 10. TIMESTAMP VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("10. TIMESTAMP VALIDATION")
print("=" * 70)

df["timestamp"] = pd.to_datetime(
    df["timestamp"]
)

if df["timestamp"].is_monotonic_increasing:
    print("✅ Dataset is globally sorted by timestamp")
else:
    print("❌ Dataset is NOT sorted by timestamp")


# ============================================================
# 11. FINAL LABEL DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("11. LABEL DISTRIBUTION")
print("=" * 70)

print(
    df["label"].value_counts()
)


# ============================================================
# DONE
# ============================================================

print("\n" + "=" * 70)
print("VALIDATION COMPLETE")
print("=" * 70)