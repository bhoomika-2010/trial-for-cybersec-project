from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from flask import Flask, jsonify, render_template


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

app = Flask(__name__)


def load_event_data() -> pd.DataFrame:
    train_df = pd.read_csv(DATA_DIR / "train_events.csv", parse_dates=["timestamp"])
    test_df = pd.read_csv(DATA_DIR / "test_events.csv", parse_dates=["timestamp"])
    combined = pd.concat([train_df, test_df], ignore_index=True)
    combined = combined.sort_values(["entity_id", "timestamp"]).reset_index(drop=True)
    return combined


def build_summary() -> dict[str, Any]:
    df = load_event_data()

    total_events = int(len(df))
    anomaly_events = df[df["label"] != "normal"]
    anomaly_count = int(len(anomaly_events))
    anomaly_rate = round((anomaly_count / total_events) * 100, 2) if total_events else 0.0

    label_counts = df["label"].value_counts().to_dict()
    overall_top_labels = [
        {"label": label, "count": int(count)}
        for label, count in sorted(label_counts.items(), key=lambda item: (-item[1], item[0]))
    ]

    entity_risk = (
        anomaly_events.groupby("entity_id").size().reset_index(name="anomaly_count").sort_values("anomaly_count", ascending=False)
    )
    most_affected_entity = (
        {
            "entity_id": str(entity_risk.iloc[0]["entity_id"]),
            "anomaly_count": int(entity_risk.iloc[0]["anomaly_count"]),
        }
        if not entity_risk.empty
        else {"entity_id": "N/A", "anomaly_count": 0}
    )

    recent_anomalies = anomaly_events.sort_values("timestamp", ascending=False).head(12)
    recent_rows = [
        {
            "timestamp": row["timestamp"].strftime("%Y-%m-%d %H:%M:%S"),
            "entity_id": row["entity_id"],
            "label": row["label"],
            "source_ip": row["source_ip"],
            "resource_accessed": row["resource_accessed"],
            "auth_method": row["auth_method"],
            "auth_result": row["auth_result"],
        }
        for _, row in recent_anomalies.iterrows()
    ]

    return {
        "total_events": total_events,
        "anomaly_count": anomaly_count,
        "anomaly_rate": anomaly_rate,
        "label_counts": overall_top_labels,
        "most_affected_entity": most_affected_entity,
        "recent_anomalies": recent_rows,
        "last_updated": pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
    }


@app.route("/")
def index():
    summary = build_summary()
    return render_template("index.html", summary=summary)


@app.route("/api/summary")
def summary_api():
    return jsonify(build_summary())


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
