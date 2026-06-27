from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


# Write operator alerts to CSV and JSON, expanding embedded JSON fields.
def write_alerts(alerts: pd.DataFrame, out_dir: Path) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "operator_alerts.csv"
    json_path = out_dir / "operator_alerts.json"
    alerts.to_csv(csv_path, index=False)
    records = alerts.to_dict(orient="records")
    for r in records:
        for key in ("top_features", "metrics_of_interest", "supporting_evidence"):
            if isinstance(r.get(key), str):
                try:
                    r[key] = json.loads(r[key])
                except json.JSONDecodeError:
                    pass
    with open(json_path, "w") as f:
        json.dump(records, f, indent=2, default=str)
    return {"csv": str(csv_path), "json": str(json_path)}
