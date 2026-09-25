"""Assemble the manuscript tables from the shipped data and analysis outputs.

This script only re-formats numbers that already exist in the repository: it
aggregates Day-180 values from the validation files, and reads the per-method
metrics produced by the analysis scripts. It does not recompute any model.

Outputs (written to results/tables/):
    table1_auxiliary_materials.csv / .md
    table2_counterfactual.csv / .md
    table3_methods.csv / .md
"""

import io
import json
import re
import sys
from pathlib import Path

import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data"
RESULTS = REPO / "results"
OUT = RESULTS / "tables"
OUT.mkdir(parents=True, exist_ok=True)


def write_table(name, frame, notes):
    if frame is None or frame.empty:
        print(f"  skipped {name}: no input available (run the analysis scripts first)")
        return
    frame.to_csv(OUT / f"{name}.csv", index=False)
    lines = [f"# {name}", ""]
    header = "| " + " | ".join(frame.columns) + " |"
    sep = "| " + " | ".join("---" for _ in frame.columns) + " |"
    lines += [header, sep]
    for row in frame.itertuples(index=False):
        lines.append("| " + " | ".join(str(v) for v in row) + " |")
    lines += ["", "Notes:", *[f"- {n}" for n in notes]]
    (OUT / f"{name}.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"  wrote {name}.csv and {name}.md")


# ----------------------------------------------------------------------------
# Table 1 - Day-180 metrics of the auxiliary materials
# ----------------------------------------------------------------------------
rows = []
for path in sorted(DATA.glob("validation_*.csv")):
    material = path.stem.replace("validation_", "")
    frame = pd.read_csv(path)
    day180 = frame[frame["Days"] == 180]
    if day180.empty:
        continue
    rows.append(
        {
            "Material": material,
            "CO2 Day 180 (g)": round(float(day180["Sample_CO2_net"].iloc[0]), 2),
            "Degradation (%)": round(float(day180["Degradation_Pct"].iloc[0]), 2),
            "Observations": len(frame),
        }
    )
write_table(
    "table1_auxiliary_materials",
    pd.DataFrame(rows),
    [
        "Day-180 values are read from data/validation_*.csv; each auxiliary material was "
        "tested under a single fixed condition.",
        "Degradation (%) is the value reported in the validation files.",
    ],
)

# ----------------------------------------------------------------------------
# Table 2 - counterfactual validation across test windows
# ----------------------------------------------------------------------------
source = RESULTS / "final_experiments.txt"
text = source.read_text(encoding="utf-8", errors="replace") if source.exists() else ""
window_re = re.compile(r"^(\d+-\d+) -> (\d+-\d+) (.*)$")
groups = {}
order = []
for line in text.splitlines():
    match = window_re.match(line)
    if match:
        key = f"{match.group(1)} / {match.group(2)}"
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(match.group(3))
    else:
        for key in order:
            groups[key].append(line)

rows = []
for key in order:
    body = "\n".join(groups[key])
    persistence = re.search(r"Persistence \(last known\): R2=([-+][\d.]+)", body)
    lsmpr = re.search(r"LS-MPR \(5 features\): R2=([-+][\d.]+)", body)
    if not (persistence and lsmpr):
        continue
    rows.append(
        {
            "Train / Test": key,
            "Persistence R2": persistence.group(1),
            "LS-MPR R2": lsmpr.group(1),
            "Source": "results/final_experiments.txt",
        }
    )
write_table(
    "table2_counterfactual",
    pd.DataFrame(rows),
    [
        "Values are read from a counterfactual-window report in results/, when present.",
        "Only the configurations present in that report are listed.",
    ],
)

# ----------------------------------------------------------------------------
# Table 3 - method comparison under both evaluation protocols
# ----------------------------------------------------------------------------
rows = []
decay_source = RESULTS / "exp_decay_reviewer.json"
if decay_source.exists():
    exp_decay = json.loads(decay_source.read_text(encoding="utf-8"))
    rows.append(
        {
            "Method": "Exp. Decay (mechanistic ref.)",
            "Random-split R2": round(exp_decay["exp_rand"], 3),
            "Temporal-holdout R2": round(exp_decay["exp_temp"], 1),
            "Source": "results/exp_decay_reviewer.json",
        }
    )

comparison_source = RESULTS / "method_comparison_advanced.txt"
if comparison_source.exists():
    report = comparison_source.read_text(encoding="utf-8", errors="replace")
    co2_section = report.split("--- CO2_Release ---", 1)[-1].split("--- Residual_Rate ---", 1)[0]
    for match in re.finditer(
        r"^\s*(LS-MPR \(baseline\)|GPR)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)s",
        co2_section,
        re.MULTILINE,
    ):
        rows.append(
            {
                "Method": match.group(1),
                "Random-split R2": match.group(2),
                "Temporal-holdout R2": "-",
                "Source": "results/method_comparison_advanced.txt",
            }
        )

write_table(
    "table3_methods",
    pd.DataFrame(rows),
    [
        "Random-split values are read from the regenerated analysis outputs; no model is refitted here.",
        "Only methods produced by the included scripts are listed.",
    ],
)

print(f"\nTables written to {OUT}")
