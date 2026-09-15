#!/usr/bin/env bash

# Generate a paper-ready LaTeX table from an evaluation-metrics JSON file.
#
# Usage:
#   ./generate_latex_metrics_table.sh metrics.json
#   ./generate_latex_metrics_table.sh metrics.json output.tex

set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
    echo "Usage: $0 METRICS_JSON [OUTPUT_TEX]" >&2
    exit 1
fi

METRICS_JSON="$1"
OUTPUT_TEX="${2:-}"

python3 - "$METRICS_JSON" "$OUTPUT_TEX" <<'PY'
import json
import sys

metrics_path = sys.argv[1]
output_path = sys.argv[2]

with open(metrics_path, encoding="utf-8") as file:
    metrics = json.load(file)


def fmt_error(value):
    value = float(value)

    if value == 0:
        return "0"

    return f"{value:.3e}"


methods = (
    ("ODE", "ode", "inference_speed_ode", None),
    ("Fourier", "fourier", "inference_speed_fourier", None),
    (
        "Fourier + Jacobian correction",
        "fourier_c",
        "inference_speed_fourier",
        "fourier_correction_time",
    ),
    ("FM", "fm", "inference_speed_fm", None),
    (
        "FM + Jacobian correction",
        "c",
        "inference_speed_fm",
        "correction_time",
    ),
)


rows = []

for label, metric_key, inference_key, correction_key in methods:

    inference_time = float(metrics[inference_key])

    if correction_key is not None:
        inference_time += float(metrics[correction_key])

    error_values = (
        metrics[f"ep_{metric_key}"],
        metrics[f"eo_{metric_key}"],
        metrics[f"err_{metric_key}"],
    )

    formatted_errors = " & ".join(
        fmt_error(value)
        for value in error_values
    )

    # Stored times are seconds; display times are milliseconds.
    inference_time_ms = inference_time * 1000.0

    rows.append(
        f"        {label} & {formatted_errors} & "
        f"{inference_time_ms:.2f} "
        + r"\\"
    )


table = "\n".join(
    [
        r"\begin{table*}[t]",
        r"    \centering",
        r"    \caption{Evaluation metrics for the different methods, including Jacobian-corrected variants.}",
        r"    \label{tab:evaluation-metrics}",
        r"    \begin{tabular}{lcccc}",
        r"        \toprule",
        r"        Method & $e_p$ & $e_o$ & $e$ & Inference time (ms) \\",
        r"        \midrule",
        *rows,
        r"        \bottomrule",
        r"    \end{tabular}",
        r"\end{table*}",
    ]
) + "\n"


if output_path:
    with open(output_path, "w", encoding="utf-8") as file:
        file.write(table)
else:
    print(table, end="")
PY
