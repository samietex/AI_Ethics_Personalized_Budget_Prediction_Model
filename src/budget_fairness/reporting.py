from __future__ import annotations

import argparse
import json
from pathlib import Path


def _fmt(x) -> str:
    try:
        return f"{float(x):.4f}"
    except Exception:
        return str(x)


def render_model_card(template: str, payload: dict) -> str:
    run_id = payload.get("run_id", "unknown")

    base_perf = payload.get("baseline", {}).get("metrics", {})
    base_fair = payload.get("baseline", {}).get("fairness", {})
    mit_perf = payload.get("mitigated", {}).get("metrics", {})
    mit_fair = payload.get("mitigated", {}).get("fairness", {})

    mapping = {
        "{{RUN_ID}}": str(run_id),

        "{{BASE_ACC}}": _fmt(base_perf.get("accuracy")),
        "{{BASE_F1}}": _fmt(base_perf.get("f1")),
        "{{BASE_PREC}}": _fmt(base_perf.get("precision")),
        "{{BASE_REC}}": _fmt(base_perf.get("recall")),
        "{{BASE_AUC}}": _fmt(base_perf.get("roc_auc")),

        "{{MIT_ACC}}": _fmt(mit_perf.get("accuracy")),
        "{{MIT_F1}}": _fmt(mit_perf.get("f1")),
        "{{MIT_PREC}}": _fmt(mit_perf.get("precision")),
        "{{MIT_REC}}": _fmt(mit_perf.get("recall")),
        "{{MIT_AUC}}": _fmt(mit_perf.get("roc_auc")),

        "{{BASE_SPD}}": _fmt(base_fair.get("statistical_parity_difference")),
        "{{BASE_DI}}": _fmt(base_fair.get("disparate_impact")),
        "{{BASE_EOD}}": _fmt(base_fair.get("equal_opportunity_difference")),
        "{{BASE_AOD}}": _fmt(base_fair.get("average_odds_difference")),
        "{{BASE_PPV}}": _fmt(base_fair.get("predictive_parity_difference")),

        "{{MIT_SPD}}": _fmt(mit_fair.get("statistical_parity_difference")),
        "{{MIT_DI}}": _fmt(mit_fair.get("disparate_impact")),
        "{{MIT_EOD}}": _fmt(mit_fair.get("equal_opportunity_difference")),
        "{{MIT_AOD}}": _fmt(mit_fair.get("average_odds_difference")),
        "{{MIT_PPV}}": _fmt(mit_fair.get("predictive_parity_difference")),
    }

    out = template
    for k, v in mapping.items():
        out = out.replace(k, v)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Generate Responsible AI docs from last_run.json")
    p.add_argument("--run-json", default="reports/artifacts/last_run.json", help="Path to last_run.json")
    p.add_argument("--template", default="docs/model_card.md", help="Path to model card template")
    p.add_argument("--out", default="docs/model_card.filled.md", help="Output path for filled model card")
    args = p.parse_args()

    run_json = Path(args.run_json)
    template_path = Path(args.template)
    out_path = Path(args.out)

    payload = json.loads(run_json.read_text(encoding="utf-8"))
    template = template_path.read_text(encoding="utf-8")

    rendered = render_model_card(template, payload)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(rendered, encoding="utf-8")

    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
