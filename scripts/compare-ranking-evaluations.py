from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any


RATE_METRICS = ("top1", "top2", "strict_f1", "coverage", "dedup")
SAFETY_METRICS = ("top2", "strict_f1", "coverage", "dedup")
EPSILON = 1e-12


def load_summary(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"评测汇总必须是 JSON 对象：{path}")
    return payload


def _nested(payload: dict[str, Any], *keys: str) -> Any:
    value: Any = payload
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            raise ValueError(f"评测汇总缺少字段：{'.'.join(keys)}")
        value = value[key]
    return value


def _finite_rate(payload: dict[str, Any], *keys: str) -> float:
    value = float(_nested(payload, *keys))
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"评测比例字段无效：{'.'.join(keys)}={value}")
    return value


def _coverage_rate(payload: dict[str, Any]) -> float:
    guard = _nested(payload, "engine", "coverage_guard")
    if not isinstance(guard, dict):
        raise ValueError("engine.coverage_guard 必须是对象")
    required = int(guard.get("required_cells", 0))
    unresolved = int(guard.get("unresolved_cells", 0))
    if required < 0 or unresolved < 0 or unresolved > required:
        raise ValueError("人物覆盖单元计数无效")
    return 1.0 if required == 0 else (required - unresolved) / required


def _extract_metrics(payload: dict[str, Any]) -> dict[str, float | int]:
    decisive_closed = int(
        _nested(payload, "eye_evidence_evaluation", "manual_with_decisive_closed")
    )
    if decisive_closed < 0:
        raise ValueError("高可信闭眼误杀数量不能为负数")
    return {
        "top1": _finite_rate(payload, "group_evaluation", "top1_exact_group_hit_rate"),
        "top2": _finite_rate(payload, "group_evaluation", "group_hit_at_2"),
        "strict_f1": _finite_rate(payload, "strict_ai_best", "f1"),
        "coverage": _coverage_rate(payload),
        "dedup": _finite_rate(payload, "grouping_summary", "top1_reduction_rate"),
        "manual_decisive_closed": decisive_closed,
    }


def compare_dataset(
    name: str,
    baseline: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, Any]:
    baseline_metrics = _extract_metrics(baseline)
    candidate_metrics = _extract_metrics(candidate)
    delta = {
        key: float(candidate_metrics[key]) - float(baseline_metrics[key])
        for key in (*RATE_METRICS, "manual_decisive_closed")
    }
    return {
        "name": name,
        "baseline": baseline_metrics,
        "candidate": candidate_metrics,
        "delta": delta,
    }


def _append_check(
    checks: list[dict[str, Any]],
    name: str,
    passed: bool,
    actual: float | int,
    requirement: str,
) -> None:
    checks.append(
        {
            "name": name,
            "passed": bool(passed),
            "actual": actual,
            "requirement": requirement,
        }
    )


def evaluate_gates(mode: str, comparisons: list[dict[str, Any]]) -> dict[str, Any]:
    if mode not in {"development", "holdout"}:
        raise ValueError(f"未知比较模式：{mode}")
    if mode == "development" and len(comparisons) < 3:
        raise ValueError("开发模式至少三个完整活动")
    if mode == "holdout" and len(comparisons) != 1:
        raise ValueError("盲测模式要求恰好一个数据集")

    checks: list[dict[str, Any]] = []
    if mode == "development":
        for comparison in comparisons:
            name = str(comparison["name"])
            top1_delta = float(comparison["delta"]["top1"])
            _append_check(
                checks,
                f"{name}.top1_not_down_more_than_1pp",
                top1_delta + EPSILON >= -0.01,
                top1_delta,
                ">= -0.01",
            )
            for metric in SAFETY_METRICS:
                delta = float(comparison["delta"][metric])
                _append_check(
                    checks,
                    f"{name}.{metric}_not_down_more_than_1pp",
                    delta + EPSILON >= -0.01,
                    delta,
                    ">= -0.01",
                )
            baseline_kills = int(comparison["baseline"]["manual_decisive_closed"])
            candidate_kills = int(comparison["candidate"]["manual_decisive_closed"])
            _append_check(
                checks,
                f"{name}.decisive_eye_kills_not_increased",
                candidate_kills <= baseline_kills,
                candidate_kills - baseline_kills,
                "<= 0",
            )
        mean_top1_delta = sum(float(item["delta"]["top1"]) for item in comparisons) / len(comparisons)
        _append_check(
            checks,
            "development.mean_top1_gain_at_least_3pp",
            mean_top1_delta + EPSILON >= 0.03,
            mean_top1_delta,
            ">= 0.03",
        )
        target_80_met = all(float(item["candidate"]["top1"]) + EPSILON >= 0.80 for item in comparisons)
    else:
        comparison = comparisons[0]
        top1_delta = float(comparison["delta"]["top1"])
        top2_delta = float(comparison["delta"]["top2"])
        _append_check(checks, "holdout.top1_strictly_improved", top1_delta > EPSILON, top1_delta, "> 0")
        _append_check(checks, "holdout.top2_not_decreased", top2_delta + EPSILON >= 0.0, top2_delta, ">= 0")
        for metric in ("strict_f1", "coverage", "dedup"):
            delta = float(comparison["delta"][metric])
            _append_check(
                checks,
                f"holdout.{metric}_not_down_more_than_1pp",
                delta + EPSILON >= -0.01,
                delta,
                ">= -0.01",
            )
        baseline_kills = int(comparison["baseline"]["manual_decisive_closed"])
        candidate_kills = int(comparison["candidate"]["manual_decisive_closed"])
        _append_check(
            checks,
            "holdout.decisive_eye_kills_not_increased",
            candidate_kills <= baseline_kills,
            candidate_kills - baseline_kills,
            "<= 0",
        )
        mean_top1_delta = top1_delta
        target_80_met = float(comparison["candidate"]["top1"]) + EPSILON >= 0.80

    return {
        "mode": mode,
        "passed": all(check["passed"] for check in checks),
        "target_80_met": target_80_met,
        "mean_top1_delta": mean_top1_delta,
        "comparisons": comparisons,
        "checks": checks,
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# 排名 A/B 比较",
        "",
        f"- 模式：{report['mode']}",
        f"- 验收闸门：{'通过' if report['passed'] else '未通过'}",
        f"- Top1 达到 80%：{'是' if report['target_80_met'] else '否'}",
        f"- 平均 Top1 差值：{float(report['mean_top1_delta']):+.2%}",
        "",
        "## 数据集指标",
        "",
        "| 数据集 | 版本 | Top1 | Top2 | 严格 F1 | 人物覆盖 | 去重率 | 闭眼硬误杀 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for comparison in report["comparisons"]:
        for label, key in (("基线", "baseline"), ("候选", "candidate")):
            metrics = comparison[key]
            lines.append(
                f"| {comparison['name']} | {label} | {metrics['top1']:.2%} | {metrics['top2']:.2%} | "
                f"{metrics['strict_f1']:.2%} | {metrics['coverage']:.2%} | {metrics['dedup']:.2%} | "
                f"{metrics['manual_decisive_closed']} |"
            )
    lines.extend(["", "## 验收检查", ""])
    for check in report["checks"]:
        marker = "通过" if check["passed"] else "失败"
        actual = check["actual"]
        actual_text = f"{actual:+.4f}" if isinstance(actual, float) else str(actual)
        lines.append(f"- [{marker}] `{check['name']}`：实际 {actual_text}，要求 {check['requirement']}")
    lines.append("")
    return "\n".join(lines)


def _parse_pair(value: str) -> tuple[str, Path, Path]:
    try:
        name, paths = value.split("=", 1)
        baseline, candidate = paths.split(",", 1)
    except ValueError as error:
        raise argparse.ArgumentTypeError("pair 格式必须为 name=baseline.json,candidate.json") from error
    if not name.strip() or not baseline.strip() or not candidate.strip():
        raise argparse.ArgumentTypeError("pair 的名称和两个路径都不能为空")
    return name.strip(), Path(baseline.strip()), Path(candidate.strip())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="比较新旧组内排序评测并执行固定验收闸门")
    parser.add_argument("--mode", choices=("development", "holdout"), required=True)
    parser.add_argument("--pair", action="append", type=_parse_pair, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        comparisons = [
            compare_dataset(name, load_summary(baseline), load_summary(candidate))
            for name, baseline, candidate in args.pair
        ]
        report = evaluate_gates(args.mode, comparisons)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "comparison.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (args.output_dir / "comparison.md").write_text(render_markdown(report), encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False))
        return 0 if report["passed"] else 2
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"比较失败：{error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
