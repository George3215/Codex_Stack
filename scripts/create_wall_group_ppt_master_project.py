from __future__ import annotations

import csv
import json
import math
import shutil
from dataclasses import dataclass
from html import escape
from pathlib import Path


ROOT = Path(r"D:\MoonStack")
BATCH_ROOT = ROOT / "experiments" / "moon_rock_stack" / "batch_runs"
PROJECT = ROOT / "Asset" / "ppt-master" / "projects" / "moon_wall_group_meeting_ppt169_20260619"
REPORTS = ROOT / "Asset" / "Reports"
SUMMARY_OUT = REPORTS / "MoonRockStack_wall_group_meeting_20260619.summary.json"

W, H = 1280, 720
FONT = "Microsoft YaHei, Arial, sans-serif"

INK = "#202124"
MUTED = "#5f6368"
LIGHT = "#f7f8fa"
LINE = "#d9dde5"
BLUE = "#2457a6"
TEAL = "#0b7a75"
RUST = "#b9542b"
AMBER = "#b88700"
GREEN = "#2f7d32"
RED = "#b3261e"
PURPLE = "#6f4fbf"


@dataclass(frozen=True)
class Case:
    key: str
    label: str
    run: str
    target: str
    strategy: str
    gravity: str
    capture: Path
    primary_image: str
    depth_image: str | None


CASES = [
    Case(
        key="wall3_earth",
        label="3层 wall_segment / Earth / success",
        run="20260616_structured_parallel_smoke",
        target="wall_segment_v1",
        strategy="geometry_bonded",
        gravity="earth",
        capture=BATCH_ROOT
        / "20260616_structured_parallel_smoke"
        / "captures_960x720"
        / "02_wall_segment_v1_success_geometry_bonded_earth_trial_00",
        primary_image="front_rgb.png",
        depth_image="top_depth.png",
    ),
    Case(
        key="wall4_earth",
        label="4层 single-face wall / Earth / strict success",
        run="20260618_single_face_wall_4course_v1_assignment_fallback5_gated_smoke",
        target="single_face_wall_4course_v1",
        strategy="wall_bonded",
        gravity="earth",
        capture=BATCH_ROOT
        / "20260618_single_face_wall_4course_v1_assignment_fallback5_gated_smoke"
        / "captures_960x720"
        / "00_single_face_wall_4course_v1_success_wall_bonded_earth_trial_00",
        primary_image="right_rgb.png",
        depth_image="wall_top_depth.png",
    ),
    Case(
        key="wall4_moon",
        label="4层 single-face wall / Moon / drift failure",
        run="20260618_single_face_wall_4course_v1_assignment_fallback5_gated_smoke",
        target="single_face_wall_4course_v1",
        strategy="wall_bonded",
        gravity="moon",
        capture=BATCH_ROOT
        / "20260618_single_face_wall_4course_v1_assignment_fallback5_gated_smoke"
        / "captures_960x720"
        / "01_single_face_wall_4course_v1_failure_wall_bonded_moon_trial_00",
        primary_image="right_rgb.png",
        depth_image="wall_top_depth.png",
    ),
    Case(
        key="wall5_earth",
        label="5层 tall_wall_v2 / Earth / target success",
        run="20260617_tall_wall_v2_smoke",
        target="tall_wall_v2",
        strategy="wall_bonded",
        gravity="earth",
        capture=BATCH_ROOT
        / "20260617_tall_wall_v2_smoke"
        / "captures_wall_camera_v2_960x720"
        / "00_tall_wall_v2_success_wall_bonded_earth_trial_00",
        primary_image="right_rgb.png",
        depth_image="wall_top_depth.png",
    ),
    Case(
        key="high8_earth",
        label="8层 high wall / Earth / strict failure",
        run="20260618_high_single_face_wall_v1_earth_ranker_top3",
        target="single_face_wall_high_v1",
        strategy="statics_wall",
        gravity="earth",
        capture=BATCH_ROOT
        / "20260618_high_single_face_wall_v1_earth_ranker_top3"
        / "captures_960x720"
        / "00_single_face_wall_high_v1_failure_statics_wall_earth_trial_00",
        primary_image="right_rgb.png",
        depth_image="wall_top_depth.png",
    ),
    Case(
        key="high8_moon",
        label="8层 high wall / Moon / neuralized failure",
        run="20260618_neural_stonefit_pose_high_wall_top8_earth_moon_v1",
        target="single_face_wall_high_v1",
        strategy="statics_wall",
        gravity="moon",
        capture=BATCH_ROOT
        / "20260618_neural_stonefit_pose_high_wall_top8_earth_moon_v1"
        / "captures_960x720"
        / "01_single_face_wall_high_v1_failure_statics_wall_moon_trial_00",
        primary_image="right_rgb.png",
        depth_image="wall_top_depth.png",
    ),
]


def fnum(value: str | None, default: float = 0.0) -> float:
    try:
        return float(value or "")
    except ValueError:
        return default


def inum(value: str | None, default: int = 0) -> int:
    try:
        return int(float(value or ""))
    except ValueError:
        return default


def read_results() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in BATCH_ROOT.rglob("results.csv"):
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                row["_run_dir"] = str(path.parent)
                row["_run_name"] = path.parent.name
                rows.append(row)
    return rows


def is_wall_target(row: dict[str, str]) -> bool:
    target = (row.get("target_name") or "").lower()
    if "wall" not in target:
        return False
    return "pillar" not in target and "column" not in target


def find_case_row(rows: list[dict[str, str]], case: Case) -> dict[str, str]:
    for row in rows:
        if (
            row.get("_run_name") == case.run
            and row.get("target_name") == case.target
            and row.get("strategy") == case.strategy
            and row.get("gravity") == case.gravity
        ):
            return row
    return {}


def summarize(rows: list[dict[str, str]]) -> dict[str, object]:
    wall_rows = [row for row in rows if is_wall_target(row)]
    by_target: dict[str, list[dict[str, str]]] = {}
    for row in wall_rows:
        by_target.setdefault(row.get("target_name", "unknown"), []).append(row)

    target_summary = []
    for target, target_rows in sorted(by_target.items()):
        attempts = len(target_rows)
        successes = sum(inum(row.get("success")) for row in target_rows)
        best_courses = max((fnum(row.get("visible_courses")) for row in target_rows), default=0)
        best_height = max((fnum(row.get("stack_height_m")) for row in target_rows), default=0)
        target_summary.append(
            {
                "target": target,
                "attempts": attempts,
                "successes": successes,
                "success_rate": successes / attempts if attempts else 0,
                "best_visible_courses": best_courses,
                "best_height_m": best_height,
            }
        )

    strategies: dict[str, list[dict[str, str]]] = {}
    for row in wall_rows:
        strategies.setdefault(row.get("strategy", "unknown"), []).append(row)
    strategy_summary = []
    for strategy, strategy_rows in sorted(strategies.items()):
        attempts = len(strategy_rows)
        successes = sum(inum(row.get("success")) for row in strategy_rows)
        strategy_summary.append(
            {
                "strategy": strategy,
                "attempts": attempts,
                "successes": successes,
                "success_rate": successes / attempts if attempts else 0,
            }
        )

    case_rows = {case.key: find_case_row(rows, case) for case in CASES}
    return {
        "total_result_rows": len(rows),
        "wall_target_attempts": len(wall_rows),
        "wall_target_successes": sum(inum(row.get("success")) for row in wall_rows),
        "target_summary": target_summary,
        "strategy_summary": strategy_summary,
        "case_rows": case_rows,
    }


def img_assets() -> dict[str, str]:
    image_dir = PROJECT / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    assets: dict[str, str] = {}
    for case in CASES:
        for suffix, filename in [("view", case.primary_image), ("depth", case.depth_image)]:
            if not filename:
                continue
            src = case.capture / filename
            if not src.exists():
                continue
            dst_name = f"{case.key}_{suffix}{src.suffix.lower()}"
            shutil.copy2(src, image_dir / dst_name)
            assets[f"{case.key}_{suffix}"] = dst_name
    return assets


def tx(text: object) -> str:
    return escape(str(text), quote=True)


def svg_open() -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
        f'<rect x="0" y="0" width="{W}" height="{H}" fill="#ffffff"/>',
    ]


def text(
    x: float,
    y: float,
    body: object,
    size: int = 28,
    fill: str = INK,
    weight: int | str = 400,
    anchor: str = "start",
) -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="{FONT}" font-size="{size}" '
        f'font-weight="{weight}" fill="{fill}" text-anchor="{anchor}">{tx(body)}</text>'
    )


def line_text(x: float, y: float, lines: list[str], size: int = 24, gap: int = 34, fill: str = INK) -> list[str]:
    return [text(x, y + i * gap, line, size=size, fill=fill) for i, line in enumerate(lines)]


def footer(slide_no: int, title: str) -> list[str]:
    return [
        f'<line x1="56" y1="670" x2="1224" y2="670" stroke="{LINE}" stroke-width="1"/>',
        text(56, 700, "MoonStack dry stone walling experiment", 16, MUTED),
        text(1224, 700, f"{slide_no:02d} / {title}", 16, MUTED, anchor="end"),
    ]


def title_block(title: str, subtitle: str | None = None) -> list[str]:
    out = [
        f'<rect x="0" y="0" width="{W}" height="74" fill="{INK}"/>',
        text(56, 48, title, 31, "#ffffff", 700),
    ]
    if subtitle:
        out.append(text(1224, 48, subtitle, 17, "#dfe3ea", 500, anchor="end"))
    return out


def panel(x: float, y: float, w: float, h: float, fill: str = LIGHT, stroke: str = LINE) -> str:
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="{fill}" stroke="{stroke}" stroke-width="1"/>'


def image(asset_name: str, x: float, y: float, w: float, h: float, mode: str = "meet") -> str:
    return (
        f'<image href="../images/{tx(asset_name)}" x="{x}" y="{y}" width="{w}" height="{h}" '
        f'preserveAspectRatio="xMidYMid {mode}"/>'
    )


def metric_card(x: float, y: float, w: float, label: str, value: str, color: str = BLUE) -> list[str]:
    return [
        panel(x, y, w, 96, "#ffffff", LINE),
        text(x + 18, y + 32, label, 17, MUTED, 500),
        text(x + 18, y + 74, value, 34, color, 800),
    ]


def bullets(x: float, y: float, items: list[str], size: int = 23, gap: int = 38, color: str = INK) -> list[str]:
    out: list[str] = []
    for i, item in enumerate(items):
        yy = y + i * gap
        out.append(f'<circle cx="{x}" cy="{yy - 8}" r="4" fill="{TEAL}"/>')
        out.append(text(x + 18, yy, item, size, color))
    return out


def target_wall_diagram(x: float, y: float, courses: list[int], scale: float = 1.0, title: str = "") -> list[str]:
    out = []
    if title:
        out.append(text(x, y - 18, title, 19, MUTED, 600))
    stone_h = 34 * scale
    stone_w = 70 * scale
    colors = ["#ded2b5", "#c9b991", "#bfae86", "#d8c7a0"]
    for ci, count in enumerate(courses):
        yy = y + (len(courses) - ci - 1) * stone_h * 0.88
        xoff = x + (max(courses) - count) * stone_w * 0.34 + (ci % 2) * stone_w * 0.23
        for si in range(count):
            xx = xoff + si * stone_w * 0.86
            phase = (ci * 7 + si * 11) % 9
            pts = [
                (xx + 8 * scale, yy + 7 * scale + phase * 0.2),
                (xx + stone_w - 11 * scale, yy + 4 * scale),
                (xx + stone_w - 3 * scale, yy + stone_h * 0.50),
                (xx + stone_w - 17 * scale, yy + stone_h - 4 * scale),
                (xx + 12 * scale, yy + stone_h - 2 * scale),
                (xx + 2 * scale, yy + stone_h * 0.42),
            ]
            poly = " ".join(f"{px:.1f},{py:.1f}" for px, py in pts)
            out.append(
                f'<polygon points="{poly}" fill="{colors[(ci + si) % len(colors)]}" '
                f'stroke="#7a715e" stroke-width="{1.2 * scale:.1f}"/>'
            )
    out.append(
        f'<line x1="{x - 10}" y1="{y + len(courses) * stone_h * 0.88 + 8}" '
        f'x2="{x + max(courses) * stone_w * 0.90 + 12}" '
        f'y2="{y + len(courses) * stone_h * 0.88 + 8}" stroke="{INK}" stroke-width="2"/>'
    )
    return out


def bar_chart(x: float, y: float, data: list[tuple[str, float, str]], width: float = 460) -> list[str]:
    out = [panel(x, y, width, 270, "#ffffff", LINE)]
    out.append(text(x + 20, y + 34, "按目标统计的成功率", 21, INK, 700))
    max_bar = width - 190
    for i, (label, value, note) in enumerate(data):
        yy = y + 72 + i * 43
        out.append(text(x + 20, yy + 18, label, 16, INK, 600))
        out.append(f'<rect x="{x + 160}" y="{yy}" width="{max_bar}" height="21" fill="#edf0f5"/>')
        out.append(f'<rect x="{x + 160}" y="{yy}" width="{max_bar * min(value, 1):.1f}" height="21" fill="{BLUE if value > 0 else RED}"/>')
        out.append(text(x + 160 + max_bar + 10, yy + 18, note, 15, MUTED))
    return out


def table(x: float, y: float, headers: list[str], rows: list[list[str]], widths: list[int], row_h: int = 36) -> list[str]:
    out: list[str] = []
    total_w = sum(widths)
    out.append(panel(x, y, total_w, row_h * (len(rows) + 1), "#ffffff", LINE))
    out.append(f'<rect x="{x}" y="{y}" width="{total_w}" height="{row_h}" fill="#eef3fb" stroke="{LINE}" stroke-width="1"/>')
    cx = x
    for h, w in zip(headers, widths):
        out.append(text(cx + 10, y + 24, h, 15, INK, 700))
        cx += w
        out.append(f'<line x1="{cx}" y1="{y}" x2="{cx}" y2="{y + row_h * (len(rows) + 1)}" stroke="{LINE}" stroke-width="1"/>')
    for ri, row in enumerate(rows):
        yy = y + row_h * (ri + 1)
        if ri % 2 == 1:
            out.append(f'<rect x="{x}" y="{yy}" width="{total_w}" height="{row_h}" fill="#fafbfc"/>')
        cx = x
        for cell, w in zip(row, widths):
            out.append(text(cx + 10, yy + 24, cell, 15, INK))
            cx += w
        out.append(f'<line x1="{x}" y1="{yy}" x2="{x + total_w}" y2="{yy}" stroke="{LINE}" stroke-width="1"/>')
    return out


def case_metric(row: dict[str, str], field: str, default: str = "-") -> str:
    return row.get(field) or default


def write_slide(name: str, body: list[str], notes: str) -> None:
    svg_path = PROJECT / "svg_output" / name
    note_path = PROJECT / "notes" / f"{Path(name).stem}.md"
    svg_path.parent.mkdir(parents=True, exist_ok=True)
    note_path.parent.mkdir(parents=True, exist_ok=True)
    svg_path.write_text("\n".join(body + ["</svg>\n"]), encoding="utf-8")
    note_path.write_text(notes + "\n", encoding="utf-8")


def pct(successes: int, attempts: int) -> str:
    if attempts == 0:
        return "-"
    return f"{successes / attempts * 100:.1f}%"


def fmt_m(value: str | float | None) -> str:
    val = fnum(str(value) if value is not None else "")
    return f"{val:.3f} m"


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    rows = read_results()
    summary = summarize(rows)
    assets = img_assets()
    case_rows: dict[str, dict[str, str]] = summary["case_rows"]  # type: ignore[assignment]

    target_summary = {
        item["target"]: item for item in summary["target_summary"]  # type: ignore[index]
    }

    def target_stats(target: str) -> tuple[int, int, float, float]:
        item = target_summary.get(target, {})
        return (
            int(item.get("attempts", 0)),
            int(item.get("successes", 0)),
            float(item.get("best_visible_courses", 0)),
            float(item.get("best_height_m", 0)),
        )

    wall_segment = target_stats("wall_segment_v1")
    wall4 = target_stats("single_face_wall_4course_v1")
    tall_wall = target_stats("tall_wall_v2")
    high_wall = target_stats("single_face_wall_high_v1")
    thick_wall = target_stats("tall_wall_thick_v1")

    slides: list[tuple[str, list[str], str]] = []

    # 1
    s = svg_open()
    s += [
        f'<rect x="0" y="0" width="{W}" height="{H}" fill="#101418"/>',
        image(assets["wall4_earth_view"], 0, 0, 1280, 720, "slice"),
        '<rect x="0" y="0" width="1280" height="720" fill="#000000" opacity="0.56"/>',
        text(72, 118, "月面干砌石墙堆叠实验", 48, "#ffffff", 800),
        text(76, 174, "组会汇报 | wall-focused revision | 2026-06-19", 23, "#d8dee9", 500),
        text(76, 250, "重点：2/3/4层墙段、墙体失败模式、下一步网络化位置输出", 28, "#ffffff", 700),
        text(76, 607, "当前结论：4层目标已出现阶段性成功；视觉墙面与高墙稳定性仍未达工程要求。", 24, "#f1c27d", 700),
    ]
    s += footer(1, "cover")
    slides.append(("01_cover.svg", s, "标题页。强调这是 wall-focused 版本，不再以石堆或石柱为主。"))

    # 2
    s = svg_open()
    s += title_block("阶段性结论", "不是石堆，而是可验证的干砌墙目标")
    s += metric_card(70, 120, 250, "墙体相关结果行", str(summary["wall_target_attempts"]), BLUE)
    s += metric_card(350, 120, 250, "严格成功行", str(summary["wall_target_successes"]), GREEN)
    s += metric_card(630, 120, 250, "最高可见层数", "8 courses", AMBER)
    s += metric_card(910, 120, 250, "4层里程碑", "Earth success", TEAL)
    s += bullets(
        90,
        285,
        [
            "本阶段真正完成的是：目标锁定、候选筛选、MuJoCo验证、失败日志与多视角图像链路。",
            "4层墙段在地球重力下通过当前 gate；月球重力下同结构仍有明显侧向漂移。",
            "高层墙能达到 7-8 个可见层，但多数视觉上仍偏核心/堆状，不应宣称为成熟墙。",
            "下一步不再依赖随机尝试：用小网络先做石头角色、候选姿态和风险筛选，再由仿真校验。",
        ],
        size=24,
        gap=50,
    )
    s += footer(2, "status")
    slides.append(("02_status.svg", s, "概览页。给出总尝试、成功、最高层数和最重要限制。"))

    # 3
    s = svg_open()
    s += title_block("论文与工程先验", "把 dry stacking 经验转成可检验的规则")
    s += table(
        64,
        118,
        ["来源", "抽象出的先验", "进入实验的形式"],
        [
            ["Furrer 2017", "在线 next-best object / target pose", "候选姿态生成 + 逐候选 MuJoCo settle"],
            ["Johns 2020", "Autonomous dry stone / 结构化墙体目标", "target slots、分层、墙厚/错缝"],
            ["Liu 2018/2021/2023", "RL/规划/稳定性序列规划", "role assignment、sequence、失败回放"],
            ["Menezes 2021", "Model-free RL dry stacking", "未来策略学习与 reward 设计"],
            ["Hibbeler statics", "重心投影、摩擦、力矩、支承面", "support balance、drift、velocity gate"],
        ],
        [210, 420, 520],
        row_h=52,
    )
    s += bullets(
        84,
        610,
        ["先验只作为假设，不当真理；每条规则都要回到成功率、漂移、可见层数和图像证据。"],
        size=23,
    )
    s += footer(3, "priors")
    slides.append(("03_priors.svg", s, "文献先验页。列出论文和它们如何进入实验。"))

    # 4
    s = svg_open()
    s += title_block("石头生成与分类", "角砾状多面体，不允许平滑边缘、薄片、尖刺")
    s += target_wall_diagram(80, 160, [5, 4, 3], scale=1.05, title="墙体目标需要可错缝、多接触、宽底")
    s += [
        panel(570, 116, 600, 430, "#ffffff", LINE),
        text(600, 160, "当前几何筛选", 27, INK, 800),
    ]
    s += bullets(
        606,
        215,
        [
            "生成：subangular / fractured / wedge / wall_block / tie / cap 等角砾多面体。",
            "拒绝：spike_score > 0.16，flatness > 1.62，short_to_mid < 0.62。",
            "4层墙 catalog：360 个候选，337 个通过，23 个因尖刺先验被拒绝。",
            "聚类：10 类几何簇，再按 base / middle / cap / tie / chock 等角色筛选。",
        ],
        size=22,
        gap=48,
    )
    s += footer(4, "rocks")
    slides.append(("04_rock_generation.svg", s, "石头生成页。强调当前生成器已从平滑/尖刺转向角砾多面体。"))

    # 5
    s = svg_open()
    s += title_block("实验 pipeline", "从结构目标到仿真验证，再到学习数据")
    labels = [
        ("目标墙 slots", "2/3/4/高层课程、错缝、墙厚"),
        ("石头角色筛选", "base / middle / cap / tie"),
        ("assignment plan", "目标槽位和同角色 fallback"),
        ("candidate poses", "yaw / z / support-aware 搜索"),
        ("MuJoCo settle", "Earth 9.80665 / Moon 1.624"),
        ("gate + log", "RMSE、drift、velocity、图像"),
    ]
    x0 = 64
    for i, (a, b) in enumerate(labels):
        x = x0 + i * 198
        s += [
            panel(x, 165, 170, 190, "#ffffff", LINE),
            text(x + 18, 210, f"{i + 1}", 38, BLUE, 900),
            text(x + 18, 255, a, 19, INK, 800),
            text(x + 18, 298, b, 15, MUTED),
        ]
        if i < len(labels) - 1:
            s.append(f'<line x1="{x + 174}" y1="260" x2="{x + 196}" y2="260" stroke="{TEAL}" stroke-width="4"/>')
    s += bullets(
        90,
        470,
        [
            "每次 placement 都保留 candidate / placement / final state / capture，因此失败案例可作为训练负样本。",
            "现在的主瓶颈不是“能不能把石头放上去”，而是“能不能让网络直接给出低风险堆叠位置”。",
        ],
        size=24,
        gap=55,
    )
    s += footer(5, "pipeline")
    slides.append(("05_pipeline.svg", s, "Pipeline 页。说明整个实验链路和学习数据来源。"))

    # 6
    s = svg_open()
    s += title_block("墙体评价规则", "成功必须同时通过统计指标和视觉证据")
    s += table(
        70,
        118,
        ["维度", "当前指标", "为什么重要"],
        [
            ["课程层数", "visible_courses", "只说明有高度，不等于墙面稳定"],
            ["目标几何", "target RMSE / max error", "防止变成散堆或歪柱"],
            ["墙面 footprint", "x/y span、aspect、outlier count", "区分墙段、堆、柱"],
            ["动力学稳定", "drift、velocity、stable_count", "防止刚摆好但仍在滑/翻"],
            ["图像证据", "RGB + depth + top view", "组会必须肉眼可检查"],
        ],
        [190, 300, 650],
        row_h=58,
    )
    s += [
        panel(70, 600, 1140, 45, "#fff6e0", "#e5c86f"),
        text(92, 630, "规则：如果结果高但柱状/堆状，必须记为 wall failure，不算墙成功。", 24, RUST, 800),
    ]
    s += footer(6, "evaluation")
    slides.append(("06_evaluation.svg", s, "评价规则页。避免把石堆当墙。"))

    # 7
    r3 = case_rows["wall3_earth"]
    s = svg_open()
    s += title_block("2层基础段：现有证据与缺口", "独立 2 层批次尚未单独记录")
    s += [
        panel(62, 112, 520, 430, "#ffffff", LINE),
        text(90, 150, "目标示意：base + middle", 24, INK, 800),
    ]
    s += target_wall_diagram(105, 230, [4, 3], scale=1.35)
    s += [
        panel(625, 112, 580, 430, "#ffffff", LINE),
        text(653, 150, "真实图像：3层 wall_segment 的低层基础", 24, INK, 800),
        image(assets["wall3_earth_view"], 650, 175, 530, 330, "meet"),
    ]
    s += bullets(
        82,
        590,
        [
            "该页不把它宣称为独立 2 层成功；它是 3 层 wall_segment 中 base+middle 子结构的证据。",
            "后续需要单独跑 2层 benchmark：固定槽位、记录稳定率、作为高层墙的地基分布训练集。",
        ],
        size=20,
        gap=36,
    )
    s += footer(7, "2-layer foundation")
    slides.append(("07_two_layer_foundation.svg", s, "2层页。明确说明目前只有子结构证据，独立2层批次还要补。"))

    # 8
    s = svg_open()
    s += title_block("3层 wall_segment：结构化堆叠能稳定闭环", "wall_segment_v1")
    attempts, successes, _, best_h = wall_segment
    s += metric_card(70, 112, 220, "attempts", str(attempts), BLUE)
    s += metric_card(315, 112, 220, "success", f"{successes} ({pct(successes, attempts)})", GREEN)
    s += metric_card(560, 112, 220, "best height", f"{best_h:.3f} m", AMBER)
    s += metric_card(805, 112, 220, "selected case", "Earth / success", TEAL)
    s += [
        panel(70, 250, 535, 320, "#ffffff", LINE),
        image(assets["wall3_earth_view"], 85, 270, 505, 275, "meet"),
        text(85, 560, "正交视角：可见 3 层短墙段", 17, MUTED, 600),
        panel(655, 250, 515, 320, "#ffffff", LINE),
        image(assets.get("wall3_earth_depth", assets["wall3_earth_view"]), 670, 270, 485, 275, "meet"),
        text(670, 560, "俯视/深度：footprint 仍偏短，需要更长墙面目标", 17, MUTED, 600),
    ]
    s += footer(8, "3-layer")
    slides.append(("08_three_layer_wall.svg", s, "3层墙段页。展示 wall_segment 的指标和图像。"))

    # 9
    r4e = case_rows["wall4_earth"]
    r4m = case_rows["wall4_moon"]
    s = svg_open()
    s += title_block("4层 single-face wall：阶段性里程碑", "Earth 通过 gate，Moon 仍漂移")
    s += [
        panel(62, 110, 565, 360, "#ffffff", LINE),
        text(88, 148, "Earth | strict success", 24, GREEN, 800),
        image(assets["wall4_earth_view"], 84, 170, 250, 235, "meet"),
        image(assets["wall4_earth_depth"], 350, 170, 250, 235, "meet"),
        text(88, 435, "right/side RGB", 16, MUTED),
        text(350, 435, "top depth", 16, MUTED),
        panel(653, 110, 565, 360, "#ffffff", LINE),
        text(678, 148, "Moon | partial / drift failure", 24, RED, 800),
        image(assets["wall4_moon_view"], 675, 170, 250, 235, "meet"),
        image(assets["wall4_moon_depth"], 940, 170, 250, 235, "meet"),
        text(678, 435, "right/side RGB", 16, MUTED),
        text(940, 435, "top depth", 16, MUTED),
    ]
    s += table(
        92,
        505,
        ["Gravity", "Success", "Placed/stable", "Courses", "Height", "RMSE", "Drift"],
        [
            ["Earth", case_metric(r4e, "success"), f'{case_metric(r4e, "rock_count")}/{case_metric(r4e, "stable_count")}', case_metric(r4e, "visible_courses"), fmt_m(r4e.get("stack_height_m")), fmt_m(r4e.get("target_rmse_xy_m")), fmt_m(r4e.get("max_horizontal_drift_m"))],
            ["Moon", case_metric(r4m, "success"), f'{case_metric(r4m, "rock_count")}/{case_metric(r4m, "stable_count")}', case_metric(r4m, "visible_courses"), fmt_m(r4m.get("stack_height_m")), fmt_m(r4m.get("target_rmse_xy_m")), fmt_m(r4m.get("max_horizontal_drift_m"))],
        ],
        [120, 100, 170, 120, 135, 135, 135],
        row_h=42,
    )
    s += footer(9, "4-layer")
    slides.append(("09_four_layer_wall.svg", s, "4层页。展示地球成功和月球失败对比。"))

    # 10
    r5 = case_rows["wall5_earth"]
    rh8e = case_rows["high8_earth"]
    rh8m = case_rows["high8_moon"]
    s = svg_open()
    s += title_block("更高墙：能到 5-8 层，但还不能称为成熟石墙", "height push")
    s += [
        panel(58, 110, 355, 360, "#ffffff", LINE),
        text(82, 148, "5层 tall_wall_v2", 23, GREEN, 800),
        image(assets["wall5_earth_view"], 78, 170, 315, 220, "meet"),
        text(82, 420, f"success={case_metric(r5, 'success')} | height={fmt_m(r5.get('stack_height_m'))}", 18, INK, 700),
        panel(462, 110, 355, 360, "#ffffff", LINE),
        text(486, 148, "8层 high wall / Earth", 23, RED, 800),
        image(assets["high8_earth_view"], 482, 170, 315, 220, "meet"),
        text(486, 420, f"success={case_metric(rh8e, 'success')} | stable={case_metric(rh8e, 'stable_count')}/{case_metric(rh8e, 'rock_count')}", 18, INK, 700),
        panel(866, 110, 355, 360, "#ffffff", LINE),
        text(890, 148, "8层 high wall / Moon", 23, RED, 800),
        image(assets["high8_moon_view"], 886, 170, 315, 220, "meet"),
        text(890, 420, f"success={case_metric(rh8m, 'success')} | stable={case_metric(rh8m, 'stable_count')}/{case_metric(rh8m, 'rock_count')}", 18, INK, 700),
    ]
    s += bullets(
        82,
        540,
        [
            "5层 tall_wall_v2 有统计成功，但视觉上仍偏短厚核心，下一步要提高墙长/墙面约束。",
            "8层 high wall 是失败案例：可见层数高，但 residual velocity、drift 或 footprint 不合格。",
        ],
        size=22,
        gap=42,
    )
    s += footer(10, "height")
    slides.append(("10_higher_wall.svg", s, "高墙页。强调更高不是成功，不能把堆状结果算墙。"))

    # 11
    s = svg_open()
    s += title_block("统计：哪些策略更接近墙", "成功率不是最终答案，但能指导下一轮")
    bars = [
        ("3层 wall_segment", wall_segment[1] / wall_segment[0] if wall_segment[0] else 0, f"{wall_segment[1]}/{wall_segment[0]}"),
        ("4层 single-face", wall4[1] / wall4[0] if wall4[0] else 0, f"{wall4[1]}/{wall4[0]}"),
        ("5层 tall_wall_v2", tall_wall[1] / tall_wall[0] if tall_wall[0] else 0, f"{tall_wall[1]}/{tall_wall[0]}"),
        ("7层 thick wall", thick_wall[1] / thick_wall[0] if thick_wall[0] else 0, f"{thick_wall[1]}/{thick_wall[0]}"),
        ("8层 high wall", high_wall[1] / high_wall[0] if high_wall[0] else 0, f"{high_wall[1]}/{high_wall[0]}"),
    ]
    s += bar_chart(72, 115, bars, width=520)
    s += [
        panel(650, 115, 530, 270, "#ffffff", LINE),
        text(680, 155, "当前经验", 26, INK, 800),
    ]
    s += bullets(
        685,
        210,
        [
            "短墙段：structure slots + role screening 效果最好。",
            "4层：fallback 修复能提高地球成功；月球要更强 lateral gate。",
            "高墙：单面薄墙抽象不够，必须加入墙厚、tie stone 和低层修复。",
        ],
        size=22,
        gap=47,
    )
    s += [
        panel(72, 435, 1108, 120, "#fff6e0", "#e5c86f"),
        text(100, 480, "重要解释", 23, RUST, 800),
        text(100, 518, "这些成功率来自当前 gate；视觉上不够墙面的结果在下一版指标中会被降级为失败。", 22, INK, 600),
    ]
    s += footer(11, "statistics")
    slides.append(("11_statistics.svg", s, "统计页。给出各目标成功率和经验。"))

    # 12
    s = svg_open()
    s += title_block("失败模式", "典型失败要成为训练数据，而不是被丢掉")
    s += [
        panel(62, 118, 350, 300, "#ffffff", LINE),
        image(assets["wall4_moon_depth"], 82, 145, 310, 210, "meet"),
        text(82, 390, "Moon drift：低重力侧向漂移", 19, RED, 800),
        panel(465, 118, 350, 300, "#ffffff", LINE),
        image(assets["high8_earth_depth"], 485, 145, 310, 210, "meet"),
        text(485, 390, "High wall：可见层高但 footprint 失控", 19, RED, 800),
        panel(868, 118, 350, 300, "#ffffff", LINE),
        image(assets["high8_moon_depth"], 888, 145, 310, 210, "meet"),
        text(888, 390, "Neuralized：局部拟合不等于整体稳定", 19, RED, 800),
    ]
    s += bullets(
        92,
        500,
        [
            "失败数据字段：slot role、stone cluster、candidate pose、support overlap、drift、velocity、depth footprint。",
            "下一轮训练要把这些失败变成 hard negatives，尤其是 Moon lateral drift 和高层支承不足。",
        ],
        size=23,
        gap=45,
    )
    s += footer(12, "failures")
    slides.append(("12_failure_modes.svg", s, "失败模式页。少展示石堆，只用墙体相关失败图。"))

    # 13
    s = svg_open()
    s += title_block("网络化方向", "先组合小网络，再走端到端")
    s += table(
        70,
        112,
        ["小网络", "训练行数", "当前指标", "用途"],
        [
            ["StoneFitNet", "7,909", "F1 0.858", "给 base/middle/cap/tie 选石头"],
            ["PoseAcceptNet", "7,909", "F1 0.858", "拒绝明显坏姿态"],
            ["MoonDriftRiskNet", "7,909", "F1 0.553", "月面侧漂风险，仍需更多负样本"],
            ["CandidatePoseRankNet", "1,193 eval", "top-3 hit 0.964", "把候选姿态从 8+ 降到 3 再仿真"],
        ],
        [230, 150, 210, 550],
        row_h=62,
    )
    s += bullets(
        94,
        565,
        [
            "短期：网络做 pruning + ranking，MuJoCo 做最后物理校验。",
            "长期：加入深度图/点云/局部支撑图，训练直接输出石头-位置-姿态。",
        ],
        size=24,
        gap=45,
    )
    s += footer(13, "learning")
    slides.append(("13_neuralized_direction.svg", s, "网络化方向页。说明不是再随机试，而是用小网络组合降低候选成本。"))

    # 14
    s = svg_open()
    s += title_block("下一步实验设计", "先把墙做像墙，再追求更高")
    s += [
        panel(70, 115, 350, 420, "#ffffff", LINE),
        text(100, 155, "A. 数据补齐", 25, BLUE, 800),
    ]
    s += bullets(105, 215, ["独立 2层 wall benchmark", "所有墙加正交正视 + 俯视深度", "保存每个 slot 的局部高度图"], 22, 50)
    s += [
        panel(465, 115, 350, 420, "#ffffff", LINE),
        text(495, 155, "B. 结构改进", 25, TEAL, 800),
    ]
    s += bullets(500, 215, ["双面/厚墙而非单线墙", "tie stone / hearting / batter", "失败下层 slot 先修复再加高"], 22, 50)
    s += [
        panel(860, 115, 350, 420, "#ffffff", LINE),
        text(890, 155, "C. 学习策略", 25, RUST, 800),
    ]
    s += bullets(895, 215, ["StoneFit + PoseRank + DriftRisk", "hard negative mining", "后续再做端到端世界模型"], 22, 50)
    s += [
        panel(70, 575, 1140, 54, "#eef6f6", "#9fd3ce"),
        text(100, 611, "阶段目标：稳定、可视、可统计的短墙段；不是把石头堆高。", 26, TEAL, 800),
    ]
    s += footer(14, "next")
    slides.append(("14_next_steps.svg", s, "下一步页。提出数据、结构和学习三个方向。"))

    # 15
    s = svg_open()
    s += [
        f'<rect x="0" y="0" width="{W}" height="{H}" fill="{INK}"/>',
        text(80, 125, "Takeaways", 54, "#ffffff", 900),
    ]
    s += line_text(
        92,
        230,
        [
            "1. 4层目标墙段已经形成可复现实验链路；Earth 有当前 gate success。",
            "2. 视觉墙体仍不足：很多高层结果更像短核心/堆状体，不能过度宣称。",
            "3. 月球重力的主要问题是侧向漂移和支承不足，需要专门风险模型。",
            "4. 下一阶段应以 wall-specific data + small networks + MuJoCo verification 为主线。",
        ],
        size=29,
        gap=70,
        fill="#ffffff",
    )
    s += footer(15, "takeaways")
    slides.append(("15_takeaways.svg", s, "总结页。"))

    for name, slide, notes in slides:
        write_slide(name, slide, notes)

    (PROJECT / "spec_lock.md").write_text(
        "\n".join(
            [
                "# MoonStack Wall Group Meeting PPT Spec",
                "",
                "- Format: ppt169, 1280x720.",
                "- Style: group-meeting report; dense data, restrained colors, no marketing hero except cover.",
                "- Emphasis: stone walls, especially 2/3/4-layer discussion.",
                "- Explicit caveat: several high/nominal wall results are not visually mature walls.",
                "- Built for PPT Master SVG pipeline.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (PROJECT / "README.md").write_text(
        "\n".join(
            [
                "# MoonStack Wall Group Meeting PPT",
                "",
                "This ppt-master project was generated from current MoonStack wall experiments.",
                "",
                "Main output target:",
                "",
                "`D:\\MoonStack\\Asset\\Reports\\MoonRockStack_wall_group_meeting_20260619.pptx`",
                "",
                "Source SVG files are under `svg_output/`; copied experiment images are under `images/`.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    summary_payload = {
        "project": str(PROJECT),
        "planned_pptx": str(REPORTS / "MoonRockStack_wall_group_meeting_20260619.pptx"),
        "slides": [name for name, _, _ in slides],
        "assets": assets,
        "summary": summary,
    }
    SUMMARY_OUT.write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"project": str(PROJECT), "slides": len(slides), "summary": str(SUMMARY_OUT)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
