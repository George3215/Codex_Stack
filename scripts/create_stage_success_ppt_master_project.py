# -*- coding: utf-8 -*-
"""Build a polished PPT Master deck for MoonStack stage success reporting.

This creates a standalone ppt-master project with SVG pages, copied image
assets, speaker notes, spec files, and a summary JSON. Export is handled by
PPT Master's own finalize/svg_to_pptx tools.
"""

from __future__ import annotations

import json
import math
import shutil
from html import escape
from pathlib import Path
from textwrap import wrap


ROOT = Path(r"D:\MoonStack")
PPT_MASTER = ROOT / "Asset" / "ppt-master"
PROJECT = PPT_MASTER / "projects" / "moon_wall_stage_success_lora_ppt169_20260622"
REPORTS = ROOT / "Asset" / "Reports"
OLD_PROJECT = PPT_MASTER / "projects" / "moon_wall_group_meeting_ppt169_20260619"
SCRIPT_SUMMARY = REPORTS / "MoonRockStack_wall_stage_success_lora_pptmaster_20260622.summary.json"

W, H = 1280, 720
FONT = "Microsoft YaHei, Arial, sans-serif"

COLORS = {
    # Based on ppt169_lora_hu_2021: academic blueprint, white canvas,
    # deep technical blue, warm orange highlights, restrained borders.
    "bg": "#FFFFFF",
    "paper": "#FFFFFF",
    "secondary_bg": "#F4F7FA",
    "ink": "#1D2733",
    "muted": "#5B6776",
    "tertiary": "#8A94A1",
    "line": "#D8DEE6",
    "blue": "#1B3A5C",
    "blue2": "#3E7CB1",
    "teal": "#3E7CB1",
    "green": "#2E7D32",
    "rust": "#E8743B",
    "amber": "#C88719",
    "red": "#C62828",
    "purple": "#6B5FB5",
    "dark": "#0F2238",
    "moon": "#C9BFA8",
    "sand": "#FBF0E9",
    "slate": "#8A94A1",
}


ASSET_SOURCES = {
    "seed602_front.png": ROOT
    / "experiments/moon_rock_stack/batch_runs/20260621_course3net_upperheuristic_4to5_moon_candidates5_seed602_v1/captures_960x720_20260622/01_single_face_wall_4course_v1_failure_statics_wall_moon_trial_00/wall_front_rgb.png",
    "seed602_top_depth.png": ROOT
    / "experiments/moon_rock_stack/batch_runs/20260621_course3net_upperheuristic_4to5_moon_candidates5_seed602_v1/captures_960x720_20260622/01_single_face_wall_4course_v1_failure_statics_wall_moon_trial_00/wall_top_depth.png",
    "seed604_front.png": ROOT
    / "experiments/moon_rock_stack/batch_runs/20260622_course3net_upperheuristic_4to5_moon_candidates5_seed604_w050_v1/captures_960x720_20260622/00_single_face_wall_4course_v1_failure_statics_wall_moon_trial_00/wall_front_rgb.png",
    "seed604_top_depth.png": ROOT
    / "experiments/moon_rock_stack/batch_runs/20260622_course3net_upperheuristic_4to5_moon_candidates5_seed604_w050_v1/captures_960x720_20260622/00_single_face_wall_4course_v1_failure_statics_wall_moon_trial_00/wall_top_depth.png",
    "role_success_failure_v17.png": REPORTS / "ppt_assets_20260622_stage_success_trends/role_success_failure_v17.png",
    "course_success_failure_v17.png": REPORTS / "ppt_assets_20260622_stage_success_trends/course_success_failure_v17.png",
    "network_metrics_v15_v16.png": REPORTS / "ppt_assets_20260622_stage_success_trends/network_metrics_v15_v16.png",
    "run_tradeoffs_20260622.png": REPORTS / "ppt_assets_20260622_stage_success_trends/run_tradeoffs_20260622.png",
    "wall3_earth_view.png": OLD_PROJECT / "images/wall3_earth_view.png",
    "wall3_earth_depth.png": OLD_PROJECT / "images/wall3_earth_depth.png",
    "wall4_earth_view.png": OLD_PROJECT / "images/wall4_earth_view.png",
    "wall4_earth_depth.png": OLD_PROJECT / "images/wall4_earth_depth.png",
    "wall4_moon_view.png": OLD_PROJECT / "images/wall4_moon_view.png",
    "wall4_moon_depth.png": OLD_PROJECT / "images/wall4_moon_depth.png",
    "wall5_earth_view.png": OLD_PROJECT / "images/wall5_earth_view.png",
    "wall5_earth_depth.png": OLD_PROJECT / "images/wall5_earth_depth.png",
}


def tx(value: object) -> str:
    return escape(str(value), quote=True)


def rgb(name: str) -> str:
    return COLORS[name]


def svg_open(bg: str = "bg") -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
        f'<rect x="0" y="0" width="{W}" height="{H}" fill="{rgb(bg)}"/>',
    ]


def group_start(gid: str) -> str:
    return f'<g id="{tx(gid)}">'


def group_end() -> str:
    return "</g>"


def text(
    x: float,
    y: float,
    body: object,
    size: int = 24,
    fill: str = "ink",
    weight: int | str = 400,
    anchor: str = "start",
    opacity: float | None = None,
) -> str:
    op = f' opacity="{opacity:.2f}"' if opacity is not None else ""
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="{FONT}" font-size="{size}" '
        f'font-weight="{weight}" fill="{rgb(fill) if fill in COLORS else fill}" '
        f'text-anchor="{anchor}"{op}>{tx(body)}</text>'
    )


def multiline(
    x: float,
    y: float,
    lines: list[str],
    size: int = 22,
    fill: str = "ink",
    weight: int | str = 400,
    gap: int | None = None,
    anchor: str = "start",
) -> list[str]:
    gap = gap if gap is not None else int(size * 1.42)
    return [text(x, y + i * gap, line, size, fill, weight, anchor) for i, line in enumerate(lines)]


def wrapped(
    x: float,
    y: float,
    body: str,
    width_chars: int,
    size: int = 22,
    fill: str = "ink",
    weight: int | str = 400,
    gap: int | None = None,
    max_lines: int | None = None,
) -> list[str]:
    lines = wrap(body, width=width_chars, break_long_words=False, replace_whitespace=False)
    if max_lines is not None:
        lines = lines[:max_lines]
    return multiline(x, y, lines, size=size, fill=fill, weight=weight, gap=gap)


def rect(
    x: float,
    y: float,
    w: float,
    h: float,
    fill: str = "paper",
    stroke: str = "line",
    sw: float = 1.0,
    rx: float = 0,
    opacity: float | None = None,
) -> str:
    op = f' opacity="{opacity:.2f}"' if opacity is not None else ""
    rx_attr = f' rx="{rx}" ry="{rx}"' if rx else ""
    return (
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" '
        f'fill="{rgb(fill) if fill in COLORS else fill}" stroke="{rgb(stroke) if stroke in COLORS else stroke}" '
        f'stroke-width="{sw:.1f}"{rx_attr}{op}/>'
    )


def line(x1: float, y1: float, x2: float, y2: float, stroke: str = "line", sw: float = 1.0, opacity: float | None = None) -> str:
    op = f' opacity="{opacity:.2f}"' if opacity is not None else ""
    return f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{rgb(stroke)}" stroke-width="{sw:.1f}"{op}/>'


def circle(x: float, y: float, r: float, fill: str = "blue", stroke: str | None = None, sw: float = 1) -> str:
    st = f' stroke="{rgb(stroke)}" stroke-width="{sw}"' if stroke else ""
    return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r:.1f}" fill="{rgb(fill)}"{st}/>'


def image(name: str, x: float, y: float, w: float, h: float, mode: str = "meet", opacity: float | None = None) -> str:
    op = f' opacity="{opacity:.2f}"' if opacity is not None else ""
    return (
        f'<image href="../images/{tx(name)}" x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" '
        f'preserveAspectRatio="xMidYMid {mode}"{op}/>'
    )


def title(slide_no: int, title_text: str, subtitle: str = "", section: str = "MoonStack") -> list[str]:
    out = [
        group_start("title"),
        rect(64, 56, 6, 56, "rust", "rust", 0, 3),
        text(86, 90, title_text, 32, "blue", 800),
        text(88, 120, section.upper(), 14, "tertiary", 800),
    ]
    if subtitle:
        out += wrapped(88, 146, subtitle, 68, size=14, fill="muted", gap=21, max_lines=2)
    out += [group_end()]
    return out


def footer(slide_no: int) -> list[str]:
    return [
        group_start("footer"),
        text(64, 700, "MoonStack dry-stone wall stacking · MuJoCo · neural data flywheel", 11, "tertiary", 500),
        text(1216, 700, f"{slide_no:02d}", 11, "tertiary", 600, "end"),
        group_end(),
    ]


def metric(x: float, y: float, label: str, value: str, note: str, color: str = "blue", w: float = 220) -> list[str]:
    return [
        group_start(f"metric-{label[:10]}"),
        rect(x, y, w, 118, "paper", "line", 1, 12),
        rect(x, y, 7, 118, color, color, 0, 12),
        text(x + 22, y + 36, label, 16, "muted", 600),
        text(x + 22, y + 78, value, 32, color, 800),
        text(x + 22, y + 102, note, 12, "muted", 500),
        group_end(),
    ]


def pill(x: float, y: float, label: str, fill: str = "blue", width: float = 116) -> list[str]:
    return [
        rect(x, y, width, 30, fill, fill, 0, 15),
        text(x + width / 2, y + 21, label, 13, "#FFFFFF", 700, "middle"),
    ]


def bullet_list(x: float, y: float, items: list[str], size: int = 22, gap: int = 42, accent: str = "blue", width_chars: int = 40) -> list[str]:
    out: list[str] = [group_start("bullets")]
    yy = y
    for i, item in enumerate(items):
        out.append(circle(x, yy - 7, 5, accent))
        lines = wrap(item, width=width_chars, break_long_words=False, replace_whitespace=False)
        for j, ln in enumerate(lines[:3]):
            out.append(text(x + 20, yy + j * (size + 6), ln, size, "ink", 500))
        yy += gap + max(0, len(lines[:3]) - 1) * (size + 6)
    out.append(group_end())
    return out


def small_table(x: float, y: float, headers: list[str], rows: list[list[str]], widths: list[int], row_h: int = 38, font_size: int = 15) -> list[str]:
    out: list[str] = [group_start("table")]
    total_w = sum(widths)
    total_h = row_h * (len(rows) + 1)
    out.append(rect(x, y, total_w, total_h, "paper", "line", 1, 8))
    out.append(rect(x, y, total_w, row_h, "#E9F0FF", "#E9F0FF", 0, 8))
    cx = x
    for head, w in zip(headers, widths):
        out.append(text(cx + 12, y + 25, head, font_size, "ink", 800))
        cx += w
        if cx < x + total_w - 1:
            out.append(line(cx, y, cx, y + total_h, "line", 1))
    for ri, row in enumerate(rows):
        yy = y + row_h * (ri + 1)
        if ri % 2 == 0:
            out.append(rect(x, yy, total_w, row_h, "#FAFCFF", "#FAFCFF", 0, 0))
        cx = x
        for cell, w in zip(row, widths):
            clipped = str(cell)
            out.append(text(cx + 12, yy + 25, clipped, font_size, "ink", 500))
            cx += w
        out.append(line(x, yy, x + total_w, yy, "line", 1))
    out.append(group_end())
    return out


def arrow(x1: float, y1: float, x2: float, y2: float, color: str = "muted") -> list[str]:
    angle = math.atan2(y2 - y1, x2 - x1)
    head = 10
    p1 = (x2 - head * math.cos(angle - 0.45), y2 - head * math.sin(angle - 0.45))
    p2 = (x2 - head * math.cos(angle + 0.45), y2 - head * math.sin(angle + 0.45))
    return [
        line(x1, y1, x2, y2, color, 2),
        f'<polygon points="{x2:.1f},{y2:.1f} {p1[0]:.1f},{p1[1]:.1f} {p2[0]:.1f},{p2[1]:.1f}" fill="{rgb(color)}"/>',
    ]


def rock_wall(x: float, y: float, courses: list[int], scale: float = 1.0, drift: float = 0.0) -> list[str]:
    out: list[str] = [group_start("rock-wall")]
    stone_w = 78 * scale
    stone_h = 38 * scale
    palette = ["#C9BFA8", "#B8AA8B", "#E7D8B4", "#9A907A", "#D7C6A0"]
    max_count = max(courses)
    for ci, count in enumerate(courses):
        yy = y + (len(courses) - ci - 1) * stone_h * 0.88
        xoff = x + (max_count - count) * stone_w * 0.35 + (ci % 2) * stone_w * 0.26 + drift * ci
        for si in range(count):
            xx = xoff + si * stone_w * 0.84
            phase = (ci * 13 + si * 7) % 11
            pts = [
                (xx + 8 * scale, yy + 8 * scale + phase * 0.15),
                (xx + stone_w - 12 * scale, yy + 5 * scale),
                (xx + stone_w - 2 * scale, yy + stone_h * 0.50),
                (xx + stone_w - 17 * scale, yy + stone_h - 5 * scale),
                (xx + 13 * scale, yy + stone_h - 2 * scale),
                (xx + 2 * scale, yy + stone_h * 0.43),
            ]
            poly = " ".join(f"{px:.1f},{py:.1f}" for px, py in pts)
            out.append(f'<polygon points="{poly}" fill="{palette[(ci + si) % len(palette)]}" stroke="#6B6253" stroke-width="{1.3 * scale:.1f}"/>')
            out.append(line(xx + 16 * scale, yy + 13 * scale, xx + stone_w - 17 * scale, yy + stone_h - 10 * scale, "slate", 0.5, 0.35))
    out.append(line(x - 14, y + len(courses) * stone_h * 0.90 + 10, x + max_count * stone_w * 0.9 + 30, y + len(courses) * stone_h * 0.90 + 10, "slate", 2))
    out.append(group_end())
    return out


def copy_assets() -> list[str]:
    img_dir = PROJECT / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    for dst_name, src in ASSET_SOURCES.items():
        if src.exists():
            shutil.copy2(src, img_dir / dst_name)
            copied.append(dst_name)
    return copied


def write_project_files() -> None:
    PROJECT.mkdir(parents=True, exist_ok=True)
    for sub in ["svg_output", "notes", "images", "sources", "templates", "exports"]:
        (PROJECT / sub).mkdir(parents=True, exist_ok=True)
    design = f"""# MoonStack Stage Success Deck Design Spec

## I. Purpose
Group-meeting academic report for MoonStack dry-stone wall stacking. The deck emphasizes staged success, failure-to-success data, rule utility, neural-network integration, and layer-wise difficulty.

## II. Format
- Canvas: ppt169, 1280x720.
- Audience: supervisor and lab group.
- Tone: rigorous, scientific, data-forward, visually polished.

## III. Visual System
- Reference: `ppt169_lora_hu_2021`
- Style: academic technical blueprint, white canvas, deep blue header, warm orange highlights, dense but readable data.
- Background: {COLORS['bg']}
- Ink: {COLORS['ink']}
- Accent colors: deep academic blue, secondary blue, warm orange, success green.
- Typography: Microsoft YaHei for Chinese report readability.
- Layout: full SVG pages, large evidence figures, KPI cards, timeline rails, structured comparisons.

## IV. Key Message
The current milestone is not a fully solved wall; it is a reproducible data flywheel and a 4-layer stage-structure case that exposes the next research problem: wall-line preservation under lunar gravity.
"""
    lock = f"""# Execution Lock

## canvas
- viewBox: 0 0 1280 720
- format: PPT 16:9

## colors
- bg: {COLORS['bg']}
- secondary_bg: {COLORS['secondary_bg']}
- primary: {COLORS['blue']}
- accent: {COLORS['rust']}
- secondary_accent: {COLORS['blue2']}
- purple: {COLORS['purple']}
- amber: {COLORS['amber']}
- text: {COLORS['ink']}
- text_secondary: {COLORS['muted']}
- text_tertiary: {COLORS['tertiary']}
- border: {COLORS['line']}
- success: {COLORS['green']}
- warning: {COLORS['red']}
- tint_primary: #EAF0F6
- tint_accent: #FBF0E9
- tint_success: #EAF6EE
- cover_dark: {COLORS['dark']}
- cover_text: #D9E6FF
- cover_subtitle: #E6ECF5
- cover_text_muted: #AAB7CA
- black: #000000
- blueprint_light: #BFD3FF
- white: #FFFFFF
- table_header: #E9F0FF
- table_alt: #FAFCFF
- aqua_tint: #EAF8F6
- aqua_border: #A7DCD7
- amber_tint: #FFF7E6
- amber_border: #F2C879
- red_tint: #FFF0F0
- red_border: #F3B4B4
- rust_border: #F2C9B2
- stone_1: #C9BFA8
- stone_2: #B8AA8B
- stone_3: #E7D8B4
- stone_4: #9A907A
- stone_5: #D7C6A0
- stone_line: #6B6253
- pale_blue_text: #D6E2FF
- conclusion_text: #C8D2E0
- seed_warning: #FFCBA4
- image_rendering: blueprint
- image_palette: cool-corporate

## typography
- font_family: Microsoft YaHei, Arial, sans-serif
- code_family: Consolas, "Courier New", monospace
- body: 18
- cover_title: 60
- hero_number: 56
- title: 32
- subtitle: 24
- annotation: 14
- chart_annotation: 13
- footnote: 11

## images
- seed602_front: images/seed602_front.png
- seed602_top_depth: images/seed602_top_depth.png
- seed604_front: images/seed604_front.png
- seed604_top_depth: images/seed604_top_depth.png
- role_success_failure_v17: images/role_success_failure_v17.png
- course_success_failure_v17: images/course_success_failure_v17.png
- network_metrics_v15_v16: images/network_metrics_v15_v16.png
- run_tradeoffs_20260622: images/run_tradeoffs_20260622.png
- wall3_earth_view: images/wall3_earth_view.png
- wall3_earth_depth: images/wall3_earth_depth.png
- wall4_earth_view: images/wall4_earth_view.png
- wall4_earth_depth: images/wall4_earth_depth.png
- wall4_moon_view: images/wall4_moon_view.png
- wall4_moon_depth: images/wall4_moon_depth.png

## page_rhythm
- P01: anchor
- P02: dense
- P03: breathing
- P04: dense
- P05: dense
- P06: dense
- P07: breathing
- P08: dense
- P09: dense
- P10: anchor
- P11: anchor
- P12: dense
- P13: dense
- P14: dense
- P15: dense
- P16: dense
- P17: dense
- P18: breathing
- P19: dense
- P20: breathing
- P21: anchor

## forbidden
- Mixing icon libraries
- rgba()
- `<style>`, `class`, `<foreignObject>`, `textPath`, `@font-face`, `<animate*>`, `<script>`, `<iframe>`, `<symbol>`+`<use>`
- `<g opacity>` (set opacity on each child element individually)
- HTML named entities in text (`&nbsp;`, `&mdash;`, `&copy;` …)
"""
    readme = """# MoonStack Stage Success PPT Master Deck

This project was generated for the 2026-06-22 MoonStack group-meeting update.
Use PPT Master's pipeline:

```powershell
conda run -n moon-rock-stack python D:\\MoonStack\\Asset\\ppt-master\\skills\\ppt-master\\scripts\\svg_quality_checker.py D:\\MoonStack\\Asset\\ppt-master\\projects\\moon_wall_stage_success_lora_ppt169_20260622
conda run -n moon-rock-stack python D:\\MoonStack\\Asset\\ppt-master\\skills\\ppt-master\\scripts\\total_md_split.py D:\\MoonStack\\Asset\\ppt-master\\projects\\moon_wall_stage_success_lora_ppt169_20260622
conda run -n moon-rock-stack python D:\\MoonStack\\Asset\\ppt-master\\skills\\ppt-master\\scripts\\finalize_svg.py D:\\MoonStack\\Asset\\ppt-master\\projects\\moon_wall_stage_success_lora_ppt169_20260622
conda run -n moon-rock-stack python D:\\MoonStack\\Asset\\ppt-master\\skills\\ppt-master\\scripts\\svg_to_pptx.py D:\\MoonStack\\Asset\\ppt-master\\projects\\moon_wall_stage_success_lora_ppt169_20260622
```
"""
    (PROJECT / "design_spec.md").write_text(design, encoding="utf-8")
    (PROJECT / "spec_lock.md").write_text(lock, encoding="utf-8")
    (PROJECT / "README.md").write_text(readme, encoding="utf-8")


def slide_cover() -> tuple[str, list[str], str]:
    s = svg_open("dark")
    s += [
        image("seed602_front.png", 0, 0, 1280, 720, "slice"),
        rect(0, 0, 1280, 720, "#000000", "#000000", 0, 0, 0.62),
        rect(0, 0, 1280, 720, "dark", "dark", 0, 0, 0.28),
        group_start("cover-title"),
        text(76, 104, "MoonStack", 22, "#BFD3FF", 800),
        text(76, 172, "月面干砌石墙堆叠实验", 50, "#FFFFFF", 800),
        text(80, 224, "阶段性成功、负样本数据飞轮与神经网络化路线", 27, "#E6ECF5", 600),
        line(80, 266, 560, 266, "blue2", 4),
        text(80, 620, "组会汇报版 | 2026-06-22 | PPT Master SVG deck", 20, "#D9E6FF", 600),
        group_end(),
        group_start("cover-kpi"),
        rect(780, 94, 360, 286, "#FFFFFF", "#FFFFFF", 0, 18, 0.88),
        text(810, 142, "阶段判断", 18, "blue", 800),
        text(810, 188, "4 层结构已出现", 35, "ink", 800),
        text(810, 232, "但 strict wall success 尚未稳定", 20, "rust", 700),
        text(810, 288, "下一步：先提升 3-4 层闭环成功率，", 18, "ink", 500),
        text(810, 320, "再推进 5 层以上高墙。", 18, "ink", 500),
        group_end(),
    ]
    note = "封面页：说明本报告不是宣称完全解决，而是汇报阶段性结构案例、数据飞轮和神经网络化路线。"
    return "01_cover.svg", s, note


def slide_summary() -> tuple[str, list[str], str]:
    s = svg_open()
    s += title(2, "本阶段要向老师说明的四件事", "把“失败很多”转化成科学实验价值，而不是只展示漂亮图片。", "Executive summary")
    s += metric(70, 200, "run examples", "41", "完整实验记录", "blue", 250)
    s += metric(350, 200, "placements", "1061", "逐石放置样本", "teal", 250)
    s += metric(630, 200, "candidate poses", "34836", "候选位姿样本", "purple", 250)
    s += metric(910, 200, "Moon positives", "592", "placement 正样本", "green", 250)
    s += [
        rect(70, 380, 1110, 185, "paper", "line", 1, 16),
        text(100, 427, "汇报主线", 22, "blue", 800),
    ]
    s += bullet_list(
        110,
        475,
        [
            "阶段性成功：seed602 出现 4 层竖向结构，但 drift 未过严格阈值。",
            "失败样本价值：middle/cap 失败集中，是训练网络替代启发式的核心监督来源。",
            "规则筛选：支承面、重心、墙线目标有效；随机、单纯 settling、单纯低 drift 不足。",
            "网络路线：先用小网络接管前三层候选排序，再发展到 wall-state / future-support critic。",
        ],
        size=20,
        gap=36,
        width_chars=52,
    )
    s += footer(2)
    note = "概览页：用 41/1061/34836/592 这些数字建立数据规模，同时说明阶段成功和严格成功的区别。"
    return "02_executive_summary.svg", s, note


def slide_target() -> tuple[str, list[str], str]:
    s = svg_open()
    s += title(3, "研究目标：从月面路标到干砌结构", "短期做可识别、可复现的小型路标墙；长期目标是月面石材干砌建筑。", "Research target")
    s += [
        rect(78, 190, 330, 330, "paper", "line", 1, 18),
        text(110, 236, "短期目标", 22, "blue", 800),
    ]
    s += rock_wall(118, 330, [4, 3, 3], 0.78, 0)
    s += multiline(110, 455, ["2-4 层可复现墙段", "可作为月面路标/标识结构"], 18, "ink", 600, gap=30)
    s += [
        rect(475, 190, 330, 330, "paper", "line", 1, 18),
        text(507, 236, "中期目标", 22, "teal", 800),
    ]
    s += rock_wall(510, 304, [5, 4, 4, 3, 3], 0.72, 4)
    s += multiline(507, 455, ["5-6 层单面墙", "提高成功率而非偶然冲高"], 18, "ink", 600, gap=30)
    s += [
        rect(872, 190, 330, 330, "paper", "line", 1, 18),
        text(904, 236, "长期目标", 22, "rust", 800),
    ]
    s += rock_wall(905, 285, [6, 5, 5, 4, 4, 3], 0.62, 3)
    s += multiline(904, 455, ["干砌石屋/防护墙", "石头选择 + 放置策略 + 物理验证"], 18, "ink", 600, gap=30)
    s += footer(3)
    note = "目标页：把用户的小目标和长期目标串起来，说明为什么先做单面墙而不是石头堆。"
    return "03_research_target.svg", s, note


def slide_wall_milestones() -> tuple[str, list[str], str]:
    s = svg_open()
    s += title(4, "墙体阶段证据：2 层基础、3 层墙段、4 层对比", "按老师汇报口径，重点展示石墙而不是石堆；2 层是基础子结构证据，3/4 层有图像和深度记录。", "WALL MILESTONES · 2/3/4 COURSES")
    s += [
        rect(64, 184, 340, 344, "paper", "line", 1, 16),
        text(92, 225, "2 层基础", 24, "blue", 800),
        text(94, 252, "foundation substructure", 12, "tertiary", 800),
    ]
    s += rock_wall(104, 330, [4, 3], 0.86, 0)
    s += wrapped(92, 476, "当前 2 层主要作为更高墙体中的基础子结构证据；重点看 footprint、墙线起点和支承连续性。", 24, 16, "muted", 500, 22, 3)
    s += [
        rect(470, 184, 340, 344, "paper", "line", 1, 16),
        text(498, 225, "3 层墙段", 24, "teal", 800),
        text(500, 252, "wall segment / Earth", 12, "tertiary", 800),
        image("wall3_earth_view.png", 498, 276, 284, 160, "meet"),
        image("wall3_earth_depth.png", 570, 432, 140, 92, "meet"),
        text(498, 556, "3 层开始体现石-石支承与墙线保持。", 16, "muted", 600),
    ]
    s += [
        rect(876, 184, 340, 344, "paper", "line", 1, 16),
        text(904, 225, "4 层地/月对比", 24, "rust", 800),
        text(906, 252, "Earth strict vs Moon drift", 12, "tertiary", 800),
        image("wall4_earth_view.png", 904, 276, 132, 146, "meet"),
        image("wall4_moon_view.png", 1052, 276, 132, 146, "meet"),
        image("wall4_earth_depth.png", 916, 432, 108, 80, "meet"),
        image("wall4_moon_depth.png", 1064, 432, 108, 80, "meet"),
        text(904, 556, "4 层能到结构状态；月面低重力下 drift 更突出。", 16, "muted", 600),
    ]
    s += [
        rect(138, 590, 1004, 44, "#FBF0E9", "#F2C9B2", 1, 12),
        text(640, 619, "本页目的：把“能堆石头”与“能形成墙体结构”分开，用 RGB + top depth 做证据。", 18, "rust", 800, "middle"),
    ]
    s += footer(4)
    note = "墙体阶段证据页：补足用户要求的 2/3/4 层墙展示。2 层说明为基础子结构，3 层和 4 层展示实际 RGB/depth 证据。"
    return "04_wall_milestones.svg", s, note


def slide_priors() -> tuple[str, list[str], str]:
    s = svg_open()
    s += title(5, "文献与力学先验如何进入实验", "先验不是结论，而是生成候选、构造损失和解释失败的约束。", "DRY-STACKING PRIORS")
    headers = ["先验来源", "转化为实验规则", "当前观察"]
    rows = [
        ["dry stacking", "错缝、重心投影、支承连续性", "能减少显然滑落的候选"],
        ["材料力学/静力学", "接触面、摩擦、力矩平衡", "base 成功率高，瓶颈在中高层"],
        ["岩石几何先验", "棱角多面体、多主面、拒绝尖刺/圆滑", "几何规则决定可支承性"],
        ["机器人抓取/GraspNet 思路", "候选生成 + 网络排序 + 物理校验", "比随机搜索更接近真实部署"],
        ["世界模型/critic 思路", "预测放后状态和未来支承", "下一阶段重点"],
    ]
    s += small_table(82, 185, headers, rows, [250, 420, 390], row_h=52, font_size=17)
    s += [
        rect(90, 560, 1090, 70, "#FFF7E6", "#F2C879", 1, 14),
        text(122, 604, "关键口径：所有运行时输入必须是放置前可观测信息，历史成功率只能做训练标签/评估统计。", 21, "ink", 700),
    ]
    s += footer(5)
    note = "先验页：强调规则来自文献和力学，不是手工拍脑袋；同时强调后验统计不能作为真实部署输入。"
    return "05_priors.svg", s, note


def slide_pipeline() -> tuple[str, list[str], str]:
    s = svg_open()
    s += title(6, "当前自动实验 pipeline", "从石头生成到 MuJoCo 地球/月球验证，再到正负样本回流。", "SIMULATION PIPELINE")
    steps = [
        ("岩石生成", "棱角多面体\n主面/粗糙度/包围盒", "blue"),
        ("候选槽位", "目标墙线\n层级/角色/base-middle-cap", "teal"),
        ("候选位姿", "yaw/roll\n支承/重心约束", "amber"),
        ("小网络排序", "StoneSlotNet\nSupportMap/PoseRisk", "purple"),
        ("MuJoCo 验证", "Earth/Moon\n漂移/速度/RMSE", "green"),
        ("数据回流", "positive/negative\nhard negatives", "red"),
    ]
    x, y = 70, 205
    for i, (name, body, color) in enumerate(steps):
        xx = x + i * 195
        s += [
            rect(xx, y, 155, 128, "paper", color, 2, 16),
            text(xx + 78, y + 42, name, 19, color, 800, "middle"),
        ]
        s += multiline(xx + 78, y + 74, body.split("\n"), 14, "ink", 500, gap=24, anchor="middle")
        if i < len(steps) - 1:
            s += arrow(xx + 160, y + 65, xx + 190, y + 65, "muted")
    s += [
        rect(130, 430, 1020, 112, "paper", "line", 1, 18),
        text(162, 474, "为什么不能随机试？", 24, "rust", 800),
        text(162, 512, "真实任务中每次候选放置都需要机械臂执行、视觉观测和物理等待；随机搜索的数据成本不可接受。", 22, "ink", 500),
        text(162, 546, "环境：MuJoCo 接触动力学；Earth g≈9.81 m/s²，Moon g≈1.62 m/s²；摩擦/漂移/残余速度都写入日志。", 17, "muted", 600),
    ]
    s += footer(6)
    note = "Pipeline 页：说明实验不是单脚本，而是生成、筛选、网络排序、物理验证、数据回流。"
    return "06_environment_pipeline.svg", s, note


def slide_flywheel() -> tuple[str, list[str], str]:
    s = svg_open()
    s += title(7, "数据飞轮：失败不是终点", "完整墙失败也能拆出可学习的局部成功和 hard negative。", "DATA FLYWHEEL")
    cx, cy = 640, 350
    nodes = [
        (640, 150, "运行实验", "MuJoCo 地/月重力", "blue"),
        (930, 290, "记录结果", "placement/candidate log", "teal"),
        (820, 545, "训练小网络", "ranker / risk / critic", "purple"),
        (460, 545, "更新策略", "先验 + loss + gate", "green"),
        (350, 290, "分析失败", "墙线/漂移/速度", "rust"),
    ]
    for i, (nx, ny, name, sub, color) in enumerate(nodes):
        s += [
            rect(nx - 105, ny - 52, 210, 104, "paper", color, 2, 18),
            text(nx, ny - 8, name, 21, color, 800, "middle"),
            text(nx, ny + 23, sub, 14, "muted", 600, "middle"),
        ]
        nx2, ny2 = nodes[(i + 1) % len(nodes)][0], nodes[(i + 1) % len(nodes)][1]
        s += arrow(nx + (nx2 - nx) * 0.32, ny + (ny2 - ny) * 0.32, nx + (nx2 - nx) * 0.58, ny + (ny2 - ny) * 0.58, "muted")
    s += [
        circle(cx, cy, 82, "dark"),
        text(cx, cy - 16, "正负样本", 24, "#FFFFFF", 800, "middle"),
        text(cx, cy + 22, "持续回流", 20, "#D6E2FF", 700, "middle"),
    ]
    s += footer(7)
    note = "数据飞轮页：用闭环图说明失败样本进入学习，不是简单失败。"
    return "07_flywheel.svg", s, note


def slide_data_ledger() -> tuple[str, list[str], str]:
    s = svg_open()
    s += title(8, "失败到成功：v17 数据账本", "回答“多少负样本才产生阶段性正样本”的问题。", "DATA LEDGER")
    s += metric(72, 176, "run dirs", "36", "实验目录", "blue", 205)
    s += metric(302, 176, "run examples", "41", "完整 run 样本", "teal", 205)
    s += metric(532, 176, "placements", "1061", "逐石放置", "purple", 205)
    s += metric(762, 176, "candidate poses", "34836", "候选位姿", "rust", 205)
    s += metric(992, 176, "assign candidates", "11360", "选石候选", "amber", 205)
    rows = [
        ["Moon placement", "1037", "592", "378", "67"],
        ["Earth placement", "24", "15", "9", "0"],
        ["base role", "286", "277", "7", "2"],
        ["middle role", "573", "234", "299", "40"],
        ["cap role", "202", "96", "81", "25"],
    ]
    s += small_table(116, 355, ["scope", "total", "success", "failure", "skipped"], rows, [250, 160, 160, 160, 160], row_h=42, font_size=17)
    s += footer(8)
    note = "数据账本页：核心数字必须讲清楚。run 级失败不等于没有正样本，placement 和 candidate 级别的数据已经积累。"
    return "08_data_ledger.svg", s, note


def slide_negative_positive() -> tuple[str, list[str], str]:
    s = svg_open()
    s += title(9, "负样本集中在哪里？", "base 已经不是瓶颈，middle/cap 和 course1-course3 是主要学习对象。", "FAILURE DISTRIBUTION")
    s += [
        rect(68, 170, 545, 350, "paper", "line", 1, 18),
        image("role_success_failure_v17.png", 84, 188, 513, 285, "meet"),
        text(96, 502, "按角色：middle failure=299，cap failure=81，base failure 只有 7。", 15, "muted", 600),
        rect(668, 170, 545, 350, "paper", "line", 1, 18),
        image("course_success_failure_v17.png", 684, 188, 513, 285, "meet"),
        text(696, 502, "按层级：第 2-4 层开始体现支承、墙线和误差累积。", 15, "muted", 600),
        rect(155, 565, 970, 50, "#EAF8F6", "#A7DCD7", 1, 16),
        text(640, 597, "结论：后续采样和 loss 应对 middle/cap 加权，不能平均看所有层。", 21, "teal", 800, "middle"),
    ]
    s += footer(9)
    note = "负样本分布页：说明失败不是随机分布，而是集中在中高层和 cap，对网络训练有方向性。"
    return "09_negative_to_positive.svg", s, note


def slide_seed602() -> tuple[str, list[str], str]:
    s = svg_open("dark")
    s += [
        image("seed602_front.png", 0, 0, 760, 720, "slice"),
        rect(0, 0, 760, 720, "#000000", "#000000", 0, 0, 0.18),
        rect(720, 0, 560, 720, "dark", "dark", 0),
        text(792, 72, "阶段性成功案例", 23, "blue2", 800),
        text(792, 126, "seed602: 4 层结构已出现", 35, "#FFFFFF", 800),
        text(794, 168, "但 drift 未过 strict success 阈值", 21, "#FFCBA4", 700),
        image("seed602_top_depth.png", 798, 218, 355, 250, "meet"),
        text(798, 496, "关键指标", 20, "#D6E2FF", 800),
    ]
    rows = [
        ["stable/failure", "20 / 4"],
        ["RMSE", "0.1158 m"],
        ["height", "0.3611 m"],
        ["drift", "0.4136 m"],
        ["velocity", "0.1446"],
    ]
    yy = 528
    for k, v in rows:
        s += [
            text(800, yy, k, 16, "#AAB7CA", 600),
            text(1128, yy, v, 16, "#FFFFFF", 800, "end"),
            line(800, yy + 11, 1130, yy + 11, "slate", 0.7, 0.55),
        ]
        yy += 30
    s += [text(640, 696, "10", 14, "#AAB7CA", 600, "middle")]
    note = "seed602 页：这是阶段性结构案例。视觉上有竖向结构，RMSE 和高度较好，但 drift 大，所以严格失败。"
    return "10_stage_success_seed602.svg", s, note


def slide_seed604() -> tuple[str, list[str], str]:
    s = svg_open()
    s += title(11, "典型失败：稳定但不是墙", "seed604 证明低 drift / 低 velocity 不能等价于结构成功。", "HARD NEGATIVE")
    s += [
        rect(66, 178, 540, 360, "paper", "line", 1, 18),
        image("seed604_front.png", 82, 194, 508, 285, "meet"),
        text(96, 510, "正视 RGB：局部堆散，缺少连续单面墙。", 16, "muted", 600),
        rect(674, 178, 540, 360, "paper", "line", 1, 18),
        image("seed604_top_depth.png", 690, 194, 508, 285, "meet"),
        text(704, 510, "俯视 depth：支承点散开，墙线目标没有保持。", 16, "muted", 600),
        rect(160, 570, 960, 52, "#FFF0F0", "#F3B4B4", 1, 16),
        text(640, 603, "数值：stable/failure=17/7，drift=0.0954 m，velocity=0.0108，但 RMSE=0.2209 m。", 20, "red", 800, "middle"),
    ]
    s += footer(11)
    note = "seed604 页：作为 hard negative，说明只优化动力学稳定会得到稳定但错误的结构。"
    return "11_typical_failure_seed604.svg", s, note


def slide_rules() -> tuple[str, list[str], str]:
    s = svg_open()
    s += title(12, "哪些规则有用？哪些规则没用？", "规则的价值在于定义候选空间、训练标签和安全约束。", "RULE AUDIT")
    rows = [
        ["有用", "棱角多面体；拒绝尖刺/圆滑边", "符合石头先验，避免非物理支承"],
        ["有用", "主面/支承面、重心投影、支承重叠", "显著减少必然滑落候选"],
        ["有用", "前三层网络化，4-5 层启发式/MuJoCo", "降低早期搜索，同时避免高层误剪"],
        ["有用", "正视 RGB + 俯视 depth + RMSE/drift/velocity", "防止把石堆误判为墙"],
        ["不足", "随机放置/随机搜索", "可收集数据，但真实任务成本过高"],
        ["不足", "只增加候选数量或 settling", "可能稳定地堆出错误形状"],
        ["不足", "只压 drift/velocity", "低漂移不等于墙线正确"],
    ]
    s += small_table(74, 172, ["判断", "规则/方法", "实验观察"], rows, [110, 390, 650], row_h=56, font_size=17)
    s += footer(12)
    note = "规则审计页：清楚回答哪些规则有效，哪些规则已经证明不足。"
    return "12_rules.svg", s, note


def slide_network_entry() -> tuple[str, list[str], str]:
    s = svg_open()
    s += title(13, "神经网络是如何接入的？", "当前不是端到端大模型，而是小网络先替代最昂贵的候选排序环节。", "NEURAL ENTRY")
    timeline = [
        ("0", "随机/候选试放", "能探索，成本过高", "red"),
        ("1", "几何启发式", "主面、重心、支承", "amber"),
        ("2", "前三层小网络", "StoneSlot + CNN + Risk", "blue"),
        ("3", "高层混合闭环", "启发式 + MuJoCo", "teal"),
        ("4", "未来 critic", "墙线 + 未来支承", "green"),
    ]
    x0, y0 = 92, 215
    for i, (idx, name, sub, color) in enumerate(timeline):
        x = x0 + i * 235
        s += [
            circle(x + 54, y0 - 45, 24, color),
            text(x + 54, y0 - 36, idx, 16, "#FFFFFF", 800, "middle"),
            rect(x, y0, 142, 112, "paper", color, 2, 16),
            text(x + 71, y0 + 40, name, 17, color, 800, "middle"),
            text(x + 71, y0 + 75, sub, 13, "muted", 600, "middle"),
        ]
        if i < len(timeline) - 1:
            s += arrow(x + 150, y0 + 55, x + 225, y0 + 55, "muted")
    rows = [
        ["StoneSlotNet", "石头几何 + 槽位/层级", "石头-槽位分数", "减少不合适石头"],
        ["support-map CNN", "支承高度/占据/目标/footprint", "候选位姿 rank", "减少候选搜索"],
        ["PoseRiskNet", "候选位姿 + 支承指标 + 几何", "风险分数", "降低滑移/漂移/速度风险"],
    ]
    s += small_table(118, 410, ["网络", "输入", "输出", "目的"], rows, [220, 390, 210, 290], row_h=50, font_size=16)
    s += footer(13)
    note = "神经网络接入页：说明最初接入方式、位置和目的，不是随机试错。"
    return "13_network_entry.svg", s, note


def slide_network_io() -> tuple[str, list[str], str]:
    s = svg_open()
    s += title(14, "神经网络输入/输出的演变", "从单石头几何拟合，演变为“石头 + 当前墙体观测 + 候选位姿”的联合评价。", "NETWORK I/O")
    left = [
        ("早期输入", "石头几何\n包围盒/主面/粗糙度", "blue"),
        ("现在输入", "石头几何 + support map\n俯视 depth + target slot + pose", "teal"),
        ("未来输入", "当前墙体状态 + 放后预测\n未来支承区域 + 重力", "green"),
    ]
    for i, (name, body, color) in enumerate(left):
        y = 190 + i * 125
        s += [
            rect(92, y, 330, 88, "paper", color, 2, 16),
            text(120, y + 34, name, 20, color, 800),
        ]
        s += multiline(120, y + 62, body.split("\n"), 15, "ink", 500, gap=23)
        s += arrow(430, y + 44, 520, y + 44, "muted")
    right = [
        ("早期输出", "石头选择/候选排名", "blue"),
        ("现在输出", "pose rank + risk score", "teal"),
        ("未来输出", "结构质量 + 未来支承 + 动力学风险", "green"),
    ]
    for i, (name, body, color) in enumerate(right):
        y = 190 + i * 125
        s += [
            rect(545, y, 330, 88, "paper", color, 2, 16),
            text(573, y + 34, name, 20, color, 800),
            text(573, y + 66, body, 15, "ink", 500),
        ]
    s += [
        rect(930, 210, 240, 300, "#FFF7E6", "#F2C879", 1, 18),
        text(1050, 260, "关键约束", 23, "rust", 800, "middle"),
    ]
    s += bullet_list(965, 320, ["不能把历史成功率作为运行时输入", "仿真后指标只能作为标签", "模型必须依赖几何与观测泛化"], size=17, gap=55, accent="rust", width_chars=18)
    s += footer(14)
    note = "网络输入输出页：回应用户关切，网络不能只拟合历史成功率，需要结合当前石堆观测。"
    return "14_network_io.svg", s, note


def slide_model_metrics() -> tuple[str, list[str], str]:
    s = svg_open()
    s += title(15, "新网络当前能做什么？不能做什么？", "top-3 高说明候选池里有答案；top-1 不够说明还不能完全替代启发式。", "MODEL METRICS")
    s += [
        rect(72, 174, 560, 332, "paper", "line", 1, 18),
        image("network_metrics_v15_v16.png", 92, 196, 520, 248, "meet"),
        text(102, 478, "v15/v16 和 critic 当前主要用于诊断与辅助排序。", 16, "muted", 600),
    ]
    rows = [
        ["v15 support-map", "top1 0.440 / top3 0.924", "辅助分析"],
        ["v16 support-map", "top1 0.396 / top3 0.911", "不能直接接管"],
        ["WallStateCritic", "top1 0.362 / top3 0.877", "离线诊断"],
        ["PoseRiskNet v16", "acc 0.790 / F1 0.882", "风险保守"],
    ]
    s += small_table(685, 184, ["模型", "指标", "闭环判断"], rows, [210, 220, 150], row_h=58, font_size=15)
    s += [
        rect(140, 560, 1000, 52, "#EAF8F6", "#A7DCD7", 1, 16),
        text(640, 593, "当前结论：成熟旧网络用于前三层，新 v15/v16 不直接替换默认闭环。", 21, "teal", 800, "middle"),
    ]
    s += footer(15)
    note = "模型指标页：解释 top3 和 top1 的意义，避免老师误以为网络已经能端到端放置。"
    return "15_model_metrics.svg", s, note


def slide_layer_difficulty() -> tuple[str, list[str], str]:
    s = svg_open()
    s += title(16, "各层级遇到的难题不同", "层数越高，问题越从局部稳定转向全局墙线和未来支承。", "LAYER-WISE DIFFICULTY")
    rows = [
        ["第1层 base", "277/286 success", "基础 footprint、墙线起点", "地面平整，支承不是瓶颈"],
        ["第2层", "104/245 success", "石-石支承、yaw/roll、重心投影", "第一次真正面对不平支承"],
        ["第3层", "92/255 success", "跨接/互锁、保留第4层支承", "不只是当前稳定，还要服务未来"],
        ["第4层", "100/214 success", "累积误差、全局墙线、drift", "局部稳定可能破坏整体墙"],
        ["第5层/cap", "course4 32/54 success", "支承选项少，容易稳定但形状错", "需要闭合顶部且不推散墙体"],
    ]
    s += small_table(66, 174, ["层级", "当前统计", "主要难题", "与前一层的区别"], rows, [185, 185, 430, 350], row_h=66, font_size=16)
    s += footer(16)
    note = "层级难点页：逐条回答第二层、第三层、第四层的相同与不同问题。"
    return "16_layer_difficulty.svg", s, note


def slide_tradeoff() -> tuple[str, list[str], str]:
    s = svg_open()
    s += title(17, "4-5 层揭示的主要矛盾", "高度、墙线、漂移和速度不是单调兼容，需要多目标训练。", "4-5 COURSE TRADE-OFF")
    s += [
        rect(70, 174, 640, 350, "paper", "line", 1, 18),
        image("run_tradeoffs_20260622.png", 92, 202, 596, 265, "meet"),
        text(100, 498, "一个指标好不能代表墙成功：seed605/606 是典型“稳定但形状错误”。", 15, "muted", 600),
    ]
    rows = [
        ["高度不错但速度高", "5L seed603", "月面残余运动危险"],
        ["漂移低但墙线差", "5L seed605", "稳定地堆歪不是成功"],
        ["候选更多仍失败", "5L seed606", "搜索量不是结构目标"],
        ["4层结构但漂移大", "4L seed602", "需要全局姿态保持"],
    ]
    s += small_table(752, 194, ["现象", "例子", "含义"], rows, [190, 150, 210], row_h=62, font_size=15)
    s += footer(17)
    note = "tradeoff 页：说明 4-5 层难度来自多目标冲突，不能只看高度或稳定性。"
    return "17_four_five_tradeoff.svg", s, note


def slide_current_strategy() -> tuple[str, list[str], str]:
    s = svg_open()
    s += title(18, "当前堆叠策略：前三层网络，高层保守验证", "这是为了让数据飞轮转动，同时避免上层被不成熟网络误剪。", "CURRENT STRATEGY")
    s += [
        rect(96, 190, 310, 330, "paper", "blue", 2, 20),
        text(125, 238, "course ≤ 2", 25, "blue", 800),
        text(125, 285, "网络辅助", 28, "ink", 800),
    ]
    s += bullet_list(132, 345, ["StoneSlotNet 先选石头", "support-map CNN 保留 top-3", "PoseRisk 对高风险位姿加惩罚"], 18, 48, "blue", 22)
    s += [
        rect(486, 190, 310, 330, "paper", "teal", 2, 20),
        text(515, 238, "course ≥ 3", 25, "teal", 800),
        text(515, 285, "启发式 + MuJoCo", 28, "ink", 800),
    ]
    s += bullet_list(522, 345, ["保留完整候选池", "用干叠几何先验过滤", "短仿真验证漂移/速度"], 18, 48, "teal", 22)
    s += [
        rect(876, 190, 310, 330, "paper", "green", 2, 20),
        text(905, 238, "所有层", 25, "green", 800),
        text(905, 285, "日志回流", 28, "ink", 800),
    ]
    s += bullet_list(912, 345, ["candidate_pose_log", "placement_log", "图像与深度图", "hard negative"], 18, 42, "green", 20)
    s += footer(18)
    note = "当前策略页：准确说明目前如何放石头，以及为什么不是随机放。"
    return "18_current_strategy.svg", s, note


def slide_scheduler() -> tuple[str, list[str], str]:
    s = svg_open()
    s += title(19, "异步调度：让数据持续产生", "本端 2080Ti 主训练，远端 1080Ti 轻量采样；当前暂停实验，先完成汇报材料。", "SCHEDULING")
    rows = [
        ["本端 RTX 2080Ti + 64G", "主训练/评估", "support-map、PoseRisk、WallStateCritic", "训练吞吐更好"],
        ["远端 GTX 1080Ti", "轻量采样", "低风险 run、候选日志汇总", "补充数据，不阻塞主训练"],
        ["MuJoCo 仿真", "物理验证", "地球/月球重力、短仿真稳定性", "保证标签来自物理闭环"],
        ["日志系统", "数据资产", "正负样本、skipped、RGB/depth、metrics", "失败可复用"],
    ]
    s += small_table(78, 190, ["资源", "角色", "任务", "原因"], rows, [270, 180, 430, 250], row_h=70, font_size=16)
    s += [
        rect(165, 585, 950, 48, "#FFF7E6", "#F2C879", 1, 14),
        text(640, 616, "当前状态：响应暂停要求，没有启动新实验；只做 PPT Master 汇报材料。", 20, "rust", 800, "middle"),
    ]
    s += footer(19)
    note = "调度页：说明两台电脑角色和数据飞轮运行方式，但当前暂停实验没有启动新 run。"
    return "19_scheduler.svg", s, note


def slide_next() -> tuple[str, list[str], str]:
    s = svg_open()
    s += title(20, "下一阶段里程碑", "先把 3-4 层成功率做高，再推进 5 层以上，不追求偶然高墙。", "NEXT MILESTONE")
    stages = [
        ("A", "3 层闭环稳定", "目标：60%-80%\n形成干净正负样本", "blue"),
        ("B", "4 层网络辅助", "前三层网络 + 第4层启发式\n扩大 hard negatives", "teal"),
        ("C", "future-support critic", "预测下一层可支承性\n减少石堆化", "purple"),
        ("D", "5-6 层可复现", "成功率可解释后\n再冲 8-10 层", "green"),
    ]
    x0 = 92
    for i, (idx, name, body, color) in enumerate(stages):
        x = x0 + i * 292
        s += [
            circle(x + 26, 220, 26, color),
            text(x + 26, 229, idx, 18, "#FFFFFF", 800, "middle"),
            rect(x, 275, 236, 210, "paper", color, 2, 20),
            text(x + 118, 328, name, 24, color, 800, "middle"),
        ]
        s += multiline(x + 118, 382, body.split("\n"), 17, "ink", 600, gap=34, anchor="middle")
        if i < len(stages) - 1:
            s += arrow(x + 244, 380, x + 286, 380, "muted")
    s += [
        rect(150, 565, 980, 52, "#EAF8F6", "#A7DCD7", 1, 16),
        text(640, 598, "关键假设：加入 wall-state / future-support 监督后，3-4 层闭环成功率应显著提升。", 20, "teal", 800, "middle"),
    ]
    s += footer(20)
    note = "下一阶段页：说明为什么不盲目冲 10 层，而是先提高 3-4 层成功率。"
    return "20_next_milestone.svg", s, note


def slide_takeaways() -> tuple[str, list[str], str]:
    s = svg_open("dark")
    s += [
        text(76, 88, "阶段性结论", 28, "blue2", 800),
        text(76, 154, "当前成果不是“墙已经完全成功”，而是实验路线开始收敛。", 43, "#FFFFFF", 800),
        line(80, 192, 920, 192, "blue2", 3),
    ]
    takeaways = [
        ("1", "4 层阶段性结构出现", "seed602 有明显竖向结构，但 drift 暴露下一阶段问题。"),
        ("2", "负样本已经成为资产", "v17: 1061 placement、34836 candidate pose，middle/cap 是重点。"),
        ("3", "规则审计变清楚", "几何支承与墙线规则有效；随机、单指标稳定性不足。"),
        ("4", "网络路线合理", "从小网络组合进入，逐步发展到结构 critic，而不是直接大模型碰运气。"),
    ]
    y = 260
    for idx, head, body in takeaways:
        s += [
            circle(105, y - 10, 20, "blue"),
            text(105, y - 2, idx, 15, "#FFFFFF", 800, "middle"),
            text(145, y, head, 24, "#FFFFFF", 800),
            text(145, y + 34, body, 18, "#C8D2E0", 500),
        ]
        y += 88
    s += [
        rect(760, 488, 390, 82, "#FFFFFF", "#FFFFFF", 0, 18, 0.92),
        text(790, 522, "建议下一步汇报指标", 18, "blue", 800),
        text(790, 552, "3-4 层成功率提升 + network top-1 提升 + 墙线误差下降", 16, "ink", 600),
        text(640, 696, "21", 14, "#AAB7CA", 600, "middle"),
    ]
    note = "总结页：收束汇报，强调科学价值在路线收敛和数据飞轮，而不是夸大成功。"
    return "21_takeaways.svg", s, note


def build_slides() -> list[tuple[str, list[str], str]]:
    return [
        slide_cover(),
        slide_summary(),
        slide_target(),
        slide_wall_milestones(),
        slide_priors(),
        slide_pipeline(),
        slide_flywheel(),
        slide_data_ledger(),
        slide_negative_positive(),
        slide_seed602(),
        slide_seed604(),
        slide_rules(),
        slide_network_entry(),
        slide_network_io(),
        slide_model_metrics(),
        slide_layer_difficulty(),
        slide_tradeoff(),
        slide_current_strategy(),
        slide_scheduler(),
        slide_next(),
        slide_takeaways(),
    ]


def write_slides(slides: list[tuple[str, list[str], str]]) -> None:
    total_md: list[str] = ["# MoonStack Stage Success PPT Speaker Notes", ""]
    for idx, (name, body, note) in enumerate(slides, 1):
        svg_path = PROJECT / "svg_output" / name
        note_path = PROJECT / "notes" / f"{Path(name).stem}.md"
        svg_path.write_text("\n".join(body + ["</svg>\n"]), encoding="utf-8")
        note_path.write_text(f"# {Path(name).stem}\n\n{note}\n", encoding="utf-8")
        title_guess = Path(name).stem
        total_md += [f"## {title_guess}", "", note, ""]
    (PROJECT / "notes" / "total.md").write_text("\n".join(total_md), encoding="utf-8")


def main() -> None:
    write_project_files()
    copied = copy_assets()
    slides = build_slides()
    write_slides(slides)
    summary = {
        "project": str(PROJECT),
        "slide_count": len(slides),
        "slides": [name for name, _, _ in slides],
        "copied_assets": copied,
        "missing_assets": [name for name, src in ASSET_SOURCES.items() if not src.exists()],
        "expected_export_dir": str(PROJECT / "exports"),
    }
    SCRIPT_SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
