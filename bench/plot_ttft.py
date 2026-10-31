"""Draw turn-2 TTFT against context length as an SVG, from committed results only.

    python -m bench.plot_ttft results/restore-qwen3-8b.json \\
        results/baseline-resident-prefix-cache-qwen3-8b.json --output docs/ttft-qwen3-8b.svg

The first file is a restore_ttft result (recompute and the three kvsnap
tiers); any further files are resident-regime resume_baselines results,
drawn as vLLM's prefix-cache hit. No plotting library is needed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

WIDTH, HEIGHT, LEFT, RIGHT, TOP, BOTTOM = 720, 420, 64, 190, 24, 48
SERIES = (("recompute", "Recompute (prefix caching off)", "#c0392b"),
          ("restore-cold", "kvsnap, disk (cold)", "#8e44ad"),
          ("restore-disk", "kvsnap, disk (page cache)", "#2471a3"),
          ("restore-host", "kvsnap, host memory", "#17a589"),
          ("prefix-cache", "vLLM prefix-cache hit (KV still on GPU)", "#7f8c8d"))


def series(restore: dict, resident: list[dict]) -> dict:
    lines = {}
    for row in restore["rows"]:
        lines.setdefault(row["arm"], []).append((row["context_tokens"], row["ttft_ms_median"]))
    for result in resident:
        for row in result["summary"]:
            lines.setdefault("prefix-cache", []).append((row["context_tokens"], row["ttft_ms_median"]))
    return {name: sorted(points) for name, points in lines.items()}


def svg(lines: dict, title: str) -> str:
    xs = sorted({x for points in lines.values() for x, _ in points})
    top = max(y for points in lines.values() for _, y in points)
    step = 100 if top <= 1000 else 200
    ymax = (int(top / step) + 1) * step
    plot_w, plot_h = WIDTH - LEFT - RIGHT, HEIGHT - TOP - BOTTOM

    def px(x):
        return LEFT + plot_w * xs.index(x) / max(1, len(xs) - 1)

    def py(y):
        return TOP + plot_h * (1 - y / ymax)

    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {WIDTH} {HEIGHT}" '
           f'font-family="Helvetica, Arial, sans-serif" font-size="12">',
           f'<rect width="{WIDTH}" height="{HEIGHT}" fill="#ffffff"/>',
           f'<text x="{LEFT}" y="16" font-size="13" font-weight="bold" fill="#222">{title}</text>']
    for tick in range(0, ymax + 1, step):
        y = py(tick)
        out.append(f'<line x1="{LEFT}" y1="{y:.1f}" x2="{LEFT + plot_w}" y2="{y:.1f}" stroke="#e5e5e5"/>')
        out.append(f'<text x="{LEFT - 8}" y="{y + 4:.1f}" text-anchor="end" fill="#555">{tick}</text>')
    for x in xs:
        out.append(f'<text x="{px(x):.1f}" y="{TOP + plot_h + 18}" text-anchor="middle" fill="#555">'
                   f'{x // 1024}k</text>')
    out.append(f'<text x="{LEFT + plot_w / 2:.1f}" y="{HEIGHT - 8}" text-anchor="middle" fill="#333">'
               'context tokens before the turn</text>')
    out.append(f'<text x="16" y="{TOP + plot_h / 2:.1f}" text-anchor="middle" fill="#333" '
               f'transform="rotate(-90 16 {TOP + plot_h / 2:.1f})">turn-2 TTFT, ms (median)</text>')
    legend_y = TOP + 8
    for key, label, colour in SERIES:
        points = lines.get(key)
        if not points:
            continue
        path = " ".join(f"{px(x):.1f},{py(y):.1f}" for x, y in points)
        out.append(f'<polyline points="{path}" fill="none" stroke="{colour}" stroke-width="2.5"/>')
        out.extend(f'<circle cx="{px(x):.1f}" cy="{py(y):.1f}" r="3.5" fill="{colour}"/>' for x, y in points)
        out.append(f'<rect x="{WIDTH - RIGHT + 16}" y="{legend_y - 9}" width="12" height="3" fill="{colour}"/>')
        words, line = label.split(), ""
        for word in words:                                 # wrap the legend at about 24 characters
            if len(line) + len(word) > 24 and line:
                out.append(f'<text x="{WIDTH - RIGHT + 34}" y="{legend_y}" fill="#333">{line.strip()}</text>')
                legend_y += 14
                line = ""
            line += word + " "
        out.append(f'<text x="{WIDTH - RIGHT + 34}" y="{legend_y}" fill="#333">{line.strip()}</text>')
        legend_y += 24
    out.append("</svg>")
    return "\n".join(out) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("restore", type=Path)
    parser.add_argument("resident", type=Path, nargs="*")
    parser.add_argument("--title", default="Qwen3-8B on one RTX 4090: turn 2 appends 64 tokens")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    lines = series(json.loads(args.restore.read_text()), [json.loads(p.read_text()) for p in args.resident])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(svg(lines, args.title))
    print(args.output)


if __name__ == "__main__":
    main()
