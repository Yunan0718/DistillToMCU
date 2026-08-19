#!/usr/bin/env python3
"""Generate the Elsevier graphical abstract for DistillToMCU.

Elsevier guidance: separate file, 531 x 1328 px (height x width) or
proportionally scaled, legible at 5 x 13 cm; PDF/TIFF/EPS preferred.

Outputs (in ../paper):
  graphical_abstract.pdf   - vector, 13.28 x 5.31 cm
  graphical_abstract.png   - 1328 x 531 px
  graphical_abstract.tiff  - 1328 x 531 px, RGB, LZW
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrow, FancyBboxPatch, Rectangle
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "paper")
os.makedirs(OUT, exist_ok=True)

# Okabe-Ito color-blind-safe palette
BLUE = "#0072B2"
ORANGE = "#E69F00"
GREEN = "#009E73"
GRAY = "#8C8C8C"
DARK = "#111111"
BODY = "#333333"
MUTED = "#5A5A5A"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "text.color": DARK,
})

W_CM, H_CM = 13.28, 5.31
fig = plt.figure(figsize=(W_CM / 2.54, H_CM / 2.54), dpi=254)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 100)
ax.set_ylim(0, 40)
ax.set_aspect("equal")
ax.axis("off")


def box(x0, y0, x1, y1, facecolor, edgecolor, lw=2.0):
    return FancyBboxPatch(
        (x0, y0), x1 - x0, y1 - y0,
        boxstyle="round,pad=0,rounding_size=1.1",
        linewidth=lw, edgecolor=edgecolor, facecolor=facecolor, zorder=2,
    )


def arrow(x0, y0, x1, y1, color="#666666", lw=2.4):
    ax.add_patch(FancyArrow(
        x0, y0, x1 - x0, y1 - y0,
        width=0.9, head_width=2.4, head_length=1.4,
        length_includes_head=True, color=color, zorder=3))


def cloud(cx, cy, s=1.0):
    parts = [
        Circle((cx, cy), 2.9 * s, fc="#EAF2FB", ec=BLUE, lw=1.2, zorder=5),
        Circle((cx - 3.7 * s, cy - 0.9 * s), 2.3 * s, fc="#EAF2FB", ec=BLUE, lw=1.2, zorder=5),
        Circle((cx + 3.7 * s, cy - 0.9 * s), 2.3 * s, fc="#EAF2FB", ec=BLUE, lw=1.2, zorder=5),
        Rectangle((cx - 5.5 * s, cy - 2.7 * s), 11.0 * s, 1.8 * s,
                  fc="#EAF2FB", ec=BLUE, lw=1.2, zorder=5),
    ]
    for p in parts:
        ax.add_patch(p)


def chip(cx, cy, s=1.0):
    ax.add_patch(Rectangle((cx - 2.9 * s, cy - 2.9 * s), 5.8 * s, 5.8 * s,
                           fc="#E9F6EF", ec=GREEN, lw=1.2, zorder=5))
    for dx in (-1.8 * s, 0.0, 1.8 * s):
        ax.plot([cx + dx, cx + dx], [cy + 2.9 * s, cy + 4.3 * s],
                color=GREEN, lw=1.1, zorder=5)
        ax.plot([cx + dx, cx + dx], [cy - 2.9 * s, cy - 4.3 * s],
                color=GREEN, lw=1.1, zorder=5)
        ax.plot([cx + 2.9 * s, cx + 4.3 * s], [cy + dx, cy + dx],
                color=GREEN, lw=1.1, zorder=5)
        ax.plot([cx - 2.9 * s, cx - 4.3 * s], [cy + dx, cy + dx],
                color=GREEN, lw=1.1, zorder=5)


def rule_card(cx, cy, s=1.0):
    ax.add_patch(FancyBboxPatch(
        (cx - 4.8 * s, cy - 3.0 * s), 9.6 * s, 6.0 * s,
        boxstyle="round,pad=0,rounding_size=1.0",
        linewidth=1.2, edgecolor=ORANGE, facecolor="white", zorder=5))
    for dy in (1.2 * s, -1.2 * s):
        ax.plot([cx - 3.0 * s, cx + 3.0 * s], [cy + dy, cy + dy],
                color="#D9A441", lw=1.0, zorder=5)


# ---------------- header ----------------
ax.text(50, 38.1, "DistillToMCU", ha="center", va="center",
        fontsize=16, fontweight="bold", zorder=4)
ax.text(50, 35.9, "behavior distillation for LLM-free MCU autonomy",
        ha="center", va="center", fontsize=7.5, color=MUTED, zorder=4)

# ---------------- stage boxes ----------------
ax.add_patch(box(2, 12, 27, 34, "#EAF2FB", BLUE))
ax.add_patch(box(36, 12, 64, 34, "#FDF3E3", ORANGE))
ax.add_patch(box(73, 12, 98, 34, "#E9F6EF", GREEN))

cloud(14.5, 30.2)
ax.text(14.5, 26.4, "Cloud LLM", ha="center", va="center",
        fontsize=12.5, fontweight="bold", color=BLUE, zorder=4)
ax.text(14.5, 22.4, "DeepSeek / Qwen", ha="center", va="center",
        fontsize=8.5, color=BODY, zorder=4)
ax.text(14.5, 19.4, "tool-call decisions", ha="center", va="center",
        fontsize=8.5, color=BODY, zorder=4)
ax.text(14.5, 16.4, "per sensor event", ha="center", va="center",
        fontsize=8.5, color=BODY, zorder=4)

rule_card(50, 30.2)
ax.text(50, 26.4, "COMIC distillation", ha="center", va="center",
        fontsize=12.5, fontweight="bold", color="#B26E00", zorder=4)
ax.text(50, 22.4, "online quantiles", ha="center", va="center",
        fontsize=8.5, color=BODY, zorder=4)
ax.text(50, 19.4, "conformal calibration", ha="center", va="center",
        fontsize=8.5, color=BODY, zorder=4)
ax.text(50, 16.4, "MDL consolidation", ha="center", va="center",
        fontsize=8.5, color=BODY, zorder=4)

chip(84, 30.2)
ax.text(84, 26.4, "ESP32-S3", ha="center", va="center",
        fontsize=12.5, fontweight="bold", color=GREEN, zorder=4)
ax.text(84, 22.4, "local rule matching", ha="center", va="center",
        fontsize=8.5, color=BODY, zorder=4)
ax.text(84, 19.4, "1.48 ms (p50)", ha="center", va="center",
        fontsize=8.5, color=BODY, zorder=4)
ax.text(84, 16.4, "no LLM at runtime", ha="center", va="center",
        fontsize=8.5, fontweight="bold", color="#1B6B3F", zorder=4)

# ---------------- arrows ----------------
arrow(27.4, 23.0, 35.4, 23.0)
arrow(64.6, 23.0, 72.4, 23.0)

# ---------------- distilled rule chip ----------------
ax.add_patch(FancyBboxPatch(
    (39, 5.3), 22, 5.4,
    boxstyle="round,pad=0,rounding_size=0.8",
    linewidth=1.2, edgecolor=GRAY, facecolor="white", zorder=2))
ax.text(50, 8.7, "temp in [22.5, 28.0]", ha="center", va="center",
        fontsize=8.0, color=DARK, zorder=4)
ax.text(50, 6.4, "conf 0.92  \u00b7  2 B/rule", ha="center", va="center",
        fontsize=7.5, color=MUTED, zorder=4)

# ---------------- bottom metrics ----------------
ax.plot([11.2], [2.7], "o", ms=5.0, color=BLUE, zorder=4)
ax.text(13.0, 2.7, "85.4% held-out autonomy", ha="left", va="center",
        fontsize=8.0, color=BODY, zorder=4)
ax.plot([46.7], [2.7], "o", ms=5.0, color=ORANGE, zorder=4)
ax.text(48.5, 2.7, "50.0% precision on UCI replay", ha="left", va="center",
        fontsize=8.0, color=BODY, zorder=4)
ax.plot([81.7], [2.7], "o", ms=5.0, color=GREEN, zorder=4)
ax.text(83.5, 2.7, "teacher shift 1.7\u20133.7 pt", ha="left", va="center",
        fontsize=8.0, color=BODY, zorder=4)

pdf_path = os.path.join(OUT, "graphical_abstract.pdf")
png_path = os.path.join(OUT, "graphical_abstract.png")
tiff_path = os.path.join(OUT, "graphical_abstract.tiff")

fig.savefig(pdf_path, format="pdf")
fig.savefig(png_path, format="png", dpi=254)
plt.close(fig)

# Force exact 1328 x 531 px raster (PDF page is 13.28 x 5.31 cm).
img = Image.open(png_path).convert("RGB")
img = img.resize((1328, 531), Image.LANCZOS)
img.save(png_path, "PNG")
img.save(tiff_path, "TIFF", compression="tiff_lzw")

print("wrote", pdf_path)
print("wrote", png_path, img.size)
print("wrote", tiff_path)
