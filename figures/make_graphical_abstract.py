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
from matplotlib.patches import FancyArrow, FancyBboxPatch
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "paper")
os.makedirs(OUT, exist_ok=True)

# Okabe-Ito color-blind-safe palette
BLUE = "#0072B2"
ORANGE = "#E69F00"
GREEN = "#009E73"
GRAY = "#8C8C8C"
LIGHT = "#F2F6FA"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "text.color": "#111111",
})

W_CM, H_CM = 13.28, 5.31
fig = plt.figure(figsize=(W_CM / 2.54, H_CM / 2.54), dpi=254)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis("off")


def box(x0, y0, x1, y1, facecolor=LIGHT, edgecolor=BLUE, lw=2.2, radius=2.0):
    return FancyBboxPatch(
        (x0, y0), x1 - x0, y1 - y0,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        linewidth=lw, edgecolor=edgecolor, facecolor=facecolor, zorder=2,
    )


def arrow(x0, y0, x1, y1, color=GRAY, lw=3.0):
    ax.add_patch(FancyArrow(x0, y0, x1 - x0, y1 - y0,
                            width=1.8, head_width=4.2, head_length=2.2,
                            length_includes_head=True,
                            color=color, zorder=3))


# ---- Stage 1: cloud LLM ----
ax.add_patch(box(2, 38, 27, 88, facecolor="#EAF2FB", edgecolor=BLUE))
ax.text(14.5, 79, "Cloud LLM", ha="center", va="center",
        fontsize=13, fontweight="bold", color=BLUE, zorder=4)
ax.text(14.5, 66, "DeepSeek / Qwen", ha="center", va="center",
        fontsize=8.5, zorder=4)
ax.text(14.5, 55, "structured tool-call", ha="center", va="center",
        fontsize=8.5, zorder=4)
ax.text(14.5, 47, "decisions per event", ha="center", va="center",
        fontsize=8.5, zorder=4)

# ---- Stage 2: COMIC distillation ----
ax.add_patch(box(36, 38, 64, 88, facecolor="#FDF3E3", edgecolor=ORANGE))
ax.text(50, 80, "COMIC distillation", ha="center", va="center",
        fontsize=12, fontweight="bold", color="#B26E00", zorder=4)
ax.text(50, 67, "online quantiles", ha="center", va="center",
        fontsize=8.5, zorder=4)
ax.text(50, 58, "conformal calibration", ha="center", va="center",
        fontsize=8.5, zorder=4)
ax.text(50, 49, "MDL consolidation", ha="center", va="center",
        fontsize=8.5, zorder=4)

# Rule chips below stage 2
chip = FancyBboxPatch((38, 13), 24, 16,
                      boxstyle="round,pad=0,rounding_size=2.5",
                      linewidth=1.4, edgecolor=GRAY, facecolor="white", zorder=2)
ax.add_patch(chip)
ax.text(50, 22.5, "temp in [22.5, 28.0]", ha="center", va="center",
        fontsize=8.0, zorder=4)
ax.text(50, 14.5, "conf 0.92  \u00b7  2 B/rule", ha="center", va="center",
        fontsize=7.5, color="#444444", zorder=4)

# ---- Stage 3: ESP32-S3 ----
ax.add_patch(box(70, 38, 98, 88, facecolor="#E9F6EF", edgecolor=GREEN))
ax.text(84, 80, "ESP32-S3", ha="center", va="center",
        fontsize=13, fontweight="bold", color=GREEN, zorder=4)
ax.text(84, 66, "local rule matching", ha="center", va="center",
        fontsize=8.5, zorder=4)
ax.text(84, 56, "1.48 ms  (p50)", ha="center", va="center",
        fontsize=8.5, zorder=4)
ax.text(84, 46, "no LLM at runtime", ha="center", va="center",
        fontsize=8.5, fontweight="bold", color="#1B6B3F", zorder=4)

# ---- Arrows between stages ----
arrow(27, 63, 35.5, 63)
arrow(64, 63, 69.5, 63)

# ---- Bottom metric strip ----
ax.text(14.5, 6.5, "85.4% held-out autonomy", ha="center", va="center",
        fontsize=8.0, color="#333333", zorder=4)
ax.text(50, 6.5, "50.0% precision on UCI replay", ha="center", va="center",
        fontsize=8.0, color="#333333", zorder=4)
ax.text(84, 6.5, "teacher shift 1.7\u20133.7 pt", ha="center", va="center",
        fontsize=8.0, color="#333333", zorder=4)

# System name
ax.text(50, 96, "DistillToMCU", ha="center", va="center",
        fontsize=15, fontweight="bold", color="#111111", zorder=4)

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
