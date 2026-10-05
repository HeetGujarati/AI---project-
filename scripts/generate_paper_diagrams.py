#!/usr/bin/env python3
"""
generate_paper_diagrams.py
Generates publication-quality IEEE diagrams for LoRA+, Dyn-LoRA+, and low-rank matrix decomposition:
1. arch_decomposition_diagram.png: Architectural schematic with frozen base model & trainable adapters (LoRA vs LoRA+ vs Dyn-LoRA+).
2. svd_decomposition_diagram.png: Mathematical matrix decomposition, SVD representation, and gradient flow at initialization & finetuning.
Directly mirrors the user's uploaded reference diagram style (with drawn ice block and fire flame symbols)!
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Rectangle, Polygon
import numpy as np


def draw_ice_cube(ax, x, y, size=1.5):
    """Draws an aesthetic stylized ice cube block for frozen weights."""
    s = size
    # Front face
    front = Polygon([[x - s*0.5, y - s*0.5], [x + s*0.5, y - s*0.5],
                     [x + s*0.5, y + s*0.5], [x - s*0.5, y + s*0.5]],
                    closed=True, facecolor="#bae6fd", edgecolor="#0284c7", linewidth=0.9, zorder=10)
    ax.add_patch(front)
    # Top face
    top = Polygon([[x - s*0.5, y + s*0.5], [x + s*0.5, y + s*0.5],
                   [x + s*0.8, y + s*0.8], [x - s*0.2, y + s*0.8]],
                  closed=True, facecolor="#e0f2fe", edgecolor="#0284c7", linewidth=0.9, zorder=10)
    ax.add_patch(top)
    # Right face
    right = Polygon([[x + s*0.5, y - s*0.5], [x + s*0.8, y - s*0.2],
                     [x + s*0.8, y + s*0.8], [x + s*0.5, y + s*0.5]],
                    closed=True, facecolor="#7dd3fc", edgecolor="#0284c7", linewidth=0.9, zorder=10)
    ax.add_patch(right)
    # Ice highlight shine
    ax.plot([x - s*0.25, x + s*0.2], [y + s*0.2, y - s*0.25], color="#ffffff", lw=1.1, zorder=11)


def draw_flame(ax, x, y, size=1.5):
    """Draws an aesthetic stylized fire flame for trainable weights."""
    s = size
    # Outer orange-red flame
    outer_verts = [
        [x, y - s*0.5],
        [x + s*0.4, y - s*0.2],
        [x + s*0.5, y + s*0.2],
        [x + s*0.3, y + s*0.6],
        [x + s*0.08, y + s*0.95],
        [x - s*0.08, y + s*0.55],
        [x - s*0.35, y + s*0.35],
        [x - s*0.45, y - s*0.1],
        [x - s*0.25, y - s*0.4],
        [x, y - s*0.5]
    ]
    flame_outer = Polygon(outer_verts, closed=True, facecolor="#ef4444", edgecolor="#b91c1c", linewidth=0.8, zorder=10)
    ax.add_patch(flame_outer)

    # Inner bright yellow flame
    inner_verts = [
        [x, y - s*0.38],
        [x + s*0.2, y - s*0.15],
        [x + s*0.15, y + s*0.15],
        [x, y + s*0.5],
        [x - s*0.15, y + s*0.15],
        [x - s*0.15, y - s*0.15],
        [x, y - s*0.38]
    ]
    flame_inner = Polygon(inner_verts, closed=True, facecolor="#fef08a", edgecolor="#f59e0b", linewidth=0.6, zorder=11)
    ax.add_patch(flame_inner)


def create_arch_decomposition_diagram(output_path: str = "experiments/plots/arch_decomposition_diagram.png"):
    """
    Creates Figure 1: Architectural decomposition of LoRA, LoRA+, and Dyn-LoRA+.
    """
    fig, ax = plt.subplots(figsize=(7.6, 4.2), dpi=300)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 52)
    ax.axis("off")

    # Main outer card
    bg = FancyBboxPatch((0.5, 0.5), 99, 51, boxstyle="round,pad=0.2,rounding_size=1.0",
                        facecolor="#fbfcfd", edgecolor="#cbd5e1", linewidth=1.0)
    ax.add_patch(bg)

    # Title Banner
    ax.text(50, 48.8, "LoRA / LoRA+ / Dyn-LoRA+ Architectural Decomposition & Finetuning Mechanics",
            ha="center", va="center", fontsize=9.2, fontweight="bold", color="#0f172a", fontfamily="serif")

    # ================= LEFT PANEL: Forward Bypass (x: 2 to 49) =================
    box_fwd = FancyBboxPatch((2, 2.5), 47, 44, boxstyle="round,pad=0.3,rounding_size=0.8",
                             facecolor="#ffffff", edgecolor="#e2e8f0", linewidth=0.8)
    ax.add_patch(box_fwd)
    ax.text(25.5, 44.5, "(a) Forward Computation & Bypass Flow", ha="center", va="center",
            fontsize=8.3, fontweight="bold", color="#1e293b", fontfamily="serif")

    # Input x
    x_box = FancyBboxPatch((3.5, 21.5), 6.5, 7, boxstyle="round,pad=0.1,rounding_size=0.5",
                           facecolor="#f1f5f9", edgecolor="#64748b", linewidth=1.0)
    ax.add_patch(x_box)
    ax.text(6.75, 25, "Input\n$x \\in \\mathbb{R}^{d_{in}}$", ha="center", va="center", fontsize=7.0, color="#1e293b", fontfamily="serif")

    # Upper Branch: Frozen Base Model W0
    w0_box = FancyBboxPatch((16, 31.5), 17, 10, boxstyle="round,pad=0.2,rounding_size=0.6",
                            facecolor="#e0f2fe", edgecolor="#0284c7", linewidth=1.2)
    ax.add_patch(w0_box)
    ax.text(24.5, 38, "Frozen Base Model\n$W_0 \\in \\mathbb{R}^{d_{out} \\times d_{in}}$",
            ha="center", va="center", fontsize=7.2, fontweight="bold", color="#0369a1", fontfamily="serif")
    
    # Draw ice cube
    draw_ice_cube(ax, 17.5, 33.5, size=1.6)
    ax.text(25.5, 33.5, "FROZEN (No Grad)", ha="center", va="center", fontsize=6.0, fontweight="bold", color="#0369a1")

    # Lower Branch: Low-Rank Adapter Bypass
    # Matrix A
    a_box = FancyBboxPatch((15, 6), 12.5, 11, boxstyle="round,pad=0.2,rounding_size=0.6",
                           facecolor="#dcfce7", edgecolor="#16a34a", linewidth=1.2)
    ax.add_patch(a_box)
    ax.text(21.25, 13.5, "Down-Proj $A$\n$r \\times d_{in}$\nInit: $\\mathcal{N}(0, \\sigma^2)$",
            ha="center", va="center", fontsize=6.4, fontweight="bold", color="#15803d", fontfamily="serif")
    draw_flame(ax, 18.0, 8.5, size=1.4)
    ax.text(23.5, 8.5, "TRAIN ($\\eta_A$)", ha="center", va="center", fontsize=5.8, fontweight="bold", color="#15803d")

    # Matrix B
    b_box = FancyBboxPatch((31.5, 6), 12.5, 11, boxstyle="round,pad=0.2,rounding_size=0.6",
                           facecolor="#fee2e2", edgecolor="#dc2626", linewidth=1.2)
    ax.add_patch(b_box)
    ax.text(37.75, 13.5, "Up-Proj $B$\n$d_{out} \\times r$\nInit: $B = 0$",
            ha="center", va="center", fontsize=6.4, fontweight="bold", color="#b91c1c", fontfamily="serif")
    draw_flame(ax, 34.5, 8.5, size=1.4)
    ax.text(40.0, 8.5, "TRAIN ($\\eta_B$)", ha="center", va="center", fontsize=5.8, fontweight="bold", color="#b91c1c")

    # Arrows in flow graph
    ax.annotate("", xy=(16, 36.5), xytext=(10, 27), arrowprops=dict(arrowstyle="->", color="#0284c7", lw=1.3))
    ax.annotate("", xy=(15, 11.5), xytext=(10, 23), arrowprops=dict(arrowstyle="->", color="#16a34a", lw=1.3))
    ax.annotate("", xy=(31.5, 11.5), xytext=(27.5, 11.5), arrowprops=dict(arrowstyle="->", color="#dc2626", lw=1.3))
    ax.text(29.5, 13.5, "$r$", ha="center", va="center", fontsize=6.8, fontstyle="italic", color="#64748b")

    # Summation node
    sum_circle = plt.Circle((42.5, 27), 2.2, facecolor="#f8fafc", edgecolor="#475569", linewidth=1.2)
    ax.add_patch(sum_circle)
    ax.text(42.5, 27, "+", ha="center", va="center", fontsize=10.5, fontweight="bold", color="#1e293b")

    # Upper to sum
    ax.annotate("", xy=(40.5, 28), xytext=(33, 35), arrowprops=dict(arrowstyle="->", color="#0284c7", lw=1.3))
    # Lower to sum with scaling
    ax.annotate("", xy=(41.5, 25), xytext=(38.5, 17), arrowprops=dict(arrowstyle="->", color="#dc2626", lw=1.3))
    ax.text(44.2, 19, "$\\times \\frac{\\alpha}{r}$", fontsize=7.2, color="#b91c1c", fontweight="bold")

    # Output representation h
    ax.annotate("", xy=(47, 27), xytext=(44.7, 27), arrowprops=dict(arrowstyle="->", color="#1e293b", lw=1.4))
    ax.text(47.5, 27, "$h$", ha="left", va="center", fontsize=8.4, fontweight="bold", color="#1e293b")

    # ================= RIGHT PANEL: Learning Rate Asymmetry (x: 51 to 98) =================
    box_opt = FancyBboxPatch((51, 2.5), 47, 44, boxstyle="round,pad=0.3,rounding_size=0.8",
                            facecolor="#ffffff", edgecolor="#e2e8f0", linewidth=0.8)
    ax.add_patch(box_opt)
    ax.text(74.5, 44.5, "(b) Learning Rate Asymmetry Schemes", ha="center", va="center",
            fontsize=8.3, fontweight="bold", color="#1e293b", fontfamily="serif")

    # Card 1: Standard LoRA
    c1 = FancyBboxPatch((52.5, 30.5), 44.5, 11.5, boxstyle="round,pad=0.2,rounding_size=0.5",
                        facecolor="#f8fafc", edgecolor="#cbd5e1", linewidth=0.8)
    ax.add_patch(c1)
    ax.text(54, 39.5, "1. Standard LoRA (Hu et al., 2021)", fontsize=6.8, fontweight="bold", color="#334155")
    b1 = FancyBboxPatch((81.5, 38.3), 14.5, 2.4, boxstyle="round,pad=0.1,rounding_size=0.3",
                        facecolor="#94a3b8", edgecolor="none")
    ax.add_patch(b1)
    ax.text(88.75, 39.5, "SYMMETRIC ($\\lambda=1$)", ha="center", va="center", fontsize=4.8, fontweight="bold", color="#ffffff")
    ax.text(54, 34.5, "• Uniform step size across adapters: $\\eta_A = \\eta_B = \\eta$\n"
                      "• Bottleneck: At step 0, $B=0 \\rightarrow \\nabla_A L = 0$ (lagging updates)\n"
                      "• Under NTK infinite-width limit, matrix $B$ updates too slowly.",
            fontsize=5.6, color="#475569", linespacing=1.25)

    # Card 2: Static LoRA+
    c2 = FancyBboxPatch((52.5, 17.0), 44.5, 12, boxstyle="round,pad=0.2,rounding_size=0.5",
                        facecolor="#eff6ff", edgecolor="#bfdbfe", linewidth=0.8)
    ax.add_patch(c2)
    ax.text(54, 26.2, "2. LoRA+ (Hayou et al., 2024)", fontsize=6.8, fontweight="bold", color="#1d4ed8")
    b2 = FancyBboxPatch((80.5, 25.0), 15.5, 2.4, boxstyle="round,pad=0.1,rounding_size=0.3",
                        facecolor="#2563eb", edgecolor="none")
    ax.add_patch(b2)
    ax.text(88.25, 26.2, "STATIC ASYM ($\\lambda=4..8$)", ha="center", va="center", fontsize=4.8, fontweight="bold", color="#ffffff")
    ax.text(54, 21.0, "• Decoupled rates: $\\eta_B = \\lambda \\cdot \\eta_A$ with static ratio $\\lambda > 1$\n"
                      "• Rapid symmetry breaking allows $A$ and $B$ to harmonize early.\n"
                      "• Vulnerability: Fixed high ratio ($\\lambda=32$) causes late chatter.",
            fontsize=5.6, color="#1e3a8a", linespacing=1.25)

    # Card 3: Dyn-LoRA+ (Proposed Extension)
    c3 = FancyBboxPatch((52.5, 3.5), 44.5, 12, boxstyle="round,pad=0.2,rounding_size=0.5",
                        facecolor="#fef2f2", edgecolor="#fecaca", linewidth=1.0)
    ax.add_patch(c3)
    ax.text(54, 12.8, "3. Dyn-LoRA+ (Proposed)", fontsize=6.8, fontweight="bold", color="#b91c1c")
    b3 = FancyBboxPatch((82.5, 11.6), 13.5, 2.4, boxstyle="round,pad=0.1,rounding_size=0.3",
                        facecolor="#dc2626", edgecolor="none")
    ax.add_patch(b3)
    ax.text(89.25, 12.8, "DYNAMIC ($\\lambda: 16\\to 1$)", ha="center", va="center", fontsize=4.8, fontweight="bold", color="#ffffff")
    ax.text(54, 7.6, "• Dynamic cosine decay: $\\lambda(t) \\in [\\lambda_{min}, \\lambda_{max}]$ annealed per step\n"
                     "• Starts high ($\\lambda=16$) for fast symmetry breaking; anneals to 1.0\n"
                     "• Optimal validation loss (1.0342) and eliminates late chatter.",
            fontsize=5.6, color="#7f1d1d", linespacing=1.25)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Generated Figure 1: {output_path}")


def create_svd_decomposition_diagram(output_path: str = "experiments/plots/svd_decomposition_diagram.png"):
    """
    Creates Figure 2: Pure LoRA+ & Dyn-LoRA+ Optimization Dynamics, Parameter States,
    and Backward Gradient Velocity Flow (No PiSSA/SVD confusion; 100% authentic LoRA+ theory).
    """
    fig, ax = plt.subplots(figsize=(7.6, 4.5), dpi=300)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 58)
    ax.axis("off")

    # Main outer frame
    bg = FancyBboxPatch((0.5, 0.5), 99, 57, boxstyle="round,pad=0.2,rounding_size=1.0",
                        facecolor="#ffffff", edgecolor="#cbd5e1", linewidth=1.0)
    ax.add_patch(bg)

    # Title
    ax.text(50, 55.0, "LoRA+ Asymmetric Gradient Flow, Velocity Dynamics & Dynamic Scheduling",
            ha="center", va="center", fontsize=9.2, fontweight="bold", color="#0f172a", fontfamily="serif")

    # ================= ROW 1: (a) Forward Low-Rank Adapter Formulation & Initial States =================
    ax.text(3.5, 51.5, "(a) Forward Low-Rank Adapter Formulation & Initial Parameter States:",
            fontsize=8.1, fontweight="bold", color="#1e293b", fontfamily="serif")

    # W box
    w_box = FancyBboxPatch((4.0, 40.0), 9.5, 9.5, boxstyle="square,pad=0", facecolor="#3b82f6", edgecolor="#1d4ed8", linewidth=1.0)
    ax.add_patch(w_box)
    ax.text(8.75, 45.8, "$W$", ha="center", va="center", fontsize=8.2, fontweight="bold", color="#ffffff")
    ax.text(8.75, 42.0, "$d_{out} \\times d_{in}$", ha="center", va="center", fontsize=5.2, color="#dbeafe")

    ax.text(15.5, 44.75, "$=$", ha="center", va="center", fontsize=12, fontweight="bold", color="#334155")

    # W_frozen box with ice cube
    w_frz = FancyBboxPatch((18.0, 40.0), 15.0, 9.5, boxstyle="square,pad=0", facecolor="#60a5fa", edgecolor="#2563eb", linewidth=1.0)
    ax.add_patch(w_frz)
    draw_ice_cube(ax, 21.5, 44.75, size=1.6)
    ax.text(27.5, 46.5, "$W_0$ (Frozen)", ha="center", va="center", fontsize=6.8, fontweight="bold", color="#ffffff")
    ax.text(27.5, 42.5, "requires_grad=F", ha="center", va="center", fontsize=5.2, color="#ffffff")

    ax.text(35.0, 44.75, "$+$", ha="center", va="center", fontsize=12, fontweight="bold", color="#334155")

    # Scaling factor alpha/r
    ax.text(38.0, 44.75, "$\\frac{\\alpha}{r}$", ha="center", va="center", fontsize=10.5, fontweight="bold", color="#b91c1c")
    ax.text(40.8, 44.75, "(", ha="center", va="center", fontsize=16, color="#475569")

    # B block (Up-projection)
    b_ft = FancyBboxPatch((42.5, 40.0), 6.5, 9.5, boxstyle="square,pad=0", facecolor="#ef4444", edgecolor="#b91c1c", linewidth=1.0)
    ax.add_patch(b_ft)
    ax.text(45.75, 46.2, "$B$", ha="center", va="center", fontsize=7.2, fontweight="bold", color="#ffffff")
    draw_flame(ax, 45.75, 42.6, size=1.4)

    ax.text(51.5, 44.75, "$\\times$", ha="center", va="center", fontsize=9, color="#475569")

    # A block (Down-projection)
    a_ft = FancyBboxPatch((54.0, 42.5), 15.0, 4.5, boxstyle="square,pad=0", facecolor="#10b981", edgecolor="#047857", linewidth=1.0)
    ax.add_patch(a_ft)
    ax.text(61.5, 44.75, "$A$", ha="center", va="center", fontsize=7.2, fontweight="bold", color="#ffffff")
    draw_flame(ax, 61.5, 41.2, size=1.4)

    ax.text(70.5, 44.75, ")", ha="center", va="center", fontsize=16, color="#475569")

    # Annotations below B and A
    ax.text(45.75, 37.8, "$B_0 = 0$\n[Trainable: $\\eta_B$]", ha="center", va="center", fontsize=5.6, fontweight="bold", color="#b91c1c")
    ax.text(61.5, 37.8, "$A_0 \\sim \\mathcal{N}(0, \\sigma^2)$\n[Trainable: $\\eta_A$]", ha="center", va="center", fontsize=5.6, fontweight="bold", color="#047857")

    # Dimensions annotation badge
    dim_box = FancyBboxPatch((73.0, 40.5), 23.5, 8.5, boxstyle="round,pad=0.2,rounding_size=0.5",
                             facecolor="#f8fafc", edgecolor="#cbd5e1", linewidth=0.8)
    ax.add_patch(dim_box)
    ax.text(84.75, 46.2, "Model Hyperparameters:", ha="center", va="center", fontsize=6.0, fontweight="bold", color="#334155")
    ax.text(84.75, 42.6, "$d_{in}=d_{out}=1536$\n$r=8, \\alpha=16$ (scaling $= 2.0$)\nActive Params: 9.23M (0.59%)",
            ha="center", va="center", fontsize=5.0, color="#475569", linespacing=1.2)

    # ================= ROW 2: (b) Backward Gradient Flow at Step t=0 =================
    ax.plot([2, 98], [35.5, 35.5], color="#e2e8f0", linewidth=0.8, linestyle="--")
    ax.text(3.5, 33.2, "(b) Backward Gradient Flow at Step t=0: The Origin of Velocity Starvation",
            fontsize=8.1, fontweight="bold", color="#1e293b", fontfamily="serif")

    # Card Left: Gradient on A
    ga_box = FancyBboxPatch((3.0, 20.0), 45.5, 11.5, boxstyle="round,pad=0.2,rounding_size=0.5",
                            facecolor="#f0fdf4", edgecolor="#86efac", linewidth=0.8)
    ax.add_patch(ga_box)
    ax.text(4.8, 29.2, "1. Backpropagated Gradient on Matrix A:", fontsize=6.4, fontweight="bold", color="#166534")
    ax.text(4.8, 24.5, "$\\nabla_A L = \\frac{\\alpha}{r} B^T (\\partial L / \\partial h) x^T$\n"
                     "• Step $t=0$: $B=0 \\rightarrow \\|\\nabla_A L\\| \\approx 0$ (Velocity Starvation)\n"
                     "• Matrix $A$ remains frozen until $B$ departs origin.",
            fontsize=5.4, color="#14532d", linespacing=1.25)

    # Card Right: Gradient on B
    gb_box = FancyBboxPatch((50.5, 20.0), 46.5, 11.5, boxstyle="round,pad=0.2,rounding_size=0.5",
                            facecolor="#fef2f2", edgecolor="#fca5a5", linewidth=0.8)
    ax.add_patch(gb_box)
    ax.text(52.5, 29.2, "2. Gradient on Matrix B & NTK Width Scaling Bottleneck:", fontsize=6.4, fontweight="bold", color="#991b1b")
    ax.text(52.5, 24.5, "$\\nabla_B L = \\frac{\\alpha}{r} (\\partial L / \\partial h) (A x)^T \\rightarrow \\|\\nabla_B L\\| \\sim \\Theta(1)$\n"
                     "• As width $d \\to \\infty$, uniform step $\\eta_B = \\eta_A$ causes update speed $\\Delta B \\sim \\mathcal{O}(1/d)$.\n"
                     "• Matrix $B$ lags behind, causing sub-optimal feature learning.",
            fontsize=5.4, color="#7f1d1d", linespacing=1.25)

    # ================= ROW 3: (c) Optimization Solutions =================
    ax.plot([2, 98], [17.5, 17.5], color="#e2e8f0", linewidth=0.8, linestyle="--")
    ax.text(3.5, 15.2, "(c) Optimization Solutions: LoRA+ Static Asymmetry vs. Dyn-LoRA+ Dynamic Annealing",
            fontsize=8.1, fontweight="bold", color="#1e293b", fontfamily="serif")

    # Card Left: LoRA+
    sol_loraplus = FancyBboxPatch((3.0, 2.0), 45.5, 11.5, boxstyle="round,pad=0.2,rounding_size=0.5",
                                  facecolor="#eff6ff", edgecolor="#93c5fd", linewidth=0.8)
    ax.add_patch(sol_loraplus)
    ax.text(4.8, 11.2, "1. LoRA+ (Hayou et al., ICML 2024):", fontsize=6.4, fontweight="bold", color="#1e40af")
    ax.text(4.8, 6.5, "Decoupled Ratio: $\\eta_B = \\lambda \\cdot \\eta_A$ with static ratio $\\lambda > 1$ (optimal $\\lambda \\in [4, 8]$)\n"
                     "• Accelerates matrix $B$ to kickstart gradients on $A$.\n"
                     "• Vulnerability: Fixed high ratio ($\\lambda=32$) causes late chatter & loss rebound.",
            fontsize=5.4, color="#1e3a8a", linespacing=1.25)

    # Card Right: Dyn-LoRA+
    sol_dyn = FancyBboxPatch((50.5, 2.0), 46.5, 11.5, boxstyle="round,pad=0.2,rounding_size=0.5",
                             facecolor="#fff1f2", edgecolor="#fda4af", linewidth=0.8)
    ax.add_patch(sol_dyn)
    ax.text(52.5, 11.2, "2. Dyn-LoRA+ (Proposed Dynamic Extension):", fontsize=6.4, fontweight="bold", color="#9f1239")
    ax.text(52.5, 6.5, "Cosine Schedule: $\\lambda(t) = \\lambda_{min} + \\frac{1}{2}(\\lambda_{max} - \\lambda_{min}) [1 + \\cos(\\pi t / T)]$\n"
                     "• Starts high ($\\lambda=16$) for fast symmetry breaking; anneals smoothly to 1.0.\n"
                     "• Eliminates late-stage gradient chatter, reaching optimal validation loss 1.0342.",
            fontsize=5.4, color="#881337", linespacing=1.25)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Generated Figure 2: {output_path}")


if __name__ == "__main__":
    create_arch_decomposition_diagram()
    create_svd_decomposition_diagram()
