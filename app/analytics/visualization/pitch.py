"""Provider-aware pitch geometry and source-labelled football figure output."""

import math
import textwrap

from .style import COLORS, save


def draw_pitch(ax, length=120, width=80, *, centered=False, invert_y=True):
    """Draw StatsBomb event units, or regulation markings on a metric pitch."""
    from matplotlib.patches import Arc, Circle, Rectangle

    # StatsBomb uses yards-like event coordinates; tracking providers use metres.
    statsbomb = length == 120 and width == 80 and not centered
    penalty_depth, penalty_width = (18, 44) if statsbomb else (16.5, 40.32)
    goal_depth, goal_area_width = (6, 20) if statsbomb else (5.5, 18.32)
    spot, radius, goal_width = (12, 10, 8) if statsbomb else (11, 9.15, 7.32)
    x0, y0 = (-length / 2, -width / 2) if centered else (0, 0)
    cy = y0 + width / 2
    line = COLORS["pitch_line"]
    ax.set_facecolor(COLORS["pitch"])

    def rectangle(x, y, w, h):
        ax.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor=line, lw=1, zorder=1))

    rectangle(x0, y0, length, width)
    ax.plot([x0 + length / 2] * 2, [y0, y0 + width], color=line, lw=1, zorder=1)
    ax.add_patch(Circle((x0 + length / 2, cy), radius, fill=False, color=line, lw=1, zorder=1))
    ax.scatter(x0 + length / 2, cy, s=5, color=line, zorder=1)
    arc_angle = math.degrees(math.acos((penalty_depth - spot) / radius))
    for left in (True, False):
        goal_line = x0 if left else x0 + length
        rectangle(goal_line if left else goal_line - penalty_depth,
                  cy - penalty_width / 2, penalty_depth, penalty_width)
        rectangle(goal_line if left else goal_line - goal_depth,
                  cy - goal_area_width / 2, goal_depth, goal_area_width)
        rectangle(goal_line - 1.6 if left else goal_line, cy - goal_width / 2, 1.6, goal_width)
        penalty_x = goal_line + spot if left else goal_line - spot
        ax.scatter(penalty_x, cy, s=5, color=line, zorder=1)
        angles = (-arc_angle, arc_angle) if left else (180 - arc_angle, 180 + arc_angle)
        ax.add_patch(Arc((penalty_x, cy), 2 * radius, 2 * radius,
                         theta1=angles[0], theta2=angles[1], color=line, lw=1, zorder=1))
    for x, y, start in [(x0, y0, 0), (x0 + length, y0, 90),
                         (x0 + length, y0 + width, 180), (x0, y0 + width, 270)]:
        ax.add_patch(Arc((x, y), 2, 2, theta1=start, theta2=start + 90,
                         color=line, lw=.8, zorder=1))
    ax.set_xlim(x0 - 2.5, x0 + length + 2.5)
    ax.set_ylim((y0 + width + 2, y0 - 2) if invert_y else (y0 - 2, y0 + width + 2))
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def direction_arrow(ax, label="Attacking direction"):
    ax.annotate("", xy=(.72, -.055), xytext=(.28, -.055), xycoords="axes fraction",
                arrowprops={"arrowstyle": "->", "color": COLORS["muted"], "lw": 1},
                annotation_clip=False)
    ax.text(.5, -.09, label, transform=ax.transAxes, ha="center", va="top",
            color=COLORS["muted"], fontsize=10)


def save_figure(fig, name, note, *, scouting_root, output_dir):
    if note:
        fig.text(.055, .035, "\n".join(textwrap.fill(line, 138) for line in note.splitlines()),
                 fontsize=10, color=COLORS["muted"], va="bottom", linespacing=1.45)
    if name != "running-position":
        logo = scouting_root / "attribution" / "statsbomb_icon.png"
        if logo.exists():
            from matplotlib import pyplot as plt

            badge = fig.add_axes([.955, .03, .025, .05])
            badge.imshow(plt.imread(logo))
            badge.set_axis_off()
    return save(fig, output_dir, name)
