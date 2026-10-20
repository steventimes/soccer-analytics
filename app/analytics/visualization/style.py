"""Shared typography and semantic colours for the project's analytical figures."""

from pathlib import Path
import textwrap

COLORS = {
    "ink": "#203344",
    "muted": "#647382",
    "grid": "#E2E8EE",
    "blue": "#2468A0",
    "orange": "#C5722C",
    "neutral": "#98A5B1",
    "light": "#F3F6F9",
    "pitch": "#F6F8FA",
    "pitch_line": "#B6C2CD",
}


def apply_style():
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 11,
        "axes.labelsize": 11, "axes.titlesize": 12,
        "axes.titleweight": "bold", "axes.titlepad": 14,
        "text.color": COLORS["ink"], "axes.labelcolor": COLORS["muted"],
        "xtick.color": COLORS["muted"], "ytick.color": COLORS["ink"],
        "axes.edgecolor": COLORS["grid"], "axes.linewidth": .8,
        "axes.spines.top": False, "axes.spines.right": False,
        "xtick.labelsize": 11, "ytick.labelsize": 11,
        "grid.color": COLORS["grid"], "grid.alpha": 1, "grid.linewidth": .7,
        "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white", "savefig.dpi": 300,
        "svg.fonttype": "none", "pdf.fonttype": 42,
        "axes.prop_cycle": plt.cycler(color=[COLORS["blue"], COLORS["orange"], COLORS["neutral"]]),
    })


def decorate(fig, title, subtitle="", note="", *, bottom=.16, top=.79):
    """Reserve figure-level space instead of squeezing prose into plotting axes."""
    fig.subplots_adjust(left=.18, right=.94, bottom=bottom, top=top, wspace=.45, hspace=.65)
    fig.text(.055, .95, title, fontsize=18, weight="bold", va="top", color=COLORS["ink"])
    if subtitle:
        fig.text(.055, .895, subtitle, fontsize=10, va="top", color=COLORS["muted"])
    if note:
        wrapped = "\n".join(textwrap.fill(line, 120) for line in note.splitlines())
        fig.text(.055, .035, wrapped, fontsize=10, va="bottom", color=COLORS["muted"], linespacing=1.5)


def save(fig, output, name, formats=("png", "svg")):
    import matplotlib.pyplot as plt

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    paths = []
    for suffix in formats:
        path = output / f"{name}.{suffix}"
        metadata = {"Creator": "soccer-analytics"}
        if suffix == "png": metadata = {"Software": "soccer-analytics"}
        if suffix == "svg": metadata["Date"] = None
        if suffix == "pdf": metadata.update(CreationDate=None, ModDate=None)
        fig.savefig(path, dpi=300, metadata=metadata)
        if suffix == "svg":
            _describe_svg(path, fig)
        paths.append(path)
    plt.close(fig)
    return paths


def _describe_svg(path, fig):
    """Keep the figure's title and scope available when its SVG is opened alone."""
    import xml.etree.ElementTree as ET

    namespace = "http://www.w3.org/2000/svg"
    ET.register_namespace("", namespace)
    ET.register_namespace("xlink", "http://www.w3.org/1999/xlink")
    tree = ET.parse(path)
    root = tree.getroot()
    labels = [text.get_text() for text in fig.texts if text.get_text().strip()]
    title = labels[0] if labels else path.stem.replace("-", " ")
    description = " ".join(labels[1:]) or title
    identity = "figure-" + path.stem.replace("_", "-").lower()
    root.set("role", "img")
    root.set("aria-labelledby", f"{identity}-title {identity}-desc")
    for index, (tag, text) in enumerate((("title", title), ("desc", description))):
        element = ET.Element(f"{{{namespace}}}{tag}", {"id": f"{identity}-{tag}"})
        element.text = text
        root.insert(index, element)
    tree.write(path, encoding="utf-8", xml_declaration=True)
