# plotting/palettes.py

PALETTES = {
    "nature": ["#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e", "#e6ab02"],
    "muted": ["#4C72B0", "#55A868", "#C44E52", "#8172B3", "#CCB974", "#64B5CD"],
}


def get_palette(name: str) -> list[str]:
    return PALETTES[name]
