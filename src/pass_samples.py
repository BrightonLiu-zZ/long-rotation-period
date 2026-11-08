"""
Theissen 2026-10-30: could Emily Pass's samples fill the 12-20 d gap in the MEarth truth table?
Pass+2023 (AJ 166, 16): Table 1 machine-readable file, 123 active M dwarfs within 15 pc, with the source of each period.
Pass+2022 (ApJ 936, 109, arXiv 2206.15318): no machine-readable table found, so the M dwarf periods are parsed from the LaTeX source.
Inputs live in data/pass/ (downloaded by hand: AAS MRT and arXiv e-print).
"""

import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.io import ascii

from step2_ls import repo, truth_path

pass_dir = repo / "data" / "pass"
gap = (12.0, 20.0)  # d, the range the MEarth table cannot resolve
tess_refs = {8: "Pass+2023, TESS", 10: "Pass+2023, TESS + MEarth"}

# Pass+2022 Sec. 3: where each 10-25 d M dwarf period came from (read from the per-pair subsections of mmbinary.tex)
pass2022_sources = {
    "TIC 450297524": ("TESS sector 21, variability on the order of a sector", "no: TESS-measured"),
    "TIC 416857959": ("TESS SAP, sector 21; 'removed along with the TESS systematics in the PDCSAP light curve'", "no: TESS-measured"),
    "LHS 1377": ("MEarth", "no: never observed by TESS"),
    "LP 613-49": ("K2 and MEarth (Newton+2016 11.66 d)", "no: already in our MEarth table"),
    "GJ 669 A": ("MEarth (Newton+2016)", "no: already in our MEarth table"),
    "GJ 49": ("TESS SAP (9.45 d harmonic, 18.9 d adopted)", "no: TESS-measured"),
}


def leading_float(cell: str) -> float:
    """
    First number in a LaTeX table cell: '7.611$^{a}$' -> 7.611, '0.534/0.444' -> 0.534, '' -> nan.
    """
    match = re.search(r"\d+(\.\d+)?", cell.replace("$\\leq$", ""))
    return float(match.group()) if match else np.nan


def table_rows(tex: str, label: str) -> list[list[str]]:
    """
    Data rows of the deluxetable whose caption contains label, split on '&'.
    """
    start = tex.index(label)
    body = tex[tex.index("\\startdata", start):tex.index("\\enddata", start)]
    rows = []
    for line in body.splitlines():
        if "&" in line and "multicolumn" not in line:
            rows.append([c.strip() for c in line.strip().rstrip("\\").split("&")])
    return rows


def load_pass2022() -> pd.DataFrame:
    """
    One row per M dwarf with a period in the three Pass+2022 tables that list M dwarf periods.
    M-M pairs: both components; WD-M and FGK-M pairs: the M dwarf only.
    """
    tex = (pass_dir / "src_2206.15318" / "mmbinary.tex").read_text(encoding="utf-8")
    rows = []
    for cells in table_rows(tex, "Masses and rotation periods for 25 nearby wide M-M multiples"):
        rows.append({"star": cells[0], "pair": "M + M", "period": leading_float(cells[4])})
        rows.append({"star": cells[1], "pair": "M + M", "period": leading_float(cells[5])})
    for cells in table_rows(tex, "Properties of the white dwarf -- M dwarf pairs"):
        rows.append({"star": cells[0], "pair": "WD + M", "period": leading_float(cells[2])})
    for cells in table_rows(tex, "Properties of the active M dwarfs with FGK primaries"):
        rows.append({"star": cells[0], "pair": "FGK + M", "period": leading_float(cells[3])})
    table = pd.DataFrame(rows).dropna(subset=["period"])
    table["star"] = table["star"].str.replace(r"\*+$", "", regex=True).str.replace(r"\s+", " ", regex=True)
    return table.reset_index(drop=True)


def load_pass2023() -> pd.DataFrame:
    """
    Pass+2023 Table 1, with each star labelled: already in our MEarth table / new, period from TESS / new, other source.
    """
    table = ascii.read(pass_dir / "pass2023_t1_mrt.txt", format="cds").to_pandas()
    table["twomass"] = table["2MASS"].str.replace("2MASS J", "", regex=False).str.strip()
    truth = pd.read_csv(truth_path)
    ours = set(truth[truth["sample"].str.startswith("mearth")]["twomass"].astype(str).str.replace("J", "", regex=False).str.strip())
    refs = pd.to_numeric(table["Ref"], errors="coerce")
    table["group"] = np.where(table["twomass"].isin(ours), "already in our MEarth table", np.where(refs.isin(tess_refs.keys()), "new, period measured with TESS", "new, other source"))
    return table[["Name", "twomass", "Per", "Ref", "group"]].rename(columns={"Per": "period"})


def plot_periods(pass2023: pd.DataFrame, pass2022: pd.DataFrame, log_x: bool = False) -> plt.Figure:
    """
    Period histograms for both samples; shaded band = 12-20 d gap.
    Linear axis with 2-day bins by default; log_x=True spreads out the < 2 d pile-up.
    """
    if log_x:
        bins = np.logspace(-1, np.log10(200), 34)
    else:
        bins = np.arange(0, 162, 2)  # d, 2-day bins so the 12-20 d band lines up with bin edges
    colors = {"already in our MEarth table": "#a8a7a3", "new, period measured with TESS": "#eb6834", "new, other source": "#2a78d6"}
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
    stacks = [pass2023[pass2023["group"] == g]["period"].dropna() for g in colors]
    axes[0].hist(stacks, bins=bins, stacked=True, color=list(colors.values()), label=list(colors.keys()))
    axes[0].set_title(f"Pass+2023: {pass2023['period'].notna().sum()} active M dwarfs with a period", fontsize=10)
    axes[0].legend(frameon=False, fontsize=8, loc="upper right")
    axes[1].hist(pass2022["period"], bins=bins, color="#2a78d6")
    axes[1].set_title(f"Pass+2022: {len(pass2022)} M dwarfs in wide binaries", fontsize=10)
    for ax in axes:
        ax.axvspan(*gap, color="#eda100", alpha=0.25, linewidth=0, zorder=0)
        if log_x:
            ax.set_xscale("log")
        ax.set_xlabel("rotation period (d); shaded = 12-20 d")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("stars")
    fig.tight_layout()
    return fig
