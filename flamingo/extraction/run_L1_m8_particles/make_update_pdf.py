"""FLAMINGO IA status update PDF. Tables are filled from stitch_summary.txt, the params files and logs/run.log;
edit DATE, NEW and the prose for each update."""
import os
import re

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (CondPageBreak, Image, KeepTogether, PageBreak, Paragraph,
                                SimpleDocTemplate, Spacer, Table, TableStyle)

RUN = "/Users/yijiezhu/ia_power_flamingo/run_L1_m8_particles"
DATE = "2026-10-27"
OUT = f"/Users/yijiezhu/ia_power_flamingo/FLAMINGO_IA_update_{DATE}.pdf"
PLOTS = os.path.join(RUN, "results", "plots")

# results from the run's own summary file (written by plot_stitch_flamingo.py)
summ = [l.split() for l in open(os.path.join(RUN, "results", "stitched_all", "stitch_summary.txt"))
        if l.strip() and not l.startswith("#")]
# TNG300 a_nla(z) +- err, by TNG snapshot redshift
TNG = {float(r[1]): (float(r[2]), float(r[3])) for r in
       (l.split() for l in open("/Users/yijiezhu/Downloads/ia_halofit_v2/stitch_demo/stitch_summary.txt"))
       if r and not r[0].startswith("#")}
NEW = (48, 38, 18, 14)


def tng_at(z):
    """TNG300 a_nla at the nearest TNG redshift (within 0.06), as (z_tng, a, err), else None."""
    zt = min(TNG, key=lambda t: abs(t - z))
    return (zt,) + TNG[zt] if abs(zt - z) <= 0.06 else None


def params(sn):
    """(M_min, M_max, log M* range string) from params/particle_shapes_<sn>.dat."""
    d = dict(l.split()[:2] for l in open(os.path.join(RUN, "params", f"particle_shapes_{sn:03d}.dat"))
             if l.strip() and not l.startswith("%"))
    lo, hi = (10 + __import__("math").log10(float(d[k]) / 0.681) for k in ("M_min", "M_max"))
    return d["M_min"], d["M_max"], f"{lo:.2f} - {hi:.2f}"


MASS = {int(r[0]): params(int(r[0]))[2] for r in summ}


def wall_times():
    """Wall time per snapshot (minutes) from the '== snapshot' lines of logs/run.log."""
    from datetime import datetime
    ev = [(datetime.strptime(l[:19], "%Y-%m-%d %H:%M:%S"), l) for l in open(os.path.join(RUN, "logs", "run.log"))
          if re.match(r"\d{4}-", l) and "== snapshot" in l]
    out, start = {}, {}
    for t, l in ev:
        sn = int(re.search(r"snapshot (\d+)", l).group(1))
        if "finished" in l:
            out[sn] = round((t - start[sn]).total_seconds() / 60)
        else:
            start[sn] = t
    return out

ss = getSampleStyleSheet()
body = ParagraphStyle("b", parent=ss["Normal"], fontName="Helvetica", fontSize=9.5, leading=13, alignment=TA_LEFT,
                      spaceAfter=5)
small = ParagraphStyle("s", parent=body, fontSize=8, leading=10, spaceAfter=2)
cell = ParagraphStyle("c", parent=body, fontSize=8.3, leading=10.3, spaceAfter=0)
cellb = ParagraphStyle("cb", parent=cell, fontName="Helvetica-Bold")
h1 = ParagraphStyle("h1", parent=ss["Title"], fontSize=17, leading=21, spaceAfter=2, alignment=TA_LEFT)
h2 = ParagraphStyle("h2", parent=ss["Heading2"], fontSize=12, leading=15, spaceBefore=9, spaceAfter=4,
                    textColor=colors.HexColor("#1f3b5c"))
sub = ParagraphStyle("sub", parent=body, textColor=colors.HexColor("#555555"), fontSize=9)
bullet = ParagraphStyle("bl", parent=body, leftIndent=12, bulletIndent=2, spaceAfter=3)
TT = "<font face='Courier'>%s</font>"
MSUN = "M<sub>sun</sub>"


def P(t, st=body):
    return Paragraph(t, st)


def bl(t):
    return Paragraph(t, bullet, bulletText="•")


def table(rows, widths, hl=()):
    data = [[P(c, cellb if r == 0 else cell) for c in row] for r, row in enumerate(rows)]
    t = Table(data, colWidths=widths, repeatRows=1)
    st = [("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dde6f0")),
          ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#9aa8b8")),
          ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
          ("TOPPADDING", (0, 0), (-1, -1), 2.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5)]
    for r in hl:
        st.append(("BACKGROUND", (0, r), (-1, r), colors.HexColor("#fff4d6")))
    t.setStyle(TableStyle(st))
    return t


def img(path, width):
    from PIL import Image as PI
    w, h = PI.open(path).size
    return Image(path, width=width, height=width * h / w)



story = [P("FLAMINGO Intrinsic-Alignment Power Spectra: Status Update", h1),
         P(f"{DATE}. FLAMINGO L1_m8 (hydro, 1 Gpc box): z = 1.5, 2, 3 and 4 (snapshots 48, 38, 18, 14) finished today, so "
           "every planned redshift bin up to z = 3 is done, plus z = 4. Follows the 2026-10-26 update (z = 0.75, 1.0). Run folder " + TT % "ia_power_flamingo/run_L1_m8_particles" + ". All numbers in the TNG / "
           "CosmoLike conventions (" + TT % "sim_IA.c" + ").", sub)]

# 1 headline
rows = [["z", "Snap", f"log M<sub>*</sub> [{MSUN}]", "Galaxies", "e_rms", "bias (k = 0.05-0.3)",
         "<b>a<sub>NLA</sub> FLAMINGO</b>", "a<sub>NLA</sub> TNG300", "ratio"]]
hl = []
for i, r in enumerate(sorted(summ, key=lambda r: float(r[1]))):
    sn, z, a, ea, erms, b, ng = int(r[0]), float(r[1]), float(r[2]), float(r[3]), float(r[7]), float(r[8]), int(r[9])
    t = tng_at(z)
    if sn in NEW:
        hl.append(i + 1)
    tcell = "-"
    if t:
        tcell = f"{t[1]:.2f} ± {t[2]:.2f}" + (f" (z = {t[0]:g})" if abs(t[0] - z) > 0.015 else "")
    rows.append([f"{z:g}", str(sn), MASS.get(sn, ""), f"{ng:,}", f"{erms:.3f}", f"{b:.2f}",
                 f"<b>{a:.2f} ± {ea:.2f}</b>", tcell, f"{a / t[1]:.2f}" if t else "-"])
story += [P("1. Headline", h2),
          P("a<sub>NLA</sub> keeps rising roughly linearly: 4.94 at z = 1.5, 6.04 at z = 2, 8.32 at z = 3 and 10.0 at z = 4, "
            "about 2.1-2.3 per unit redshift above z = 1. TNG300 rises faster, so after crossing at z = 1 FLAMINGO falls below it: "
            "the FLAMINGO / TNG300 ratio is 0.74 at z = 1.5, 0.71 at z = 2 and 0.53 at z = 3. TNG300 has no z = 4 output. "
            "New rows highlighted:"),
          table(rows, [0.42 * inch, 0.42 * inch, 0.95 * inch, 0.78 * inch, 0.52 * inch, 0.72 * inch, 1.08 * inch,
                       1.45 * inch, 0.45 * inch], hl=hl),
          Spacer(1, 3),
          P("a<sub>NLA</sub>: low-k fit of P<sub>dE</sub> = A P<sub>NL</sub> at k = 0.05-0.3 h/Mpc, "
            "A = -a<sub>NLA</sub> C<sub>1</sub>&rho;<sub>cr</sub> &Omega;<sub>m</sub>/D(z), C<sub>1</sub>&rho;<sub>cr</sub> = 0.0134, "
            "shear units, k<sub>&perp;</sub> (same recipe as " + TT % "ia_halofit_v2/stitch.py --amp lowk" + "). "
            "TNG300 values from " + TT % "stitch_summary.txt" + " at the nearest TNG output; z = 0.75 has no TNG output, "
            "so it is compared with z = 0.70 (interpolating to 0.75 gives about 3.37 and a ratio of 1.12). Errors are "
            "mode-scatter only (underestimated, see Section 5).", small),
          bl("<b>FLAMINGO below TNG300 at z &ge; 1.5.</b> The deficit is visible across the whole P<sub>dE</sub> and "
             "P<sub>EE</sub> curves, not only in the fit range. Part of it may be selection: the upper mass edge rises "
             "only slowly (11.10, 11.23, 11.35) while FLAMINGO's 10.0 lower edge (100-particle limit) sits well above "
             "TNG's, so the two samples differ more at high z. This should be checked with matched mass bins before "
             "reading it as a physics difference."),
          bl("<b>P<sub>EE</sub></b> stays about 1.35-1.40x above NLA<super>2</super> at low k up to z = 3; at z = 4 it is "
             "1.62, but with only 0.53 M galaxies the low-k P<sub>EE</sub> points are noisy."),
          bl("<b>Sample size falls fast</b>: 4.81 M (z = 1.5), 3.66 M (2), 1.61 M (3), 0.53 M (4). Bias rises to 1.94, "
             "2.36, 3.56 and 5.26, e_rms to 0.323, 0.335, 0.358 and 0.372. The a<sub>NLA</sub> error grows to 0.13 at z = 4."),
          bl("<b>z = 4 mass cut is an extrapolation.</b> The target sample has no z = 4 bin, so snapshot 14 uses the "
             "z = 3 range (10.0 &lt; log M* &lt; 11.35).")]

# 2 this run
wt = wall_times()
rows2 = [["Target z", "Snapshot (measured z)", "Files streamed (hdfstream)",
          "Adopted log M<sub>*</sub>", "M_min - M_max [10<super>10</super> " + MSUN + "/h]", "Wall time"]]
for sn in NEW:
    r = next((r for r in summ if int(r[0]) == sn), None)
    if r is None:
        continue
    mmin, mmax, lm = params(sn)
    rows2.append([f"{float(r[1]):g}", f"{sn} ({float(r[1]):.2f})",
                  TT % f"SOAP-HBT/halo_properties_{sn:04d}" + ", " + TT % f"flamingo_{sn:04d}",
                  lm.replace("10.00", "10.0"), f"{mmin} - {mmax}", f"{wt[sn]} min" if sn in wt else "-"])
story += [P("2. What was run today", h2),
          P("Same code, shape definition and settings as the earlier runs (TNG unreduced inertia tensor of all bound "
            "star particles; Select all; MinStars 100; Ngrid 1024; nfold 1, 4, 16). Snapshots and mass ranges are "
            "those of plan v2, Sections 4-5, except snapshot 14 (z = 3 range carried to z = 4):"),
          table(rows2, [0.6 * inch, 1.05 * inch, 2.45 * inch, 0.9 * inch, 1.2 * inch, 0.65 * inch]),
          Spacer(1, 3),
          bl("Total 09:14-11:25. Streaming took 39, 28, 15 and 7 min: fewer star particles at high z, and the COSMA "
             "connection was back to full speed."),
          bl("<b>2.8 Gpc box checked and not usable.</b> L2p8_m8 (FLAMINGO-10K) is dark-matter only. L2p8_m9 star "
             "particles are 7.9e8 " + MSUN + " (1.0e8 at m8), so log M* = 10 is about 13 particles and MinStars 100 "
             "means log M* &gt; 10.9; the 2026-10-22 resolution test also gave a 16% lower a<sub>NLA</sub> at m9."),
          bl("A harmless " + TT % "RuntimeWarning" + " at z = 3 and 4: FLAMINGO's matter P(k) has 1-2 negative values at "
             "k &gt; 500 h/Mpc, far outside the range used; no NaN reaches the outputs."),
          bl("<b>Storage.</b> The 1024<super>3</super> grids (13 GB per nfold) were deleted right after their spectra "
             "were written. The galaxy catalogues (0.3-2.6 GB each, 24 GB for all 13) are kept because the plot and "
             "CosmoLike export scripts read them.")]

# 3 figures
story += [CondPageBreak(5.5 * inch), P("3. Results", h2),
          P(f"Density-shape and shape-shape spectra for all {len(summ)} redshifts, k<sub>&perp;</sub>, shear units, "
            "folds 1/4/16 stitched; dashed: NLA fit; green: TNG300 stitched at the nearest redshift (none within "
            "0.06 is shown as missing)."),
          img(os.path.join(PLOTS, "summary_spectra.png"), 7.0 * inch),
          Spacer(1, 6),
          KeepTogether([P("NLA amplitude against redshift:"), img(os.path.join(PLOTS, "summary_anla.png"), 4.6 * inch)])]
for sn in NEW:
    f = os.path.join(PLOTS, f"stitch_all_sn{sn}.png")
    if os.path.isfile(f):
        story += [CondPageBreak(4.2 * inch), P(f"NLA fit and stitch, snapshot {sn}:", small), img(f, 7.0 * inch)]

# 4 files
snl = ", ".join(str(int(r[0])) for r in summ)
story += [P("4. Files", h2),
          bl(TT % "run_L1_m8_particles/results/PS/" + ": raw Pds-format spectra (P{hh,hE,hB,EE,EB,BB}, l = 0, 2, 4; "
             f"nfold 1, 4, 16) for snapshots {snl}."),
          bl(TT % "results/stitched_all/" + ": stitched tables and " + TT % "stitch_summary.txt"
             + f" (all {len(summ)} snapshots)."),
          bl(TT % "results/plots/" + ": " + TT % "summary_spectra.png" + ", " + TT % "summary_anla.png" + ", "
             + TT % "stitch_all_sn&lt;sn&gt;.png" + "."),
          bl(TT % "cosmolike_all/" + f": files in the " + TT % "sim_IA.c" + f" layout for all {len(summ)} snapshots, with "
             + TT % "README_sim_IA.txt" + ". Use " + TT % "set_sim_IA_config(dir, n, sn, z, 4)" + " and "
             + TT % "set_sim_IA_window(1, 681.0, 1024)" + "."),
          bl(TT % "results/Galaxies/galaxies_&lt;sn&gt;.hdf5" + ": galaxy catalogues; " + TT % f"SUMMARY_{DATE}.md"
             + ": plain-text version of the headline table; " + TT % "make_update_pdf.py" + ": this report.")]

# 5 caveats
story += [P("5. Caveats", h2),
          bl("<b>Matter legs approximate.</b> No matter field; P<sub>dE</sub> = P<sub>hE</sub>/b(k) assumes deterministic "
             "bias and is least reliable at k &gt; 1 h/Mpc. The bias is now 1.6, so this matters more at high z."),
          bl("<b>Error bars underestimated</b> at high k (mode-scatter sigma is 3-9x too small in clustered mocks); "
             "jackknife errors not yet added."),
          bl("<b>Selection</b> on total bound stellar mass, not TNG's mass within 2 R<sub>1/2</sub>; FLAMINGO mass ranges "
             "start at log 10.0 (100-particle limit), higher than TNG's."),
          bl("<b>Mass and redshift are mixed</b> at z &le; 0.4, where the upper mass edge moves; from z = 0.5 to 1.0 it "
             "is fixed at log 11.07; above z = 1 it rises again (11.10-11.35).")]

# 6 next
story += [P("6. Next", h2),
          bl("Only snapshot 10 (z = 5) is left with full particle data; it would also need an extrapolated mass cut "
             "and would have very few galaxies."),
          bl("Matched-mass comparison with TNG300 at z &ge; 1.5 to tell selection from physics in the ratio drop."),
          bl("Jackknife error bars and fixed-mass / central-satellite splits of a<sub>NLA</sub> from the kept catalogues.")]

doc = SimpleDocTemplate(OUT, pagesize=letter, leftMargin=0.75 * inch, rightMargin=0.75 * inch,
                        topMargin=0.65 * inch, bottomMargin=0.65 * inch, title=f"FLAMINGO IA status update {DATE}")


def footer(c, d):
    c.saveState()
    c.setFont("Helvetica", 7.5)
    c.setFillColor(colors.HexColor("#777777"))
    c.drawString(0.75 * inch, 0.42 * inch, f"FLAMINGO IA status update, {DATE}")
    c.drawRightString(letter[0] - 0.75 * inch, 0.42 * inch, f"page {d.page}")
    c.restoreState()


doc.build(story, onFirstPage=footer, onLaterPages=footer)
print("wrote", OUT)
