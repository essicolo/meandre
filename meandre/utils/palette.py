"""Palettes derived from the Quebec government design system (design.quebec.ca).

Only official tokens are used, so any figure stays compliant. The ordering of
each palette is chosen to maximise perceptual separation: sequential ramps are
regular in CIE L*, the qualitative palette maximises the worst-case CIE dE76
distance under normal vision and under simulated deuteranopia, protanopia and
tritanopia, and every qualitative colour holds at least 3:1 against white.
"""

# Official tokens, light theme
BLEU_PALE = "#DAE6F0"
BLEU_150 = "#C6DBEE"
BLEU_200 = "#ADCDEB"
BLEU_300 = "#72B2EB"
BLEU_400 = "#3B95E1"
BLEU_CLAIR = "#4A98D9"
BLEU_500 = "#0078CC"
BLEU = "#1472BF"
BLEU_PIV = "#095797"
BLEU_MOYEN = "#19406C"
BLEU_FONCE = "#223654"
BLEU_800 = "#162B47"
ROSE_PALE = "#FFDBD6"
ROSE = "#E58271"
ROUGE = "#CB381F"
ROUGE_FONCE = "#692519"
VERT_PALE = "#D7F0BB"
VERT = "#4F813D"
VERT_FONCE = "#2C4024"
JAUNE_PALE = "#F8E69A"
JAUNE = "#E0AD03"
JAUNE_FONCE = "#AD781C"
VIOLET = "#6B4FA1"
GRIS_PALE = "#F1F1F2"
GRIS_CLAIR = "#C5CAD2"
GRIS = "#8893A2"
GRIS_MOYEN = "#6B778A"
GRIS_FONCE = "#4E5662"

# Continuous variables: two anchors, let the scale interpolate. Never bin.
SEQUENTIEL_BAS = BLEU_PALE
SEQUENTIEL_HAUT = BLEU_800

# Discrete sequential ramps, regular in L*. 4 and 7 classes are the regular ones.
SEQUENTIEL_3 = [BLEU_200, BLEU_500, BLEU_800]
SEQUENTIEL_4 = [BLEU_PALE, BLEU_300, BLEU_500, BLEU_MOYEN]
SEQUENTIEL_5 = [BLEU_PALE, BLEU_300, BLEU_500, BLEU_PIV, BLEU_800]
SEQUENTIEL_7 = [BLEU_PALE, BLEU_200, BLEU_300, BLEU_400, BLEU, BLEU_PIV, BLEU_MOYEN]

# Diverging, symmetric in L* around a near-neutral centre.
DIVERGENT_5 = [ROUGE, ROSE, GRIS_PALE, BLEU_300, BLEU]
DIVERGENT_7 = [ROUGE_FONCE, ROUGE, ROSE, GRIS_PALE, BLEU_300, BLEU, BLEU_MOYEN]

# Qualitative, nested: take the first n. Worst-case dE76 stays at 20 from n=3 to n=6.
QUALITATIF = [BLEU_MOYEN, ROUGE, JAUNE_FONCE, BLEU_CLAIR, VIOLET, ROUGE_FONCE]
QUALITATIF_MANQUANT = GRIS_CLAIR

# Single-series accent, and neutral reference lines.
ACCENT = BLEU_PIV
REFERENCE = GRIS_MOYEN

# Dark theme (background #121519): the identity blue lightens.
ACCENT_SOMBRE = BLEU_300
SEQUENTIEL_SOMBRE_BAS = BLEU_800
SEQUENTIEL_SOMBRE_HAUT = BLEU_150


def qualitatif(n):
    """First n colours of the nested qualitative palette."""
    if n > len(QUALITATIF):
        raise ValueError(f"{n} categories exceeds the {len(QUALITATIF)} distinguishable colours; regroup or facet")
    return QUALITATIF[:n]


# Control points of the continuous ramp: the official blue ladder, already near-regular
# in L* (90.7, 81.0, 70.5, 59.8, 46.9, 36.2, 26.6, 17.2).
_RAMPE_ANCRES = [BLEU_PALE, BLEU_200, BLEU_300, BLEU_400, BLEU, BLEU_PIV, BLEU_MOYEN, BLEU_800]


def _srgb_vers_lab(hexa):
    def lin(c):
        c = c / 255
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
    def f(t):
        return t ** (1 / 3) if t > 0.008856 else 7.787 * t + 16 / 116
    h = hexa.lstrip("#")
    r, g, b = [lin(int(h[i:i + 2], 16)) for i in (0, 2, 4)]
    x = (0.4124 * r + 0.3576 * g + 0.1805 * b) / 0.95047
    y = 0.2126 * r + 0.7152 * g + 0.0722 * b
    z = (0.0193 * r + 0.1192 * g + 0.9505 * b) / 1.08883
    fx, fy, fz = f(x), f(y), f(z)
    return (116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz))


def _lab_vers_srgb(lab):
    def finv(t):
        return t ** 3 if t ** 3 > 0.008856 else (t - 16 / 116) / 7.787
    def unlin(c):
        return 12.92 * c if c <= 0.0031308 else 1.055 * c ** (1 / 2.4) - 0.055
    ll, a, b = lab
    fy = (ll + 16) / 116
    x = 0.95047 * finv(fy + a / 500)
    y = finv(fy)
    z = 1.08883 * finv(fy - b / 200)
    r = 3.2406 * x - 1.5372 * y - 0.4986 * z
    g = -0.9689 * x + 1.8758 * y + 0.0415 * z
    bl = 0.0557 * x - 0.2040 * y + 1.0570 * z
    return "#" + "".join(f"{max(0, min(255, round(unlin(max(0.0, min(1.0, c))) * 255))):02X}" for c in (r, g, bl))


def rampe(n=256, ancres=None):
    """Continuous sequential ramp, n colours interpolated in CIE Lab.

    Some geoms (geom_imshow) take a list of colours rather than two anchors, and they
    quantise that list into bands. Passing rampe(256) gives them a ramp the reader sees
    as continuous, instead of classes that the data does not have.
    """
    ancres = ancres or _RAMPE_ANCRES
    labs = [_srgb_vers_lab(c) for c in ancres]
    m = len(labs) - 1
    out = []
    for i in range(n):
        t = i / (n - 1) * m
        k = min(int(t), m - 1)
        u = t - k
        out.append(_lab_vers_srgb(tuple(labs[k][j] + u * (labs[k + 1][j] - labs[k][j]) for j in range(3))))
    return out
