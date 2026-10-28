"""Palettes derived from the Quebec government design system (design.quebec.ca).

Only official tokens are used, so any figure stays compliant. The ordering of
each palette is chosen to maximise perceptual separation: sequential ramps are
regular in CIE L*, the qualitative palette maximises the worst-case CIE dE76
distance under normal vision and under simulated deuteranopia, protanopia and
tritanopia, and every qualitative colour holds at least 3:1 against white.
"""

# Official tokens, light theme
BLUE_PALE = "#DAE6F0"
BLUE_150 = "#C6DBEE"
BLUE_200 = "#ADCDEB"
BLUE_300 = "#72B2EB"
BLUE_400 = "#3B95E1"
BLUE_LIGHT = "#4A98D9"
BLUE_500 = "#0078CC"
BLUE = "#1472BF"
BLUE_PIV = "#095797"
BLUE_MEDIUM = "#19406C"
BLUE_DARK = "#223654"
BLUE_800 = "#162B47"
PINK_PALE = "#FFDBD6"
PINK = "#E58271"
RED = "#CB381F"
RED_DARK = "#692519"
GREEN_PALE = "#D7F0BB"
GREEN = "#4F813D"
GREEN_DARK = "#2C4024"
YELLOW_PALE = "#F8E69A"
YELLOW = "#E0AD03"
YELLOW_DARK = "#AD781C"
VIOLET = "#6B4FA1"
GRAY_PALE = "#F1F1F2"
GRAY_LIGHT = "#C5CAD2"
GRAY = "#8893A2"
GRAY_MEDIUM = "#6B778A"
GRAY_DARK = "#4E5662"

# Continuous variables: two anchors, let the scale interpolate. Never bin.
SEQUENTIAL_LOW = BLUE_PALE
SEQUENTIAL_HIGH = BLUE_800

# Discrete sequential ramps, regular in L*. 4 and 7 classes are the regular ones.
SEQUENTIAL_3 = [BLUE_200, BLUE_500, BLUE_800]
SEQUENTIAL_4 = [BLUE_PALE, BLUE_300, BLUE_500, BLUE_MEDIUM]
SEQUENTIAL_5 = [BLUE_PALE, BLUE_300, BLUE_500, BLUE_PIV, BLUE_800]
SEQUENTIAL_7 = [BLUE_PALE, BLUE_200, BLUE_300, BLUE_400, BLUE, BLUE_PIV, BLUE_MEDIUM]

# Diverging, symmetric in L* around a near-neutral centre.
DIVERGING_5 = [RED, PINK, GRAY_PALE, BLUE_300, BLUE]
DIVERGING_7 = [RED_DARK, RED, PINK, GRAY_PALE, BLUE_300, BLUE, BLUE_MEDIUM]

# Qualitative, nested: take the first n. Worst-case dE76 stays at 20 from n=3 to n=6.
QUALITATIVE = [BLUE_MEDIUM, RED, YELLOW_DARK, BLUE_LIGHT, VIOLET, RED_DARK]
QUALITATIVE_MISSING = GRAY_LIGHT

# Single-series accent, and neutral reference lines.
ACCENT = BLUE_PIV
REFERENCE = GRAY_MEDIUM

# Dark theme (background #121519): the identity blue lightens.
ACCENT_DARK = BLUE_300
SEQUENTIAL_DARK_LOW = BLUE_800
SEQUENTIAL_DARK_HIGH = BLUE_150


# Control points of the continuous ramp: the official blue ladder, already near-regular
# in L* (90.7, 81.0, 70.5, 59.8, 46.9, 36.2, 26.6, 17.2).
_RAMP_ANCHORS = [BLUE_PALE, BLUE_200, BLUE_300, BLUE_400, BLUE, BLUE_PIV, BLUE_MEDIUM, BLUE_800]


def _srgb_to_lab(hexa):
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


def _lab_to_srgb(lab):
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


def ramp(n=256, anchors=None):
    """Continuous sequential ramp, n colours interpolated in CIE Lab.

    Some geoms (geom_imshow) take a list of colours rather than two anchors, and they
    quantise that list into bands. Passing rampe(256) gives them a ramp the reader sees
    as continuous, instead of classes that the data does not have.
    """
    anchors = anchors or _RAMP_ANCHORS
    labs = [_srgb_to_lab(c) for c in anchors]
    m = len(labs) - 1
    out = []
    for i in range(n):
        t = i / (n - 1) * m
        k = min(int(t), m - 1)
        u = t - k
        out.append(_lab_to_srgb(tuple(labs[k][j] + u * (labs[k + 1][j] - labs[k][j]) for j in range(3))))
    return out
