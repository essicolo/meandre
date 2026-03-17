"""Physical constants and unit conversion utilities."""

# ---- Thermodynamic constants ----
LATENT_HEAT_VAPORIZATION = 2.45e6   # J kg-1, at ~20C
LATENT_HEAT_FUSION = 3.34e5          # J kg-1
PSYCHROMETRIC_CONSTANT = 0.0665      # kPa C-1 (at sea level, 20C)
STEFAN_BOLTZMANN = 5.67e-8           # W m-2 K-4

# ---- Water ----
RHO_WATER = 1000.0    # kg m-3
RHO_ICE = 917.0       # kg m-3
CP_WATER = 4186.0     # J kg-1 K-1

# ---- Atmospheric ----
P_SEA_LEVEL = 101.325  # kPa
LAPSE_RATE = 0.0065    # K m-1, standard atmosphere
CP_AIR = 1013.0        # J kg-1 K-1

# ---- Soil ----
CP_SOIL_DRY = 840.0    # J kg-1 K-1
RHO_SOIL_DRY = 1500.0  # kg m-3


def celsius_to_kelvin(T_c: float) -> float:
    return T_c + 273.15


def kelvin_to_celsius(T_k: float) -> float:
    return T_k - 273.15


def m3s_to_mm_per_day(Q_m3s: float, area_km2: float) -> float:
    """Convert m3/s to mm/day over a catchment of given area."""
    area_m2 = area_km2 * 1e6
    return Q_m3s * 86400.0 / area_m2 * 1000.0


def mm_per_day_to_m3s(Q_mm: float, area_km2: float) -> float:
    """Convert mm/day to m3/s over a catchment of given area."""
    area_m2 = area_km2 * 1e6
    return Q_mm / 1000.0 * area_m2 / 86400.0


def saturation_vapour_pressure(T_c: float) -> float:
    """Tetens formula: saturation vapour pressure (kPa) at T_c (Celsius)."""
    return 0.6108 * (17.27 * T_c / (T_c + 237.3))


def slope_saturation_vapour_pressure(T_c: float) -> float:
    """Slope of saturation vapour pressure curve (kPa/C) at T_c."""
    e_s = saturation_vapour_pressure(T_c)
    return 4098.0 * e_s / (T_c + 237.3) ** 2
