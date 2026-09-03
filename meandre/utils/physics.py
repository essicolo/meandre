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


def saturation_vapour_pressure(T_c: float) -> float:
    """Tetens formula: saturation vapour pressure (kPa) at T_c (Celsius)."""
    return 0.6108 * (17.27 * T_c / (T_c + 237.3))
