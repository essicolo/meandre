import sys; sys.argv=['x']
exec(open('.runs/slso-od/eval_network_ab.py').read().replace(
'''RUNS = [
    ("D8 (Copernicus, réseau actuel)", ".runs/slso-od/config/slso-od-vsafull.toml",
     ".runs/slso-od/checkpoints/best-vsafull.pt", ".runs/slso-od/data/basin.duckdb"),
    ("HydroSHEDS (conditionné)", ".runs/slso-od/config/slso-od-hs.toml",
     ".runs/slso-od/checkpoints/best-hs.pt", ".runs/slso-od/data/basin_hydrosheds.duckdb"),
]''',
'''RUNS = [
    ("HS baseline (sans codes)", ".runs/slso-od/config/slso-od-hs.toml",
     ".runs/slso-od/checkpoints/best-hs.pt", ".runs/slso-od/data/basin_hydrosheds.duckdb"),
    ("HS + codes latents ADDITIFS", ".runs/slso-od/config/slso-od-hs-latent.toml",
     ".runs/slso-od/checkpoints/best-hs-latent.pt", ".runs/slso-od/data/basin_hydrosheds.duckdb"),
]'''))
