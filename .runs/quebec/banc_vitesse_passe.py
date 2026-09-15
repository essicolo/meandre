"""Combien coute une passe avant, et la compilation du sol la reduit-elle ?

Le 2026-09-15, refaire les caches de naturalisation avec les prelevements corriges demande
trente passes avant, deux par region. Sous Windows, ou Triton est absent et ou le noyau du
sol ne peut donc pas etre compile, la premiere passe du Saguenay depasse quarante minutes.
La question est de savoir si WSL, qui est du Linux sur la meme carte, la ramene a un cout
acceptable.

On chronometre N jours de simulation sur une region, apres une passe d'echauffement qui
absorbe la compilation. Le resultat est un nombre de pas par seconde, d'ou se deduit le
cout des 9 132 jours d'une passe complete.

    ~/venv-meandre/bin/python .runs/quebec/banc_vitesse_passe.py sagu 200
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch


def main(reg, n_jours, total=9132):
    import tomllib

    import joint_data
    from meandre.model import HydroModel
    from meandre.utils.state import HydroState

    cfg = tomllib.load(open(".runs/quebec/config/gasp-v4.toml", "rb"))

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"appareil {dev} | torch {torch.__version__}", flush=True)
    try:
        import triton
        print(f"triton {triton.__version__} : compilation du sol possible", flush=True)
        dispo = True
    except Exception as e:
        print(f"triton absent ({type(e).__name__}) : compilation impossible", flush=True)
        dispo = False

    r = joint_data.load_region(reg, dict(cfg["loss"]), device=dev)
    td = r["train_data"]
    n = r["n_nodes"]
    f = td.forcing[:n_jours]
    doy = td.day_of_year[:n_jours]
    print(f"{reg} : {n} nœuds, {n_jours} jours chronométrés", flush=True)

    for compile_sol in ([False, True] if dispo else [False]):
        torch.manual_seed(1234)
        m = HydroModel(n_nodes=n, n_territorial=r["territorial"].n_features, n_forcing=6,
                       use_temporal=False, use_residual=False, use_travel_time_attn=False,
                       use_frost_rankinen=True, column_theta_init_frac=0.9,
                       param_mode="nerf", column_mode="hydrotel", et_mode="mcguinness",
                       use_temperature=False, spatial_melt=True,
                       routing_mode="operator-lagged", predict_lake_params=True,
                       compile_soil=compile_sol, use_aquifer=True).to(dev)
        m.eval()

        def passe(k):
            with torch.no_grad():
                m.simulate(forcing=f[:k], initial_state=HydroState.zeros(n, device=dev),
                           graph=td.graph, node_coords=td.node_coords,
                           territorial=td.territorial, withdrawals=td.withdrawals,
                           day_of_year=doy[:k])

        t_c = time.perf_counter()
        passe(min(20, n_jours))                       # echauffement et compilation
        if dev == "cuda":
            torch.cuda.synchronize()
        t_comp = time.perf_counter() - t_c
        t0 = time.perf_counter()
        passe(n_jours)
        if dev == "cuda":
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        vit = n_jours / dt
        print(f"  sol {'compilé' if compile_sol else 'interprété'} : échauffement "
              f"{t_comp:.1f} s | {dt:.1f} s pour {n_jours} jours, {vit:.1f} pas/s, "
              f"soit {total / vit / 60:.1f} min pour une passe de {total} jours",
              flush=True)
        del m
        if dev == "cuda":
            torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "sagu",
                  int(sys.argv[2]) if len(sys.argv) > 2 else 200))
