"""POURQUOI méandre CHOISIT-IL de raboter ? (question d'Essi : diagnostic causal, pas
observation). Test de module, quelques secondes, aucune simulation régionale.

Hypothèse : le rabotage n'est pas une préférence pour la platitude, c'est la RÉPONSE
OPTIMALE À UNE ERREUR DE CALAGE TEMPOREL. Un pic net mal daté est puni DEUX FOIS par un
écart quadratique (pic manquant au bon jour + faux pic au mauvais jour) ; l'atténuer
supprime la moitié de la double peine. Si l'hypothèse est vraie, le K qui MINIMISE la
perte doit CROÎTRE avec le décalage — et l'optimum doit être à K bas quand le décalage
est nul.

Protocole : hydrogramme observé synthétique (crue de fonte réaliste + étiage), apport
latéral identique mais DÉCALÉ de d jours, routé par le Muskingum (conservatif depuis le
correctif) pour K de 4 à 48 h. On calcule les pertes réellement utilisées à
l'entraînement (KGE par station w=1.0, PBIAS w=0.5, MSE) et on lit l'argmin.

  PYTHONIOENCODING=utf-8 python .runs/quebec/banc_pourquoi_k.py
"""
import os, sys
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.getcwd())
import numpy as np, torch
torch.set_grad_enabled(False)
from meandre.routing.kinematic import MuskingumCunge

T = 365
rng = np.random.default_rng(0)

def hydrogramme(decalage=0):
    """Crue de fonte (avril-mai) + orages d'été + étiage, en apport latéral m³/s."""
    t = np.arange(T)
    fonte = 40.0 * np.exp(-((t - 110 - decalage) / 12.0) ** 2)
    automne = 18.0 * np.exp(-((t - 290 - decalage) / 15.0) ** 2)
    orages = np.zeros(T)
    for j in rng.choice(np.arange(160, 260), 8, replace=False):
        orages[j + decalage if j + decalage < T else j] = rng.uniform(5, 20)
    return 2.0 + fonte + automne + orages

def route(apport, K_h, x=0.20, nsub=2):
    r = MuskingumCunge(dt=86400.0, n_substeps=nsub)
    K = torch.tensor([K_h * 3600.0]); X = torch.tensor([x])
    Q = torch.zeros(1); out = np.zeros(T)
    for t in range(T):
        Q = r(Q_in=torch.zeros(1), Q_out_prev=Q, q_lateral=torch.tensor([float(apport[t])]), K=K, x=X)
        out[t] = float(Q)
    return out

def metriques(sim, obs):
    r = np.corrcoef(sim, obs)[0, 1]
    beta = sim.mean() / obs.mean()
    gamma = (sim.std() / sim.mean()) / (obs.std() / obs.mean())
    kge = 1 - np.sqrt((r - 1) ** 2 + (beta - 1) ** 2 + (gamma - 1) ** 2)
    mse = np.mean((sim - obs) ** 2)
    pbias = abs(beta - 1)
    return kge, mse, r, gamma, 1.0 * (1 - kge) + 0.5 * pbias

KS = np.array([4, 6, 8, 12, 16, 20, 24, 30, 36, 42, 48], float)
# l'OBSERVÉ est l'hydrogramme routé à K faible (réponse peu diffusée, comme Hydrotel)
obs = route(hydrogramme(0), 4.0)

print("=== POURQUOI LE MODÈLE CHOISIT UN GRAND K ===")
print("Perte d'entraînement (1·(1−KGE) + 0.5·|beta−1|) en fonction de K, par décalage\n")
print(f"{'décalage':>9s} | {'K* optimal':>10s} | {'perte à K*':>11s} | {'perte à K=4h':>12s} | "
      f"{'gain du rabotage':>17s} | {'pic simulé / pic obs':>21s}")
for d in (0, 1, 2, 3, 5):
    ap = hydrogramme(d)
    L = []
    for k in KS:
        s = route(ap, k)
        L.append(metriques(s, obs))
    L = np.array(L)
    perte = L[:, 4]
    j = int(np.argmin(perte)); j4 = 0
    sopt = route(ap, KS[j])
    print(f"{d:6d} j  | {KS[j]:8.0f} h | {perte[j]:11.4f} | {perte[j4]:12.4f} | "
          f"{perte[j4] - perte[j]:17.4f} | {sopt.max() / obs.max():21.3f}")

print("\nDétail au décalage de 2 jours (le décalage réellement mesuré sur méandre) :")
ap = hydrogramme(2)
print(f"  {'K (h)':>6s} {'KGE':>8s} {'r':>8s} {'gamma':>8s} {'perte':>9s}")
for k in (4, 12, 24, 36, 48):
    kge, mse, r, g, p = metriques(route(ap, k), obs)
    print(f"  {k:6.0f} {kge:8.3f} {r:8.3f} {g:8.3f} {p:9.4f}")
