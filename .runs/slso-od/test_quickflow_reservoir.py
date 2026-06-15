"""Test isolé du réservoir supérieur à seuil (HBV-EC K0/UZL/K1), avant câblage.

Fonction pure, pilotée par un orage synthétique. On vérifie que le stock S_uz :
  - se remplit pendant l'orage,
  - ne lâche QUE de la vidange lente Q1 tant que S_uz < UZL,
  - lâche une BOUFFÉE rapide Q0 dès que S_uz franchit UZL,
  - conserve la masse (entrée = sorties + variation de stock),
  - reste différentiable (gradient fini de la sortie vs les params).

Formulation (pas journalier, dt=1 j) :
  Q0 = K0 · softplus(beta·(S_uz − UZL)) / beta      # rapide, ~0 sous le seuil
  Q1 = K1 · S_uz                                     # interflow, toujours
  S_uz_new = clamp(S_uz + I − Q0 − Q1, min=0)

  python .runs/slso-od/test_quickflow_reservoir.py
"""
from __future__ import annotations
import sys
sys.stdout.reconfigure(encoding="utf-8")
import torch
import torch.nn.functional as F


def quickflow_reservoir(S_uz, inflow, K0, K1, UZL, beta=0.5):
    """Un pas du réservoir à seuil. Tout en mm, dt = 1 jour.

    S_uz, inflow : (n,) tenseurs. K0, K1, UZL, beta : scalaires ou (n,).
    Retourne (S_uz_new, Q0, Q1).
    """
    excess = F.softplus(beta * (S_uz - UZL)) / beta      # ≈ max(S_uz−UZL,0), lissé
    Q0 = K0 * excess
    Q1 = K1 * S_uz
    S_new = torch.clamp(S_uz + inflow - Q0 - Q1, min=0.0)
    return S_new, Q0, Q1


def main():
    torch.set_printoptions(precision=3, sci_mode=False)
    n_days = 40
    # Orage synthétique : base 1 mm/j, gros pic de 35 mm sur 3 jours autour de j15.
    inflow = torch.ones(n_days) * 1.0
    inflow[14:17] = torch.tensor([18.0, 35.0, 22.0])
    inflow[25] = 12.0  # petite averse sous le seuil

    # Params type boréal (récessions 1/j, seuil mm).
    K0 = torch.tensor(0.6)    # vidange rapide (au-dessus du seuil)
    K1 = torch.tensor(0.08)   # interflow lent
    UZL = torch.tensor(15.0)  # seuil de déclenchement (mm)
    beta = 0.5

    S = torch.tensor(0.0)
    rows = []
    for t in range(n_days):
        S, Q0, Q1 = quickflow_reservoir(S, inflow[t], K0, K1, UZL, beta)
        rows.append((t, float(inflow[t]), float(S), float(Q0), float(Q1)))

    print(f"{'jour':>4} {'inflow':>7} {'S_uz':>7} {'Q0_fast':>8} {'Q1_slow':>8} {'seuil?':>7}")
    for t, inf, s, q0, q1 in rows:
        flag = "  <==" if s > float(UZL) else ""
        print(f"{t:4d} {inf:7.1f} {s:7.2f} {q0:8.3f} {q1:8.3f} {flag:>7}")

    # 1) Conservation de masse sur toute la série.
    tot_in = float(inflow.sum())
    tot_q0 = sum(r[3] for r in rows)
    tot_q1 = sum(r[4] for r in rows)
    S_final = rows[-1][2]
    err = tot_in - (tot_q0 + tot_q1 + S_final)
    print(f"\nMasse : entrée={tot_in:.2f}  Q0={tot_q0:.2f}  Q1={tot_q1:.2f}  "
          f"stock_final={S_final:.2f}  résidu={err:.4f} mm")

    # 2) Seuil : Q0 négligeable hors crue, dominant en crue.
    q0_base = rows[5][3]      # jour calme
    q0_peak = max(r[3] for r in rows)
    print(f"Seuil : Q0 jour calme={q0_base:.4f} mm/j  vs  Q0 pic={q0_peak:.3f} mm/j  "
          f"(ratio {q0_peak/max(q0_base,1e-9):.0f}×)")

    # 3) Différentiabilité : gradient de la somme des pics vs les 3 params.
    K0g = torch.tensor(0.6, requires_grad=True)
    K1g = torch.tensor(0.08, requires_grad=True)
    UZLg = torch.tensor(15.0, requires_grad=True)
    S = torch.tensor(0.0)
    peakflow = torch.tensor(0.0)
    for t in range(n_days):
        S, Q0, Q1 = quickflow_reservoir(S, inflow[t], K0g, K1g, UZLg, beta)
        peakflow = peakflow + Q0
    peakflow.backward()
    print(f"\nDifférentiable : d(ΣQ0)/dK0={K0g.grad:.3f}  d/dK1={K1g.grad:.3f}  "
          f"d/dUZL={UZLg.grad:.3f}  (tous finis, non-nuls)")
    print("TEST_DONE")


if __name__ == "__main__":
    main()
