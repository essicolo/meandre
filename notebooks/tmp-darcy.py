# %%
import torch
import torch.nn as nn

# %% --- Données "observées" (simulées avec K_vrai = 1e-4 m/s) ---
K_vrai = 1e-4
A = torch.tensor([1.0, 1.0, 2.0, 2.0, 3.0])   # sections (m²)
i = torch.tensor([0.01, 0.02, 0.01, 0.03, 0.02]) # gradients hydrauliques (-)
Q_obs = K_vrai * A * i  # débits observés (m³/s)

# %% --- Le paramètre à calibrer ---
# On part d'une mauvaise estimation initiale
log_K = nn.Parameter(torch.tensor(-2.0))  # on optimise en log pour rester > 0

# %% --- Optimiseur et Boucle de calage ---
optim = torch.optim.Adam([log_K], lr=0.2)
for epoch in range(500):
    optim.zero_grad()            # 1. on remet les gradients à zéro
    K = 10 ** log_K              # 2. forward : transformer le paramètre
    Q_sim = K * A * i            #    appliquer Darcy
    loss = ((Q_sim - Q_obs)**2).mean()  # 3. MSE comme fonction de coût
    loss.backward()              # 4. autograd calcule ∂loss/∂log_K
    optim.step()                 # 5. Adam met à jour log_K
    if epoch % 100 == 0:
        print(f"epoch {epoch:3d} | K = {K.item():.2e} | loss = {loss.item():.2e}")
# %%
