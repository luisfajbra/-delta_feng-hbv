"""
=============================================================================
  dPL-HBV Demo: Aprender parámetros HBV con gradientes
  Basado en Feng et al. (2022) — Water Resources Research

  Concepto demostrado:
    Forzamientos → Red neural → parámetros HBV → HBV diferenciable → Q → Loss
                                                                ↖______________↙
                                                                   backprop

  El gradiente fluye a través de TODO el modelo físico hasta la red neural.
  Esto es el núcleo de dPL-HBV (differentiable parameter learning).

  Requisitos:  pip install torch numpy matplotlib
  Ejecutar:    python dpl_hbv_demo.py
=============================================================================
"""
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ======================================================================
# 1. HBV DIFERENCIABLE en PyTorch (simplificado)
# ======================================================================

def hbv_forward(P, T, Ep, theta, fixed, warmup=0):
    """
    HBV forward completamente diferenciable.

    theta : dict de tensores con parámetros aprendibles
            {'beta': tensor[T], 'gamma': tensor[T], 'FC': tensor, ...}
    fixed : dict de tensores con parámetros fijos
    warmup : días de calentamiento

    Retorna Q : caudal simulado tensor[T-warmup]
    """
    beta  = theta['beta']
    gamma = theta['gamma']
    FC    = theta['FC']
    K0    = theta['K0']
    K1    = theta['K1']
    K2    = theta['K2']
    perc  = theta['perc']
    LP    = theta['LP']
    uzl   = theta['uzl']

    TT   = fixed['TT']
    DD   = fixed['DD']
    CWH  = fixed['CWH']
    rfz  = fixed['rfz']

    n = len(P)
    Q_raw = torch.zeros(n)
    eps = 1e-5

    Sp   = torch.tensor(0.0)
    Sliq = torch.tensor(0.0)
    Ss   = FC * 0.5
    Suz  = torch.tensor(10.0)
    Slz  = torch.tensor(20.0)

    for t in range(n):
        Nieve  = torch.where(T[t] <= TT, P[t], torch.tensor(0.0))
        Lluvia = torch.where(T[t] <= TT, torch.tensor(0.0), P[t])

        smelt = torch.clamp(DD * (T[t] - TT), min=0.0)
        smelt = torch.min(smelt, Sp + Nieve)
        Rfz   = torch.clamp(DD * rfz * torch.clamp(TT - T[t], min=0.0) * Sliq, min=0.0)
        Rfz   = torch.min(Rfz, Sliq)

        Sp   = torch.clamp(Sp + Nieve + Rfz - smelt, min=0.0)
        Sliq = torch.clamp(Sliq + smelt - Rfz, min=0.0)
        Isnow = torch.clamp(Sliq - CWH * Sp, min=0.0)
        Sliq = Sliq - Isnow

        W     = torch.clamp((Ss / (FC + eps)) ** beta[t], max=1.0)
        Peff  = W * (Lluvia + Isnow)
        Ex    = torch.clamp(Ss - FC, min=0.0)
        eta   = torch.clamp((Ss / (FC * LP + eps)) ** gamma[t], max=1.0)
        ET    = eta * Ep[t]

        Ss = torch.clamp(Ss + (Lluvia + Isnow) - Peff - Ex - ET, min=0.0, max=FC)

        Perc = torch.min(perc, Suz)
        Q0   = torch.clamp(K0 * (Suz - uzl), min=0.0)
        Q1   = K1 * Suz
        Suz  = torch.clamp(Suz + Peff + Ex - Perc - Q0 - Q1, min=0.0)

        Q2  = K2 * Slz
        Slz = torch.clamp(Slz + Perc - Q2, min=0.0)

        Q_raw[t] = Q0 + Q1 + Q2

    return Q_raw[warmup:]


# ======================================================================
# 2. RED NEURAL QUE PREDICE PARÁMETROS (g_A del paper)
# ======================================================================

class ParamNet(nn.Module):
    """
    Predice β(t) y γ(t) desde P, T y día del año.
    En el paper real esto es una LSTM; aquí usamos MLP + features acumulados.
    """
    def __init__(self, n_features=6, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),     nn.ReLU(),
            nn.Linear(hidden, hidden // 2), nn.ReLU(),
            nn.Linear(hidden // 2, 2),
        )

    def forward(self, x):
        raw = self.net(x)
        beta  = 0.5 + 6.5 * torch.sigmoid(raw[:, 0])   # [0.5, 7.0]
        gamma = 0.1 + 3.9 * torch.sigmoid(raw[:, 1])   # [0.1, 4.0]
        return beta, gamma


# ======================================================================
# 3. DATOS SINTÉTICOS
# ======================================================================

def make_data(n_years=3, seed=42):
    np.random.seed(seed)
    n = n_years * 365
    doy = (np.arange(n) % 365) + 1

    T = 10 + 15 * np.cos(2 * np.pi * (doy - 200) / 365) + 3 * np.random.randn(n)
    P = np.zeros(n)
    for i in range(n):
        prob = 0.35 * (0.8 + 0.5 * np.sin(2 * np.pi * (doy[i] - 80) / 365))
        if np.random.rand() < prob:
            P[i] = np.random.exponential(2.5)
    Ep = np.clip(0.4 * (T + 5) / 25 * 5 + 2 * np.sin(np.pi * np.maximum(0, doy - 80) / 365) * (doy < 355), 0, 8)

    beta_true  = np.clip(2.5 + 1.8 * np.cos(2 * np.pi * (doy - 260) / 365) + 0.3 * np.random.randn(n), 0.5, 7.0)
    gamma_true = np.clip(1.0 + 1.5 * np.maximum(0, np.sin(2 * np.pi * (doy - 80) / 365)) + 0.15 * np.random.randn(n), 0.1, 4.0)

    fixed_np = dict(TT=0, DD=3.5, CWH=0.1, rfz=0.05, FC=250, LP=0.7, perc=1.2, K0=0.35, uzl=15, K1=0.08, K2=0.025)
    fixed = {k: torch.tensor(v, dtype=torch.float32) for k, v in fixed_np.items()}

    Pt = torch.tensor(P, dtype=torch.float32)
    Tt = torch.tensor(T, dtype=torch.float32)
    Ept = torch.tensor(Ep, dtype=torch.float32)
    beta_t  = torch.tensor(beta_true, dtype=torch.float32)
    gamma_t = torch.tensor(gamma_true, dtype=torch.float32)

    theta_true = {'beta': beta_t, 'gamma': gamma_t, 'FC': fixed['FC'], 'K0': fixed['K0'],
                  'K1': fixed['K1'], 'K2': fixed['K2'], 'perc': fixed['perc'], 'LP': fixed['LP'], 'uzl': fixed['uzl']}

    warmup = 365
    Q_obs = hbv_forward(Pt, Tt, Ept, theta_true, fixed, warmup=warmup)
    Q_obs = torch.clamp(Q_obs + 0.2 * torch.randn_like(Q_obs), min=0.0)

    return P, T, Ep, Q_obs.numpy(), beta_true[warmup:], gamma_true[warmup:], fixed, warmup


# ======================================================================
# 4. ENTRENAMIENTO END-TO-END
# ======================================================================

def train():
    P, T, Ep, Q_obs, beta_true, gamma_true, fixed, warmup = make_data()

    Pt  = torch.tensor(P, dtype=torch.float32)
    Tt  = torch.tensor(T, dtype=torch.float32)
    Ept = torch.tensor(Ep, dtype=torch.float32)
    Qt  = torch.tensor(Q_obs, dtype=torch.float32)
    doy = (np.arange(len(P)) % 365) + 1

    P_roll = torch.tensor(np.convolve(P, np.ones(30)/30, mode='same'), dtype=torch.float32)
    T_roll = torch.tensor(np.convolve(T, np.ones(15)/15, mode='same'), dtype=torch.float32)
    X = torch.stack([Pt, Tt,
                     torch.sin(torch.tensor(2*np.pi*doy/365, dtype=torch.float32)),
                     torch.cos(torch.tensor(2*np.pi*doy/365, dtype=torch.float32)),
                     P_roll, T_roll], dim=1)

    X_train = X[warmup:]
    net = ParamNet(n_features=6, hidden=32)
    optim = torch.optim.Adam(net.parameters(), lr=0.005)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=60)

    n_ep = 60
    loss_hist = []
    print(f"{'Epoch':>5}  {'Loss':>10}  {'NSE':>8}")
    print("-" * 30)

    for ep in range(n_ep):
        optim.zero_grad()
        beta_p, gamma_p = net(X_train)
        FC_p = torch.clamp(torch.tensor(250.0) + torch.randn(1)*0.1, min=100, max=500).squeeze()

        theta = {'beta': beta_p, 'gamma': gamma_p, 'FC': FC_p,
                 'K0': torch.tensor(0.35), 'K1': torch.tensor(0.08), 'K2': torch.tensor(0.025),
                 'perc': torch.tensor(1.2), 'LP': torch.tensor(0.7), 'uzl': torch.tensor(15.0)}

        Q_sim = hbv_forward(Pt[warmup:], Tt[warmup:], Ept[warmup:], theta, fixed, warmup=0)
        loss = torch.sqrt(torch.mean((Q_sim - Qt) ** 2))
        loss.backward()
        optim.step()
        sched.step()
        loss_hist.append(loss.item())

        nse = 1 - torch.sum((Q_sim - Qt)**2) / torch.sum((Qt - Qt.mean())**2)
        if ep % 5 == 0 or ep == n_ep - 1:
            print(f"{ep+1:>5}  {loss.item():>10.4f}  {nse.item():>8.4f}")

    return net, Pt, Tt, Ept, Qt, X_train, beta_true, gamma_true, fixed, warmup, loss_hist


# ======================================================================
# 5. RESULTADOS
# ======================================================================

def plot(net, Pt, Tt, Ept, Qt, X_train, beta_true, gamma_true, fixed, warmup, loss_hist):
    net.eval()
    with torch.no_grad():
        bp, gp = net(X_train)
        FC_p = torch.tensor(250.0)
        theta = {'beta': bp, 'gamma': gp, 'FC': FC_p,
                 'K0': torch.tensor(0.35), 'K1': torch.tensor(0.08), 'K2': torch.tensor(0.025),
                 'perc': torch.tensor(1.2), 'LP': torch.tensor(0.7), 'uzl': torch.tensor(15.0)}
        Qs = hbv_forward(Pt[warmup:], Tt[warmup:], Ept[warmup:], theta, fixed, warmup=0)

    nse = 1 - torch.sum((Qs - Qt)**2) / torch.sum((Qt - Qt.mean())**2)

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle(f'dPL-HBV Demo — Feng et al. (2022)  |  NSE = {nse.item():.3f}',
                 fontsize=14, fontweight='bold')

    t = np.arange(len(Qt))
    axes[0, 0].plot(Qt.numpy(), 'k', lw=1, label='Observado', alpha=0.8)
    axes[0, 0].plot(Qs.numpy(), 'r', lw=1.2, label='Simulado', alpha=0.8)
    axes[0, 0].set_ylabel('Q (mm/d)'); axes[0, 0].set_xlabel('Día')
    axes[0, 0].set_title(f'Caudal — NSE={nse.item():.3f}'); axes[0, 0].legend(fontsize=8); axes[0, 0].grid(alpha=0.3)

    axes[0, 1].plot(loss_hist, 'b-', lw=1.5)
    axes[0, 1].set_ylabel('RMSE'); axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_title('Pérdida (end-to-end backprop)'); axes[0, 1].grid(alpha=0.3)

    axes[0, 2].plot(Qt.numpy(), Qs.numpy(), 'r.', alpha=0.3, markersize=3)
    mx = max(Qt.max().item(), Qs.max().item())
    axes[0, 2].plot([0, mx], [0, mx], 'k--', lw=1)
    axes[0, 2].set_xlabel('Q obs'); axes[0, 2].set_ylabel('Q sim')
    axes[0, 2].set_title('Scatter 1:1'); axes[0, 2].grid(alpha=0.3)

    idx = slice(0, min(365, len(Qt)))
    axes[1, 0].plot(beta_true[idx], 'k--', lw=1.5, label='β verdadero', alpha=0.7)
    axes[1, 0].plot(bp.numpy()[idx], 'r-', lw=1, label='β aprendido', alpha=0.7)
    axes[1, 0].set_ylabel('β'); axes[1, 0].set_xlabel('Día')
    axes[1, 0].set_title('β (runoff) — verdadero vs aprendido'); axes[1, 0].legend(fontsize=8); axes[1, 0].grid(alpha=0.3)

    axes[1, 1].plot(gamma_true[idx], 'k--', lw=1.5, label='γ verdadero', alpha=0.7)
    axes[1, 1].plot(gp.numpy()[idx], 'r-', lw=1, label='γ aprendido', alpha=0.7)
    axes[1, 1].set_ylabel('γ'); axes[1, 1].set_xlabel('Día')
    axes[1, 1].set_title('γ (ET) — verdadero vs aprendido'); axes[1, 1].legend(fontsize=8); axes[1, 1].grid(alpha=0.3)

    doy_p = (np.arange(len(Qt)) % 365) + 1
    for m, c in zip([1,4,7,10], ['#2196F3','#4CAF50','#FF9800','#F44336']):
        mask = (doy_p >= m*30-15) & (doy_p < m*30+15)
        axes[1, 2].scatter(Pt[warmup:][mask].numpy(), Qt[mask].numpy(), alpha=0.3, s=5, c=c, label=f'Mes {m}')
    axes[1, 2].set_xlabel('P (mm/d)'); axes[1, 2].set_ylabel('Q (mm/d)')
    axes[1, 2].set_title('Relación P→Q por estación'); axes[1, 2].legend(fontsize=7); axes[1, 2].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('dpl_hbv_demo.png', dpi=150, bbox_inches='tight')
    print(f"\n  Figura: dpl_hbv_demo.png")
    plt.show()

    print(f"\n  {'='*50}")
    print(f"  NSE: {nse.item():.4f}")
    print(f"  β aprendido (media): {bp.mean().item():.2f}  (verdadero: {beta_true.mean():.2f})")
    print(f"  γ aprendido (media): {gp.mean().item():.2f}  (verdadero: {gamma_true.mean():.2f})")
    print(f"  {'='*50}")
    print("""
  FLUJO DEMOSTRADO:
    P, T, doy  →  ParamNet  →  β(t), γ(t)  →  HBV forward  →  Q_sim
       ↑                                          │
       └────────── backprop ←─────────────────────┘
                    (gradientes PyTorch)

  En el paper real:
    • La red es una LSTM (memoria temporal)
    • Se entrena en 671 cuencas CAMELS
    • Además aprende K0, K1, K2, FC, perc, etc.
    • NSE mediano = 0.732  (vs LSTM puro = 0.748)
  """)


if __name__ == '__main__':
    args = train()
    plot(*args)
