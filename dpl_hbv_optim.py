"""
=============================================================================
  dPL-HBV Demo — Aprendizaje de parámetros con gradientes
  Basado en Feng et al. (2022), Water Resources Research

  Demuestra que un modelo HBV implementado en PyTorch es DIFERENCIABLE:
    1. Se inicializan parámetros HBV con valores aleatorios
    2. Se simula caudal con el HBV forward
    3. Se calcula pérdida contra caudal observado
    4. El gradiente backpropaga a través del HBV → ajusta parámetros
    5. Los parámetros convergen a valores físicamente realistas

  Esto es la base del dPL-HBV: el proceso físico se vuelve "entrenable"
  igual que una red neural.

  pip install torch numpy matplotlib
  python dpl_hbv_optim.py
=============================================================================
"""
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ======================================================================
# HBV DIFERENCIABLE
# ======================================================================

def hbv(P, T, Ep, par, n_warmup=365):
    """
    HBV forward en PyTorch — diferenciable en todos los parámetros.
    par : tensor[13] con [TT, DD, CWH, rfz, FC, LP, beta, gamma, perc, K0, uzl, K1, K2]
    """
    TT, DD, CWH, rfz, FC, LP, beta, gamma, perc, K0, uzl, K1, K2 = par.unbind()
    n = len(P)
    Q = torch.zeros(n)
    Sp = Sliq = torch.tensor(0.0)
    Ss = FC * 0.5
    Suz, Slz = torch.tensor(10.0), torch.tensor(20.0)
    eps = 1e-5

    for t in range(n):
        Ps = torch.where(T[t] <= TT, P[t], torch.tensor(0.0))
        Pr = P[t] - Ps
        smelt = torch.min(torch.clamp(DD * (T[t] - TT), min=torch.tensor(0.0)), Sp + Ps)
        Rfz = torch.min(torch.clamp(DD * rfz * torch.clamp(TT - T[t], min=torch.tensor(0.0)) * Sliq, min=torch.tensor(0.0)), Sliq)
        Sp = torch.clamp(Sp + Ps + Rfz - smelt, min=torch.tensor(0.0))
        Sliq = torch.clamp(Sliq + smelt - Rfz, min=torch.tensor(0.0))
        Isnow = torch.clamp(Sliq - CWH * Sp, min=torch.tensor(0.0))
        Sliq -= Isnow

        W = torch.clamp((Ss / (FC + eps)) ** beta, max=torch.tensor(1.0))
        Peff = W * (Pr + Isnow)
        Ex = torch.clamp(Ss - FC, min=torch.tensor(0.0))
        eta = torch.clamp((Ss / (FC * LP + eps)) ** gamma, max=torch.tensor(1.0))
        ET = eta * Ep[t]
        Ss = torch.clamp(Ss + Pr + Isnow - Peff - Ex - ET, min=torch.tensor(0.0), max=FC)

        Perc = torch.min(perc, Suz)
        Q0 = torch.clamp(K0 * (Suz - uzl), min=torch.tensor(0.0))
        Q1 = K1 * Suz
        Suz = torch.clamp(Suz + Peff + Ex - Perc - Q0 - Q1, min=torch.tensor(0.0))
        Q2 = K2 * Slz
        Slz = torch.clamp(Slz + Perc - Q2, min=torch.tensor(0.0))
        Q[t] = Q0 + Q1 + Q2

    return Q[n_warmup:]


# ======================================================================
# PARÁMETROS REALES vs APRENDIDOS
# ======================================================================

PAR_NAMES = ['TT', 'DD', 'CWH', 'rfz', 'FC', 'LP', 'beta', 'gamma', 'perc', 'K0', 'uzl', 'K1', 'K2']
PAR_TRUE  = [0.0, 3.5, 0.1, 0.05, 250., 0.7, 2.5, 1.0, 1.2, 0.35, 15., 0.08, 0.025]
PAR_LO    = [-2., 1.0, 0.0, 0.0,  50., 0.2, 0.5, 0.1, 0.0, 0.01,  0., 0.001, 0.001]
PAR_HI    = [ 2., 6.0, 0.3, 0.2,  500., 1.0, 6.0, 4.0, 5.0, 0.8,  50., 0.5,  0.1]


def make_obs(seed=42):
    np.random.seed(seed)
    n = 3 * 365
    doy = (np.arange(n) % 365) + 1
    T = 10 + 15*np.cos(2*np.pi*(doy-200)/365) + 3*np.random.randn(n)
    P = np.zeros(n)
    for i in range(n):
        if np.random.rand() < 0.35*(0.8+0.5*np.sin(2*np.pi*(doy[i]-80)/365)):
            P[i] = np.random.exponential(2.5)
    Ep = np.clip(0.4*(T+5)/25*5 + 2*np.sin(np.pi*np.maximum(0,doy-80)/365)*(doy<355), 0, 8)

    par_true = torch.tensor(PAR_TRUE, dtype=torch.float32)
    Pt, Tt, Ept = [torch.tensor(x, dtype=torch.float32) for x in [P, T, Ep]]
    Q_obs = hbv(Pt, Tt, Ept, par_true, n_warmup=365)
    Q_obs = torch.clamp(Q_obs + 0.2*torch.randn_like(Q_obs), min=0)
    return Pt, Tt, Ept, Q_obs, par_true


def sigmoid_scale(x, lo, hi):
    """Mapea x ∈ ℝ → [lo, hi] via sigmoid"""
    return lo + (hi - lo) * torch.sigmoid(x)


def run():
    Pt, Tt, Ept, Qt, par_true = make_obs()

    # Parámetros aprendidos en espacio logit (sin restricciones)
    par_raw = nn.Parameter(torch.randn(13) * 0.5)
    optim = torch.optim.Adam([par_raw], lr=0.05)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=80)

    hist_loss, hist_par = [], []
    print(f"{'Ep':>4}  {'Loss':>8}  {'NSE':>7}  {'beta':>6}  {'gamma':>6}  {'FC':>7}")
    print("-" * 55)

    for ep in range(80):
        optim.zero_grad()
        par = sigmoid_scale(par_raw, torch.tensor(PAR_LO), torch.tensor(PAR_HI))
        Qs = hbv(Pt, Tt, Ept, par, n_warmup=365)
        loss = torch.sqrt(torch.mean((Qs - Qt)**2))
        loss.backward()
        optim.step()
        sched.step()

        nse = 1 - torch.sum((Qs - Qt)**2) / torch.sum((Qt - Qt.mean())**2)
        hist_loss.append(loss.item())
        hist_par.append(par.detach().clone())

        if ep % 5 == 0 or ep == 79:
            print(f"{ep+1:>4}  {loss.item():>8.4f}  {nse.item():>7.4f}"
                  f"  {par[6].item():>6.2f}  {par[7].item():>6.2f}  {par[4].item():>7.1f}")

    par_final = sigmoid_scale(par_raw, torch.tensor(PAR_LO), torch.tensor(PAR_HI)).detach()
    return hist_loss, hist_par, par_final, par_true, Qt, Pt, Tt, Ept


def plot(hist_loss, hist_par, par_final, par_true, Qt, Pt, Tt, Ept):
    par_all = torch.stack(hist_par)  # [80, 13]

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle('dPL-HBV Demo — Optimización de parámetros por gradientes',
                 fontsize=14, fontweight='bold')

    # Simulación final
    Qs = hbv(Pt, Tt, Ept, par_final, n_warmup=365)
    nse = 1 - torch.sum((Qs - Qt)**2) / torch.sum((Qt - Qt.mean())**2)

    t = np.arange(len(Qt))
    axes[0, 0].plot(Qt.numpy(), 'k', lw=1, label='Observado', alpha=0.8)
    axes[0, 0].plot(Qs.numpy(), 'r', lw=1.2, label='Simulado', alpha=0.8)
    axes[0, 0].set_ylabel('Q (mm/d)'); axes[0, 0].set_xlabel('Día')
    axes[0, 0].set_title(f'Caudal — NSE={nse.item():.3f}'); axes[0, 0].legend(fontsize=9); axes[0, 0].grid(alpha=0.3)

    axes[0, 1].plot(hist_loss, 'b-', lw=2)
    axes[0, 1].set_ylabel('RMSE'); axes[0, 1].set_xlabel('Iteración')
    axes[0, 1].set_title('Convergencia del loss'); axes[0, 1].grid(alpha=0.3)

    # Parámetros clave evolucionando
    for i, (name, ax) in enumerate(zip(['beta', 'gamma', 'FC'], axes[0, 2:])):
        idx = PAR_NAMES.index(name)
        ax.plot(par_all[:, idx].numpy(), 'r-', lw=1.5, label='Aprendido')
        ax.axhline(PAR_TRUE[idx], color='k', ls='--', lw=1.5, label=f'Verdadero ({PAR_TRUE[idx]})')
        ax.set_ylabel(name); ax.set_xlabel('Iteración')
        ax.set_title(f'Parámetro {name}'); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # Todos los parámetros: verdadero vs aprendido
    ax = axes[1, 0]
    x = np.arange(13)
    ax.barh(x - 0.2, PAR_TRUE, 0.4, color='k', alpha=0.6, label='Verdadero')
    ax.barh(x + 0.2, par_final.numpy(), 0.4, color='r', alpha=0.6, label='Aprendido')
    ax.set_yticks(x); ax.set_yticklabels(PAR_NAMES, fontsize=8)
    ax.set_xlabel('Valor'); ax.set_title('Parámetros: verdadero vs aprendido')
    ax.legend(fontsize=8); ax.grid(alpha=0.3, axis='x')

    # Scatter 1:1
    ax = axes[1, 1]
    ax.plot(Qt.numpy(), Qs.numpy(), 'r.', alpha=0.3, markersize=3)
    mx = max(Qt.max().item(), Qs.max().item())
    ax.plot([0, mx], [0, mx], 'k--', lw=1)
    ax.set_xlabel('Q obs'); ax.set_ylabel('Q sim')
    ax.set_title(f'Scatter 1:1 (NSE={nse.item():.3f})'); ax.grid(alpha=0.3)

    # Error por día
    ax = axes[1, 2]
    ax.fill_between(t, 0, (Qs - Qt).numpy(), alpha=0.5, color='r')
    ax.axhline(0, color='k', lw=0.5)
    ax.set_ylabel('Error (mm/d)'); ax.set_xlabel('Día')
    ax.set_title('Error de simulación'); ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('dpl_hbv_demo.png', dpi=150, bbox_inches='tight')
    print(f"\n  Figura: dpl_hbv_demo.png")
    plt.show()

    print(f"\n  {'='*55}")
    print(f"  NSE final: {nse.item():.4f}")
    print(f"  {'Par':<8} {'Verdadero':>10} {'Aprendido':>10} {'Error %':>8}")
    print(f"  {'-'*40}")
    for i, name in enumerate(PAR_NAMES):
        err = abs(par_final[i].item() - PAR_TRUE[i]) / max(abs(PAR_TRUE[i]), 1e-6) * 100
        print(f"  {name:<8} {PAR_TRUE[i]:>10.3f} {par_final[i].item():>10.3f} {err:>7.1f}%")
    print(f"  {'='*55}")
    print("""
  CONCEPTO DEMOSTRADO:
    El HBV es diferenciable → los gradientes fluyen desde el loss
    hasta cada parámetro físico → se optimizan con Adam/SGD.

    En el paper dPL-HBV:
    • En vez de optimizar parámetros directamente, una LSTM
      los PREDICE desde forzamientos y atributos de cuenca
    • Se entrena end-to-end en 671 cuencas CAMELS
    • Resultado: NSE mediano = 0.732 (casi igual a LSTM puro = 0.748)
    • Pero además produce ET, flujo base, nieve — variables físicas
  """)


if __name__ == '__main__':
    plot(*run())
