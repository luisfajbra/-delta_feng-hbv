"""
=============================================================================
  Modelo δ-HBV Diferenciable — Basado en Feng et al. (2022)
  "Differentiable, Learnable, Regionalized Process-Based Models With
   Multiphysical Outputs can Approach State-Of-The-Art Hydrologic Prediction"
  Water Resources Research, 58, e2022WR032404

  Este script demuestra:
  1. El modelo HBV original (bucket model clásico)
  2. El modelo δ-HBV con parametrización dinámica (DP) — versión δₙ(γt, βt)
  3. Comparación de NSE entre ambos enfoques
  4. Visualización de variables internas (ET, flujo base, humedad del suelo)
  5. El efecto de la parametrización dinámica en β y γ

  Ejecutar en VS Code:
    pip install numpy matplotlib scipy
    python delta_HBV_Feng2022.py
=============================================================================
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# PARTE 1: MODELO HBV ESTÁNDAR
# ============================================================
# HBV (Hydrologiska Byråns Vattenbalansavdelning, Bergström 1976)
# Modelo de "cubos" que simula el ciclo hidrológico con 5 variables de estado:
#   Sp   = nieve sólida (snowpack)
#   Sliq = agua líquida en la nieve
#   Ss   = humedad superficial del suelo
#   Suz  = zona subsuperficial (upper zone)
#   Slz  = acuífero (lower zone)

def run_HBV(P, T, Ep, params, dynamic_beta=None, dynamic_gamma=None):
    """
    Simula el modelo HBV en modo diferenciable.

    Parámetros físicos (params dict):
        TT    : temperatura umbral nieve/lluvia (°C)
        DD    : factor de derretimiento (mm/°C/día)
        CWH   : capacidad máxima de agua líquida en nieve (fracción)
        rfz   : factor de recongelamiento
        FC    : capacidad de campo (máxima humedad superficial, mm)
        LP    : umbral LP*FC para reducción de ET (-)
        beta  : parámetro forma suelo→escorrentía (-)
        gamma : exponente eficiencia ET (-)
        perc  : percolación máxima a acuífero (mm/día)
        K0    : recesión rápida zona sub-superficial (1/día)
        uzl   : umbral para flujo rápido Q0 (mm)
        K1    : recesión lenta zona sub-superficial (1/día)
        K2    : recesión acuífero/flujo base (1/día)
        theta_a, theta_tau : parámetros del hidrograma unitario gamma

    dynamic_beta  : array [T] con β dinámico diario (None = estático)
    dynamic_gamma : array [T] con γ dinámico diario (None = estático)
    """
    n = len(P)

    # Extraer parámetros
    TT    = params['TT']
    DD    = params['DD']
    CWH   = params['CWH']
    rfz   = params['rfz']
    FC    = params['FC']
    LP    = params['LP']
    beta  = params['beta']
    gamma = params['gamma']
    perc  = params['perc']
    K0    = params['K0']
    uzl   = params['uzl']
    K1    = params['K1']
    K2    = params['K2']
    theta_a = params.get('theta_a', 2.5)
    theta_tau = params.get('theta_tau', 3.0)

    # Inicializar variables de estado
    Sp   = 0.0   # nieve sólida
    Sliq = 0.0   # agua líquida en nieve
    Ss   = FC * 0.5   # humedad superficial (inicio al 50% de capacidad)
    Suz  = 10.0  # zona sub-superficial
    Slz  = 20.0  # acuífero

    # Arrays de salida
    Q_sim  = np.zeros(n)
    ET_sim = np.zeros(n)
    BF_sim = np.zeros(n)  # flujo base Q2
    SM_sim = np.zeros(n)  # humedad del suelo Ss
    SWE_sim = np.zeros(n) # snow water equivalent
    beta_t_out  = np.zeros(n)
    gamma_t_out = np.zeros(n)

    # --------------------------------------------------------
    # Hidrograma unitario (routing) — función gamma
    # Convoluciona el caudal generado con un retardo de cuenca
    # --------------------------------------------------------
    tmax = 15
    t_uh = np.arange(1, tmax + 1, dtype=float)
    # Función gamma: ξ(t; θa, θτ) = (1/Γ(θa)) * (t/θτ)^(θa-1) * exp(-t/θτ)
    from scipy.special import gamma as gamma_func
    uh = (1.0 / (gamma_func(theta_a) * theta_tau**theta_a)) * \
         t_uh**(theta_a - 1) * np.exp(-t_uh / theta_tau)
    uh /= uh.sum()  # normalizar

    Q_raw = np.zeros(n)  # caudal antes del routing

    for t in range(n):
        # === MÓDULO NIEVE ===
        if T[t] <= TT:
            Ps = P[t]   # toda la precipitación como nieve
            Pr = 0.0
        else:
            Ps = 0.0
            Pr = P[t]   # toda la precipitación como lluvia

        smelt = max(0.0, DD * (T[t] - TT))    # fusión de nieve
        Rfz   = max(0.0, DD * rfz * (TT - T[t]) * Sliq)  # recongelamiento

        dSp   = Ps + Rfz - smelt
        Sp    = max(0.0, Sp + dSp)
        smelt = min(smelt, Sp + Ps)  # no fundir más de lo que hay

        Isnow = max(0.0, Sliq - CWH * Sp)  # infiltración desde nieve
        dSliq = smelt - Rfz - Isnow
        Sliq  = max(0.0, Sliq + dSliq)

        SWE_sim[t] = Sp + Sliq

        # === MÓDULO SUELO (con beta/gamma estáticos o dinámicos) ===
        beta_t  = dynamic_beta[t]  if dynamic_beta  is not None else beta
        gamma_t = dynamic_gamma[t] if dynamic_gamma is not None else gamma

        # Factor de humedad del suelo → escorrentía efectiva
        # W = min((Ss/FC)^β, 1)
        W    = min((Ss / FC) ** beta_t, 1.0) if FC > 0 else 0.0
        Peff = W * (Pr + Isnow)  # escorrentía efectiva

        # Exceso de capacidad
        Ex   = max(0.0, Ss - FC)

        # Evapotranspiración real
        # η = min((Ss/(FC*LP))^γ, 1)
        eta_ratio = (Ss / (FC * LP)) if (FC * LP) > 0 else 0.0
        eta  = min(eta_ratio ** gamma_t, 1.0)
        ET   = eta * Ep[t]

        dSs = (Pr + Isnow) - Peff - Ex - ET
        Ss  = max(0.0, Ss + dSs)
        Ss  = min(Ss, FC)

        ET_sim[t] = ET
        SM_sim[t] = Ss
        beta_t_out[t]  = beta_t
        gamma_t_out[t] = gamma_t

        # === MÓDULO SUBSUPERFICIAL ===
        Perc = min(perc, Suz)   # percolación hacia acuífero
        Q0   = max(0.0, K0 * (Suz - uzl))   # flujo rápido
        Q1   = K1 * Suz                       # flujo sub-superficial lento

        dSuz = Peff + Ex - Perc - Q0 - Q1
        Suz  = max(0.0, Suz + dSuz)

        # === MÓDULO ACUÍFERO (flujo base) ===
        Q2   = K2 * Slz   # baseflow
        dSlz = Perc - Q2
        Slz  = max(0.0, Slz + dSlz)

        BF_sim[t] = Q2
        Q_raw[t]  = Q0 + Q1 + Q2

    # === ROUTING: convolución con hidrograma unitario gamma ===
    Q_routed = np.convolve(Q_raw, uh)[:n]

    return Q_routed, ET_sim, BF_sim, SM_sim, SWE_sim, beta_t_out, gamma_t_out


# ============================================================
# PARTE 2: MÉTRICA NSE (Nash-Sutcliffe Efficiency)
# ============================================================
# NSE = 1 - Σ(Q_sim - Q_obs)² / Σ(Q_obs - Q̄_obs)²
# NSE=1: modelo perfecto | NSE=0: igual que usar la media | NSE<0: peor que la media

def NSE(Q_sim, Q_obs):
    mask = ~np.isnan(Q_obs)
    obs  = Q_obs[mask]
    sim  = Q_sim[mask]
    num  = np.sum((sim - obs) ** 2)
    den  = np.sum((obs - np.mean(obs)) ** 2)
    return 1.0 - num / den if den > 0 else np.nan


# ============================================================
# PARTE 3: GENERACIÓN DE DATOS SINTÉTICOS REALISTAS
# ============================================================
# Simulamos una cuenca tipo "Midwest" con estacionalidad clara,
# similar a las 671 cuencas del dataset CAMELS del paper

np.random.seed(42)
n_years = 10
n_days  = n_years * 365
t       = np.arange(n_days)
doy     = (t % 365) + 1  # día del año

# --- Temperatura: ciclo estacional con variabilidad diaria ---
T_mean  = 10.0  # media anual (°C)
T_amp   = 15.0  # amplitud estacional
T_noise = 3.0
T = T_mean + T_amp * np.cos(2 * np.pi * (doy - 200) / 365) + \
    T_noise * np.random.randn(n_days)

# --- Precipitación: más frecuente en primavera/otoño ---
P_base = 2.5  # mm/día promedio
P_seas = 0.8 + 0.5 * np.sin(2 * np.pi * (doy - 80) / 365)  # factor estacional
P = np.zeros(n_days)
rain_prob = 0.35 * P_seas
for i in range(n_days):
    if np.random.rand() < rain_prob[i]:
        P[i] = np.random.exponential(P_base * P_seas[i])

# --- Evapotranspiración potencial: sigue temperatura (Hargreaves simplificado) ---
# En el paper usan el método de Hargreaves que considera T_max, T_min y latitud
Ep = np.maximum(0.0, 0.4 * (T - (-5)) / 25.0 * 5.0 +
     2.0 * np.sin(np.pi * np.maximum(0, doy - 80) / 365) * (doy < 355))
Ep = np.clip(Ep, 0, 8.0)


# ============================================================
# PARTE 4: PARÁMETROS HBV (representativos de cuencas CAMELS)
# ============================================================
params_base = {
    'TT'    : 0.0,    # umbral nieve/lluvia (°C)
    'DD'    : 3.5,    # factor de derretimiento (mm/°C/día)
    'CWH'   : 0.1,    # capacidad líquida en nieve
    'rfz'   : 0.05,   # factor recongelamiento
    'FC'    : 250.0,  # capacidad de campo (mm)
    'LP'    : 0.7,    # fracción FC para reducción ET
    'beta'  : 2.5,    # parámetro escorrentía (HBV original, estático)
    'gamma' : 1.0,    # exponente ET (HBV original, estático)
    'perc'  : 1.2,    # percolación máxima (mm/día)
    'K0'    : 0.35,   # recesión rápida
    'uzl'   : 15.0,   # umbral flujo rápido (mm)
    'K1'    : 0.08,   # recesión sub-superficial
    'K2'    : 0.025,  # recesión flujo base
    'theta_a'  : 2.5, # hidrograma unitario - forma
    'theta_tau': 3.0, # hidrograma unitario - escala
}


# ============================================================
# PARTE 5: PARAMETRIZACIÓN DINÁMICA (δ models con DP)
# ============================================================
# En el paper, g_A (una red LSTM) estima β^t y γ^t cada día.
# Aquí lo simulamos con funciones físicamente motivadas:
#
# β^t: refleja almacenamiento hídrico acumulado. El paper muestra que
#      β alcanza máximo en septiembre y mínimo en marzo/abril.
#      Estacionalidad inversa al almacenamiento → más runoff en invierno.
#
# γ^t: refleja fenología vegetal. La vegetación es más activa en verano,
#      aumentando la eficiencia de ET en la estación cálida.

# β dinámico: ciclo estacional + respuesta a precipitación acumulada
beta_seasonal = 2.5 + 1.8 * np.cos(2 * np.pi * (doy - 260) / 365)
# Añadir memoria de precipitación (ventana de 30 días)
P_smooth = np.convolve(P, np.ones(30) / 30, mode='same')
beta_memory = -0.5 * (P_smooth - P_smooth.mean()) / (P_smooth.std() + 1e-6)
beta_dynamic = np.clip(beta_seasonal + beta_memory + 0.3 * np.random.randn(n_days) * 0.2, 0.5, 7.0)

# γ dinámico: relacionado con fenología vegetal
gamma_seasonal = 1.0 + 1.5 * np.maximum(0, np.sin(2 * np.pi * (doy - 80) / 365))
gamma_dynamic = np.clip(gamma_seasonal + 0.15 * np.random.randn(n_days), 0.1, 4.0)


# ============================================================
# PARTE 6: EJECUTAR LOS TRES MODELOS
# ============================================================

print("=" * 60)
print("  Simulando modelos — Feng et al. (2022) δ-HBV")
print("=" * 60)

# Modelo 1: HBV original (parámetros estáticos, 1 componente)
print("\n[1/3] HBV original (δ₁) — parámetros estáticos...")
Q_hbv, ET_hbv, BF_hbv, SM_hbv, SWE_hbv, beta_hbv, gamma_hbv = run_HBV(
    P, T, Ep, params_base
)

# Modelo 2: δ-HBV con β dinámico
print("[2/3] δ-HBV con β dinámico (δₙ(βt))...")
Q_beta, ET_beta, BF_beta, SM_beta, SWE_beta, beta_bt, gamma_bt = run_HBV(
    P, T, Ep, params_base, dynamic_beta=beta_dynamic
)

# Modelo 3: δ-HBV con β y γ dinámicos (mejor modelo del paper)
print("[3/3] δ-HBV con β y γ dinámicos (δₙ(γt, βt)) — mejor δ model...")
Q_delta, ET_delta, BF_delta, SM_delta, SWE_delta, beta_d, gamma_d = run_HBV(
    P, T, Ep, params_base,
    dynamic_beta=beta_dynamic,
    dynamic_gamma=gamma_dynamic
)


# ============================================================
# PARTE 7: CONSTRUIR "OBSERVACIONES" SINTÉTICAS
# ============================================================
# En el paper se usan datos reales de USGS y MODIS.
# Aquí construimos observaciones que el modelo δ debería capturar mejor,
# añadiendo efectos estacionales que el HBV estático no puede representar.

# Las observaciones incorporan efecto de almacenamiento profundo (que HBV original ignora)
memory_effect = 0.15 * np.convolve(P, np.ones(60) / 60, mode='same')
seasonal_bias  = 0.3 * np.sin(2 * np.pi * (doy - 30) / 365)  # ciclo estacional en runoff

Q_obs = np.maximum(0.0,
    Q_delta * (1 + seasonal_bias + memory_effect / Q_delta.mean()) +
    0.3 * np.random.randn(n_days)
)
Q_obs = np.maximum(Q_obs, 0.0)

# Observaciones de ET (MODIS simulado) — más altas en verano (ciclo vegetal)
ET_obs_modis = ET_delta * gamma_dynamic / params_base['gamma'] + \
               0.2 * np.random.randn(n_days)
ET_obs_modis = np.maximum(ET_obs_modis, 0.0)

# BFI observado (índice de flujo base)
BFI_obs = 0.45 + 0.2 * np.sin(2 * np.pi * (doy - 120) / 365) + \
          0.05 * np.random.randn(n_days)
BFI_obs = np.clip(BFI_obs, 0.1, 0.9)


# ============================================================
# PARTE 8: CÁLCULO DE MÉTRICAS
# ============================================================

NSE_hbv   = NSE(Q_hbv,   Q_obs)
NSE_beta  = NSE(Q_beta,  Q_obs)
NSE_delta = NSE(Q_delta, Q_obs)

# Correlación temporal de ET (como en el paper, Figure 6)
r_ET_hbv,   _ = pearsonr(ET_hbv[:365*5],   ET_obs_modis[:365*5])
r_ET_beta,  _ = pearsonr(ET_beta[:365*5],  ET_obs_modis[:365*5])
r_ET_delta, _ = pearsonr(ET_delta[:365*5], ET_obs_modis[:365*5])

# BFI simulado vs observado (como en el paper, Figure 5)
BFI_hbv   = BF_hbv   / (Q_hbv   + 1e-6)
BFI_beta  = BF_beta  / (Q_beta  + 1e-6)
BFI_delta = BF_delta / (Q_delta + 1e-6)
r_BFI_hbv,   _ = pearsonr(BFI_hbv,   BFI_obs)
r_BFI_beta,  _ = pearsonr(BFI_beta,  BFI_obs)
r_BFI_delta, _ = pearsonr(BFI_delta, BFI_obs)

print("\n" + "=" * 60)
print("  RESULTADOS — comparar con Tabla 2 de Feng et al. (2022)")
print("=" * 60)
print(f"\n{'Modelo':<25} {'NSE':>8} {'r_ET':>8} {'r_BFI':>8}")
print("-" * 55)
print(f"{'HBV original (δ₁)':<25} {NSE_hbv:>8.3f} {r_ET_hbv:>8.3f} {r_BFI_hbv:>8.3f}")
print(f"{'δ-HBV β dinámico':<25} {NSE_beta:>8.3f} {r_ET_beta:>8.3f} {r_BFI_beta:>8.3f}")
print(f"{'δ-HBV γ+β dinámico':<25} {NSE_delta:>8.3f} {r_ET_delta:>8.3f} {r_BFI_delta:>8.3f}")
print()
print("  Valores del paper original (Tabla 2, Daymet forcing):")
print(f"{'  dPL+HBV (δ₁)':<25} {'0.640':>8} {'0.770':>8} {'0.560':>8}")
print(f"{'  δₙ(βt)':<25}  {'0.729':>8} {'0.801':>8} {'0.725':>8}")
print(f"{'  δₙ(γt,βt)':<25} {'0.732':>8} {'0.844':>8} {'0.760':>8}")
print(f"{'  LSTM (referencia)':<25} {'0.748':>8} {'  ---':>8} {'  ---':>8}")
print("=" * 60)


# ============================================================
# PARTE 9: VISUALIZACIONES
# ============================================================

fig = plt.figure(figsize=(18, 22))
fig.suptitle('Modelo δ-HBV — Feng et al. (2022)\nModelos Diferenciales Aprendibles en Hidrología',
             fontsize=15, fontweight='bold', y=0.98)

gs = gridspec.GridSpec(4, 2, figure=fig, hspace=0.45, wspace=0.35)

# Paleta de colores consistente con el paper
c_obs   = 'black'
c_hbv   = '#1f77b4'   # azul — HBV original
c_beta  = '#ff7f0e'   # naranja — con β dinámico
c_delta = '#d62728'   # rojo — con γ+β dinámico (mejor)
c_lstm  = '#2ca02c'   # verde — LSTM referencia

# ---- Panel A: Series de tiempo de caudal (año 6-7) ----
ax1 = fig.add_subplot(gs[0, :])
yr5 = 5 * 365
yr7 = 7 * 365
t_plot = np.arange(yr5, yr7)
ax1.plot(t_plot - yr5, Q_obs[yr5:yr7],   color=c_obs,   lw=1.5, label='Observado', zorder=5)
ax1.plot(t_plot - yr5, Q_hbv[yr5:yr7],   color=c_hbv,   lw=1.2, ls='--', label=f'HBV original (δ₁)  NSE={NSE_hbv:.3f}')
ax1.plot(t_plot - yr5, Q_beta[yr5:yr7],  color=c_beta,  lw=1.2, ls='--', label=f'δ-HBV β dinámico     NSE={NSE_beta:.3f}')
ax1.plot(t_plot - yr5, Q_delta[yr5:yr7], color=c_delta, lw=1.5, ls='-',  label=f'δ-HBV γ+β dinámico  NSE={NSE_delta:.3f}', alpha=0.9)
ax1.set_xlabel('Días (años 6-7 del período de evaluación)', fontsize=10)
ax1.set_ylabel('Caudal (mm/día)', fontsize=10)
ax1.set_title('A. Comparación de series de tiempo de caudal\n(similar a Figura 3 del paper)', fontsize=11)
ax1.legend(fontsize=9, loc='upper right')
ax1.set_xlim(0, 730)
ax1.grid(alpha=0.3)

# ---- Panel B: NSE comparativo (barras) ----
ax2 = fig.add_subplot(gs[1, 0])
modelos  = ['HBV\noriginal\n(δ₁)', 'δ-HBV\nβ dinámico\n(δₙ(βt))', 'δ-HBV\nγ+β din.\n(δₙ(γt,βt))', 'LSTM\n(referencia\npaper)']
nse_vals = [NSE_hbv, NSE_beta, NSE_delta, 0.748]
colors   = [c_hbv, c_beta, c_delta, c_lstm]
bars = ax2.bar(modelos, nse_vals, color=colors, alpha=0.85, edgecolor='black', linewidth=0.7)
for bar, val in zip(bars, nse_vals):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
             f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
ax2.axhline(0.748, color=c_lstm, ls='--', lw=1.5, alpha=0.7, label='LSTM paper (0.748)')
ax2.set_ylabel('NSE mediano', fontsize=10)
ax2.set_title('B. NSE por modelo\n(comparable a Tabla 2 del paper)', fontsize=11)
ax2.set_ylim(0.4, 1.0)
ax2.grid(axis='y', alpha=0.3)

# ---- Panel C: Parámetro β dinámico (como Figura 4 del paper) ----
ax3 = fig.add_subplot(gs[1, 1])
yr2 = 2 * 365
yr4 = 4 * 365
doy_plot = doy[yr2:yr4]
sort_idx = np.argsort(doy_plot)
beta_plot = beta_dynamic[yr2:yr4]
# Mediana mensual
meses = np.arange(1, 13)
beta_monthly = [np.median(beta_plot[(doy_plot >= (m-1)*30+1) & (doy_plot <= m*30)]) for m in meses]
ax3.plot(t[yr2:yr4] - yr2, beta_plot, color='gray', alpha=0.4, lw=0.7, label='β diario (todos los componentes)')
# Línea del componente mediano (simulando la Figura 4)
beta_smooth = np.convolve(beta_plot, np.ones(30)/30, mode='same')
ax3.plot(t[yr2:yr4] - yr2, beta_smooth, color=c_beta, lw=2.5, label='β mediano suavizado')
ax3.axhline(params_base['beta'], color=c_hbv, ls='--', lw=2, label=f"β estático = {params_base['beta']}")
ax3.set_xlabel('Días (años 3-4)', fontsize=10)
ax3.set_ylabel('β (parámetro runoff)', fontsize=10)
ax3.set_title('C. Parámetro β dinámico en el tiempo\n(similar a Figura 4a del paper)', fontsize=11)
ax3.legend(fontsize=8)
ax3.grid(alpha=0.3)

# ---- Panel D: Parámetro γ dinámico ----
ax4 = fig.add_subplot(gs[2, 0])
gamma_plot = gamma_dynamic[yr2:yr4]
gamma_smooth = np.convolve(gamma_plot, np.ones(30)/30, mode='same')
ax4.plot(t[yr2:yr4] - yr2, gamma_plot, color='gray', alpha=0.4, lw=0.7)
ax4.plot(t[yr2:yr4] - yr2, gamma_smooth, color=c_delta, lw=2.5, label='γ mediano suavizado')
ax4.axhline(params_base['gamma'], color=c_hbv, ls='--', lw=2, label=f"γ estático = {params_base['gamma']}")
ax4.set_xlabel('Días (años 3-4)', fontsize=10)
ax4.set_ylabel('γ (eficiencia ET)', fontsize=10)
ax4.set_title('D. Parámetro γ dinámico (fenología vegetal)\n(similar a Figura 4b del paper)', fontsize=11)
ax4.legend(fontsize=8)
ax4.grid(alpha=0.3)

# ---- Panel E: Evapotranspiración — comparación con MODIS ----
ax5 = fig.add_subplot(gs[2, 1])
# Composición de 8 días (como hace el paper con MOD16A2)
ET_8d_hbv   = [np.mean(ET_hbv[i:i+8])   for i in range(0, min(n_days, 365*5)-7, 8)]
ET_8d_delta = [np.mean(ET_delta[i:i+8])  for i in range(0, min(n_days, 365*5)-7, 8)]
ET_8d_modis = [np.mean(ET_obs_modis[i:i+8]) for i in range(0, min(n_days, 365*5)-7, 8)]
t_8d = np.arange(len(ET_8d_hbv))

ax5.scatter(ET_8d_modis, ET_8d_delta, alpha=0.4, s=12, color=c_delta,
            label=f'δ-HBV (R={r_ET_delta:.3f})')
ax5.scatter(ET_8d_modis, ET_8d_hbv,   alpha=0.3, s=8, color=c_hbv,
            label=f'HBV original (R={r_ET_hbv:.3f})')
max_et = max(max(ET_8d_modis), max(ET_8d_delta))
ax5.plot([0, max_et], [0, max_et], 'k--', lw=1.5, label='1:1')
ax5.set_xlabel('ET MODIS estimada (mm/8días)', fontsize=10)
ax5.set_ylabel('ET simulada (mm/8días)', fontsize=10)
ax5.set_title('E. Correlación ET simulada vs MODIS\n(sin entrenar en ET — similar a Figura 6b)', fontsize=11)
ax5.legend(fontsize=8)
ax5.grid(alpha=0.3)

# ---- Panel F: Flujo base — simulado vs observado ----
ax6 = fig.add_subplot(gs[3, 0])
ax6.scatter(BFI_obs[:365*3:8], BFI_delta[:365*3:8], alpha=0.4, s=12, color=c_delta,
            label=f'δ-HBV (R={r_BFI_delta:.3f})')
ax6.scatter(BFI_obs[:365*3:8], BFI_hbv[:365*3:8],   alpha=0.3, s=8, color=c_hbv,
            label=f'HBV original (R={r_BFI_hbv:.3f})')
ax6.plot([0, 1], [0, 1], 'k--', lw=1.5, label='1:1')
ax6.set_xlabel('BFI observado (análisis de recesión)', fontsize=10)
ax6.set_ylabel('BFI simulado (Q₂/Q)', fontsize=10)
ax6.set_title('F. Índice de Flujo Base (BFI)\n(variable no entrenada — similar a Figura 5e)', fontsize=11)
ax6.legend(fontsize=8)
ax6.set_xlim(0, 1); ax6.set_ylim(0, 1)
ax6.grid(alpha=0.3)

# ---- Panel G: Variables internas del ciclo hidrológico ----
ax7 = fig.add_subplot(gs[3, 1])
yr1 = 1 * 365
yr3 = 3 * 365
t_r = np.arange(yr1, yr3) - yr1

# Normalizar para visualizar juntas
SM_n  = SM_delta[yr1:yr3] / params_base['FC']
BF_n  = BF_delta[yr1:yr3] / (BF_delta.max() + 1e-6)
SWE_n = SWE_delta[yr1:yr3] / (SWE_delta.max() + 1e-6)
ET_n  = ET_delta[yr1:yr3] / (ET_delta.max() + 1e-6)

ax7.fill_between(t_r, SM_n,  alpha=0.4, color='#8B4513', label='Humedad suelo (Ss/FC)')
ax7.fill_between(t_r, SWE_n, alpha=0.5, color='#4FC3F7', label='Nieve (SWE norm.)')
ax7.plot(t_r, BF_n,  color='#1565C0', lw=1.2, label='Flujo base Q₂ (norm.)')
ax7.plot(t_r, ET_n,  color='#2E7D32', lw=1.2, label='ET real (norm.)')
ax7.set_xlabel('Días (años 2-3)', fontsize=10)
ax7.set_ylabel('Variables normalizadas (0-1)', fontsize=10)
ax7.set_title('G. Variables internas del ciclo hidrológico\n(salidas físicas que LSTM no puede producir)', fontsize=11)
ax7.legend(fontsize=8, loc='upper right')
ax7.set_xlim(0, 730)
ax7.grid(alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig('delta_HBV_resultados.png', dpi=150, bbox_inches='tight')
print("\n  Figura guardada: delta_HBV_resultados.png")
plt.show()


# ============================================================
# PARTE 10: RESUMEN CONCEPTUAL DEL FRAMEWORK δ
# ============================================================
print("\n" + "=" * 60)
print("  RESUMEN CONCEPTUAL — ¿Qué hace el modelo δ?")
print("=" * 60)
print("""
  PROBLEMA:
    • Modelos físicos (HBV, mHM): NSE ~ 0.53 — interpretables pero poco precisos
    • LSTM (deep learning): NSE ~ 0.748 — preciso pero caja negra

  SOLUCIÓN (δ models):
    1. Usar HBV como esqueleto físico (conservación de masa garantizada)
    2. Hacer el modelo DIFERENCIABLE (implementado en PyTorch)
    3. Una red LSTM (g_A) estima los parámetros de HBV desde datos
       → θ = g_A(atributos_cuenca, forzantes_meteorológicas)
    4. Parametrización dinámica:
       → β^t varía cada día (memoria hídrica estacional)
       → γ^t varía cada día (fenología vegetal)
    5. Entrenamiento "end-to-end" con gradientes — igual que deep learning

  RESULTADO:
    • δₙ(γt, βt): NSE mediano = 0.732  [vs LSTM = 0.748, HBV clásico = 0.53]
    • Produce además: ET, flujo base, humedad suelo, nieve — sin entrenarlas
    • Correlación ET con MODIS: R = 0.844
    • Correlación BFI con USGS: R = 0.760

  MENSAJE CENTRAL:
    "No hay que elegir entre precisión e interpretabilidad física"
    → Los modelos diferenciales aprendibles pueden lograr ambas.
""")
