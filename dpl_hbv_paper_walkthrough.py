"""
Walkthrough didactico de los modelos del paper dPL-HBV.

Este script esta pensado para explicar, de forma cercana al paper de
Feng et al. (2022), como se plantean las variantes principales:

1. dPL+HBV estatico (delta_1)
   LSTM(P, T, PET, attrs) -> un set de parametros estaticos -> HBV -> Q

2. delta_n(beta_t)
   LSTM(P, T, PET, attrs) -> parametros estaticos + beta_t diario -> HBV -> Q

3. delta_n(gamma_t, beta_t)
   LSTM(P, T, PET, attrs) -> parametros estaticos + beta_t + gamma_t diarios -> HBV -> Q

Para que sea ejecutable en un entorno sin PyTorch, aqui la LSTM del paper se
reemplaza por un "emulador" explicativo que genera parametros desde forzamientos
y atributos de cuenca. La parte hidrologica (HBV, parametros dinamicos, estados,
salidas fisicas y comparacion de variantes) si se mantiene muy cerca de la idea
central del paper.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


np.random.seed(42)

N_WARMUP = 365
UH_LEN = 15
EPS = 1e-6

PARAM_RANGES = {
    "TT": (-2.5, 2.5),
    "CFMAX": (0.5, 10.0),
    "CFR": (0.0, 0.1),
    "CWH": (0.0, 0.2),
    "FC": (50.0, 1000.0),
    "LP": (0.2, 1.0),
    "BETA": (1.0, 6.0),
    "PERC": (0.0, 10.0),
    "UZL": (0.0, 100.0),
    "K0": (0.05, 0.9),
    "K1": (0.01, 0.5),
    "K2": (0.001, 0.2),
    "GAMMA": (0.3, 5.0),
    "A": (0.2, 2.9),
    "B": (0.5, 6.5),
}

BASE_PARAMS = {
    "TT": 0.0,
    "CFMAX": 3.5,
    "CFR": 0.05,
    "CWH": 0.10,
    "FC": 250.0,
    "LP": 0.70,
    "BETA": 2.5,
    "PERC": 1.20,
    "UZL": 15.0,
    "K0": 0.35,
    "K1": 0.08,
    "K2": 0.025,
    "GAMMA": 1.0,
    "A": 1.5,
    "B": 3.0,
}

VARIANT_LABELS = {
    "static": "dPL+HBV estatico (delta_1)",
    "dynamic_beta": "delta_n(beta_t)",
    "dynamic_beta_gamma": "delta_n(gamma_t, beta_t)",
}


@dataclass
class BasinData:
    P: np.ndarray
    T: np.ndarray
    Ep: np.ndarray
    attrs: np.ndarray
    Q_obs: np.ndarray
    ET_obs: np.ndarray
    BFI_obs: np.ndarray
    beta_true: np.ndarray
    gamma_true: np.ndarray
    teacher: Dict[str, np.ndarray]
    warmup: int = N_WARMUP


@dataclass
class VariantResult:
    variant: str
    label: str
    params: Dict[str, float]
    beta: np.ndarray | None
    gamma: np.ndarray | None
    outputs: Dict[str, np.ndarray]
    metrics: Dict[str, float]


def clip_to_range(name: str, value):
    lo, hi = PARAM_RANGES[name]
    return np.clip(value, lo, hi)


def moving_average(x: np.ndarray, window: int) -> np.ndarray:
    kernel = np.ones(window, dtype=float) / float(window)
    return np.convolve(x, kernel, mode="same")


def zscore(x: np.ndarray) -> np.ndarray:
    return (x - x.mean()) / (x.std() + EPS)


def pearson_np(x: np.ndarray, y: np.ndarray) -> float:
    x0 = x - x.mean()
    y0 = y - y.mean()
    den = np.sqrt(np.sum(x0 * x0) * np.sum(y0 * y0)) + EPS
    return float(np.sum(x0 * y0) / den)


def nse_np(sim: np.ndarray, obs: np.ndarray) -> float:
    den = np.sum((obs - obs.mean()) ** 2) + EPS
    return float(1.0 - np.sum((sim - obs) ** 2) / den)


def gamma_unit_hydrograph(a: float, b: float, length: int = UH_LEN) -> np.ndarray:
    t = np.arange(1, length + 1, dtype=float)
    uh = (1.0 / (math.gamma(a) * (b ** a))) * (t ** (a - 1.0)) * np.exp(-t / b)
    return uh / (uh.sum() + EPS)


def run_hbv(
    P: np.ndarray,
    T: np.ndarray,
    Ep: np.ndarray,
    params: Dict[str, float],
    dynamic_beta: np.ndarray | None = None,
    dynamic_gamma: np.ndarray | None = None,
) -> Dict[str, np.ndarray]:
    n = len(P)

    TT = params["TT"]
    CFMAX = params["CFMAX"]
    CFR = params["CFR"]
    CWH = params["CWH"]
    FC = params["FC"]
    LP = params["LP"]
    BETA = params["BETA"]
    PERC = params["PERC"]
    UZL = params["UZL"]
    K0 = params["K0"]
    K1 = params["K1"]
    K2 = params["K2"]
    GAMMA = params["GAMMA"]

    Sp = 0.0
    Sliq = 0.0
    Ss = FC * 0.5
    Suz = 10.0
    Slz = 20.0

    Q_raw = np.zeros(n)
    Q0_all = np.zeros(n)
    Q1_all = np.zeros(n)
    Q2_all = np.zeros(n)
    ET_all = np.zeros(n)
    SM_all = np.zeros(n)
    SWE_all = np.zeros(n)
    PEFF_all = np.zeros(n)
    BETA_all = np.zeros(n)
    GAMMA_all = np.zeros(n)

    for t in range(n):
        beta_t = dynamic_beta[t] if dynamic_beta is not None else BETA
        gamma_t = dynamic_gamma[t] if dynamic_gamma is not None else GAMMA

        if T[t] <= TT:
            Ps = P[t]
            Pr = 0.0
        else:
            Ps = 0.0
            Pr = P[t]

        smelt = max(0.0, CFMAX * (T[t] - TT))
        smelt = min(smelt, Sp + Ps)

        refreeze = max(0.0, CFR * max(TT - T[t], 0.0) * Sliq)
        refreeze = min(refreeze, Sliq)

        Sp = max(0.0, Sp + Ps + refreeze - smelt)
        Sliq = max(0.0, Sliq + smelt - refreeze)

        Isnow = max(0.0, Sliq - CWH * Sp)
        Sliq = Sliq - Isnow

        W = min((Ss / (FC + EPS)) ** beta_t, 1.0)
        Peff = W * (Pr + Isnow)
        Ex = max(0.0, Ss - FC)

        eta = min((Ss / (FC * LP + EPS)) ** gamma_t, 1.0)
        ET = eta * Ep[t]

        Ss = max(0.0, Ss + (Pr + Isnow) - Peff - Ex - ET)
        Ss = min(Ss, FC)

        Perc = min(PERC, Suz)
        Q0 = max(0.0, K0 * (Suz - UZL))
        Q1 = K1 * Suz
        Suz = max(0.0, Suz + Peff + Ex - Perc - Q0 - Q1)

        Q2 = K2 * Slz
        Slz = max(0.0, Slz + Perc - Q2)

        Q_raw[t] = Q0 + Q1 + Q2
        Q0_all[t] = Q0
        Q1_all[t] = Q1
        Q2_all[t] = Q2
        ET_all[t] = ET
        SM_all[t] = Ss / (FC + EPS)
        SWE_all[t] = Sp + Sliq
        PEFF_all[t] = Peff
        BETA_all[t] = beta_t
        GAMMA_all[t] = gamma_t

    uh = gamma_unit_hydrograph(params["A"], params["B"])
    Q = np.convolve(Q_raw, uh, mode="full")[:n]
    BFI = np.clip(Q2_all / (Q_raw + EPS), 0.0, 1.0)

    return {
        "Q": Q,
        "Q_raw": Q_raw,
        "Q0": Q0_all,
        "Q1": Q1_all,
        "Q2": Q2_all,
        "ET": ET_all,
        "SM": SM_all,
        "SWE": SWE_all,
        "PEFF": PEFF_all,
        "BFI": BFI,
        "BETA": BETA_all,
        "GAMMA": GAMMA_all,
    }


def make_synthetic_basin(n_years: int = 4) -> BasinData:
    n = n_years * 365
    t = np.arange(n)
    doy = (t % 365) + 1

    T = 10.0 + 15.0 * np.cos(2.0 * np.pi * (doy - 200) / 365.0) + 2.8 * np.random.randn(n)

    P = np.zeros(n, dtype=float)
    rain_prob = 0.34 * (0.85 + 0.45 * np.sin(2.0 * np.pi * (doy - 80) / 365.0))
    for i in range(n):
        if np.random.rand() < rain_prob[i]:
            P[i] = np.random.exponential(2.7)

    Ep = np.clip(
        0.4 * (T + 5.0) / 25.0 * 5.0
        + 2.2 * np.sin(np.pi * np.maximum(0.0, doy - 80) / 365.0) * (doy < 355),
        0.0,
        8.0,
    )

    attrs = np.array(
        [
            0.35, 0.15, 0.08, 0.25, 0.12, 0.30, 0.45, 0.20, 0.60,
            0.10, 0.05, 0.03, 0.15, 0.08, 0.25, 0.30, 0.40, 0.35,
            0.20, 0.15, 0.10, 0.25, 0.40, 0.05, 0.30, 0.20, 0.35,
        ],
        dtype=float,
    )
    attrs = np.clip(attrs + 0.02 * np.random.randn(len(attrs)), 0.0, 1.0)

    p30 = moving_average(P, 30)
    p60 = moving_average(P, 60)
    dryness = zscore(p30)
    memory = zscore(p60)
    pet_signal = zscore(Ep)

    beta_true = 2.6 + 1.0 * np.cos(2.0 * np.pi * (doy - 255) / 365.0) - 0.45 * dryness - 0.15 * memory
    beta_true = clip_to_range("BETA", beta_true + 0.08 * np.random.randn(n))

    gamma_true = 1.0 + 1.4 * np.maximum(0.0, np.sin(2.0 * np.pi * (doy - 80) / 365.0)) + 0.20 * pet_signal
    gamma_true = clip_to_range("GAMMA", gamma_true + 0.06 * np.random.randn(n))

    teacher_params = dict(BASE_PARAMS)
    teacher = run_hbv(P, T, Ep, teacher_params, dynamic_beta=beta_true, dynamic_gamma=gamma_true)

    q_noise = 0.05 * np.std(teacher["Q"][N_WARMUP:]) * np.random.randn(n)
    et_noise = 0.06 * np.std(teacher["ET"][N_WARMUP:]) * np.random.randn(n)
    bfi_noise = 0.03 * np.random.randn(n)

    Q_obs = np.maximum(0.0, teacher["Q"] + q_noise)
    ET_obs = np.maximum(0.0, teacher["ET"] + et_noise)
    BFI_obs = np.clip(teacher["BFI"] + bfi_noise, 0.0, 1.0)

    return BasinData(
        P=P,
        T=T,
        Ep=Ep,
        attrs=attrs,
        Q_obs=Q_obs,
        ET_obs=ET_obs,
        BFI_obs=BFI_obs,
        beta_true=beta_true,
        gamma_true=gamma_true,
        teacher=teacher,
    )


def inverse_model_emulator(
    P: np.ndarray,
    T: np.ndarray,
    Ep: np.ndarray,
    attrs: np.ndarray,
    variant: str,
) -> tuple[Dict[str, float], np.ndarray | None, np.ndarray | None]:
    params = dict(BASE_PARAMS)

    aridity = attrs[0]
    frac_snow = attrs[1]
    slope = attrs[7]
    frac_forest = attrs[9]
    soil_porosity = attrs[16]
    soil_conductivity = attrs[17]
    permeability = attrs[24]
    storage_min = attrs[25]
    storage_max = attrs[26]

    params["TT"] = float(clip_to_range("TT", -1.0 + 4.0 * frac_snow))
    params["CFMAX"] = float(clip_to_range("CFMAX", 2.6 + 1.8 * (1.0 - frac_snow)))
    params["FC"] = float(clip_to_range("FC", 170.0 + 320.0 * soil_porosity + 90.0 * storage_max))
    params["LP"] = float(clip_to_range("LP", 0.45 + 0.30 * frac_forest + 0.08 * aridity))
    params["BETA"] = float(clip_to_range("BETA", 2.0 + 0.9 * storage_max - 0.5 * slope))
    params["PERC"] = float(clip_to_range("PERC", 0.8 + 2.8 * permeability))
    params["UZL"] = float(clip_to_range("UZL", 8.0 + 18.0 * slope + 10.0 * storage_min))
    params["K0"] = float(clip_to_range("K0", 0.18 + 0.35 * slope))
    params["K1"] = float(clip_to_range("K1", 0.04 + 0.16 * soil_conductivity))
    params["K2"] = float(clip_to_range("K2", 0.01 + 0.05 * storage_max))
    params["GAMMA"] = float(clip_to_range("GAMMA", 0.85 + 0.55 * aridity + 0.20 * frac_forest))
    params["A"] = float(clip_to_range("A", 1.1 + 0.7 * slope))
    params["B"] = float(clip_to_range("B", 2.2 + 1.0 * storage_max))

    beta_hat = None
    gamma_hat = None

    doy = (np.arange(len(P)) % 365) + 1
    p30 = moving_average(P, 30)
    p60 = moving_average(P, 60)
    dryness = zscore(p30)
    memory = zscore(p60)
    pet_signal = zscore(Ep)

    if variant in {"dynamic_beta", "dynamic_beta_gamma"}:
        beta_hat = params["BETA"] + 0.85 * np.cos(2.0 * np.pi * (doy - 255) / 365.0) - 0.32 * dryness - 0.10 * memory
        beta_hat = clip_to_range("BETA", beta_hat)

    if variant == "dynamic_beta_gamma":
        gamma_hat = params["GAMMA"] + 1.10 * np.maximum(0.0, np.sin(2.0 * np.pi * (doy - 80) / 365.0)) + 0.12 * pet_signal
        gamma_hat = clip_to_range("GAMMA", gamma_hat)

    return params, beta_hat, gamma_hat


def evaluate_variant(outputs: Dict[str, np.ndarray], basin: BasinData) -> Dict[str, float]:
    sl = slice(basin.warmup, None)
    return {
        "nse": nse_np(outputs["Q"][sl], basin.Q_obs[sl]),
        "r_et": pearson_np(outputs["ET"][sl], basin.ET_obs[sl]),
        "r_bfi": pearson_np(outputs["BFI"][sl], basin.BFI_obs[sl]),
    }


def run_variants(basin: BasinData) -> List[VariantResult]:
    variants = ["static", "dynamic_beta", "dynamic_beta_gamma"]
    results: List[VariantResult] = []

    for variant in variants:
        params, beta_hat, gamma_hat = inverse_model_emulator(basin.P, basin.T, basin.Ep, basin.attrs, variant)
        outputs = run_hbv(basin.P, basin.T, basin.Ep, params, dynamic_beta=beta_hat, dynamic_gamma=gamma_hat)
        results.append(
            VariantResult(
                variant=variant,
                label=VARIANT_LABELS[variant],
                params=params,
                beta=beta_hat,
                gamma=gamma_hat,
                outputs=outputs,
                metrics=evaluate_variant(outputs, basin),
            )
        )
    return results


def print_summary(results: List[VariantResult]) -> None:
    print("=" * 74)
    print("Comparacion de variantes del paper")
    print("=" * 74)
    print(f"{'Modelo':<32} {'NSE':>8} {'r_ET':>8} {'r_BFI':>8}")
    print("-" * 60)
    for result in results:
        print(
            f"{result.label:<32} "
            f"{result.metrics['nse']:>8.3f} "
            f"{result.metrics['r_et']:>8.3f} "
            f"{result.metrics['r_bfi']:>8.3f}"
        )


def print_walkthrough(basin: BasinData, best: VariantResult) -> None:
    sl = slice(basin.warmup, None)
    peak_local = int(np.argmax(basin.Q_obs[sl]))
    peak_idx = basin.warmup + peak_local
    outputs = best.outputs

    print("\nLectura conceptual:")
    print("- delta_1: una LSTM produciria un solo vector de parametros por cuenca.")
    print("- delta_n(beta_t): la red mantiene estaticos casi todos los parametros y deja variar beta_t.")
    print("- delta_n(gamma_t, beta_t): la red deja variar beta_t y gamma_t, por eso se adapta mejor a estacionalidad y memoria.")
    print("\nDia representativo para explicar el flujo interno:")
    print(f"  dia = {peak_idx}")
    print(f"  P    = {basin.P[peak_idx]:7.2f} mm/d")
    print(f"  T    = {basin.T[peak_idx]:7.2f} C")
    print(f"  PET  = {basin.Ep[peak_idx]:7.2f} mm/d")
    print(f"  beta = {outputs['BETA'][peak_idx]:7.2f}")
    print(f"  gamma= {outputs['GAMMA'][peak_idx]:7.2f}")
    print(f"  Peff = {outputs['PEFF'][peak_idx]:7.2f} mm/d")
    print(f"  ET   = {outputs['ET'][peak_idx]:7.2f} mm/d")
    print(f"  Q0   = {outputs['Q0'][peak_idx]:7.2f} mm/d")
    print(f"  Q1   = {outputs['Q1'][peak_idx]:7.2f} mm/d")
    print(f"  Q2   = {outputs['Q2'][peak_idx]:7.2f} mm/d")
    print(f"  Q    = {outputs['Q'][peak_idx]:7.2f} mm/d")


def create_figure(basin: BasinData, results: List[VariantResult]) -> str:
    colors = {
        "static": "#1f77b4",
        "dynamic_beta": "#ff7f0e",
        "dynamic_beta_gamma": "#d62728",
    }
    best = max(results, key=lambda item: item.metrics["nse"])

    fig, axes = plt.subplots(3, 2, figsize=(16, 15))
    fig.suptitle(
        "dPL-HBV Walkthrough - formulacion, variantes y funcionamiento",
        fontsize=15,
        fontweight="bold",
    )

    ax = axes[0, 0]
    ax.axis("off")
    ax.text(
        0.02,
        0.98,
        "\n".join(
            [
                "Como se plantean los modelos del paper:",
                "",
                "1) dPL+HBV estatico (delta_1)",
                "   LSTM(P,T,PET,attrs) -> pars estaticos -> HBV -> Q",
                "",
                "2) delta_n(beta_t)",
                "   LSTM -> pars estaticos + beta_t diario -> HBV -> Q",
                "",
                "3) delta_n(gamma_t,beta_t)",
                "   LSTM -> pars estaticos + beta_t + gamma_t diarios -> HBV -> Q",
                "",
                "Modulo diario del HBV:",
                "   nieve -> suelo/ET -> aguas sub -> routing gamma",
                "",
                "beta_t controla la particion lluvia->escorrentia.",
                "gamma_t controla la eficiencia de ET.",
                "",
                "En esta version didactica la LSTM se reemplaza",
                "por un emulador explicativo para que el script",
                "corra sin PyTorch.",
            ]
        ),
        ha="left",
        va="top",
        fontsize=10,
        family="monospace",
    )
    ax.set_title("A. Arquitectura conceptual")

    ax = axes[0, 1]
    start = basin.warmup + 180
    end = min(start + 365, len(basin.Q_obs))
    tt = np.arange(end - start)
    ax.plot(tt, basin.Q_obs[start:end], color="black", lw=1.5, label="Q observada")
    for result in results:
        ax.plot(
            tt,
            result.outputs["Q"][start:end],
            color=colors[result.variant],
            lw=2.0 if result.variant == best.variant else 1.2,
            label=f"{result.label} | NSE={result.metrics['nse']:.3f}",
        )
    ax.set_title("B. Hidrograma de evaluacion")
    ax.set_xlabel("Dia")
    ax.set_ylabel("Q (mm/d)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    x = np.arange(len(results))
    width = 0.24
    ax.bar(x - width, [r.metrics["nse"] for r in results], width=width, color="#4C78A8", label="NSE")
    ax.bar(x, [r.metrics["r_et"] for r in results], width=width, color="#59A14F", label="r_ET")
    ax.bar(x + width, [r.metrics["r_bfi"] for r in results], width=width, color="#E15759", label="r_BFI")
    ax.set_xticks(x)
    ax.set_xticklabels([r.label.replace(" estatico", "") for r in results], rotation=12, ha="right")
    ax.set_ylim(0.0, 1.02)
    ax.set_title("C. Comparacion de metricas")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    win = slice(basin.warmup, basin.warmup + 365)
    beta_pred = best.outputs["BETA"]
    gamma_pred = best.outputs["GAMMA"]
    ax.plot(basin.beta_true[win], color="#ff7f0e", lw=2.0, label="beta real")
    ax.plot(beta_pred[win], color="#ff7f0e", lw=1.2, ls="--", label="beta estimado")
    ax2 = ax.twinx()
    ax2.plot(basin.gamma_true[win], color="#2ca02c", lw=2.0, label="gamma real")
    ax2.plot(gamma_pred[win], color="#2ca02c", lw=1.2, ls="--", label="gamma estimado")
    ax.set_title(f"D. Parametros dinamicos del mejor modelo\n({best.label})")
    ax.set_xlabel("Dia despues del warmup")
    ax.set_ylabel("beta")
    ax2.set_ylabel("gamma")
    ax.grid(alpha=0.3)
    lines = ax.get_lines() + ax2.get_lines()
    labels = [line.get_label() for line in lines]
    ax.legend(lines, labels, fontsize=8, loc="upper right")

    ax = axes[2, 0]
    ax.plot(best.outputs["SM"][win], color="#8C564B", lw=1.5, label="SM mejor modelo")
    ax.plot(basin.teacher["SM"][win], color="#8C564B", lw=1.0, ls="--", label="SM referencia")
    ax.plot(best.outputs["ET"][win] / (best.outputs["ET"][win].max() + EPS), color="#2ca02c", lw=1.5, label="ET normalizada")
    ax.plot(best.outputs["BFI"][win], color="#1f77b4", lw=1.5, label="BFI")
    ax.plot(best.outputs["SWE"][win] / (best.outputs["SWE"][win].max() + EPS), color="#17becf", lw=1.5, label="SWE normalizada")
    ax.set_title("E. Estados y salidas internas")
    ax.set_xlabel("Dia despues del warmup")
    ax.set_ylabel("Valor normalizado")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[2, 1]
    flow_start = basin.warmup + 120
    flow_end = min(flow_start + 220, len(basin.Q_obs))
    flow_t = np.arange(flow_end - flow_start)
    ax.plot(flow_t, best.outputs["Q0"][flow_start:flow_end], color="#d62728", lw=1.4, label="Q0 rapido")
    ax.plot(flow_t, best.outputs["Q1"][flow_start:flow_end], color="#ff7f0e", lw=1.4, label="Q1 intermedio")
    ax.plot(flow_t, best.outputs["Q2"][flow_start:flow_end], color="#1f77b4", lw=1.4, label="Q2 baseflow")
    ax.plot(flow_t, best.outputs["Q"][flow_start:flow_end], color="black", lw=1.2, ls="--", label="Q total")
    ax.set_title("F. Descomposicion de caudal")
    ax.set_xlabel("Dia")
    ax.set_ylabel("Flujo (mm/d)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out_path = "dpl_hbv_paper_walkthrough.png"
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    basin = make_synthetic_basin()
    results = run_variants(basin)
    best = max(results, key=lambda item: item.metrics["nse"])

    print_summary(results)
    print_walkthrough(basin, best)
    figure_path = create_figure(basin, results)

    print("\nParametros estaticos del mejor modelo:")
    for key in ["FC", "LP", "PERC", "K0", "K1", "K2", "A", "B"]:
        print(f"  {key:<5} {best.params[key]:8.3f}")

    print("\nArchivo generado:")
    print(f"  {figure_path}")
    print("\nReferencias utiles dentro de la carpeta:")
    print("  - delta_HBV_Feng2022.py  -> mejor comparador de variantes del paper")
    print("  - dpl_hbv_demo.py        -> mejor para explicar diferenciabilidad")
    print("  - dpl_hbv_paper.py       -> replica parcial de la arquitectura")
    print("  - dpl_hbv_paper_walkthrough.py -> guion didactico integrador")


if __name__ == "__main__":
    main()
