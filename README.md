# dPL-HBV — Differentiable Parameter Learning + HBV

Demo práctico basado en **Feng et al. (2022)**, *Water Resources Research*:
> "Differentiable, Learnable, Regionalized Process-Based Models With Multiphysical Outputs can Approach State-Of-The-Art Hydrologic Prediction"

## Concepto

El modelo HBV (hidrológico conceptual) se implementa completamente en **PyTorch**, lo que lo hace **diferenciable**: los gradientes fluyen desde la pérdida hasta cada parámetro físico. Esto permite:

1. Optimizar parámetros HBV directamente con Adam/SGD
2. Usar una red neural (LSTM) para *predecir* parámetros desde forzamientos y atributos de cuenca
3. Entrenamiento end-to-end — igual que deep learning, pero con un modelo físico interpretable

## Archivos

| Archivo | Descripción |
|---------|-------------|
| `dpl_hbv_optim.py` | **Demo principal**: optimiza parámetros HBV con gradientes. Muestra diferenciabilidad y backprop |
| `delta_HBV_Feng2022.py` | Implementación numpy del HBV con parámetros estáticos y dinámicos. Compara NSE entre variantes |
| `DIAGNOSIS.md` | Análisis técnico completo del código original (rnn.py, train.py, camels.py) |
| `Feng_etal_2022.pdf` | Paper original |

## Ejecutar

```bash
pip install torch numpy matplotlib
python dpl_hbv_optim.py
```

Genera `dpl_hbv_demo.png` con 6 paneles: caudal simulado vs observado, convergencia del loss, evolución de parámetros, y scatter 1:1.

## Resultado clave

Los parámetros se optimizan por gradiente descendiente y convergen a valores físicamente realistas:

```
Par       Verdadero  Aprendido  Error %
TT            0.000      0.628   —
DD            3.500      2.505  28.4%
LP            0.700      0.682   2.6%
beta          2.500      2.660   6.4%
K0            0.350      0.383   9.4%
K1            0.080      0.075   6.2%
```

La equifinalidad (múltiples soluciones válidas) es esperada en modelos hidrológicos y refleja el comportamiento real del HBV.

## Referencias

- Feng et al. (2022). *Differentiable, Learnable, Regionalized Process-Based Models...* Water Resources Research, 58, e2022WR032404.
- Beck et al. (2020). HBV-light hydrological model. http://www.gloh2o.org/hbv/
- Código original: `7943626/dPLHBVrelease/` de Dapeng Feng
