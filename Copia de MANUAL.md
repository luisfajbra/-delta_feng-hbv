## 3. Archivo 1: `delta_HBV_Feng2022.py`

> **Propósito**: Comparar HBV estático vs δ-HBV con parámetros dinámicos.
> **Tecnología**: NumPy (NO PyTorch) — no hay diferenciabilidad aquí.
> **Qué NO hace**: No entrena, no optimiza, no usa LSTM. Solo simula.

### 3.1 Función `run_HBV` (líneas 38-178)

**Es el motor del modelo HBV en NumPy**. Es la versión "estática" del forward, pero acepta parámetros dinámicos opcionales.

**Firma** (línea 38):
```python
def run_HBV(P, T, Ep, params, dynamic_beta=None, dynamic_gamma=None):
```

**Entradas**:
- `P`, `T`, `Ep`: arrays de longitud N con precipitación, temperatura, PET
- `params`: diccionario con los 15 parámetros (13 HBV + 2 routing)
- `dynamic_beta`, `dynamic_gamma`: arrays opcionales [N] para parametrización dinámica

**Líneas clave**:

| Línea | Qué hace | Por qué importa |
|-------|----------|-----------------|
| 64-78 | Extrae cada parámetro del dict | Nombres exactos del paper |
| 81-85 | Inicializa los 5 estados | Valores iniciales clásicos del HBV |
| 88-94 | Crea arrays de salida | `Q_sim`, `ET_sim`, `BF_sim`, `SM_sim`, `SWE_sim` |
| 100-106 | Construye el hidrograma unitario gamma | Routing — retardo del cauce |
| 112-117 | Partición lluvia/nieve | El umbral TT decide |
| 119-128 | Módulo de nieve completo | smelt, Rfz, Sp, Sliq, Isnow |
| 133-134 | **Punto clave**: si hay dynamic_beta/gamma, úsalos | Aquí está la diferencia δ₁ vs δₙ |
| 138-139 | Calcula W y Peff | La no-linealidad del suelo está en BETA |
| 146-148 | Calcula ET real | BETAET controla la eficiencia |
| 160-173 | Módulo groundwater | Q0, Q1, Q2 — los tres componentes |
| 176 | Convolución con UH gamma | Routing final |

**Diferencia clave con PyTorch**: Usa `max()`, `min()`, `**` de NumPy. No hay gradientes. Es pura simulación forward.

### 3.2 Función `NSE` (líneas 187-193)

**Métrica estándar de hidrología**:

```python
def NSE(Q_sim, Q_obs):
    mask = ~np.isnan(Q_obs)
    obs  = Q_obs[mask]
    sim  = Q_sim[mask]
    num  = np.sum((sim - obs) ** 2)           # error del modelo
    den  = np.sum((obs - np.mean(obs)) ** 2)  # error de usar la media
    return 1.0 - num / den if den > 0 else np.nan
```

**Interpretación**:
- NSE = 1 → perfecto
- NSE = 0 → tan bueno como usar la media de las observaciones
- NSE < 0 → peor que usar la media

### 3.3 Generación de datos sintéticos (líneas 202-228)

Crea un escenario realista tipo CAMELS:

| Línea | Variable | Fórmula | Significado |
|-------|----------|---------|-------------|
| 212-213 | T | 10 + 15×cos(...) + ruido | Temperatura estacional (verano ~25°C, invierno ~-5°C) |
| 219-222 | P | Exponencial con probabilidad estacional | Lluvia más frecuente en primavera/otoño |
| 226-228 | Ep | Basada en T + ciclo estacional | PET estilo Hargreaves simplificado |

### 3.4 Parámetros base (líneas 234-250)

Valores típicos de una cuenca húmeda del CAMELS:

```python
params_base = {
    'TT'    : 0.0,     # Nieva cuando T ≤ 0°C
    'DD'    : 3.5,     # 3.5 mm de nieve se derriten por cada °C sobre 0
    'CWH'   : 0.1,     # El snowpack retiene 10% de agua líquida
    'rfz'   : 0.05,    # 5% de recongelamiento
    'FC'    : 250.0,   # El suelo retiene hasta 250 mm
    'LP'    : 0.7,     # ET empieza a reducirse cuando Ss < 0.7×FC
    'beta'  : 2.5,     # Respuesta del suelo (moderadamente no-lineal)
    'gamma' : 1.0,     # ET proporcional a humedad (lineal)
    'perc'  : 1.2,     # Máx 1.2 mm/día percola al acuífero
    'K0'    : 0.35,    # Flujo rápido: 35% del exceso sobre UZL
    'uzl'   : 15.0,    # Flujo rápido solo si Suz > 15 mm
    'K1'    : 0.08,    # Flujo intermedio: 8% de Suz
    'K2'    : 0.025,   # Flujo base: 2.5% de Slz
    'theta_a'  : 2.5,  # Forma del UH gamma
    'theta_tau': 3.0,  # Escala del UH gamma
}
```

### 3.5 Parametrización dinámica (líneas 266-275)

**β dinámico** (línea 267-271):
```python
beta_seasonal = 2.5 + 1.8 * np.cos(2 * np.pi * (doy - 260) / 365)
# Pico en septiembre (doy=260), mínimo en marzo
P_smooth = np.convolve(P, np.ones(30) / 30, mode='same')  # memoria de 30 días
beta_memory = -0.5 * (P_smooth - P_smooth.mean()) / (P_smooth.std() + 1e-6)
beta_dynamic = np.clip(beta_seasonal + beta_memory + ruido, 0.5, 7.0)
```

**Por qué esta forma**: El paper muestra que β tiene un pico a finales del verano/otoño (la cuenca está "saturada" después del verano seco) y un mínimo en primavera (la cuenca se vació en invierno).

**γ dinámico** (línea 274-275):
```python
gamma_seasonal = 1.0 + 1.5 * np.maximum(0, np.sin(2 * np.pi * (doy - 80) / 365))
gamma_dynamic = np.clip(gamma_seasonal + ruido, 0.1, 4.0)
```

**Por qué esta forma**: γ sigue la fenología vegetal — máximo en verano cuando las plantas están más activas y la ET es más eficiente.

### 3.6 Construcción de "observaciones" (líneas 314-332)

Como no tenemos datos reales de USGS, se construyen observaciones artificiales:

```python
memory_effect = 0.15 * np.convolve(P, np.ones(60) / 60, mode='same')
seasonal_bias  = 0.3 * np.sin(2 * np.pi * (doy - 30) / 365)
Q_obs = Q_delta * (1 + seasonal_bias + memory_effect / Q_delta.mean()) + ruido
```

**Truco conceptual**: Las observaciones se basan en la simulación δ-HBV (la mejor) pero con efectos adicionales que el HBV estático NO puede capturar. Esto hace que el δ-HBV tenga ventaja justa sobre el HBV estático.

### 3.7 Métricas (líneas 339-354)

Tres métricas por modelo:
1. **NSE** en caudal — precisión del hidrograma
2. **Pearson r** en ET — correlación temporal con "MODIS" simulado
3. **Pearson r** en BFI — correlación del índice de flujo base

### 3.8 Visualización (líneas 377-515)

7 paneles en un layout 4×2:

| Panel | Contenido | Análogo en el paper |
|-------|-----------|---------------------|
| A | Series de tiempo Q (años 6-7) | Figura 3 |
| B | Barras NSE comparativo | Tabla 2 |
| C | β dinámico en el tiempo | Figura 4a |
| D | γ dinámico en el tiempo | Figura 4b |
| E | Scatter ET simulada vs MODIS | Figura 6b |
| F | Scatter BFI simulado vs observado | Figura 5e |
| G | Variables internas (Ss, SWE, Q2, ET) | No hay equivalente directo |

---

## 4. Archivo 2: `dpl_hbv_demo.py`

> **Propósito**: Demostrar que una red neural puede predecir parámetros HBV y el gradiente fluye end-to-end.
> **Tecnología**: PyTorch — diferenciabilidad real.
> **Qué SÍ hace**: Entrena una red, backprop a través del HBV.

### 4.1 Función `hbv_forward` (líneas 29-97)

**HBV en PyTorch** — la versión diferenciable del modelo.

**Diferencia con NumPy**: Todas las operaciones usan `torch.*`:
- `torch.where()` en vez de `if/else`
- `torch.clamp()` en vez de `max()/min()`
- `torch.min()`, `torch.tensor(0.0)` para mantener el grafo computacional

**Firma** (línea 29):
```python
def hbv_forward(P, T, Ep, theta, fixed, warmup=0):
```

**Dos dicts de parámetros** — concepto clave del paper:

| Dict | Contenido | ¿Aprendible? |
|------|-----------|--------------|
| `theta` | β, γ, FC, K0, K1, K2, perc, LP, uzl | **Sí** — salen de la red |
| `fixed` | TT, DD, CWH, rfz | **No** — fijos por convención |

**Líneas clave**:

| Línea | Qué hace | Diferencia con NumPy |
|-------|----------|---------------------|
| 66-67 | Partición lluvia/nieve con `torch.where` | Necesita ser diferenciable en TT |
| 79 | `W = (Ss / FC) ** beta[t]` — β puede ser **por timestep** | Aquí β[t] es un tensor que la red predijo |
| 82 | `eta = (Ss / (FC*LP)) ** gamma[t]` — γ por timestep | Aquí γ[t] es un tensor |
| 97 | Retorna `Q_raw[warmup:]` | Solo los días post-warmup se usan en el loss |

### 4.2 Clase `ParamNet` (líneas 104-122)

**La red neural que predice parámetros** (equivalente simplificado de la LSTM del paper).

```python
class ParamNet(nn.Module):
    def __init__(self, n_features=6, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden), nn.ReLU(),    # capa 1: 6→32
            nn.Linear(hidden, hidden),     nn.ReLU(),    # capa 2: 32→32
            nn.Linear(hidden, hidden//2),  nn.ReLU(),    # capa 3: 32→16
            nn.Linear(hidden//2, 2),                     # output: 16→2
        )

    def forward(self, x):
        raw = self.net(x)                        # [T, 2] sin restricciones
        beta  = 0.5 + 6.5 * torch.sigmoid(raw[:, 0])   # mapea a [0.5, 7.0]
        gamma = 0.1 + 3.9 * torch.sigmoid(raw[:, 1])   # mapea a [0.1, 4.0]
        return beta, gamma
```

**¿Por qué sigmoid?** Los parámetros tienen rangos físicos. El sigmoid mapea ℝ → (0,1), y luego se escala al rango deseado.

**Inputs de la red** (6 features):
1. P (precipitación del día)
2. T (temperatura del día)
3. sin(día del año) — estacionalidad
4. cos(día del año) — estacionalidad
5. P rolling mean 30 días — memoria hídrica
6. T rolling mean 15 días — memoria térmica

### 4.3 Función `make_data` (líneas 129-161)

**Genera datos sintéticos con parámetros DINÁMICOS** (a diferencia de `delta_HBV_Feng2022.py` que usa estáticos):

| Línea | Qué genera |
|-------|-----------|
| 142-143 | `beta_true` y `gamma_true` — series temporales con estacionalidad + ruido |
| 154-155 | `theta_true` — dict con β(t), γ(t) como tensores |
| 158 | Ejecuta HBV con parámetros dinámicos para generar Q_obs |
| 159 | Añade ruido gaussiano (5%) para simular error de observación |

### 4.4 Función `train` (líneas 168-214)

**El loop de entrenamiento end-to-end**:

```python
for ep in range(n_ep):
    optim.zero_grad()                              # limpiar gradientes
    
    beta_p, gamma_p = net(X_train)                 # red predice β(t), γ(t)
    FC_p = torch.clamp(torch.tensor(250.0) + torch.randn(1)*0.1, 100, 500)

    theta = {
        'beta': beta_p, 'gamma': gamma_p, 'FC': FC_p,   # dinámicos/aprendidos
        'K0': torch.tensor(0.35), 'K1': ...,             # fijos
        ...
    }

    Q_sim = hbv_forward(Pt[warmup:], Tt[warmup:], Ept[warmup:], theta, fixed, 0)
    
    loss = torch.sqrt(torch.mean((Q_sim - Qt) ** 2))    # RMSE
    loss.backward()                                      # ← BACKPROP aquí
    optim.step()                                         # update pesos
```

**El momento clave**: `loss.backward()` (línea 205) calcula gradientes que fluyen:
1. Desde el loss hacia Q_sim
2. Desde Q_sim a través del HBV (cada estado, cada ecuación)
3. Hasta β(t), γ(t) y los pesos de la red

Esto es la **diferenciabilidad** en acción.

### 4.5 Función `plot` (líneas 221-297)

**6 paneles** que muestran:
1. Q obs vs Q sim
2. Convergencia del loss
3. Scatter 1:1
4. β verdadero vs β aprendido
5. γ verdadero vs γ aprendido
6. Relación P→Q por estación

**Resultado esperado**: NSE ~0.8-0.9 en datos sintéticos. La red aprende a aproximar la estacionalidad de β y γ.

---

## 5. Archivo 3: `dpl_hbv_paper.py`

> **Propósito**: Replicación fiel de la arquitectura del paper: LSTM + HBV + Routing Gamma.
> **Tecnología**: PyTorch con `nn.LSTM`.
> **Qué hace**: End-to-end training con la arquitectura exacta del paper.

### 5.1 Rangos de parámetros (líneas 26-50)

```python
HBV_PARAM_RANGES = {
    'TT':     (-2.5,   2.5,   '°C'),
    'CFMAX':  ( 0.5,  10.0,   'mm/°C/d'),
    'CFR':    ( 0.0,   0.1,   ''),
    'CWH':    ( 0.0,   0.2,   ''),
    'FC':     (50.0, 1000.0,   'mm'),
    'LP':     ( 0.2,   1.0,   ''),
    'BETA':   ( 1.0,   6.0,   ''),
    'PERC':   ( 0.0,  10.0,   'mm/d'),
    'UZL':    ( 0.0, 100.0,   'mm'),
    'K0':     ( 0.05,  0.9,   ''),
    'K1':     ( 0.01,  0.5,   ''),
    'K2':     ( 0.001, 0.2,   ''),
    'BETAET': ( 0.3,   5.0,   ''),
}
```

**Estos rangos son EXACTOS** a los del paper (`parascaLst` en `rnn.py` del código original). Si alguien te pregunta de dónde salen: son los rangos físicos establecidos en la literatura hidrológica para cuencas del CONUS.

### 5.2 Clase `HBVForward` (líneas 55-175)

**HBV diferenciable completo con routing Gamma**. Es el forward más fiel al paper de los tres archivos.

**Constructor** (líneas 71-75):
```python
def __init__(self, n_warmup=365, uh_len=15):
    super().__init__()
    self.n_warmup = n_warmup   # 365 días de calentamiento
    self.uh_len = uh_len       # 15 días de ventana de routing
    self.eps = 1e-5            # protección numérica (evita división por 0)
```

**Forward** (línea 77):
```python
def forward(self, P, T, Ep, pars, route_pars=None):
```

**Secciones del forward**:

| Líneas | Módulo | Variables de estado |
|--------|--------|---------------------|
| 98-111 | Nieve | Sp, Sliq |
| 116-134 | Suelo | Ss |
| 139-150 | Groundwater | Suz, Slz |
| 155-167 | Routing Gamma | (ninguna, es convolución) |
| 170-173 | Warmup + retorno | Q, Q0, Q1, Q2, ET |

**Detalles clave por módulo**:

#### Nieve (líneas 98-111)
```python
if T[t] <= TT:
    Ps = P[t]   # nieve
    Pr = torch.tensor(0.0)
else:
    Ps = torch.tensor(0.0)
    Pr = P[t]   # lluvia
```
**Por qué no `torch.where` aquí**: El if/else de Python NO rompe la diferenciabilidad porque solo controla el flujo, no los valores. Las operaciones dentro (`Sp + Ps + Rfz - smelt`) sí son torch operations.

```python
smelt = torch.clamp(CFMAX * (T[t] - TT), min=torch.tensor(0.0))
smelt = torch.min(smelt, Sp + Ps)  # no derretir más de lo que hay
```

#### Suelo (líneas 116-134)
```python
W = torch.clamp((Ss / (FC + eps)) ** BETA, max=torch.tensor(1.0))
```
El `eps` evita división por cero si FC=0 (protección numérica del paper).

```python
eta_ratio = Ss / (FC * LP + eps)
eta = torch.clamp(eta_ratio ** BETAET, max=torch.tensor(1.0))
ET_act = eta * Ep[t]
```
Si Ss < FC×LP → eta_ratio < 1 → eta < 1 → ET reducida. Esto modela que las plantas no pueden transpirar eficientemente cuando el suelo está seco.

#### Groundwater (líneas 139-150)
```python
Perc = torch.min(PERC, Suz)
Q0 = torch.clamp(K0 * (Suz - UZL), min=torch.tensor(0.0))
```
Q0 solo existe cuando Suz > UZL. El `clamp` asegura que no haya flujo negativo.

#### Routing Gamma (líneas 177-192)
```python
def _gamma_routing(self, Q_raw, route_pars):
    a = route_pars['a']
    b = route_pars['b']
    lenF = self.uh_len

    t_uh = torch.arange(1, lenF + 1, dtype=torch.float32)
    uh = (1.0 / (gamma_func(a.item()) * b ** a)) * t_uh ** (a - 1) * torch.exp(-t_uh / b)
    uh = uh / uh.sum()  # normalizar

    Q_routed = torch.conv1d(
        Q_raw.view(1, 1, -1),
        uh.flip(0).view(1, 1, -1),
        padding=lenF - 1
    ).squeeze()
    return Q_routed[:len(Q_raw)]
```

**La ecuación del hidrograma unitario gamma**:
```
UH(t) = (1/Γ(a)) × (t/b)^(a-1) × exp(-t/b)
```

Donde:
- `a` = parámetro de forma (controla la "punta" del hidrograma)
- `b` = parámetro de escala (controla el retardo temporal)
- `Γ(a)` = función gamma de Euler

**¿Qué hace el routing?** Convoluciona el caudal "instantáneo" con un kernel que simula el tiempo que tarda el agua en llegar al cauce. Sin routing, el caudal responde inmediatamente a la lluvia. Con routing, hay un retardo de ~15 días típico.

### 5.3 Clase `ParameterLSTM` (líneas 181-229)

**La LSTM inversa del paper** — el componente g_A.

```python
class ParameterLSTM(nn.Module):
    def __init__(self, n_forcing=3, n_attr=27, hidden=128, n_layers=1,
                 n_params=N_STATIC_PARAMS + N_ROUTING):
        super().__init__()
        # LSTM: input = forcing + attributes en cada timestep
        self.lstm = nn.LSTM(
            input_size=n_forcing + n_attr,  # 3 + 27 = 30
            hidden_size=hidden,              # 128
            num_layers=n_layers,             # 1
            batch_first=True
        )
        # Dense layers: hidden → parámetros
        self.fc = nn.Sequential(
            nn.Linear(hidden, hidden),       # 128→128
            nn.ReLU(),
            nn.Linear(hidden, n_params),     # 128→15
        )
```

**Arquitectura**:
```
Input:  [T, 30]  ← 3 forcing + 27 attributes
  ↓
LSTM:   [1, T, 30] → [1, T, 128]  (hidden states en cada timestep)
  ↓
Tomar última salida: [128]
  ↓
FC:     [128] → [128] → [15]
  ↓
Sigmoid: [15] en [0, 1]
```

**Forward** (línea 202):
```python
def forward(self, forcing, attr):
    T = forcing.shape[0]
    attr_expanded = attr.unsqueeze(0).expand(T, -1)  # [T, 27]
    x = torch.cat([forcing, attr_expanded], dim=1).unsqueeze(0)  # [1, T, 30]

    lstm_out, (h_n, c_n) = self.lstm(x)       # [1, T, 128]
    last_hidden = lstm_out[0, -1, :]          # [128] — última salida

    params_norm = torch.sigmoid(self.fc(last_hidden))  # [15] en [0,1]
    return params_norm
```

**¿Por qué solo la última salida de la LSTM?** Porque en el modelo de parámetros **estáticos** del paper, la LSTM procesa toda la serie temporal y luego produce UN SOLO vector de parámetros para toda la cuenca. Es como si la LSTM "resumiera" la información de los forzamientos para decidir qué parámetros son mejores.

**¿Por qué sigmoid?** Para que los outputs estén en [0,1]. Luego se escalan al rango físico con `_scale_params`.

### 5.4 Clase `DPLHBV` (líneas 235-275)

**Modelo completo end-to-end**:

```python
class DPLHBV(nn.Module):
    def __init__(self, hidden=128, n_layers=1, n_warmup=365):
        super().__init__()
        self.lstm = ParameterLSTM(hidden=hidden, n_layers=n_layers)
        self.hbv = HBVForward(n_warmup=n_warmup)
        self._build_param_ranges()
```

**`_scale_params`** (línea 249):
```python
def _scale_params(self, params_norm):
    return self.param_lo + (self.param_hi - self.param_lo) * params_norm
```
Ejemplo: si `params_norm[4] = 0.3` para FC:
- FC = 50 + (1000 - 50) × 0.3 = 50 + 285 = 335 mm

**`_unpack_params`** (línea 255):
```python
def _unpack_params(self, scaled):
    pars = {}  # 13 parámetros HBV
    for i, name in enumerate(param_names[:13]):
        pars[name] = scaled[i]
    route_pars = {'a': scaled[13], 'b': scaled[14]}
    return pars, route_pars
```

**Forward** (línea 263):
```python
def forward(self, P, T, Ep, attr):
    forcing = torch.stack([P, T, Ep], dim=1)     # [T, 3]
    params_norm = self.lstm(forcing, attr)        # [15] en [0,1]
    scaled = self._scale_params(params_norm)      # rango físico
    pars, route_pars = self._unpack_params(scaled)

    Q, Q0, Q1, Q2, ET = self.hbv(P, T, Ep, pars, route_pars)
    return Q, Q0, Q1, Q2, ET, scaled
```

### 5.5 Función `make_camels_synthetic` (líneas 281-363)

**Genera datos con estructura CAMELS**:

| Componente | Líneas | Descripción |
|------------|--------|-------------|
| Forcing T | 297-298 | Temperatura estacional con ruido |
| Forcing P | 301-305 | Precipitación con eventos aleatorios |
| Forcing Ep | 308-310 | PET estilo Hargreaves |
| Atributos | 314-342 | 27 descriptores de cuenca (simulados) |
| Q_obs | 345-356 | HBV verdadero + 5% ruido |

**Los 27 atributos** simulan los del dataset CAMELS:
- Aridity index, snow fraction, elevation, slope, area
- Forest fraction, LAI, GVF (green vegetation fraction)
- Soil depth, porosity, conductivity, water content
- Geology (igneous, metamorphic, sedimentary, carbonates)
- Permeability, storage min/max

**Truco importante**: Q_obs se genera con los parámetros verdaderos + ruido del 5%. Esto significa que existe una solución "perfecta" y el entrenamiento debería converger hacia ella.

### 5.6 Función `train_dpl_hbv` (líneas 369-518)

**Loop de entrenamiento — réplica del paper**:

```python
def train_dpl_hbv(n_epochs=100, lr=0.005):
    # 1. Crear datos
    P, T, Ep, Q_obs_np, attr_true, Pt, Tt, Ept, Q_obs_t, warmup = make_camels_synthetic()

    # 2. Crear modelo
    model = DPLHBV(hidden=128, n_layers=1, n_warmup=warmup)

    # 3. Optimizer y scheduler (exactos del paper)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
```

**Loop principal** (líneas 484-513):

| Línea | Operación | Significado |
|-------|-----------|-------------|
| 485 | `optimizer.zero_grad()` | Limpiar gradientes anteriores |
| 488 | `model(Pt, Tt, Ept, attr_tensor)` | Forward completo |
| 491 | `loss = sqrt(mean((Q_sim - Q_obs)²))` | RMSE |
| 492 | `loss.backward()` | **Backprop**: gradiente fluye por HBV → LSTM |
| 495 | `clip_grad_norm_(max_norm=1.0)` | Evitar gradientes explosivos |
| 497 | `optimizer.step()` | Actualizar pesos |
| 498 | `scheduler.step()` | Decay del learning rate |

**¿Qué imprime?** Cada 10 epochs:
- Loss (RMSE)
- NSE
- BETA (qué tan no-lineal es el suelo)
- FC (capacidad de campo)
- K0 (velocidad flujo rápido)
- K2 (velocidad flujo base)

### 5.7 Función `plot_results` (líneas 526-653)

**9 paneles** en layout 3×3:

| Panel | Qué muestra | Por qué importa |
|-------|-------------|-----------------|
| 1 | Hidrograma Q obs vs sim | ¿El modelo captura los picos? |
| 2 | Componentes apilados (Q0, Q1, Q2) | ¿Cuánto es flujo base vs rápido? |
| 3 | ET simulada | Variable física no entrenada |
| 4 | Convergencia del loss | ¿Está aprendiendo? |
| 5 | NSE por epoch | ¿Mejora la eficiencia? |
| 6 | Scatter 1:1 | ¿Hay bias sistemático? |
| 7 | Parámetros verdadero vs aprendido | ¿Converge a los valores reales? |
| 8 | Error porcentual por parámetro | ¿Cuáles son difíciles de aprender? |
| 9 | Forzamientos del último año | Contexto meteorológico |

---

## 6. Glosario de Parámetros

### Parámetros HBV (los 13)

| Símbolo | Nombre | Rango | Unidad | Descripción intuitiva |
|---------|--------|-------|--------|----------------------|
| TT | Temperature Threshold | [-2.5, 2.5] | °C | Si T < TT → nieva, si T > TT → llueve |
| CFMAX | Degree-day factor | [0.5, 10] | mm/°C/d | Cada grado sobre TT derrite CFMAX mm de nieve |
| CFR | Re-freezing coefficient | [0, 0.1] | - | Fracción del agua líquida que se recongela cuando T < TT |
| CWH | Water holding capacity | [0, 0.2] | - | El snowpack puede retener CWH×Sp de agua líquida antes de drenar |
| FC | Field Capacity | [50, 1000] | mm | Máxima agua que el suelo puede almacenar |
| LP | Limit of potential ET | [0.2, 1.0] | - | Cuando Ss < LP×FC, la ET se reduce |
| BETA | Shape parameter | [1, 6] | - | Controla qué tan "no-lineal" es la respuesta del suelo. Alto = más runoff para mismo input |
| PERC | Max percolation | [0, 10] | mm/d | Máxima agua que pasa del suelo superior al acuífero por día |
| UZL | Upper zone threshold | [0, 100] | mm | Flujo rápido solo se activa cuando Suz > UZL |
| K0 | Fast flow coefficient | [0.05, 0.9] | 1/d | Velocidad del flujo rápido (crecidas) |
| K1 | Intermediate flow coefficient | [0.01, 0.5] | 1/d | Velocidad del flujo intermedio |
| K2 | Baseflow coefficient | [0.001, 0.2] | 1/d | Velocidad del flujo base (ríos en época seca) |
| BETAET | ET shape parameter | [0.3, 5] | - | Controla qué tan sensible es la ET a la humedad del suelo |

### Parámetros de Routing

| Símbolo | Nombre | Rango | Descripción |
|---------|--------|-------|-------------|
| a | Gamma shape | [0, 2.9] | Forma del hidrograma unitario |
| b | Gamma scale | [0, 6.5] | Escala temporal del hidrograma unitario |

---

## 7. Diagrama de Flujo Completo

```
┌─────────────────────────────────────────────────────────────────┐
│                     FORZAMIENTOS DIARIOS                        │
│   P(t): Precipitación    T(t): Temperatura    Ep(t): PET        │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    MÓDULO 1: NIEVE                              │
│                                                                 │
│   ¿T ≤ TT? → Sí: Ps=P (nieve), No: Pr=P (lluvia)              │
│   smelt = CFMAX × (T - TT)    Rfz = CFR × (TT-T) × Sliq       │
│   Sp += Ps + Rfz - smelt      Sliq += smelt - Rfz             │
│   Isnow = max(0, Sliq - CWH×Sp)  → va al suelo                │
└────────────────────────┬────────────────────────────────────────┘
                         │ Pr + Isnow
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    MÓDULO 2: SUELO                              │
│                                                                 │
│   W = min((Ss/FC)^BETA, 1)     Peff = W × (Pr + Isnow)        │
│   Ex = max(0, Ss - FC)                                        │
│   eta = min((Ss/(FC×LP))^BETAET, 1)   ET = eta × Ep           │
│   Ss = clamp(Ss + Pr + Isnow - Peff - Ex - ET, 0, FC)         │
│                                                                 │
│   Peff + Ex → van al groundwater                                │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                  MÓDULO 3: GROUNDWATER                          │
│                                                                 │
│   Perc = min(PERC, Suz)                                         │
│   Q0 = max(0, K0 × (Suz - UZL))   ← flujo rápido (crecidas)   │
│   Q1 = K1 × Suz                     ← flujo intermedio         │
│   Suz = max(0, Suz + Peff + Ex - Perc - Q0 - Q1)               │
│   Q2 = K2 × Slz                     ← flujo base               │
│   Slz = max(0, Slz + Perc - Q2)                                │
│                                                                 │
│   Q_raw = Q0 + Q1 + Q2                                          │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                  MÓDULO 4: ROUTING                              │
│                                                                 │
│   UH_gamma(t) = (1/Γ(a)) × (t/b)^(a-1) × exp(-t/b)            │
│   Q_routed = Q_raw * UH_gamma  (convolución, ventana 15 días)  │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                      SALIDAS                                    │
│                                                                 │
│   Q(t): Caudal simulado  (comparar con Q_obs → loss)           │
│   ET(t): Evapotranspiración real  (no se entrena, pero sale)   │
│   Q0, Q1, Q2: Componentes de caudal                            │
└─────────────────────────────────────────────────────────────────┘
```

### Flujo de entrenamiento (backprop):

```
Q_obs  ─→  Loss(RMSE)  ─→  Q_sim  ─→  HBV Forward  ─→  Parámetros
   ↑                                                          │
   │                                                          ▼
   └──────────────── LSTM (g_A) ←────────────────────── Gradientes
```

---

## Preguntas Frecuentes para tu Preparación

### ¿Qué es el "warmup" y por qué 365 días?

Los estados internos (Sp, Sliq, Ss, Suz, Slz) se inicializan con valores arbitrarios (0, FC×0.5, etc.). Los primeros 365 días se "queman" para que los estados converjan a valores realistas antes de empezar a calcular el loss. Es como "precalentar" el modelo.

### ¿Por qué el NSE es ~0.99 en datos sintéticos pero ~0.73 en el paper?

Porque en datos sintéticos, Q_obs se genera con el MISMO HBV que intentamos calibrar. Es un problema de "calibración inversa" donde la solución exacta existe. En el paper, Q_obs son datos reales de USGS con procesos que el HBV no modela (agua subterránea profunda, efectos humanos, etc.).

### ¿Qué diferencia hay entre los 3 archivos?

| Archivo | Tecnología | Objetivo | ¿Entrena? | ¿LSTM? |
|---------|-----------|----------|-----------|--------|
| `delta_HBV_Feng2022.py` | NumPy | Comparar estático vs dinámico | No | No |
| `dpl_hbv_demo.py` | PyTorch | Demo MLP + β(t),γ(t) dinámicos | Sí | No (MLP) |
| `dpl_hbv_paper.py` | PyTorch | Replicación fiel LSTM + HBV + routing | Sí | Sí |

### ¿Qué es la "parametrización dinámica"?

En vez de usar un solo valor de BETA para todo el período de simulación, se usa un valor diferente BETA(t) para cada día. En el paper, la LSTM predice estos valores diarios. En nuestro demo, los simulamos con funciones sinusoidales.

### ¿Qué es el routing Gamma UH?

Es una convolución que simula el tiempo que tarda el agua en viajar desde donde cae la lluvia hasta el punto de medición del caudal. Sin routing, el caudal responde instantáneamente. Con routing, hay un retardo y suavizado.
