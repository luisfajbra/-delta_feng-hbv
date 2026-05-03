# Diagnóstico Técnico y Estrategia de Modularización Didáctica - dPL-HBV

## 1. Estado Actual del Proyecto

### 1.1 Resumen General
El proyecto implementa **dPL-HBV** (Differentiable Parameter Learning + HBV), un modelo hidrológico conceptual que combina una red LSTM inversa para aprender parámetros del modelo HBV a partir de datos meteorológicos y atributos de cuenca, con un forward HBV completamente diferenciable en PyTorch.

### 1.2 Estructura de Directorios Actual
```
7943626/
├── dPLHBVrelease/
│   ├── environment.yml              # Entorno Conda (Python 3.6, PyTorch 1.0.1, CUDA 10.0)
│   ├── Instructions_README.pdf      # Documentación original
│   └── hydroDL-dev/
│       ├── hydroDL/
│       │   ├── model/
│       │   │   ├── rnn.py           # Modelo HBV + LSTM (1850 líneas, archivo central)
│       │   │   ├── train.py         # Bucles de entrenamiento/testing (399 líneas)
│       │   │   ├── crit.py          # Funciones de pérdida (555 líneas)
│       │   │   ├── cnn.py           # Capas CNN auxiliares
│       │   │   └── dropout.py       # Máscaras de dropout personalizadas
│       │   ├── data/
│       │   │   ├── camels.py        # Carga y normalización de datos CAMELS (647 líneas)
│       │   │   └── Dataframe.py     # Clase base de dataframe
│       │   ├── master/              # Utilidades de configuración
│       │   └── utils/               # Utilidades generales (tiempo, etc.)
│       └── example/dPLHBV/
│           ├── traindPLHBV.py       # Script de entrenamiento principal
│           ├── testdPLHBV-Static.py # Evaluación con parámetros estáticos
│           └── testdPLHBV-Dynamic.py# Evaluación con parámetros dinámicos
└── pet_harg/
    ├── daymet/    # PET calculada con método Hargreaves (forcing Daymet)
    ├── maurer/    # PET (forcing Maurer)
    └── nldas/     # PET (forcing NLDAS)
```

---

## 2. Arquitectura del Modelo dPL-HBV

### 2.1 Flujo de Datos
```
Forzamientos (P, T, PET) + Atributos de Cenca
                    │
                    ▼
        ┌───────────────────────┐
        │  LSTM Inversa (gA)    │  ← CudnnLstmModel
        │  Entradas: z (series) │
        │            c (attrs)  │
        │  Salida: parámetros   │
        │    HBV normalizados   │
        └───────────┬───────────┘
                    │ sigmoid/softmax
                    ▼
        ┌───────────────────────┐
        │  Escalado de Pars     │  ← [0,1] → rango físico
        │  Routing (a, b)       │
        │  Component Weights    │
        └───────────┬───────────┘
                    ▼
        ┌───────────────────────┐
        │  HBV Forward          │  ← HBVMul / HBVMulTD
        │  Forzamiento: x       │
        │  (P, T, PET)          │
        │  Loop temporal        │
        └───────────┬───────────┘
                    ▼
        ┌───────────────────────┐
        │  Gamma UH Routing     │  ← UH_gamma + UH_conv
        │  (opcional)           │
        └───────────┬───────────┘
                    ▼
              Q simulado (5 salidas)
              [Qs, Q0, Q1, Q2, ET]
```

### 2.2 Componentes Clave

#### A. Variantes del Modelo HBV (`rnn.py`)

| Clase | Parámetros | ET Shape | Routing | Línea |
|-------|-----------|----------|---------|-------|
| `HBVMul` | 12 estáticos | No | Sí | 903 |
| `HBVMulET` | 13 estáticos (incl. BETAET) | Sí | Sí | 1101 |
| `HBVMulTD` | 12 dinámicos + estáticos | No | Sí | 1341 |
| `HBVMulTDET` | 13 dinámicos + estáticos | Sí | Sí | 1557 |

**12 parámetros HBV originales:**
1. `BETA` [1, 6] - Forma de la curva de respuesta del suelo
2. `FC` [50, 1000] - Capacidad máxima del almacenamiento del suelo (mm)
3. `K0` [0.05, 0.9] - Coeficiente de descarga rápida (superficial)
4. `K1` [0.01, 0.5] - Coeficiente de descarga intermedia
5. `K2` [0.001, 0.2] - Coeficiente de descarga base (lenta)
6. `LP` [0.2, 1] - Umbral de evapotranspiración potencial
7. `PERC` [0, 10] - Percolación máxima (mm/d)
8. `UZL` [0, 100] - Umbral de la zona superior (mm)
9. `TT` [-2.5, 2.5] - Temperatura umbral lluvia/nieve (°C)
10. `CFMAX` [0.5, 10] - Factor de grado-día para fusión de nieve
11. `CFR` [0, 0.1] - Coeficiente de recongelación
12. `CWH` [0, 0.2] - Capacidad de retención de agua en nieve

**Parámetro adicional (ET):**
13. `BETAET` [0.3, 5] - Forma de la ecuación de evapotranspiración

**Parámetros de routing:**
- `a` [0, 2.9], `b` [0, 6.5] - Parámetros de la distribución Gamma del Hidrograma Unitario

#### B. Wrappers LSTM Inversos (`rnn.py`)

| Clase | Parámetros | Línea |
|-------|-----------|-------|
| `MultiInv_HBVModel` | Estáticos + routing + pesos | 1272 |
| `MultiInv_HBVTDModel` | Dinámicos + estáticos + routing + pesos | 1783 |

#### C. Módulos Auxiliares

- `UH_gamma(a, b, lenF)` - Genera el Hidrograma Unitario Gamma (`rnn.py:871`)
- `UH_conv(x, UH)` - Convolución 1D del UH con el exceso de lluvia (`rnn.py:843`)
- `CudnnLstmModel` - LSTM optimizado para GPU (`rnn.py:252`)

### 2.3 Módulos del HBV Forward (loop temporal)

El forward del HBV sigue esta secuencia en cada paso de tiempo (`rnn.py:996-1051`):

1. **Partición de precipitación** - Líquida (lluvia) vs Sólida (nieve) según TT
2. **Módulo de nieve** - Acumulación, fusión (degree-day), recongelación, liberación al suelo
3. **Módulo de suelo** - Humedad relativa, recarga, exceso, evaporación real
4. **Módulo de aguas subterráneas** - Zona superior (Q0, Q1), percolación, zona inferior (Q2)
5. **Agregación** - Promedio ponderado de componentes múltiples
6. **Routing** (opcional) - Convolución UH Gamma

---

## 3. Hallazgos y Problemas Detectados

### 3.1 Problemas de Arquitectura
1. **Acoplamiento excesivo**: `rnn.py` contiene 1850 líneas mezclando LSTM, HBV, routing, CNNs y modelos auxiliares.
2. **Lógica de entrenamiento mezclada con definición de modelo**: `train.py` usa `type(model)` en cadenas `if` extensas para determinar el flujo.
3. **Rutas hardcodeadas**: `traindPLHBV.py` tiene rutas como `/scratch/Camels` y `/data/rnnStreamflow`.
4. **Dependencias implícitas**: El módulo `camels.py` inicializa variables globales (`gageDict`, `statDict`) al importar.

### 3.2 Problemas de Compatibilidad
1. **Python 3.6** - Obsoleto; el `environment.yml` incluye paquetes de 2019.
2. **PyTorch 1.0.1 + CUDA 10.0** - Versiones muy antiguas; la compatibilidad con hardware moderno es dudosa.
3. **`torch._cudnn_rnn`** - API interna de PyTorch que ha cambiado entre versiones (`rnn.py:316-323`).
4. **`F.sigmoid`** - Deprecated en PyTorch moderno; se usa `torch.sigmoid`.

### 3.3 Problemas de Usabilidad Didáctica
1. **Sin comentarios explicativos** en el forward del HBV; la matemática no está documentada paso a paso.
2. **Escalado de parámetros opaco**: Los rangos `parascaLst` están hardcodeados sin explicación.
3. **Componentes múltiples (Nmul)**: El concepto de "multi-componente" no está explicado en el código.
4. **Normalización compleja**: `transNormbyDic` aplica transformaciones log-sqrt para precipitación/caudal sin documentación clara.
5. **El bucle de entrenamiento** usa minibatches aleatorios en tiempo y espacio, lo cual es eficiente pero difícil de seguir.

### 3.4 Riesgos Técnicos
1. **`torch.clamp(SM, min=PRECS)`** - Truco numérico para mantener gradientes (`PRECS=1e-5`); puede causar inestabilidad.
2. **`.cuda()` hardcodeado** en múltiples clases; no funciona en CPU sin modificación.
3. **`torch.no_grad()`** para warm-up en HBV (`rnn.py:938`) - Correcto para inference pero corta el grafo si el warm-up está dentro del training loop.

---

## 4. Estrategia de Modularización Didáctica

### 4.1 Estructura de Carpetas Propuesta

```
dpl_hbv_demo/
├── README.md                     # Visión general del proyecto
├── requirements.txt              # Dependencias modernas (PyTorch >= 2.0)
├── config.py                     # Configuración centralizada (rutas, hiperparámetros)
│
├── data/
│   ├── __init__.py
│   ├── loader.py                 # Carga de datos CAMELS (P, T, PET, atributos)
│   ├── pet.py                    # Carga de PET desde CSVs
│   ├── normalizer.py             # Normalización y desnormalización
│   └── camels_attrs.py           # Lista de atributos y descripciones
│
├── models/
│   ├── __init__.py
│   ├── lstm_inverse.py           # LSTM inversa (gA) - predice parámetros
│   ├── routing.py                # UH Gamma + convolución
│   └── hbv/
│       ├── __init__.py
│       ├── snow.py               # Módulo de nieve (degree-day)
│       ├── soil.py               # Módulo de suelo y ET
│       ├── groundwater.py        # Módulo de aguas subterráneas (Q0, Q1, Q2)
│       ├── forward.py            # Orquestador del forward HBV
│       └── parameters.py         # Escalado de parámetros [0,1] → físico
│
├── training/
│   ├── __init__.py
│   ├── losses.py                 # Funciones de pérdida (RMSE, RMSE combinada)
│   ├── trainer.py                # Bucle de entrenamiento simplificado
│   └── sampler.py                # Muestreo aleatorio de minibatches
│
├── notebooks/
│   ├── 01_data_exploration.ipynb   # Exploración de datos CAMELS
│   ├── 02_hbv_components.ipynb     # Cada módulo HBV explicado paso a paso
│   ├── 03_parameter_learning.ipynb # Cómo la LSTM inversa aprende parámetros
│   ├── 04_routing.ipynb            # Explicación del routing Gamma UH
│   ├── 05_full_model.ipynb         # Modelo completo dPL-HBV
│   └── 06_training_demo.ipynb      # Entrenamiento con 1 cuenca (demo rápida)
│
├── visualize/
│   ├── __init__.py
│   ├── hydrograph.py              # Plot de hidrogramas simulados vs observados
│   ├── parameters.py              # Visualización de parámetros aprendidos
│   └── states.py                  # Visualización de estados internos (SM, SUZ, SLZ)
│
└── scripts/
    ├── train_static.py            # Entrenamiento con parámetros estáticos
    ├── train_dynamic.py           # Entrenamiento con parámetros dinámicos
    └── evaluate.py                # Evaluación y métricas
```

### 4.2 Flujo Paso a Paso para la Refactorización

**Fase 1 - Separación de responsabilidades:**
1. Extraer `HBVMul.forward()` → `models/hbv/forward.py` con submódulos
2. Extraer `UH_gamma` y `UH_conv` → `models/routing.py`
3. Extraer escalado de parámetros → `models/hbv/parameters.py`
4. Extraer `MultiInv_HBVModel` → `models/lstm_inverse.py`

**Fase 2 - Simplificación del entrenamiento:**
1. Crear `training/trainer.py` con un bucle limpio (sin `type(model)` chains)
2. Crear `training/sampler.py` para el muestreo aleatorio
3. Consolidar funciones de pérdida en `training/losses.py`

**Fase 3 - Datos y normalización:**
1. Crear `data/loader.py` con una interfaz clara
2. Crear `data/normalizer.py` con funciones documentadas
3. Centralizar configuraciones en `config.py`

**Fase 4 - Notebooks didácticos:**
1. Cada notebook cubre un concepto con visualizaciones
2. Usar datos de 1-3 cuencas para ejemplos rápidos
3. Incluir diagramas del modelo HBV con ecuaciones

### 4.3 Funciones Principales a Desarrollar

#### `models/hbv/parameters.py`
```
- ParameterScaler(ranges: dict)
  - scale(params_normalized: Tensor) -> Tensor  # [0,1] → rango físico
  - inverse(params_physical: Tensor) -> Tensor  # rango físico → [0,1]
```

#### `models/hbv/snow.py`
```
- SnowModule()
  - forward(P, T, TT, CFMAX, CFR, CWH, states) -> (rain, tosoil, new_states)
```

#### `models/hbv/soil.py`
```
- SoilModule()
  - forward(rain, tosoil, ETpot, SM, FC, BETA, LP, BETAET, states) -> (recharge, excess, ETact, new_states)
```

#### `models/hbv/groundwater.py`
```
- GroundwaterModule()
  - forward(recharge, excess, SUZ, SLZ, K0, K1, K2, PERC, UZL) -> (Q0, Q1, Q2, new_states)
```

#### `models/hbv/forward.py`
```
- HBVForward(n_components: int, use_routing: bool, ...)
  - warmup(forcing, params, bufftime) -> states
  - forward(forcing, params, routing_params, states) -> Q, components, ET
```

#### `models/routing.py`
```
- GammaUH(lenF: int)
  - forward(a, b) -> UH_vector
- UHRouting()
  - forward(Q, UH) -> Q_routed
```

#### `models/lstm_inverse.py`
```
- ParameterLSTM(n_forcing, n_attrs, n_params, n_components, hidden_size)
  - forward(forcing_series, basin_attrs) -> params_normalized
```

### 4.4 Datos Requeridos para Ejecución

| Dato | Fuente | Formato | Descripción |
|------|--------|---------|-------------|
| Precipitación (P) | CAMELS | mm/día | Forzamiento meteorológico |
| Temperatura (T) | CAMELS | °C | Media diaria |
| PET | pet_harg/*.csv | mm/día | Evapotranspiración potencial (Hargreaves) |
| Caudal observado (Q) | CAMELS | ft³/s → mm/día | Variable objetivo |
| Atributos de cuenca | CAMELS | estáticos | Topografía, clima, suelo, vegetación, geología |

**Mínimo para demo:** 1 cuenca con 5-10 años de datos diarios.

### 4.5 Visualizaciones Recomendadas

1. **Diagrama del HBV** - Cajas y flujos entre módulos (nieve → suelo → groundwater → Q)
2. **Hidrograma** - Q observado vs simulado con componentes (Q0, Q1, Q2) apilados
3. **Series de estados** - SM, SUZ, SLZ, SNOWPACK a lo largo del tiempo
4. **Parámetros aprendidos** - Distribución por cuenca, comparación con rangos físicos
5. **Scatter plot** - Q obs vs Q sim con línea 1:1 y métricas (NSE, RMSE)
6. **Mapa** (opcional) - Ubicación de cuencas CAMELS con performance codificada por color

### 4.6 Partes a Reutilizar vs Refactorizar

| Componente | Acción | Razón |
|------------|--------|-------|
| HBV forward logic | **Refactorizar** | Separar en submódulos, añadir comentarios matemáticos |
| UH_gamma / UH_conv | **Reutilizar** | Correctos, solo mover y documentar |
| CudnnLstmModel | **Reemplazar** | Usar `nn.LSTM` estándar de PyTorch (más portable) |
| LSTM cell manual | **Eliminar** | No necesario para la demo; `nn.LSTM` basta |
| CNN models | **Eliminar** | No usados en dPL-HBV |
| Loss functions | **Refactorizar** | Mantener solo `RmseLossComb`, documentar |
| Training loop | **Reescribir** | Eliminar `type()` chains, hacer genérico |
| Data loading (camels.py) | **Refactorizar** | Separar carga de normalización, eliminar globales |
| Normalización | **Reutilizar** | Lógica correcta, solo documentar transformaciones |

### 4.7 Riesgos y Mitigación

| Riesgo | Impacto | Mitigación |
|--------|---------|------------|
| Sin acceso a datos CAMELS completos | Alto | Crear datos sintéticos para demo; usar 1-3 cuencas |
| PyTorch antiguo incompatible | Alto | Actualizar a PyTorch >= 2.0, reemplazar `torch._cudnn_rnn` |
| Inestabilidad numérica en HBV | Medio | Validar con valores de referencia del paper; añadir asserts |
| Overfitting con parámetros dinámicos | Medio | Mantener `dydrop` configurable; recomendar estáticos para demo |
| Complejidad excesiva | Medio | Empezar con Nmul=1 (un solo componente HBV), sin routing |

---

## 5. Recomendaciones de Priorización

### Para presentación/demo rápida (semana 1-2):
1. Extraer HBV forward en submódulos limpios con comentarios
2. Crear notebook `02_hbv_components.ipynb` explicando cada módulo
3. Ejecutar con datos de 1 cuenca y parámetros manuales (sin LSTM)
4. Visualizar hidrograma y estados internos

### Para demostración completa (semana 3-4):
5. Implementar LSTM inversa simplificada con `nn.LSTM`
6. Notebook `03_parameter_learning.ipynb` y `05_full_model.ipynb`
7. Entrenamiento completo con 10-20 cuencas
8. Evaluación con métricas estándar (NSE, KGE, RMSE)

### Para extensión avanzada (semana 5+):
9. Parámetros dinámicos (TDOpt=True)
10. Routing por componente
11. Análisis de regionalización (PUB/PUR)

---

## 6. Información No Confirmada (Marcar)

- **Rutas exactas de datos**: No se verificó si los archivos CSV de PET existen y están completos en `pet_harg/`.
- **Disponibilidad de CAMELS**: No se confirmó si el dataset CAMELS está descargado en la máquina local.
- **GPU disponible**: No se verificó si hay GPU compatible con CUDA en el entorno actual.
- **Paper asociado**: Se menciona el paper de Feng et al. pero no se localizó el PDF en la estructura de archivos.
- **Resultados de referencia**: No se encontraron modelos pre-entrenados (.pt) en el repositorio.
