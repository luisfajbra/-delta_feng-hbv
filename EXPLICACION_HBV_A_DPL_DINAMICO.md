# De HBV Estatico a dPL-HBV Dinamico

Este archivo explica, paso a paso y sin saltos, como el codigo original pasa de:

- `HBV estatico`
- a `dPL-HBV con parametros dinamicos`

La idea es que puedas leer `rnn.py` y entender que esta pasando sin adivinar nada.


## 1. Idea en 20 segundos

El HBV original usa un solo set de parametros para toda la serie.

El dPL-HBV dinamico hace esto:

1. La LSTM no produce un solo vector final, sino una serie temporal de parametros.
2. De esa serie temporal, el codigo arma una version "casi estatica".
3. Luego reemplaza solo algunos parametros por su valor diario.
4. El HBV sigue siendo el mismo HBV.
5. Lo unico que cambia es que algunos parametros ahora cambian con el tiempo.


## 2. El esquema mental correcto

### Caso estatico

```text
z (forzamientos + atributos)
    |
    v
LSTM
    |
    v
Gen[time, basin, para]
    |
    +--> tomar SOLO el ultimo tiempo: Gen[-1, :, :]
             |
             v
         parametros HBV fijos por cuenca
             |
             v
            HBV
             |
             v
             Q
```

### Caso dinamico

```text
z (forzamientos + atributos)
    |
    v
LSTM
    |
    v
Params0[time, basin, para]
    |
    +--> escalar a rango fisico
    |
    +--> crear una copia estatica desde un solo timestep
    |
    +--> reemplazar SOLO los indices en tdlst por su version diaria
             |
             v
      parametros mezclados:
      - algunos estaticos
      - algunos dinamicos
             |
             v
            HBV
             |
             v
            Q
```


## 3. Mapa rapido de tensores

Cuando leas `rnn.py`, piensa en estas formas:

```text
x           = [time, basin, var]
z           = [time, basin, ninv]
Gen         = [time, basin, para_total]
Params0     = [basin, para_total]            en el modelo estatico
Params0     = [time, basin, para_total]      en el modelo dinamico
hbvpara     = [basin, nfea, nmul]            en el modelo estatico
hbvpara     = [time, basin, nfea, nmul]      en el modelo dinamico
parAllTrans = [time, basin, nfea, nmul]
parstaFull  = [time, basin, nfea, nmul]
parhbvFull  = [time, basin, nfea, nmul]
```

Traduccion humana:

- `x`: lo que entra al HBV forward
- `z`: lo que entra a la LSTM inversa
- `Gen` o `Params0`: lo que la LSTM produce
- `parAllTrans`: parametros ya escalados a rango fisico
- `parstaFull`: copia estatica repetida en todos los dias
- `parhbvFull`: mezcla final que de verdad usa el HBV

Si esta seccion te queda clara, el resto del archivo se vuelve mucho mas facil.


## 4. Donde se activa el modo dinamico

En `traindPLHBV.py` esta el interruptor real.

Referencia:

- `example/dPLHBV/traindPLHBV.py:64-74`
- `example/dPLHBV/traindPLHBV.py:277-280`

Lo importante es esto:

```python
if TDOpt is True:
    tdRep = [1, 13]
    ETMod = True
    Nfea = 13
    dydrop = 0.0
    staind = -1
```

Eso significa:

- `TDOpt=True`: usar la version dinamica
- `tdRep=[1, 13]`: los parametros dinamicos son el 1 y el 13
- `ETMod=True`: activan el parametro extra de ET
- `Nfea=13`: ya no usan 12 parametros HBV, ahora usan 13
- `dydrop=0.0`: NO fuerzan esos parametros a comportarse como estaticos
- `staind=-1`: para los parametros estaticos toman el ultimo timestep aprendido


## 5. Ojo: que parametros son 1 y 13

En este codigo el orden de parametros es:

```text
 1 -> BETA
 2 -> FC
 3 -> K0
 4 -> K1
 5 -> K2
 6 -> LP
 7 -> PERC
 8 -> UZL
 9 -> TT
10 -> CFMAX
11 -> CFR
12 -> CWH
13 -> BETAET
```

Eso sale del escalado en `rnn.py`.

Referencia:

- `hydroDL/model/rnn.py:1404-1410`
- `hydroDL/model/rnn.py:1622-1628`

Entonces:

- `tdRep=[1, 13]` quiere decir: hacer dinamicos `BETA` y `BETAET`

No quiere decir "todos los parametros son dinamicos".


## 6. La confusion mas comun: que es `Params0`

La palabra `Params0` aparece en los dos modelos, pero NO significa lo mismo en la practica.

### En el modelo estatico

Referencia:

- `hydroDL/model/rnn.py:1314-1338`

Codigo clave:

```python
Gen = self.lstminv(z)
Params0 = Gen[-1, :, :]
hbvpara0 = Params0[:, 0:self.nhbvpm]
hbvpara = torch.sigmoid(hbvpara0).view(ngage, self.nfea, self.nmul)
```

Traduccion humana:

1. `self.lstminv(z)` produce una serie temporal completa.
2. Pero el codigo NO usa toda la serie.
3. Se queda solo con el ultimo tiempo: `Gen[-1, :, :]`.
4. Ese ultimo tiempo se llama `Params0`.
5. A partir de ahi, arma un solo set de parametros por cuenca.

En otras palabras:

```text
La LSTM produce muchos timesteps
pero el modelo estatico tira casi todo
y se queda con el ultimo.
```

### En el modelo dinamico

Referencia:

- `hydroDL/model/rnn.py:1828-1850`

Codigo clave:

```python
Params0 = self.lstminv(z)
ntstep = Params0.shape[0]
hbvpara0 = Params0[:, :, 0:self.nhbvpm]
hbvpara = torch.sigmoid(hbvpara0).view(ntstep, ngage, self.nfea, self.nmul)
```

Traduccion humana:

1. `self.lstminv(z)` vuelve a producir una serie temporal completa.
2. Esta vez NO la colapsan al ultimo paso.
3. `Params0` ahora contiene parametros para cada dia.
4. Despues los acomodan como:

```text
[tiempo, cuenca, parametro, componente]
```

Eso es el corazon del cambio.


## 7. El cambio mas importante, casi linea por linea

Esta es la comparacion mas importante del archivo.

### Antes: modelo estatico

Referencia:

- `hydroDL/model/rnn.py:1315-1320`

```python
Gen = self.lstminv(z)
Params0 = Gen[-1, :, :]
hbvpara0 = Params0[:, 0:self.nhbvpm]
hbvpara = torch.sigmoid(hbvpara0).view(ngage, self.nfea, self.nmul)
```

Que hace cada linea:

1. `Gen = self.lstminv(z)`
   La LSTM devuelve una salida por cada tiempo.

2. `Params0 = Gen[-1, :, :]`
   Toma solo el ultimo tiempo.

3. `hbvpara0 = Params0[:, 0:self.nhbvpm]`
   Corta la parte que corresponde a parametros HBV.

4. `hbvpara = sigmoid(...).view(ngage, self.nfea, self.nmul)`
   Los lleva a `[0,1]` y los reordena como parametros fijos por cuenca/componente.


### Despues: modelo dinamico

Referencia:

- `hydroDL/model/rnn.py:1829-1835`

```python
Params0 = self.lstminv(z)
ntstep = Params0.shape[0]
hbvpara0 = Params0[:, :, 0:self.nhbvpm]
hbvpara = torch.sigmoid(hbvpara0).view(ntstep, ngage, self.nfea, self.nmul)
```

Que hace cada linea:

1. `Params0 = self.lstminv(z)`
   La LSTM devuelve una salida por cada tiempo.

2. `ntstep = Params0.shape[0]`
   Guarda cuantos dias hay.

3. `hbvpara0 = Params0[:, :, 0:self.nhbvpm]`
   Corta la parte HBV, pero ahora sin perder la dimension tiempo.

4. `hbvpara = sigmoid(...).view(ntstep, ngage, self.nfea, self.nmul)`
   Ahora cada dia tiene su propio set de parametros normalizados.


## 8. Todavia falta una parte: no todos los parametros se vuelven dinamicos

Este es el detalle que mas confunde.

Muchos piensan:

```text
Si la LSTM produce parametros diarios,
entonces TODOS los parametros HBV se vuelven diarios.
```

Pero el codigo NO hace eso.

Hace esto:

1. Construye una version estatica de todos los parametros.
2. Copia esa version estatica a todos los dias.
3. Reemplaza solo los indices de `tdlst`.

Referencia:

- `hydroDL/model/rnn.py:1414-1424`
- `hydroDL/model/rnn.py:1632-1642`

Codigo clave:

```python
parstaFull = parAllTrans[staind, :, :, :].unsqueeze(0).repeat([Nstep, 1, 1, 1])
parhbvFull = torch.clone(parstaFull)
pmat = torch.ones([1, Ngrid, 1]) * dydrop
for ix in tdlst:
    staPar = parstaFull[:, :, ix-1, :]
    dynPar = parAllTrans[:, :, ix-1, :]
    drmask = torch.bernoulli(pmat).detach_().cuda()
    comPar = dynPar * (1-drmask) + staPar * drmask
    parhbvFull[:, :, ix-1, :] = comPar
```


## 9. Traduccion a prueba de tontos de ese bloque

### Linea 1

```python
parstaFull = parAllTrans[staind, :, :, :].unsqueeze(0).repeat([Nstep, 1, 1, 1])
```

Esto significa:

- "Voy a agarrar UN solo timestep de parametros"
- "Ese timestep sera mi version estatica"
- "Lo voy a repetir en todos los dias"

Como `staind = -1`, el timestep elegido es el ultimo.

O sea:

```text
Tomo el ultimo vector de parametros aprendidos
y lo copio para todos los dias.
```


### Linea 2

```python
parhbvFull = torch.clone(parstaFull)
```

Esto significa:

- "Arranco suponiendo que TODO sera estatico"


### Linea 3

```python
pmat = torch.ones([1, Ngrid, 1]) * dydrop
```

Esto arma una probabilidad de dropout para los parametros dinamicos.

Si `dydrop=0.0`, entonces:

```text
probabilidad de apagar lo dinamico = 0
```

O sea:

```text
no apaga nada
```


### Linea 4

```python
for ix in tdlst:
```

Esto significa:

- "Voy a recorrer solo los parametros que quiero hacer dinamicos"

Como `tdlst = [1, 13]`, el loop solo toca:

- `BETA`
- `BETAET`


### Linea 5

```python
staPar = parstaFull[:, :, ix-1, :]
```

Esto saca la version estatica del parametro `ix`.


### Linea 6

```python
dynPar = parAllTrans[:, :, ix-1, :]
```

Esto saca la version diaria del mismo parametro.


### Linea 7

```python
drmask = torch.bernoulli(pmat).detach_().cuda()
```

Esto decide si para alguna cuenca ese parametro dinamico se fuerza a comportarse como estatico.

Pero con `dydrop=0.0`, el resultado practico es:

```text
drmask = 0
```


### Linea 8

```python
comPar = dynPar * (1-drmask) + staPar * drmask
```

Esto mezcla ambas opciones.

Si `drmask = 0`:

```text
comPar = dynPar
```

Si `drmask = 1`:

```text
comPar = staPar
```


### Linea 9

```python
parhbvFull[:, :, ix-1, :] = comPar
```

Esto reemplaza en la matriz final el parametro `ix`.

Resultado final:

- `BETA` queda dinamico
- `BETAET` queda dinamico
- todo lo demas queda estatico


## 10. Visualmente, que sale de ese bloque

Piensa en una tabla por parametro.

### Antes de reemplazar

```text
Dia      BETA   FC    K0   ...   BETAET
1        fijo   fijo  fijo       fijo
2        fijo   fijo  fijo       fijo
3        fijo   fijo  fijo       fijo
4        fijo   fijo  fijo       fijo
```

### Despues de reemplazar `tdlst=[1,13]`

```text
Dia      BETA   FC    K0   ...   BETAET
1        dyn    fijo  fijo       dyn
2        dyn    fijo  fijo       dyn
3        dyn    fijo  fijo       dyn
4        dyn    fijo  fijo       dyn
```

Eso es exactamente el modelo dinamico de este experimento.


## 11. Donde pegan esos parametros dentro del HBV

### `BETA`

Referencia:

- `hydroDL/model/rnn.py:1470-1475`
- `hydroDL/model/rnn.py:1688-1691`

Codigo:

```python
soil_wetness = (SM / parFC) ** parBETA
recharge = (RAIN + tosoil) * soil_wetness
```

Traduccion humana:

- `BETA` controla la no linealidad del paso suelo -> recarga/escorrentia
- Si `BETA` cambia dia a dia, cambia la sensibilidad del suelo dia a dia


### `BETAET`

Referencia:

- `hydroDL/model/rnn.py:1703-1708`

Codigo:

```python
evapfactor = (SM / (parLP * parFC)) ** parBETAET
ETact = ETpm[t, :, :] * evapfactor
```

Traduccion humana:

- `BETAET` controla la forma de la funcion de evapotranspiracion
- Si `BETAET` cambia dia a dia, la ET real responde distinto segun el dia


## 12. Muy importante: el routing NO se vuelve diario

Este detalle suele pasar desapercibido.

En el modelo dinamico:

```python
routpara0 = Params0[-1, :, self.nhbvpm:self.nhbvpm+self.nroutpm]
```

Referencia:

- `hydroDL/model/rnn.py:1836`

Eso significa:

- Los parametros de routing siguen saliendo del ultimo timestep
- O sea, el routing sigue siendo estatico dentro de esta implementacion

Entonces el cambio dinamico fuerte esta en:

- `BETA`
- `BETAET`

No en el routing.


## 13. Ejemplo mini con numeros inventados

Supongamos:

- `Nstep = 3`
- `tdlst = [1, 13]`
- `staind = -1`
- `dydrop = 0.0`

La LSTM produce, ya escalado:

```text
Dia 1: BETA=2.0, FC=300, ..., BETAET=0.8
Dia 2: BETA=2.7, FC=280, ..., BETAET=1.1
Dia 3: BETA=1.9, FC=310, ..., BETAET=0.9
```

### Paso A: construir la copia estatica con `staind=-1`

Toma el ultimo dia y lo repite:

```text
Dia 1: BETA=1.9, FC=310, ..., BETAET=0.9
Dia 2: BETA=1.9, FC=310, ..., BETAET=0.9
Dia 3: BETA=1.9, FC=310, ..., BETAET=0.9
```

### Paso B: reemplazar solo `BETA` y `BETAET`

Resultado final:

```text
Dia 1: BETA=2.0, FC=310, ..., BETAET=0.8
Dia 2: BETA=2.7, FC=310, ..., BETAET=1.1
Dia 3: BETA=1.9, FC=310, ..., BETAET=0.9
```

Fijate bien:

- `FC` quedo estatico
- `BETA` quedo dinamico
- `BETAET` quedo dinamico

Eso es exactamente lo que hace el codigo.


## 14. Entonces, cual es el "delta" real del modelo

El salto conceptual no es:

```text
HBV -> otro modelo totalmente distinto
```

El salto real es:

```text
HBV con parametros fijos
-> HBV con mezcla de parametros fijos y diarios aprendidos por LSTM
```

El HBV sigue teniendo:

- nieve
- suelo
- aguas subterraneas
- routing

Lo que cambia es la forma de alimentar los parametros.


## 15. Diferencia exacta entre las clases

### `MultiInv_HBVModel`

- Usa `HBVMul`
- Toma solo el ultimo timestep de la LSTM
- Todos los parametros HBV quedan estaticos

Referencia:

- `hydroDL/model/rnn.py:1272-1338`


### `MultiInv_HBVTDModel`

- Usa `HBVMulTD` o `HBVMulTDET`
- Conserva toda la serie temporal de la LSTM
- Convierte solo algunos parametros en dinamicos

Referencia:

- `hydroDL/model/rnn.py:1783-1850`


### `HBVMulTD`

- Version dinamica para 12 parametros HBV
- Permite volver dinamico cualquier indice de `tdlst`

Referencia:

- `hydroDL/model/rnn.py:1341-1557`


### `HBVMulTDET`

- Igual que `HBVMulTD`
- Pero agrega `BETAET`
- Por eso puede usar `tdRep=[1,13]`

Referencia:

- `hydroDL/model/rnn.py:1557-1783`


## 16. La trampa de nombre mas peligrosa

En algunas explicaciones se habla de `gamma_t`.

Pero en ESTE codigo original:

- el parametro dinamico de ET se llama `BETAET`
- la palabra `gamma` aparece tambien en el routing (`UH_gamma`)

No mezcles estas dos cosas:

```text
BETAET dinamico != parametros gamma del unit hydrograph
```


## 17. Resumen brutalmente corto

Si quieres una version ultra corta, es esta:

1. La LSTM siempre produce una serie temporal de salidas.
2. En el modelo estatico, el codigo tira todo menos el ultimo paso.
3. En el modelo dinamico, conserva toda la serie.
4. Luego construye una base estatica usando `staind=-1`.
5. Despues reemplaza solo los parametros listados en `tdlst`.
6. En este experimento, `tdlst=[1,13]`, o sea `BETA` y `BETAET`.
7. Con `dydrop=0.0`, esos parametros son realmente dinamicos siempre.
8. El resto de parametros sigue siendo estatico.


## 18. Si quieres leer el codigo en este orden

Para entenderlo sin marearte, leelo asi:

1. `example/dPLHBV/traindPLHBV.py:64-74`
   Aqui decides si el experimento sera dinamico.

2. `hydroDL/model/rnn.py:1314-1338`
   Aqui ves el caso estatico.

3. `hydroDL/model/rnn.py:1828-1850`
   Aqui ves el caso dinamico.

4. `hydroDL/model/rnn.py:1414-1424`
   Aqui ves como mezclan estatico + dinamico.

5. `hydroDL/model/rnn.py:1470-1499`
   Aqui ves donde actua `BETA` en el HBV.

6. `hydroDL/model/rnn.py:1703-1708`
   Aqui ves donde actua `BETAET`.


## 19. Una sola frase final

El truco del modelo dinamico no es "inventar otro HBV", sino usar la LSTM para fabricar una pelicula temporal de parametros y luego enchufar solo algunos de esos parametros, dia por dia, dentro del mismo HBV.
