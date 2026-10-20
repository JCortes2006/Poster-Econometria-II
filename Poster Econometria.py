#%% --------------------------------------------------------
from pathlib import Path

# Módulos de numpy, pandas, matplotlib y scipy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import jarque_bera, probplot

# Módulos de statsmodels
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller, kpss

#%% Ruta relativa -------------------------------------------------------- --------------------------

BASE = Path(__file__).resolve().parent
Ruta = BASE/"IPC Ajustado(ENERO 2018 - SEPTIEMBRE 2026).csv"

Base_datos = pd.read_csv(Ruta, sep =";", decimal = ",")  

# %% Visualizar datos

print(Base_datos.head())
print(Base_datos.tail())

#%% Se le quita la columna IPC Bogota --------------------------------------------------------
Base_datos = Base_datos.drop(columns = ["IPC Bogota"])

print(Base_datos.head())
print(Base_datos.tail())

# %% Se convierte en serie --------------------------------------------------------

Base_datos["Fecha"] = pd.to_datetime(Base_datos["Fecha"])
Serie = Base_datos.set_index("Fecha")["IPC Ajustado 2026"]

#%% Se grafica la serie ---------------------------------------------------------------

plt.plot(Serie)
plt.title("IPC como Proporción del de Septiembre de 2026")
plt.xlabel("Año")
plt.ylabel("Proporción")
plt.show()

#   FAC y FACP ----------------------------------------------------------------------
#%%
plot_acf(Serie, lags=100)
plt.show()

plot_pacf(Serie, lags=20)
plt.show()
##  Se sugiere un AR(1) dado la FAC y la FACP

# %% Test DF ------------------------------------------------------------------------------

test_df = adfuller(Serie)
print("Estadístico ADF:", test_df[0])
print("p-valor:", test_df[1])
print("Rezagos usados:", test_df[2]) 
print("Observaciones:", test_df[3])
print("Valores críticos:")
for nivel, valor in test_df[4].items():
    print(f"{nivel}: {valor}")
    
if test_df[1] < 0.05:
    print("R H0, la serie es estacionaria")   
else:
    print("NR H0, la serie no es estacionaria")
    
# %% Se diferencia la Serie 1 Vez y se grafica --------------------------------------------------------v

Base_diferenciada = Serie.diff().dropna()

plt.plot(Base_diferenciada)
plt.xlabel("Fecha")
plt.ylabel("Diferencia del IPC (en pp)")
plt.title("Serie Diferenciada 1 Vez")

# %% ¿Sera estacionaria? -------------------------------------------------------------------
test_df_dif = adfuller(Base_diferenciada)

print("Estadístico ADF:", test_df_dif[0])
print("p-valor:", test_df_dif[1])
print("Rezagos usados:", test_df_dif[2]) 
print("Observaciones:", test_df_dif[3])
print("Valores críticos:")

for nivel, valor in test_df_dif[4].items():
    print(f"{nivel}: {valor}")
    
if test_df_dif[1] < 0.05:
    print("R H0, la serie es estacionaria")    
else:
    print("NR H0, la serie no es estacionaria")

#   NO resulta ser estacionaria

# %% Se podría diferenciar 2 veces ----------------------------------------------------------

Base_dif2 = Serie.diff().diff().dropna()

# Se grafica

plt.plot(Base_dif2)
plt.xlabel("Fecha")
plt.ylabel("Diferencia de IPC (en pp)")
plt.title("Serie Diferenciada 2 Veces")
plt.show()

# %% ¿Es es estacionaria?

test_df_dif2 = adfuller(Base_dif2)
print("p_valor:", test_df_dif2[1])
if test_df_dif2[1] < 0.05:
    print("R H0, la serie es estacionaria")
else:
    print("NR H0, la serie no es estacionaria")

#   SI resulta ser estacionaria
#   Dado que se diferenció 2 veces, se propondría un ARIMA(1,2,0)
#   Dicho ARIMA se encuentra modelando la diferencia en puntos porcentuales
#   entre el IPC de cada mes, estando todos ajustados al IPC de septiembre de este año.

# %% ESTIMACIÓN ---------------------------------------------------------------------------

arima_propuesto = SARIMAX(Serie, order=(1, 2, 0))
arima_estimado = arima_propuesto.fit(disp=False)
print(arima_estimado.summary()) #   AIC = 38.17, BIC = 43.44

# %% VALIDACIÓN -----------------------------------------------------------------------

Residuos = arima_estimado.resid.dropna()

#   Se grafican los residuos
plt.plot(Residuos)
plt.show()

# %%    Graficamos sus FAC y FACP-------------------------------------------------------

plot_acf(Residuos, lags = 80)
plot_pacf(Residuos, lags = 50)
plt.show()

#   Ocurre que en ambas gráficas, el primer rezago es significativo, por lo que puede
#   considerarse la posibilidad de agregar otro rezago a AR

# %%    Agregamos AR(2) -----------------------------------------------------------

arima_propuesto_2 = SARIMAX(Serie, order=(2, 2, 0))
arima_estimado_2 = arima_propuesto_2.fit(disp=False)
print(arima_estimado_2.summary())   #   AIC = 39.85, BIC = 47.75

# %%    Se grafican sus residuales ----------------------------------------------------

Residuos_2 = arima_estimado_2.resid.dropna()

plt.plot(Residuos_2)
plt.show()

# %% Y sus FAC y FACP --------------------------------------------------------------

plot_acf(Residuos_2, lags = 80)
plot_pacf(Residuos_2, lags = 50)
plt.show()

#   La situación no cambia, de hecho parece empeorar, pues los criterios AIC y BIC
#   son mayores en este modelo

# %%    ARIMA(1,2,1) -------------------------------------

arima_propuesto_3 = SARIMAX(Serie, order=(1,2,1))
arima_estimado_3 = arima_propuesto_3.fit(disp=False)
print(arima_estimado_3.summary())

#   Los criterios AIC y BIC son 26.4 y 34.3 respectivamente, los más bajos hasta
#   el momento. PERO la problemática del rezago permanece.

# %% Evaluar supuestos AUTOCORRELACIÓN -------------------------------------------------------------

Residuos_3 = arima_estimado_3.resid.dropna()

ljung_box = acorr_ljungbox(Residuos_3, lags=[6, 12, 18, 24], return_df=True )

print("Prueba de Ljung Box")
print(ljung_box)

#   Todos los p-valores son mayores a 0.05 en todos lso rezagos
#   por lo tanto NO HAY AUTOCORRELACIÓN ni en el corto ni largo plazo

# %%    HETEROCEDASTICIDAD -------------------------------------------------------------

arch_test = het_arch(Residuos_3, nlags=10) 

print("Prueba ARCH")
print("LM p-value:", arch_test[1])

#   P-valor es mayor a 0.05, por lo que NO HAY HETEROCEDASTICIDAD
