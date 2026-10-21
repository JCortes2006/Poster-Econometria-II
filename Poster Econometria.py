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

#%% Ruta relativa -------------------------------------------------------- -------------------------
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

#   %%  Descripción de la Serie --------------------------------------------------------
print(Serie.describe())

# El reconteo muestra 105 observaciones (CONCUERDA CON EXCEL)

#%% Se grafica la serie ---------------------------------------------------------------
plt.plot(Serie)
plt.title("IPC Mensual como Proporción del IPC de Septiembre de 2026")
plt.xlabel("Año")
plt.ylabel("Proporción (%)")
plt.show()

# Va hasta SEPTIEMBRE 2026, por eso aparece 2027 en el eje X

#   FAC y FACP ----------------------------------------------------------------------
#%%
plot_acf(Serie, lags=100)
plt.show()

plot_pacf(Serie, lags=20)
plt.show()
# Podriamos sugerir AR(1) debido a que en la FACP hay 1 rezago significativo

# %% HAREMOS LOS TEST ADF ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# SIN EMBARGO: puede ser sin constante, con constante o con constante y tendencia lineal
# H0: La raíz es unitaria, por ende NO es estacionario
# Ha: La raíz no es unitaria, SI es estacionario

# NO se usará sin constante, pq gráficamente se evidencia una tendencia además de que
# la media es obviamente distinta de 0 

##  COMENZANDO CON ADF CON CONSTANTE    ##

test_df_uno = adfuller(Serie, regression ="c")
print("Estadístico ADF:", test_df_uno[0])
print("p-valor:", test_df_uno[1])
    
if test_df_uno[1] < 0.05:
    print("R H0, la serie SI es estacionaria")   
else:
    print("NR H0, la serie NO es estacionaria")
#%% CONTINUANDO CON ADF CON CONSTANTE Y TENDENCIA LINEAL    ##

test_df_dos = adfuller(Serie, regression ="c")
print("Estadístico ADF:", test_df_dos[0])
print("p-valor:", test_df_dos[1])
    
if test_df_dos[1] < 0.05:
    print("R H0, la serie SI es estacionaria")   
else:
    print("NR H0, la serie NO es estacionaria")
    

#%% PARA COMPLEMENTAR HACEMOS EL TEST KPSS  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# USANDO CON CONSTANTE CON CONSTANTE Y TENDENCIA LINEAL

# H0: La serie SI es estacionaria
# Ha: La serie NO es estacionaria
#%% CON CONSTANTE   ##

test_kpss_uno = kpss(Serie, regression = "c")
print("Estadístico KPSS:",test_kpss_uno[0])
print("P-valor KPSS:",test_kpss_uno[1])

if test_kpss_uno[1] < 0.05:
    print("R H0, la serie NO es estacionaria")
else:
    print("NR H0, la serie SI es estacionaria")
    
# %% CON CONSTANTE Y TENDENCIA LINEAL   ##

test_kpss_dos = kpss(Serie, regression = "ct")
print("Estadístico KPSS:",test_kpss_dos[0])
print("P-valor KPSS:",test_kpss_dos[1])

if test_kpss_dos[1] < 0.05:
    print("R H0, la serie NO es estacionaria")
else:
    print("NR H0, la serie SI es estacionaria")
    
# %% AMBOS TEST CONCUERDAN QUE LA SERIE NO ES ESTACIONARIA

#%% DIFERENCIAMOS LA SERIE 1 VEZ ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Serie_diff_uno = Serie.diff().dropna()

plt.plot(Serie_diff_uno)
plt.title("Serie Diferenciada 1 Vez")
plt.show()

# %% HACEMOS EL TEST ADF    ##

test_adf_diff = adfuller(Serie_diff_uno, regression = "c")
if test_adf_diff[0] <0.05:
    print("R H0, la seire NO tiene raíz unitaria")
else:
    print("NR H0, la serie SI tiene raíz unitaria")

# %% HACEMOS EL TEST KPSS   ##
test_kpss_diff = kpss(Serie_diff_uno, regression = "c")
if test_kpss_uno[0] <0.05:
    print("R H0, la serie NO es estacionaria")
else:
    print("NR H0, la serie SI es estacionaria")

# %% POR ENDE, SE CONCLUYE QUE d = 1, AHORA IDENTIFICAMOS EL MODELO

FAC_dif = plot_acf(Serie_diff_uno, lags=20)
plt.show()

FACP_diff = plot_pacf(Serie_diff_uno, lags=20)
plt.show()

# Con estas gráficas creemos que se puede tratar:
# AR(1)     AR(2)   MA(1)   MA(2)   MA (1,1)    ARMA(1,2)   MA(2,1)
# Todas lógicamente con d=1

# %% PROBAMOS CON AR(1) ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

modelo_ar1 = SARIMAX(Serie, order=(1,1,0))
ar1_estimado = modelo_ar1.fit(disp=False)

print(ar1_estimado.summary())

#   AIC: 34.125     BIC = 39.414
# %% Graficamos los errores:

Residual_ar1 = ar1_estimado.resid.dropna()

plt.title("Residuos del AR(1)")
plt.plot(Residual_ar1)
plt.show()

FAC_residual1 = plot_acf(Residual_ar1, lags = 30)
plt.title("FAC del AR(1)")
plt.xlabel("Rezagos")
plt.show()

FACP_residual1 = plot_pacf(Residual_ar1, lags = 30)
plt.title("FACP del AR(1)")
plt.xlabel("Rezagos")
plt.show()

# El primer residuo es gigantesco, esto ocurre porque el sistema trata de adivinar de
# donde partió el modelo, entonces lo coloca artifilmente, por eso es que
# como diferenciamos una vez, perdemos la primera observación

#%% Le quitamos la primera observacion para que ese residuo tan grande no influya

Residuos_ar1 = Residual_ar1.iloc[1:]
plt.title("Residuos del AR(1)")
plt.plot(Residuos_ar1)
plt.show()

FAC_residuosar1 = plot_acf(Residuos_ar1, lags = 30)
plt.title("FAC del AR(1)")
plt.xlabel("Rezagos")
plt.show()

FACP_residuosar1 = plot_pacf(Residuos_ar1, lags = 30)
plt.title("FACP del AR(1)")
plt.xlabel("Rezagos")
plt.show()
#%% VALIDAMOS SUPUESTOS: ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

##  AUTOCORRELACIÓN   ##
LB_ar1 = acorr_ljungbox(Residuos_ar1, lags=[6, 12, 18, 24, 30],
    return_df=True)

print(LB_ar1)
if(LB_ar1["lb_pvalue"] < 0.05).any():
    print("R H0, SI hay autocorrelación")
else:
    print("NR H0, NO hay autocorrelación")

#%%  NORMALIDAD  ##

Nrm_ar1 = jarque_bera(Residuos_ar1)
print("P-value:", Nrm_ar1[1])

if Nrm_ar1[1]<0.05:
    print("R H0, los residuos NO son normales")
else:
    print("NR H0, los residuos SI son normales")
    
# %% HOMOSCEDASTICIDAD:

arch_ar1 = het_arch(Residuos_ar1, nlags=12)
print("P-value:", arch_ar1[1])

if arch_ar1[1] < 0.05:
    print("R H0, se presenta HETEROSCEDASTICIDAD")
else:
    print("NR H0, se presentea HOMOSCEDASTICIDAD")


# HAY autocorrelación, NO normalidad, y HOMOSCEDASTICIDAD

# %% PROBAMOS CON AR(2) ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

modelo_ar2 = SARIMAX(Serie, order=(2,1,0))
ar2_estimado = modelo_ar2.fit(disp=False)

print(ar2_estimado.summary())

#   AIC = 34.791    BIC = 42.724

# PERO, si miramos en .summary, ocurre que el SEGUNDO REZAGO NO ES SIGNIFICATIVO
# no solo eso, sino que los criterios empeoran ligeramente, por lo que no hay
# necesidad de evaluar los supuestos de los residuos, sino descartarlo directamente 


# %% PROBAMOS CON MA(1) ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

modelo_ma1 = SARIMAX(Serie, order=(0,1,1))
ma1_estimado = modelo_ma1.fit(disp=False)

print(ma1_estimado.summary())

#   AIC = 87.541        BIC = 92.829

#%% GRAFICAMOS SUS RESIDUOS

# Sucederá lo mismo que con AR(1), así que quitamos el primer residuo
Residual_ma1 = ma1_estimado.resid.dropna()
Residuos_ma1 = Residual_ma1.iloc[1:]

plt.plot(Residuos_ma1)
plt.title("Residuos del Modelo MA()1")
plt.show()

FAC_residuosma1 = plot_acf(Residuos_ma1, lags = 15)
plt.title("FAC del MA(1)")
plt.xlabel("Rezagos")
plt.show()

FACP_residuosma1 = plot_pacf(Residuos_ma1, lags = 15)
plt.title("FACP del MA(1)")
plt.xlabel("Rezagos")
plt.show()
#%% VALIDAMOS SUPUESTOS: ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

##  AUTOCORRELACIÓN   ##
LB_ma1 = acorr_ljungbox(Residuos_ma1, lags=[6, 12, 18, 24],
    return_df=True)

print(LB_ma1)
if(LB_ma1["lb_pvalue"] < 0.05).any():
    print("R H0, SI hay autocorrelación")
else:
    print("NR H0, NO hay autocorrelación")

#%%  NORMALIDAD  ##

Nrm_ma1 = jarque_bera(Residuos_ma1)
print("P-value:", Nrm_ma1[1])

if Nrm_ma1[1]<0.05:
    print("R H0, los residuos NO son normales")
else:
    print("NR H0, los residuos SI son normales")
    
# %% HOMOSCEDASTICIDAD:

arch_ma1 = het_arch(Residuos_ma1, nlags=12)
print("P-value:", arch_ma1[1])

if arch_ma1[1] < 0.05:
    print("R H0, se presenta HETEROSCEDASTICIDAD")
else:
    print("NR H0, se presentea HOMOSCEDASTICIDAD")


# HAY autocorrelación, NO normalidad, y HETEROSCEDASTICIDAD
# NO CUMPLE NINGÚN SUPUESTO

# %% PROBAMOS CON MA(2) ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Como la FAC de AR(1) tiene significancia en el rezago 2, implica que 
# deberíamos de agregar otro q a MA()

modelo_ma2 = SARIMAX(Serie, order=(0,1,2))
ma2_estimado = modelo_ma2.fit(disp=False)

print(ma2_estimado.summary())

#   AIC = 65.360        BIC = 73.293

#%% GRAFICAMOS SUS RESIDUOS 
# Quitamos la primera observación

Residual_ma2 = ma2_estimado.resid.dropna()
Residuos_ma2 = Residual_ma2.iloc[1:]

plt.title("Residuos del MA(2)")
plt.plot(Residuos_ma2)
plt.show()

FAC_residuosma2 = plot_acf(Residuos_ma2, lags = 15)
plt.title("FAC del MA(2)")
plt.xlabel("Rezagos")
plt.show()

FACP_residuosma2 = plot_pacf(Residuos_ma2, lags = 15)
plt.title("FACP del MA(2)")
plt.xlabel("Rezagos")
plt.show()
#%% VALIDAMOS SUPUESTOS: ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

##  AUTOCORRELACIÓN   ##
LB_ma2 = acorr_ljungbox(Residuos_ma2, lags=[6, 12, 18, 24],
    return_df=True)

print(LB_ma2)
if(LB_ma2["lb_pvalue"] < 0.05).any():
    print("R H0, SI hay autocorrelación")
else:
    print("NR H0, NO hay autocorrelación")

#%%  NORMALIDAD  ##

Nrm_ma2 = jarque_bera(Residuos_ma2)
print("P-value:", Nrm_ma2[1])

if Nrm_ma2[1]<0.05:
    print("R H0, los residuos NO son normales")
else:
    print("NR H0, los residuos SI son normales")
    
# %% HOMOSCEDASTICIDAD:

arch_ma2 = het_arch(Residuos_ma2, nlags=12)
print("P-value:", arch_ma2[1])

if arch_ma2[1] < 0.05:
    print("R H0, se presenta HETEROSCEDASTICIDAD")
else:
    print("NR H0, se presentea HOMOSCEDASTICIDAD")


# HAY autocorrelación, NORMALIDAD, y HETEROSCEDASTICIDAD
# (ahora solo cumple normalidad)

# %% PROBAMOS CON ARMA(1,1) ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

modelo_arma11 = SARIMAX(Serie, order=(1,1,1))
arma11_estimado = modelo_arma11.fit(disp=False)

print(arma11_estimado.summary())

#   AIC = 34.786       BIC = 42.719

# PASA LO MISMO, el coeficiente MA tiene un IC que incluye a 0, y sumado a que los criterios
# AIC y BIC empeoran ligeramente a comparación de AR(1) se descarta este modelo sin nenecsidad
# de evaluar residuos

#%% ¿Qué tenemos?

# De todos los posibles modelos propuestos, el mejor es el AR(1), pues si bien MA(1) y MA(2)
# tienen rezagos significativos, debido a los supuestos son los PEORES modelos. Mientras que
# AR(1) seria el "mejor" propuesto en el sentido de que incumple solo la no autocorrelación.

# En otras palabras, dado que en ARMA(1,1) q=1 no es significativo, y p >= 2 tampoco lo
# es, y debido a que los criterios empeoran al pasar de de AR(1) a ARMA(1,1); o al
# pasar de AR(1) a AR(2), se descartA: PRIMERO, ARMA(1,1) o q>=2; SEGUNDO AR p>=2.
# Entonces nos quedamos con AR(1), sin embargo debemos solucionar el problema de no autocorrelación

#%% VEAMOS CÓMO SOLUCIONAR LB

# FAC y FACP de los residuos
FAC_residuosar1 = plot_acf(Residuos_ar1, lags=15)
plt.title("FAC de los Residuos del AR(1) Propuesto")
plt.xlabel("Rezago")
plt.show()

FACP_residuosar1 = plot_pacf(Residuos_ar1, lags=15)
plt.title("FACP de los Residuos del AR(1) Propuesto")
plt.xlabel("Rezago")
plt.show()

# Dada estas gráficas, es posible que hay estacionalidad cada 12 periodos
# es decir, como la serie es mensual, el IPC de enero se relaciona con el IPC
# de enero del año pasado, lo cual tiene sentido

# No solo eso, sino que como las FAC y FACP de AR(1) tienen un rezago significativo
# en 12, tenemos que meter un P = 1 a la parte estacional
# %% LE AGREGAMOS COMPONENTE ESTACIONAL CADA 12 MESES ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

modelo_sar1 = SARIMAX( Serie, order=(1,1,0), seasonal_order=(1,0,0,12))
sar1_estimado = modelo_sar1.fit(disp=False)

print(sar1_estimado.summary())

#   AIC= -0.584     BIC = 7.349

#%% GRAFICAMOS RESIDUOS 

Residual_sar1 = sar1_estimado.resid.dropna()
Residuos_sar1 = Residual_sar1.iloc[1:]

plt.title("Residuos del SARIMA(1,1,0)(1,0,0,12)")
plt.plot(Residuos_sar1)
plt.show()

FAC_residuos_sar1 = plot_acf(Residuos_sar1, lags=36)
plt.title("FAC de los Residuos del SARIMA(1,1,0)(1,0,0,12)")
plt.xlabel("Rezago")
plt.show() 

FACP_residuos_sar1 = plot_pacf(Residuos_sar1, lags=36)
plt.title("FACP de los Residuos del SARIMA(1,1,0)(1,0,0,12)")
plt.xlabel("Rezago")
plt.show()
# %% VALIDACIÓN DE SUPUESTOS 

#   NO AUTOCORRELACIÓN
LB_sar1 = acorr_ljungbox(Residuos_sar1, lags=[6, 12, 18, 24],
    return_df=True)

print(LB_sar1)

if (LB_sar1["lb_pvalue"] < 0.05).any():
    print("R H0, SI hay autocorrelación")
else:
    print("NR H0, NO hay autocorrelación.")

# %% NORMALIDAD
Nrm_sar1 = jarque_bera(Residuos_sar1)
print("P-valor:",Nrm_sar1[1])

if Nrm_sar1[1] < 0.05:
    print("R H0, los residuos NO son normales")
else:
    print("NR H0, los residuos SON normales")

# %% HOMOSCEDASTICIDAD
arch_sar1 = het_arch(Residuos_sar1, nlags = 12)
print("P-valor:",arch_sar1[1])

if arch_sar1[1] <0.05:
    print("R H0, hay HETEROSCEDASTICIDAD")
else:
    print("NR H0, hay HOMOSCEDASTICIDAD")

# HAY aucotorrelación, errores NORMALES, hay HOMOSCEDASTICIDAD

# %% Como todavia hay autocorrelación está quedando algo que no capta el modelo por completo
# NO puede ser AR(2) ya que anteriormente vimos que el 2do rezago es insignificativo

# Entonces vamos a meter MA(1), que si bien anteriormente dio que era insignificativo
# tmb presentó rezago en el periodo 12 en sus FAC y FACP, por lo que no solo agregamos
# MA(1) sino que metemos Q=1 a la parte estacional

modelo_sar11 = SARIMAX( Serie, order=(1,1,1), seasonal_order=(1,0,1,12), trend = "t")
sar11_estimado = modelo_sar11.fit(disp=False)

print(sar11_estimado.summary())
#   AIC = -17.642       BIC = -1.776
# Algo bueno es que AHORA TODOS LOS COEFICIENTES SON SIGNIFICATIVOS

# %%   GRAFICAR RESIDUOS

Residual_sar11 = sar11_estimado.resid.dropna()
Residuos_sar11 = Residual_sar11.iloc[1:]

plt.plot(Residuos_sar11)
plt.title("Residuos del Modelo SARIMA(1,1,1)(1,0,1,10)")
plt.show()

FAC_residuos_sar11 = plot_acf(Residuos_sar11, lags = 15)
plt.title("FAC de los Residuos del Modelo SARIMA(1,1,1)(1,0,1,12)")
plt.xlabel("Rezagos")
plt.show()

FACP_residuos_sar11 = plot_pacf(Residuos_sar11, lags = 15)
plt.title("FACP de los Residuos del Modelo SARIMA(1,1,1)(1,0,1,12)")
plt.xlabel("Rezagos")
plt.show()

# %% VERIFICAR SUPUESTOS

#   NO AUTOCORRELACIÓN:
LB_sar11 = acorr_ljungbox(Residuos_sar11, lags=[6, 12, 18, 24],
    return_df=True)

print(LB_sar11)

if (LB_sar11["lb_pvalue"] < 0.05).any():
    print("R H0, SI hay autocorrelación")
else:
    print("NR H0, NO hay autocorrelación.")

# %% NORMALIDAD
Nrm_sar11 = jarque_bera(Residuos_sar11)
print("P-valor:",Nrm_sar11[1])

if Nrm_sar11[1] < 0.05:
    print("R H0, los residuos NO son normales")
else:
    print("NR H0, los residuos SON normales")

# %% HOMOSCEDASTICIDAD
arch_sar11 = het_arch(Residuos_sar11, nlags = 12)
print("P-valor:",arch_sar11[1])

if arch_sar11[1] <0.05:
    print("R H0, hay HETEROSCEDASTICIDAD")
else:
    print("NR H0, hay HOMOSCEDASTICIDAD")

# NO HAY AUTOCORRELACIÓN, HAY NORMALIDAD, HAY HOMOSCEDASTICIDAD

#   Finalmente, nos quedamos con un modelo SARIMA(1,1,1)(1,0,1,12)

# %% PRONOSTICAMOS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Prediccion_sarima11 = sar11_estimado.get_forecast(steps=15)
# Por curiosidad coloqué steps = 24 para ver dos años adelante
Valores_predichos = Prediccion_sarima11.predicted_mean
IC_sarima11 = Prediccion_sarima11.conf_int()

tabla_pronostico = pd.DataFrame({
    "pronostico": Valores_predichos,
    "limite_inferior": IC_sarima11.iloc[:, 0],
    "limite_superior": IC_sarima11.iloc[:, 1]
})
print(tabla_pronostico)

# GRÁFICA PRONÓSTICO
plt.plot(Serie, label="Datos históricos")
plt.title("Precio Relativo de la Canasta Básica Familiar en Bogotá (relación a precio de sept 2026)")
plt.xlabel("Fecha")
plt.ylabel("Precio Relativo")
plt.plot(Valores_predichos, color="orange")
plt.fill_between(
    Valores_predichos.index,
    IC_sarima11.iloc[:, 0],
    IC_sarima11.iloc[:, 1],
    color="orange",
    alpha=0.3,  
)

# %%
