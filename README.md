# Poster Econometria II
Base de datos y código en Python para la elaboración del poster (SERIE DE TIEMPO IPC PARA LA CIUDAD DE BOGOTÁ DESDE ENERO 2018 HASTA SEPTIEMBRE 2026)
La base de datos de Excel.csv es aquella con la que se trabaja dentro de Python, aquella .xlsx es donde se encuentra la base de datos original y modificada en la hoja IPC Ajustado.
EL modelo final que por los cirterios AIC y BIC es el mejor es un ARIMA(2,2,0)
Sin embargo hay un leve problema de rezagos dentro de la FAC y FACP de este modelo propuetso, a pesar de que tanto los supuestos de homoscedasticidad y no autocorrelación se cumplen

PARA ANACONDA ENVIRONMENT:
Usar Numpy, Pandas, Matplotlib, Statsmodels, Scipy

Uso de branches:
Pueden bien sea desde la aplicación o desde la web crear una branch, a la cual subirán la actualización del documento, para evitar alterra el documento principal