# Poster Econometria II
<<<<<<< Updated upstream
Base de datos y código en Python para la elaboración del poster (SERIE DE TIEMPO IPC PARA LA CIUDAD DE BOGOTÁ DESDE ENERO 2018 HASTA SEPTIEMBRE 2026)
=======
Base de datos y código en Python para la elaboración del Póster 1 de la materia Econometria II en el semestre 2026-2.
Los datos usados consisten en una SERIE DE TIEMPO IPC PARA LA CIUDAD DE BOGOTÁ DESDE ENERO 2018 HASTA SEPTIEMBRE 2026 producida por el Banco de la República, que se puede encontrar en https://uba.banrep.gov.co/htmlcommons/SeriesHistoricas/precios-inflacion.html , seleccionando Por Ciudad y posteriormente Índice.

El documento tipo csv es aquel con la que se trabaja dentro de Python, aquel .xlsx es donde se encuentra la base de datos original, junto a los Metadatos formulados por el Banrep. La base de datos modificada y con la que se trabajó corresponde a la hoja Serie Ajustada.
La base de datos original recolecta el IPC desde 1979 para todas las ciudades de Colombia, pero para este trabajo se optó por reducir la muestra al IPC solo para la ciudad de Bogotá y desde el año 2018 hasta el mes de septiembre de 2026.

Se creó un enviroment en Anaconda con las librerias: Numpy, Pandas, Matplotlib y Statsmodels

El modelo que mejor se ajustó a la serie de datos consistió en un SARIMA(1,1,1)(1,0,0,12); la diferenciación d=1 fue necesaria para estandarizar la serie y que fuese acepata por los test de Duckey Fuller y KPSS (realizados con constante)
Tuvo que agregarse un componente estacionario debido a que esta serie es mensual, por lo que se evidenciaba una relación entre la observación en un mes con la observación del mismo mes pero en el año anterior.
Finalmente la predicción resultó acorde a las observaciones, pues, como se trabajó con un modelo base 100, era de esperarse que se predijeran observaciones cercanas a este valor (la sustentación de esto se realizará en el póster y durante su exposición).
>>>>>>> Stashed changes
