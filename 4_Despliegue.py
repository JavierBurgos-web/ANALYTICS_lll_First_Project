import Funciones as funciones  ###archivo de funciones propias
import pandas as pd ### para manejo de datos
import sqlite3 as sql
import joblib
import openpyxl ## para exportar a excel
import numpy as np


###### el despliegue consiste en dejar todo el código listo para una ejecucion automática en el periodo definido:
###### en este caso se ejecutara el proceso de entrenamiento y prediccion anualmente.
# Cargar datos desde el archivo CSV
df_final = "DATA/df_final.csv"
df = pd.read_csv(df_final)


df_t= funciones.preparar_datos(df_final)



# Cargar modelo entrenado
m_lreg = joblib.load("salidas\\m_lreg.pkl")

# Realizar predicciones
predicciones = m_lreg.predict(df)
pd_pred = pd.DataFrame(predicciones, columns=['pred_perf_2024'])

# Crear DataFrame con predicciones
perf_pred = pd.concat([df['EmployeeID'], df_t, pd_pred], axis=1)

# Guardar predicciones en archivos
perf_pred[['EmployeeID', 'pred_perf_2024']].to_excel("salidas\\prediccion.xlsx")
coeficientes = pd.DataFrame(np.append(m_lreg.intercept_, m_lreg.coef_), columns=['coeficientes'])
coeficientes.to_excel("salidas\\coeficientes.xlsx")

# Ver las 10 predicciones más bajas
emp_pred_bajo = perf_pred.sort_values(by=["pred_perf_2024"], ascending=True).head(10)
emp_pred_bajo.set_index('EmpID2', inplace=True)
pred = emp_pred_bajo.T
print(pred)