
import pandas as pd ### para manejo de datos
import sqlite3 as sql #### para bases de datos sql
#import a_funciones as funciones ### archivo de funciones propias
#import matplotlib as mpl ## gráficos
import seaborn as sns
from pandas.plotting import scatter_matrix  ## para matriz de correlaciones
from sklearn import tree ###para ajustar arboles de decisión
from sklearn.tree import export_text ## para exportar reglas del árbol
import sys ## saber ruta de la que carga paquetes
import matplotlib.pyplot as plt

###Ruta directorio qué tiene paquetes
sys.path
sys.path.append('c:\\cod\\LEA3_HR\\data') ## este comanoa agrega una ruta
df_final=("DATA/df_final.csv")  
df_final=pd.read_csv(df_final)
df_final['IDEmpleado'] = df_final['IDEmpleado'].astype(str)
df_final.columns

### explorar variable respuesta ###
fig=df_final.Retirado.hist(bins=50,ec='black') ## no hay atípicos
fig.grid(False)
plt.show()

### explorar variables numéricas  ###
df_final.hist(figsize=(15, 15), bins=20)
plt.suptitle('Distribución de Variables Numéricas', y=1.02)
plt.show()

# Converir la variable 'FechaDeRetiro' a datetime
df_final['FechaDeRetiro'] = pd.to_datetime(df_final['FechaDeRetiro'], errors='coerce')
# Crear grafica de retiros por mes
df_final.set_index('FechaDeRetiro').resample('M').size().plot()
plt.title('Retiros por mes')
plt.xlabel('FechaDeRetiro')
plt.ylabel('Empleados retirados')
plt.show()

### explorar variables categóricas  ###
for column in ['ViajesDeNegocios', 'Departamento', 'CampoDeEducacion', 'Genero', 'RolDeTrabajo', 'EstadoCivil','MotivoDeRenuncia']:
    plt.figure(figsize=(10, 6))
    sns.countplot(x=column, data=df_final)
    plt.title(f'Distribución de {column}')
    plt.show()


