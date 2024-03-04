
import pandas as pd ### para manejo de datos
import sqlite3 as sql #### para bases de datos sql
#import a_funciones as funciones ### archivo de funciones propias
#import matplotlib as mpl ## gráficos
#import matplotlib.pyplot as plt ### gráficos
from pandas.plotting import scatter_matrix  ## para matriz de correlaciones
from sklearn import tree ###para ajustar arboles de decisión
from sklearn.tree import export_text ## para exportar reglas del árbol

###Ruta directorio qué tiene paquetes
sys.path
sys.path.append('c:\\cod\\LEA3_HR\\data') ## este comanoa agrega una ruta

### explorar variable respuesta ###
fig=df_final.FechaDeRetiro.hist(bins=50,ec='black') ## no hay atípicos
fig.grid(False)
plt.show()

boxprops = dict(linestyle='-', color='black')
medianprops = dict(linestyle='-',  color='black')
fig=df.boxplot("perf_2023",patch_artist=True,
                boxprops=boxprops,
                medianprops=medianprops,
                whiskerprops=dict(color='black'),
                showmeans=True)
fig.grid(False)
plt.show()