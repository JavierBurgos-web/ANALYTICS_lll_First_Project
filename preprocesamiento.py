
#### Cargar paquetes siempre al inicio
import pandas as pd ### para manejo de datos
import sqlite3 as sql #### para bases de datos sql
#import a_funciones as funciones  ###archivo de funciones propias
import sys ## saber ruta de la que carga paquetes

###Ruta directorio qué tiene paquetes
sys.path
sys.path.append('c:\\cod\\LEA3_HR\\data') ## este comanda agrega una ruta

### Cargar tablas de datos desde github ###

employees=("DATA/employee_survey_data.csv")  
general=("DATA/general_data.csv")  
manager=("DATA/manager_survey.csv")
retirement=("DATA/retirement_info.csv")

df_employees=pd.read_csv(employees)
df_general=pd.read_csv(general)
df_manager=pd.read_csv(manager)
df_retirement=pd.read_csv(retirement)

###### Verificar lectura correcta de los datos
df_employees.sort_values(by=['EmployeeID'],ascending=1).head(100)
df_general.sort_values(by=['EmployeeID'],ascending=0).head(5)
df_manager.sort_values(by=['EmployeeID'],ascending=0).head(100)
df_retirement.sort_values(by=['EmployeeID'],ascending=0).head(100)

##### resumen con información tablas faltantes y tipos de variables y hacer correcciones

df_general.info(verbose=True)
df_employees.info()
df_manager.info()
df_retirement.info()

# Eliminar duplicados basados en la columna 'EmployeeID'
df_employees = df_employees.drop_duplicates(subset='EmployeeID', keep='first')
df_general = df_general.drop_duplicates(subset='EmployeeID', keep='first')
df_manager = df_manager.drop_duplicates(subset='EmployeeID', keep='first')
df_retirement = df_retirement.drop_duplicates(subset='EmployeeID', keep='first')

# Eliminar las variables con mismo valor
columnas_a_eliminar = ['EmployeeCount', 'Over18', 'StandardHours', 'InfoDate', 'InfoDate']
df_general = df_general.drop(columns=columnas_a_eliminar, errors='ignore')

# Verifica las columnas actuales
print(df_general.columns)

# Eliminar la columna sin nombre 
df_general = df_general.drop(columns=[col for col in df_general.columns if 'Unnamed' in col])

# Verifica las columnas después de la eliminación
print(df_general.columns)



