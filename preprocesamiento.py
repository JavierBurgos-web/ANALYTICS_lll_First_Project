
import subprocess
#subprocess.run(['pip', 'install', 'seaborn'])
#subprocess.run(['pip', 'install', 'matplotlib'])

#### Cargar paquetes siempre al inicio
import pandas as pd ### para manejo de datos
import sqlite3 as sql #### para bases de datos sql
#import a_funciones as funciones  ###archivo de funciones propias
import sys ## saber ruta de la que carga paquetes
import seaborn as sns
import matplotlib.pyplot as plt

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

# Eliminar la columna sin nombre 
df_employees = df_employees.drop(columns=['Unnamed: 0'])
df_general = df_general.drop(columns=['Unnamed: 0'])
df_manager = df_manager.drop(columns=['Unnamed: 0'])
df_retirement = df_retirement.drop(columns=['Unnamed: 0.1', 'Unnamed: 0'])

# Verifica las columnas actuales
print(df_employees.columns)
print(df_general.columns)
print(df_manager.columns)
print(df_retirement.columns)

# Eliminar las variables con mismo valor
df_employees = df_employees.drop(columns=['DateSurvey'])
df_general = df_general.drop(columns=['EmployeeCount','Over18','StandardHours','InfoDate'])
df_manager = df_manager.drop(columns=['SurveyDate'])
df_retirement = df_retirement.drop(columns=['Attrition','retirementType'])

# Verifica las columnas actuales
print(df_employees.columns)
print(df_general.columns)
print(df_manager.columns)
print(df_retirement.columns)

# Reemplazar valores nulos por 0
df_general['NumCompaniesWorked'].fillna(0, inplace=True)
df_general['TotalWorkingYears'].fillna(0, inplace=True)
df_employees['EnvironmentSatisfaction'].fillna(0, inplace=True)
df_employees['JobSatisfaction'].fillna(0, inplace=True)
df_employees['WorkLifeBalance'].fillna(0, inplace=True)

# Eliminar filas con valores nulos o en blanco en la columna 'resignationReason'
df_retirement = df_retirement.dropna(subset=['resignationReason'], inplace=False)

# Verificar el resultado
df_general.info(verbose=True)
df_employees.info()
df_manager.info()
df_retirement.info()

# Combinar los primeros tres DataFrames
df_combined = pd.merge(df_general, df_employees, on='EmployeeID', how='inner')
df_combined = pd.merge(df_combined, df_manager, on='EmployeeID', how='inner')

# Combinar con el DataFrame de retiro
df_final = pd.merge(df_combined, df_retirement, on='EmployeeID', how='left')

# Diccionario de mapeo para traducir los nombres
nombres_traducidos = {
    'Age': 'Edad',
    'BusinessTravel': 'ViajesDeNegocios',
    'Department': 'Departamento',
    'DistanceFromHome': 'DistanciaDesdeCasa',
    'Education': 'Educacion',
    'EducationField': 'CampoDeEducacion',
    'EmployeeID': 'IDEmpleado',
    'Gender': 'Genero',
    'JobLevel': 'NivelDeTrabajo',
    'JobRole': 'RolDeTrabajo',
    'MaritalStatus': 'EstadoCivil',
    'MonthlyIncome': 'IngresoMensual',
    'NumCompaniesWorked': 'NumEmpresasTrabajadas',
    'PercentSalaryHike': 'AumentoSalarioPorcentual',
    'StockOptionLevel': 'NivelOpcionesAcciones',
    'TotalWorkingYears': 'AñosTotalesTrabajados',
    'TrainingTimesLastYear': 'VecesEntrenadoUltimoAnio',
    'YearsAtCompany': 'AñosEnLaEmpresa',
    'YearsSinceLastPromotion': 'AñosDesdeUltimaPromocion',
    'YearsWithCurrManager': 'AñosConActualGerente',
    'EnvironmentSatisfaction': 'SatisfaccionAmbiental',
    'JobSatisfaction': 'SatisfaccionLaboral',
    'WorkLifeBalance': 'EquilibrioVidaLaboral',
    'JobInvolvement': 'InvolucramientoLaboral',
    'PerformanceRating': 'CalificacionDeDesempeño',
    'retirementDate': 'FechaDeRetiro',
    'retirementType': 'TipoDeRetiro',
    'resignationReason': 'MotivoDeRenuncia'
}

# Aplicar el cambio de nombres al DataFrame
df_final.rename(columns=nombres_traducidos, inplace=True)

df_final.info()
df_final.describe

df_final.to_csv('DATA/df_final.csv', index=False)


## Poner en el archivo exploración datos: 
df_final.hist(figsize=(15, 15), bins=20)
plt.suptitle('Distribución de Variables Numéricas', y=1.02)
plt.show()

for column in ['ViajesDeNegocios', 'Departamento', 'CampoDeEducacion', 'Genero', 'RolDeTrabajo', 'EstadoCivil']:
    plt.figure(figsize=(10, 6))
    sns.countplot(x=column, data=df_final)
    plt.title(f'Distribución de {column}')
    plt.show()

# Selecciona solo las variables numéricas para la matriz de correlación





# Union de Bases de datos
#df_merged = df_employees.merge(df_general, on='EmployeeID', how='outer')
#df_merged = df_merged.merge(df_manager, on='EmployeeID', how='outer')
#df_merged = df_merged.merge(df_retirement, on='EmployeeID', how='outer')
#df_merged

# Exportacion del dataframe df_merge en el repositorio DATA

#df_merged.to_csv('DATA/df_merged.csv', index=False)

