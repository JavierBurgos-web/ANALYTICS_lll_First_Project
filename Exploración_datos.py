
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV, Lasso
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


###Ruta directorio qué tiene paquetes
sys.path
sys.path.append('c:\\cod\\LEA3_HR\\data') ## este comanoa agrega una ruta
df_final=("DATA/df_final.csv")  
df_final=pd.read_csv(df_final)
#df_final['EmployeeID'] = df_final['EmployeeID'].astype(str)
df_final.columns

### explorar variable respuesta ###
fig=df_final.Attrition.hist(bins=50,ec='black') ## no hay atípicos
fig.grid(False)
plt.show()

df_final.shape


#Correción del tipo de cada variable

df_final.info()
df_final.value_counts('NumCompaniesWorked')


df_final['EnvironmentSatisfaction'] = df_final['EnvironmentSatisfaction'].astype('category')
df_final['JobSatisfaction'] = df_final['JobSatisfaction'].astype('category')
df_final['WorkLifeBalance'] = df_final['WorkLifeBalance'].astype('category')
df_final['Education'] = df_final['Education'].astype('category')
df_final['JobInvolvement'] = df_final['JobInvolvement'].astype('category')
df_final['PerformanceRating'] = df_final['PerformanceRating'].astype('category')
df_final['JobLevel'] = df_final['JobLevel'].astype('category')
df_final['resignationReason'] = df_final['resignationReason'].astype('category')


df_final['JobLevel'] = df_final['JobLevel'].astype('category')
df_final['StockOptionLevel'] = df_final['StockOptionLevel'].astype('category')
df_final['TrainingTimesLastYear'] = df_final['TrainingTimesLastYear'].astype('category')
df_final['NumCompaniesWorked'] = df_final['NumCompaniesWorked'].astype('int64')

df_final['retirementDate'] = pd.to_datetime(df_final['retirementDate'], errors='coerce')

df_final['Attrition'] = df_final['Attrition'].astype('category')
df_final['BusinessTravel'] = df_final['BusinessTravel'].astype('category')
df_final['Department'] = df_final['Department'].astype('category')
df_final['EducationField'] = df_final['EducationField'].astype('category')
df_final['Gender'] = df_final['Gender'].astype('category')
df_final['JobRole'] = df_final['JobRole'].astype('category')
df_final['MaritalStatus'] = df_final['MaritalStatus'].astype('category')

df_final.info()


### explorar variables numéricas  ###
df_final.hist(figsize=(15, 15), bins=20)
plt.suptitle('Distribución de Variables Numéricas', y=1.02)
plt.show()

# Crear grafica de retiros por mes
df_final.set_index('retirementDate').resample('M').size().plot()
plt.title('Retiros por mes')
plt.xlabel('FechaDeRetiro')
plt.ylabel('Empleados retirados')
plt.show()

### explorar variables categóricas  ###
for column in ['BusinessTravel', 'Department', 'EducationField', 'Gender', 'JobRole', 'MaritalStatus','resignationReason']:
    plt.figure(figsize=(10, 6))
    sns.countplot(x=column, data=df_final)
    plt.title(f'Distribution of {column}')
    plt.show()


### Variable respuesta VS categoricas  ###
    
fig, axs = plt.subplots(3, 5, figsize=(14, 8))

gráficos = [
    'EnvironmentSatisfaction',
    'JobSatisfaction',
    'WorkLifeBalance',
    'BusinessTravel',
    'Department',
    'Education',
    'EducationField',
    'Gender',
    'JobLevel',
    'JobRole',
    'MaritalStatus',
    'StockOptionLevel',
    'TrainingTimesLastYear',
    'JobInvolvement',
    'PerformanceRating',
]

for i, gráfico in enumerate(gráficos):
    axs[i // 5, i % 5].plot(df_final.groupby(gráfico)['Attrition'].value_counts().unstack())
    axs[i // 5, i % 5].set_title(gráfico)
    # Girar las etiquetas del eje x
    axs[i // 5, i % 5].set_xticklabels(axs[i // 5, i % 5].get_xticklabels(), rotation=90)

plt.subplots_adjust(wspace=0.25, hspace=3.5)
plt.legend(['No', 'Sí'], loc='lower right')
plt.suptitle('Análisis de Attrition')
plt.show()

# Seleccionar solo las columnas categóricas
df_categoricas = df_final.select_dtypes(include=['category'])

# Crear una matriz vacía para almacenar los valores de p-valor
p_values = []

# Correlación de chi-cuadrado y los p-valores
for col1 in df_categoricas.columns:
    row_p_values = []
    for col2 in df_categoricas.columns:
        if col1 == col2:
            row_p_values.append(1.0)  # Poner 1.0 en la diagonal principal
        else:
            contingency_table = pd.crosstab(df_categoricas[col1], df_categoricas[col2])
            _, p, _, _ = chi2_contingency(contingency_table)
            row_p_values.append(p)
    p_values.append(row_p_values)

# Crear un DataFrame de p-valores
p_value_df = pd.DataFrame(p_values, columns=df_categoricas.columns, index=df_categoricas.columns)

plt.figure(figsize=(10, 8))
cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(p_value_df, annot=True, cmap=cmap, fmt=".2f")
plt.title("Matriz de Asociación (p-Valores) entre Variables Categóricas")
plt.show()

### Attrition (Categorica) vs numericas ###

# Crear una figura con tres filas y cinco columnas
fig, axs = plt.subplots(2, 5, figsize=(25, 7))

# Crear los gráficos
for i, gráfico in enumerate(
    [
        'Age',
        'DistanceFromHome',
        'MonthlyIncome',
        'NumCompaniesWorked',
        'PercentSalaryHike',
        'YearsAtCompany',
        'YearsSinceLastPromotion',
        'YearsWithCurrManager',
        'TotalWorkingYears',
    ]
):
    # Crear un gráfico de dispersión para la variable numérica
    axs[i // 5, i % 5].scatter(
        df_final[gráfico], df_final['Attrition'], alpha=0.5, s=10, 
        c=df_final['Attrition'].map({0: 'blue', 1: 'orange'}),
    )
    # Agregar una leyenda
    axs[i // 5, i % 5].set_title(gráfico)
    # Ajustar el diseño de la figura
    plt.subplots_adjust(wspace=0.2, hspace=0.6)

# Mostrar la figura
plt.show()

### Relación entre numericas  ###

df_numericas = df_final.select_dtypes(include=[np.number])
correlation_matrix = df_numericas.corr()
mask = np.triu(correlation_matrix, k=1)

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="RdBu", mask=mask)
plt.title("Matriz de Correlación de Variables Numéricas")
plt.show()

### Selección de variables ###
pd.set_option('display.max_columns', None)
print(df_final)
df_final_V2 = df_final.copy()
# Para cambiar el tipo de dato puede utilizar la función astype de pandas
df_final.BusinessTravel = df_final['BusinessTravel'].astype(str)
df_final.Department = df_final['Department'].astype(str)
df_final.Gender = df_final['Gender'].astype(str)
df_final.JobRole = df_final['JobRole'].astype(str)
df_final.MaritalStatus = df_final['MaritalStatus'].astype(str)


# Convierta las columnas en variables dummy utilizando pd.get_dummies()
df_final = pd.get_dummies(df_final)
# Imprimir primeras 3 filas
df_final.head()