### Cargar paquetes
import pandas as pd ### para manejo de datos
import sqlite3 as sql
import Funciones as funciones  ###archivo de funciones profe
from sklearn import linear_model ## para regresión lineal
from sklearn import tree ###para ajustar arboles de decisión
from sklearn.ensemble import RandomForestRegressor ##Ensamble con bagging
from sklearn.ensemble import GradientBoostingRegressor ###Ensamble boosting
from sklearn.model_selection import cross_val_predict, cross_val_score, cross_validate
from sklearn.metrics import mean_squared_error
from sklearn import metrics

import numpy as np
import matplotlib.pyplot as plt ### gráficos
from sklearn.model_selection import RandomizedSearchCV
import joblib  ### para guardar modelos
from sklearn.preprocessing import StandardScaler ## escalar variables 
#import openpyxl

#Traer base de datos
df_final=("DATA/df_final.csv")  
df_final=pd.read_csv(df_final)
### Selección de variables ###
df = df_final.copy()
df.info()

#### imputación para variables categóricas y numéricas
##se separan columnas categóricas o que se quieran tratar así de las numéricas

list_cat=['BusinessTravel', 'Department', 'Gender', 
          'JobRole', 'MaritalStatus', 'EducationField','retirementDate', 'resignationReason']

df1=funciones.imputar_f(df,list_cat)
df1.info()

# convertir las variables categoricas en dummies
list_dummies=['BusinessTravel', 'Department', 'Gender', 
          'JobRole', 'MaritalStatus', 'EducationField','retirementDate', 'resignationReason']


df_dummies=pd.get_dummies(df1,columns=list_dummies)
df_dummies.info()







### Seleccion de variables por metodo Wrapper ###
#Backward selection
df_final_V2_int = df_final_V2.select_dtypes(include = ["number"]) # filtrar solo variables númericas
#df_final_V2_int = df_final_V2_int.drop(['Attrition', 'retirementDate'], axis = 1) # excluir 'Attrition' y 'retirementDate'
y = df_final_V2['Attrition']
df_final_V2_int.head()

# Normalización de variables categoricas ordinales
from sklearn.preprocessing import MinMaxScaler
df_final_V2_norm = df_final_V2_int.copy(deep = True)  # crear una copia del DataFrame
scaler = MinMaxScaler()  # asignar el tipo de normalización
sv = scaler.fit_transform(df_final_V2_norm.iloc[:, :])  # normalizar los datos
df_final_V2_norm.iloc[:, :] = sv  # asignar los nuevos datos
df_final_V2_norm.head()

from sklearn.feature_selection import RFE
# Función recursiva de selección de características
def recursive_feature_selection(X,y,model,k): #model=modelo que me va a servir de estimador para seleccionar las variables
                                              # K = variables que se quiere tener al final
  rfe = RFE(model, n_features_to_select=k, step=1)# step=1 cada cuanto el toma la sicesion de tomar una caracteristica; paso de analisis de caracteristicas
  fit = rfe.fit(X, y)
  c2_var = fit.support_
  print("Num Features: %s" % (fit.n_features_))
  print("Selected Features: %s" % (fit.support_))
  print("Feature Ranking: %s" % (fit.ranking_))

  return c2_var # estimador de las variables seleccioandas

# Establecer Estimador
model = LinearRegression() # algoritmo a travez del cual se van a encontrar las variables

# Obtener columnas seleciconadas - (3 caracteristicas)
df_final_V2_var = recursive_feature_selection(df_final_V2_norm, y, model,7) # x_int =  conjunto caracteristicas numericas
                                                        # y = variable respueta
                                                        # model = modelo que se definio para estimar variables
                                                        # k = numero de variables que se quiere al final

# Nuevo conjunto de datos
df_final_V3 = df_final_V2_int.iloc[:,df_final_V2_var]
df_final_V3.head()

#NOTA
#En la anterior tabla se muestran las 7 variables de 
#mayor importancia que se obtuvieron  despues de 
#utilizar un algoritmo de regresion lineal en el
# modelo empleado para la seleccion de variables en el metodo wrapper



#df_final_V2.fillna(0, inplace=True)
# Separación de caracteristicas y target
#X_class = df_final_V2.drop(['Attrition'], axis=1)
#y_class = df_final_V2['Attrition']

#print(X_class.shape)
#print(y_class.shape)
# Separación de caracteristicas y Attrition (X , y)
#y = df_final_V2.Attrition
#X = df_final_V2.drop(["Attrition"], axis = 1)
# Separación en conjuntos de entrenamiento y validación con 80% de muestras para entrenamiento

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

#Imprimir Tamaño de dataset

#print("Tamaño del conjunto de entrenamiento:",X_train.shape )
#print("Tamaño del conjunto de validación:", X_test.shape )


#Nombre de caracteristicas númericas
#numeric_columns=list(X.select_dtypes('float64').columns)

#Estandarización de variables númericas
#pipeline=ColumnTransformer([("std_num", StandardScaler() , numeric_columns)], remainder='passthrough')
#X_train_std = pipeline.fit_transform(X_train)
#X_test_std = pipeline.transform(X_test)

#Separación de caracteristicas númericas y categóricas
#numeric_columns=list(X.select_dtypes('float64').columns)
#categorical_columns=list(X.select_dtypes('object').columns)
