# Importamos paquetes
import pandas as pd


# Variables globales
DF_PATH = r'../Data/pacientes_ucic_v3.csv'
OUTPUT_PATH = r'./salidas.txt'

# Leemos .csv y nos quedamos con las columnas de salida
df = pd.read_csv(DF_PATH, sep=';', index_col='Unnamed: 0')
y_cols = [col for col in df.columns if 'Situación al alta' in col]
df = df[y_cols]


# Definimos una función para retornar las salidas positivas
def write_output(df, index):
    situacion = [col for col in df.columns if df.loc[index, col] == 1]
    return situacion


# Escribimos los resultados en un fichero
with open(OUTPUT_PATH, 'w') as text:
    for i in df.index:
        text.write(str(i) + ' ' + str(write_output(df, i)) + '\n')

