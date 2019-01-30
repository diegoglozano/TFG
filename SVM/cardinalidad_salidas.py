import pandas as pd

path = r'../Data/pacientes_ucic_v3.csv'

df = pd.read_csv(path, sep=';', index_col='Unnamed: 0')

print(df.iloc[:, 220:].columns)
y_col = ['Situación al alta de UCI.Control de fallo cardiaco',
         'Situación al alta de UCI.Precisa ayuda respiratoria',
         'Situación al alta de UCI.Ayuda movilización',
         'Situación al alta de UCI.Estable no precisa cuidados especiales',
         'Situación al alta de UCI.Precisa vigilancia herida quirúrgica',
         'Situación al alta de UCI.Precisa telemetría',
         'Situación al alta de UCI.Ayuda comer',
         'Situación al alta de UCI.Vigilancia por delirio',
         'Situación al alta de UCI.Vigilancia anemia/anticoagulantes',
         'Situación al alta de UCI.Limitado esfuerzo terapéutico',
         'Situación al alta de UCI.Cuidados especiales por vía aérea artificial',
         'Situación al alta de UCI.Otro']

output = df[y_col]

cardinalidad = output.sum(axis=1).mean()  # Da 1.33

print(cardinalidad)
