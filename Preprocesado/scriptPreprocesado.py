#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 23 23:51:31 2018

@author: diego
"""

import pandas as pd
import numpy as np
from datetime import datetime

before = datetime.now()

def cifraDePlaquetasMasBaja():
    """
    0
    20.000-49.999
    50.000-99.999
    100.000-149.999
    >150.000
    1
    
    data[data['Cifra de plaquetas más baja'] == '20.000-49.999'] = 0
    data[data['Cifra de plaquetas más baja'] == '50.000-99.999'] = 1
    data[data['Cifra de plaquetas más baja'] == '100.000-149.999'] = 2
    data[data['Cifra de plaquetas más baja'] == '>150.000'] = 3
    """
    cat = pd.Categorical(data['Cifra de plaquetas más baja'], categories=['20.000-49.999','50.000-99.999','100.000-149.999','>150.000'])
    data['Cifra de plaquetas más baja'], uniques = pd.factorize(cat, sort = True)
    data['Cifra de plaquetas más baja'] = data['Cifra de plaquetas más baja'].replace(-1, np.nan)
    
def fechaNacimiento():
    # QUITAMOS LA HORA
    data['FECHA_NACIMIENTO'] = data['FECHA_NACIMIENTO'].str.split(" ", 0, True)
    
    # GUARDAMOS EN VARIABLES TEMPORALES DIA MES Y AÑO
    temporal0 = data['FECHA_NACIMIENTO'].str.split("/", 0, True)[0]
    temporal1 = data['FECHA_NACIMIENTO'].str.split("/", 0, True)[1]
    temporal2 = '19' + data['FECHA_NACIMIENTO'].str.split("/", 0, True)[2]
    
    # LAS JUNTAMOS OTRA VEZ
    data['FECHA_NACIMIENTO'] = temporal0 + '/' + temporal1 + '/' + temporal2
    
    # CONVERTTIMOS A DATETIME AMBAS COLUMNAS
    data['FECHA_NACIMIENTO'] = pd.to_datetime(data['FECHA_NACIMIENTO'])
    data['Fecha ingreso hospitalario'] = pd.to_datetime(data['Fecha ingreso hospitalario'])
    
    # CREAMOS LA COLUMNA EDAD Y RESTAMOS LAS FECHAS. LA GUARDAMOS COMO FLOAT Y COMO AÑO 
    data['EDAD'] = 0
    data['EDAD'] = data['Fecha ingreso hospitalario'] - data['FECHA_NACIMIENTO']
    data['EDAD'] = data['EDAD'] / (np.timedelta64(1, 'D') * 365)
    
def indiceDeFragilidad():
    """
    0 Mejor?
    nan
    Muy saludable. Por encima de lo esperado para su edad
    Sano: Paciente asintomático, con actividad normal
    Controlado: Paciente con sintomatología tratada, con activid
    Vulnerable:Independiente, pero con actividad limitada por su
    Fragilidad leve: Necesita ayuda para tareas difíciles
    Fragilidad moderada: Necesita ayuda dentro y fuera de la cas
    Fragilidad grave: Totalmente dependiente para las actividade
    Fragilidad muy grave: Totalmente dependiente, no tolera ni l
    Enfermedad terminal: Esperanza de vida menor de 6 meses
    1 Peor?

            
    data['Índice de Fragilidad'] = data['Índice de Fragilidad'].fillna(value = 0)
    data[data['Índice de Fragilidad'] == 'Muy saludable. Por encima de lo esperado para su edad'] = 1
    data[data['Índice de Fragilidad'] == 'Sano: Paciente asintomático, con actividad normal'] = 2
    data[data['Índice de Fragilidad'] == 'Controlado: Paciente con sintomatología tratada, con activid'] = 3
    data[data['Índice de Fragilidad'] == 'Vulnerable:Independiente, pero con actividad limitada por su'] = 4
    data[data['Índice de Fragilidad'] == 'Fragilidad leve: Necesita ayuda para tareas difíciles'] = 5
    data[data['Índice de Fragilidad'] == 'Fragilidad moderada: Necesita ayuda dentro y fuera de la cas'] = 6
    data[data['Índice de Fragilidad'] == 'Fragilidad grave: Totalmente dependiente para las actividade'] = 7
    data[data['Índice de Fragilidad'] == 'Fragilidad muy grave: Totalmente dependiente, no tolera ni l'] = 8
    data[data['Índice de Fragilidad'] == 'Enfermedad terminal: Esperanza de vida menor de 6 meses'] = 9
    """
    
    data['Índice de Fragilidad'] = data['Índice de Fragilidad'].fillna(value = 'NAN')
    cat = pd.Categorical(data['Índice de Fragilidad'], categories=['NAN', 'Muy saludable. Por encima de lo esperado para su edad','Sano: Paciente asintomático, con actividad normal','Controlado: Paciente con sintomatología tratada, con activid','Vulnerable:Independiente, pero con actividad limitada por su','Fragilidad leve: Necesita ayuda para tareas difíciles','Fragilidad moderada: Necesita ayuda dentro y fuera de la cas','Fragilidad grave: Totalmente dependiente para las actividade','Fragilidad muy grave: Totalmente dependiente, no tolera ni l','Enfermedad terminal: Esperanza de vida menor de 6 meses'])
    data['Índice de Fragilidad'], uniques = pd.factorize(cat, sort = True)   

def ingresoEn(data):
    temporal = pd.get_dummies(data['Ingreso en:'], prefix = 'Ingreso en:', prefix_sep = ' ') #dummy_na = True

    data = pd.concat([data, temporal], axis=1, sort=False)
    return data
    
def motivoDeIngresoEnUCI(data):
    datatemp = pd.get_dummies(data['Motivo de ingreso en UCI'], prefix = 'Motivo de ingreso en UCI:', prefix_sep = ' ')

    data = pd.concat([data, datatemp], axis=1, sort=False)
    return data

def sexo(data):
    temporal = pd.get_dummies(data['SEXO'])

    data = pd.concat([data, temporal], axis=1, sort=False)
    return data

def numConcentradosHematies():
    """
    0
    Menor de 3
    Mayor o igual a 3
    1
    
    data[data['núm. concentrados hematies primeras 48 h'] == 'Menor de 3'] = 0
    data[data['núm. concentrados hematies primeras 48 h'] == 'Mayor o igual a 3'] = 1
    """

    cat = pd.Categorical(data['núm. concentrados hematies primeras 48 h'], categories=['Menor de 3', 'Mayor o igual a 3'])
    data['núm. concentrados hematies primeras 48 h'], uniques = pd.factorize(cat, sort = True)
    data['núm. concentrados hematies primeras 48 h'] = data['núm. concentrados hematies primeras 48 h'].replace(-1, np.nan)
    
def otrosFactores():
    # HACEMOS UN SPLIT PARA OBTENER LAS CATEGORIAS 
    datatemp = data['Otros factores'].str.split(', ')
    
    categorias = []
    
    # RELLENAMOS LOS NAN PARA NO OBTENER ERROR
    datatemp = datatemp.fillna(value = 'a')
    
    # RECORREMOS TODAS LAS FILAS Y LOS ELEMENTOS DE CADA FILA Y LOS AÑADIMOS A LA LISTA CATEGORIAS DE FORMA QUE NO SE REPITAN
    for i in range(len(datatemp)):
        for j in range(len(datatemp.loc[i])):
            if ((datatemp[i][j] in categorias) == False):
                categorias.append(datatemp[i][j])

    # QUITAMOS LA CATEGORÍA 'A' (NAN)
    categorias.remove('a')
    
    # RELLENO TODO CON FALSE
    for i in categorias:
        data['Otros factores. ' + i] = 0
    
    for i in range(len(datatemp)): # RECORRO FILAS
        for j in range(len(datatemp.loc[i])): # RECORRO ELEMENTOS DENTRO DE UNA FILA
            for k in range(len(categorias)): # RECORRO CATEGORIAS
                if(datatemp[i][j] == categorias[k]):
                    data['Otros factores. ' + categorias[k]].loc[i] = 1
                """
                else:
                    data['Otros factores. ' + categorias[k]].loc[i] = False
                """
    # CREO QUE EL PROBLEMA ES QUE SE ESTA SOBREESCRIBIENDO
    # POR EJEMPLO: EN LA PRIMERA ITERACION LA PRIMERA FILA DE APERTURA PLEURAS SE PONE EN TRUE PERO AL COMPARAR CON LA SIGUIENTE
    # CATEGORIA SE SOBREESCRIBE POR UN FALSE DADO QUE EL BUCLE FOR DE LAS FILAS TODAVIA NO CAMBIO. HAY QUE HACER UN BREAK
    
    # LA SOLUCION PUEDE SER RELLENAR TODO CON FALSE Y DESPUES CAMBIAR A TRUE SI SE DA EL CASO. FUE BUENA LA SOLUCION

def picoDeTroponina():
    """
    0
    Menor de 500
    500-999
    1000-3999
    4000 o más
    1
    
    data[data['Pico de troponina'] == 'Menor de 500'] = 0
    data[data['Pico de troponina'] == '500-999'] = 1
    data[data['Pico de troponina'] == '1000-3999'] = 2
    data[data['Pico de troponina'] == '4000 o más'] = 3
    """
    
    cat = pd.Categorical(data['Pico de troponina'], categories=['Menor de 500', '500-999', '1000-3999', '4000 o más'])
    data['Pico de troponina'], uniques = pd.factorize(cat, sort = True)
    data['Pico de troponina'] = data['Pico de troponina'].replace(-1, np.nan)
    

def sangradoEnLasPrimeras24():
    """
    0
    Menor de 500 ml
    500-1000 ml
    Mayor de 1000 ml
    1
    
    data[data['Sangrado en las primeras 24 horas'] == 'Menor de 500 ml'] = 0
    data[data['Sangrado en las primeras 24 horas'] == '500-1000 ml'] = 1
    data[data['Sangrado en las primeras 24 horas'] == 'Mayor de 1000 ml'] = 2
    """
    
    cat = pd.Categorical(data['Sangrado en las primeras 24 horas'], categories=['Menor de 500 ml', '500-1000 ml', 'Mayor de 1000 ml'])
    data['Sangrado en las primeras 24 horas'], uniques = pd.factorize(cat, sort = True)
    data['Sangrado en las primeras 24 horas'] = data['Sangrado en las primeras 24 horas'].replace(-1, np.nan)

def tiempos():
    data['Tiempo de CEC'] = data['Tiempo de CEC'].fillna(value = 0)
    data['Tiempo de clampaje'] = data['Tiempo de clampaje'].fillna(value = 0)
    data['Tiempo de isquemia'] = data['Tiempo de isquemia'].fillna(value = 0)
    data['Tiempo de parada circulatoria'] = data['Tiempo de parada circulatoria'].fillna(value = 0)
    
def tipoDeCirugia():
    """
    0 Peor
    NaN
    Programada
    Emergente
    Urgente
    1 Mejor
    
    data['Tipo de cirugía'] = data['Tipo de cirugía'].fillna(value = 0)
    data[data['Tipo de cirugía'] == 'Programada'] = 1
    data[data['Tipo de cirugía'] == 'Emergente (intervención en menos de 48 horas del diagnóstico'] = 2
    data[data['Tipo de cirugía'] == 'Urgente (se interviene durante el ingreso de una descompensa'] = 3
    """
    data['Tipo de cirugía'] = data['Tipo de cirugía'].fillna(value = 'NAN')
    cat = pd.Categorical(data['Tipo de cirugía'], categories=['NAN', 'Programada', 'Emergente (intervención en menos de 48 horas del diagnóstico', 'Urgente (se interviene durante el ingreso de una descompensa'])
    data['Tipo de cirugía'], uniques = pd.factorize(cat, sort = True)

def cca_nyha_PF():
    data['cca_nyha_PF'] = data['cca_nyha_PF'].fillna(value = 'NAN')
    cat = pd.Categorical(data['cca_nyha_PF'], categories=['NAN', 'I', 'II', 'III', 'IV'])
    data['cca_nyha_PF'], uniques = pd.factorize(cat, sort = True)
    #data['cca_nyha_PF'] = data['cca_nyha_PF'].replace(-1, np.nan)

def diasHastaCirugia():
    # NO HACE FALTA CAMBIAR LA FECHA DE INGRESO HOSPITALARIO A DATETIME PORQUE YA LO HICE PARA LA EDAD
    data['Fecha de cirugía'] = pd.to_datetime(data['Fecha de cirugía'])
    data['Dias hasta cirugia'] = 0
    data['Dias hasta cirugia'] = data['Fecha de cirugía'] - data['Fecha ingreso hospitalario']
    data['Dias hasta cirugia'] = data['Dias hasta cirugia'] / (np.timedelta64(1, 'D'))
    
def ccaFevi():
    data['cca_fevi_PF_varab_pre-ingr'] = data['cca_fevi_PF_varab_pre-ingr'].fillna(value = 'No cuantificada')
    cat = pd.Categorical(data['cca_fevi_PF_varab_pre-ingr'], categories=['No cuantificada', 'Normal', 'Disfunción leve (>40)', 'Disfunción moderada (30-40)', 'Disfunción grave (<30)'])
    data['cca_fevi_PF_varab_pre-ingr'], uniques = pd.factorize(cat, sort = True)
    
data = pd.read_csv('../data/pacientes_ucic.csv', sep = ';')
    
# LLAMAMOS A LAS FUNCIONES 
    
cifraDePlaquetasMasBaja()
fechaNacimiento()
indiceDeFragilidad()
data = ingresoEn(data) # NO SE POR QUE SE COMPORTAN DE DISTINTA FORMA
data = motivoDeIngresoEnUCI(data)
data = sexo(data)
numConcentradosHematies()
otrosFactores()
picoDeTroponina()
sangradoEnLasPrimeras24()
tiempos()
tipoDeCirugia()
cca_nyha_PF()
#diasHastaCirugia()
ccaFevi()

cols = data.columns.tolist()

tiempo = datetime.now() - before
print(tiempo)


#['N', 'EPISODIO', 'FECHA_NACIMIENTO', 'SEXO', 'Talla', 'Peso', 'Factores de riesgo cardiovascular', 'Otros antecedentes', 'cca_nyha_PF', 'Índice de Fragilidad', 'Enfermedad cardiológica', 'Fecha diagnóstico (Se decide tto quirúr)', 'Fecha último ECO/CORONARIOGRAFÍA', 'Resultado último eco', 'cca_fevi_PF_varab_pre-ingr', 'cca_lesiones_coronariog_angioTC_PF_varia', 'Tratamiento anticoagulante', 'Fecha ingreso hospitalario', 'Ingreso en:', 'Situación funcional al ingreso Hospital', 'Tratamiento percutáneo tras el ingreso', 'Tipo de cirugía', 'Intervención realizada', 'Fecha de cirugía', 'Tiempo de CEC', 'Tiempo de clampaje', 'Tiempo de isquemia', 'Tiempo de parada circulatoria', 'Medidas soporte a la salida de quirófano', 'Otros factores', 'Motivo de ingreso en UCI', 'Otro motivo de ingreso en UCI', 'Fecha de ingreso', 'SAPS-3', 'Complicaciones hemodinámicas', 'Complicaciones respiratorias', 'Infecciones', 'Isquemia miocárdica peri-operatoria', 'Pico de troponina', 'Complicaciones hemorrágicas', 'Sangrado en las primeras 24 horas', 'Cifra de plaquetas más baja', 'núm. concentrados hematies primeras 48 h', 'Otras complicaciones', 'Complicaciones neurológicas', 'Complicaciones digestivas', 'Complicaciones renales', 'Reintervencion durante su estancia UCI', 'Terapias en UCI', 'Situación al alta de UCI', 'Fecha de alta de UCI', 'Destino alta', 'NA', 'edad', 'Factores de riesgo cardiovascular.HTA', 'Factores de riesgo cardiovascular.Enfermedad vascular cerebral', 'Factores de riesgo cardiovascular.Dislipemia', 'Factores de riesgo cardiovascular.Diabetes', 'Factores de riesgo cardiovascular.Enfermedad ileo-femoral', 'Factores de riesgo cardiovascular.Tabaco', 'Factores de riesgo cardiovascular.Enfermedad aorta abdominal', 'Enfermedad cardiológica.Coronaria', 'Enfermedad cardiológica.Intervencionismo coronario percutáneo previo', 'Enfermedad cardiológica.IAM', 'Enfermedad cardiológica.Aorta ascendente', 'Enfermedad cardiológica.Cirugía aórtica previa', 'Enfermedad cardiológica.F.A. crónica', 'Enfermedad cardiológica.Enfermedad valvular', 'Enfermedad cardiológica.Miocardiopatia dilatada crónica', 'Enfermedad cardiológica.Marcapasos tricameral/desfibrilador', 'Enfermedad cardiológica.Cirugía mitral previa', 'Enfermedad cardiológica.Marcapasos previo', 'Enfermedad cardiológica.Endocarditis protésica precoz', 'Enfermedad cardiológica.Endocarditis protésica tardía', 'Enfermedad cardiológica.Endocarditis v. nativa', 'Enfermedad cardiológica.Tumores cardiacos', 'Enfermedad cardiológica.Miocardiopatía Hipertrófica', 'Enfermedad cardiológica.Pericardio', 'Enfermedad cardiológica.Cirugía v. aórtica y raíz aórtica previa', 'Enfermedad cardiológica.no confirmada', 'Enfermedad cardiológica.BY-PASS coronario previo', 'Enfermedad cardiológica.Otro', 'Resultado último eco.Alteraciones segmentarias contractilidad', 'Resultado último eco.Patrón pseudonormal diastólico', 'Resultado último eco.V.I. hipertrófico', 'Resultado último eco.Estenosis aórtica', 'Resultado último eco.Insuficiencia mitral', 'Resultado último eco.Dilatación aurículas', 'Resultado último eco.Dilatación VI', 'Resultado último eco.TAPSE bajo', 'Resultado último eco.HTP mayor de 45 MMHG', 'Resultado último eco.Derrame pericárdico', 'Resultado último eco.Insuficiencia tricúspide', 'Resultado último eco.Insuficiencia aórtica', 'Resultado último eco.Disfunción protésica', 'Resultado último eco.Lesiones de endocarditis-imágenes vibrátiles velos valvulare', 'Resultado último eco.Disfunción anuloplastia', 'Resultado último eco.Mixoma auricular', 'Resultado último eco.Miocardiopatía hipertrófica con S.A.M.', 'Resultado último eco.Dilatación VD', 'Resultado último eco.Estenosis mitral', 'Resultado último eco.C.I.A.', 'Resultado último eco.Constricción pericárdica', 'Resultado último eco.Otro', 'cca_lesiones_coronariog_angioTC_PF_varia.DA', 'cca_lesiones_coronariog_angioTC_PF_varia.CD', 'cca_lesiones_coronariog_angioTC_PF_varia.Cx', 'cca_lesiones_coronariog_angioTC_PF_varia.TROCO', 'cca_lesiones_coronariog_angioTC_PF_varia.Dilatación aorta ascendente', 'Tratamiento anticoagulante.Antiagregantes', 'Tratamiento anticoagulante.A.C.O.', 'Intervención realizada.Bypass a DA', 'Intervención realizada.Bypass a OM', 'Intervención realizada.Cirugía de raiz Aórtica-Bental', 'Intervención realizada.Prótesis biológica aórtica', 'Intervención realizada.Bypass a CD', 'Intervención realizada.Reparación mitral', 'Intervención realizada.Bypass a CX', 'Intervención realizada.Bypass a diagonal', 'Intervención realizada.Trasplante cardiaco', 'Intervención realizada.Drenaje pericárdico', 'Intervención realizada.Cierre CIV', 'Intervención realizada.Implante asistencial', 'Intervención realizada.Prótesis mecánica aórtica', 'Intervención realizada.Explante marcapasos', 'Intervención realizada.Recambio prótesis disfuncionante', 'Intervención realizada.Prótesis biológica mitral', 'Intervención realizada.Tubo supra coronario', 'Intervención realizada.Prótesis mecánica mitral', 'Intervención realizada.Resección mixoma auricular', 'Intervención realizada.Reparación esternal', 'Intervención realizada.Reparación trucúspide', 'Intervención realizada.Miectomia septal', 'Intervención realizada.Cierre de CIA', 'Intervención realizada.Cirugía raiz aórtica-David/Jacoob', 'Intervención realizada.Limpieza o Desbridamiento', 'Intervención realizada.Resección pericardio', 'Intervención realizada.Otro', 'Medidas soporte a la salida de quirófano.Catecolaminas', 'Medidas soporte a la salida de quirófano.Balón I.A.', 'Medidas soporte a la salida de quirófano.Marcapasos', 'Medidas soporte a la salida de quirófano.Asistencia ventricular', 'Medidas soporte a la salida de quirófano.ECMO', 'Medidas soporte a la salida de quirófano.Otro', 'Complicaciones hemodinámicas.Fallo VI', 'Complicaciones hemodinámicas.FA/Flutter', 'Complicaciones hemodinámicas.Catecolaminas más de 12h', 'Complicaciones hemodinámicas.Hipotensión leve', 'Complicaciones hemodinámicas.Shock cardiogénico', 'Complicaciones hemodinámicas.Fallo VD', 'Complicaciones hemodinámicas.Bloqueos  AV', 'Complicaciones hemodinámicas.Shock séptico', 'Complicaciones hemodinámicas.Shock hemorrágico', 'Complicaciones hemodinámicas.TV', 'Complicaciones hemodinámicas.Parada cardiaca', 'Complicaciones hemodinámicas.Síndrome de bajo gasto - una o dos catecolaminas más de 12 h', 'Complicaciones hemodinámicas.Fallo VI- bajo gasto con PCE elevada', 'Complicaciones hemodinámicas.Fallo VD- bajo gasto con PVC elevada', 'Complicaciones hemodinámicas.Tormenta arrítmica (más de 3 episodios de TV sostenida en 24', 'Isquemia miocárdica peri-operatoria.Cambios en ECG', 'Isquemia miocárdica peri-operatoria.Cambios en ecocardio', 'Isquemia miocárdica peri-operatoria.Deterioro hemodinámico/arritmias ventriculares', 'Isquemia miocárdica peri-operatoria.Otro', 'Terapias en UCI.Ventilación mecánica más de 72 horas', 'Terapias en UCI.Catecolaminas más de 72 horas', 'Terapias en UCI.Cardioversión', 'Terapias en UCI.Ventilación mecánica invasiva menor de 24 horas', 'Terapias en UCI.Ventilación no invasiva', 'Terapias en UCI.Técnicas de sustitución renal', 'Terapias en UCI.Coronariografía indicada en UCI', 'Terapias en UCI.Balón I.A.', 'Terapias en UCI.Asistencia ventricular', 'Terapias en UCI.Ventilación no invasiva post-extubación', 'Terapias en UCI.Marcapasos endocavitario', 'Terapias en UCI.C. valvular durante ingreso en UCI', 'Terapias en UCI.ECMO VA', 'Terapias en UCI.TAC craneal', 'Terapias en UCI.Angiotac toraco-abdominal', 'Terapias en UCI.Óxido Nítrico', 'Terapias en UCI.Traqueotomía percutánea', 'Terapias en UCI.Otro', 'Complicaciones neurológicas.Delirio', 'Complicaciones neurológicas.Polineuropatía enfermo crítico', 'Complicaciones neurológicas.ACV', 'Complicaciones neurológicas.Convulsiones', 'Complicaciones neurológicas.Encefalopatía anoxica', 'Complicaciones respiratorias.Agudización EPOC', 'Complicaciones respiratorias.Derrame pleural', 'Complicaciones respiratorias.Dehiscencia esternal', 'Complicaciones respiratorias.Atelectasias', 'Complicaciones respiratorias.Neumotórax', 'Complicaciones respiratorias.H.T.P.', 'Complicaciones respiratorias.S. restrictivo (obesidad)', 'Complicaciones respiratorias.Traumatismo torácico post-resucitación', 'Complicaciones respiratorias.SDRA', 'Complicaciones respiratorias.Otro', 'Complicaciones renales.Inicio de tratamiento sustitutivo renal', 'Complicaciones renales.Creatinina mayor de 3 de nueva aparición', 'Complicaciones digestivas.Isquemia intestinal', 'Complicaciones digestivas.Hemorragia digestiva', 'Complicaciones digestivas.Ileo paralítico', 'Complicaciones digestivas.Otro', 'Complicaciones hemorrágicas.Sangrado por drenaje quirúrgico', 'Complicaciones hemorrágicas.Hematomas mediastinico', 'Complicaciones hemorrágicas.Parada cardiaca por rotura aorta/ventriculo', 'Complicaciones hemorrágicas.Otro', 'Tratamiento percutáneo tras el ingreso.STENT EN CD', 'Tratamiento percutáneo tras el ingreso.STENT EN DA', 'Tratamiento percutáneo tras el ingreso.STENT EN CX', 'Tratamiento percutáneo tras el ingreso.STENT EN TRONCO', 'Tratamiento percutáneo tras el ingreso.Otro', 'Infecciones.Traqueobronquitis', 'Infecciones.Infección herida quirúrgica', 'Infecciones.Infección urinaria', 'Infecciones.BGN resistentes', 'Infecciones.Mediastintis', 'Infecciones.Infección catéter', 'Infecciones.Neumonía', 'Infecciones.Bacteriemia', 'Infecciones.Infección inseción dispositivo', 'Infecciones.MARSA', 'Infecciones.Pericarditis', 'Infecciones.Otro', 'Reintervencion durante su estancia UCI.Acumulos mediastinicos pasadas 24 horas /Tamponade', 'Reintervencion durante su estancia UCI.Sangrado por drenajes en las primeras 24 horas', 'Reintervencion durante su estancia UCI.Trasplante cardíaco', 'Reintervencion durante su estancia UCI.Retirada de dispositivos', 'Reintervencion durante su estancia UCI.Otro', 'Situación al alta de UCI.Control de fallo cardiaco', 'Situación al alta de UCI.Precisa ayuda respiratoria', 'Situación al alta de UCI.Ayuda movilización', 'Situación al alta de UCI.Estable no precisa cuidados especiales', 'Situación al alta de UCI.Precisa vigilancia herida quirúrgica', 'Situación al alta de UCI.Precisa telemetría', 'Situación al alta de UCI.Ayuda comer', 'Situación al alta de UCI.Vigilancia por delirio', 'Situación al alta de UCI.Vigilancia anemia/anticoagulantes', 'Situación al alta de UCI.Limitado esfuerzo terapéutico', 'Situación al alta de UCI.Cuidados especiales por vía aérea artificial', 'Situación al alta de UCI.Otro', 'Situación funcional al ingreso Hospital.Fallo cardiaco con necesidad diuréticos IV', 'Situación funcional al ingreso Hospital.Estable', 'Situación funcional al ingreso Hospital.Ventilación no invasiva', 'Situación funcional al ingreso Hospital.Inestable necesita catecolaminas', 'Situación funcional al ingreso Hospital.En ventilación mecánica', 'Situación funcional al ingreso Hospital.Parada cardiaca recuperada', 'Situación funcional al ingreso Hospital.trasplante cardiaco', 'Situación funcional al ingreso Hospital.Otro', 'EDAD', 'Ingreso en: Otra: HSA', 'Ingreso en: Otra: QUIROFANO', 'Ingreso en: Otra: URGENCIAS', 'Ingreso en: Otra: h Alvarez Buylla', 'Ingreso en: Otra: otro hospital', 'Ingreso en: Otra: unidad de ictus', 'Ingreso en: Planta', 'Ingreso en: Reanimación Quirúrgica', 'Ingreso en: UVI Cardiaca', 'Ingreso en: UVI Polivalente', 'Ingreso en: Unidad Coronaria', 'Motivo de ingreso en UCI: Complicaciones médicas del postoperatorio', 'Motivo de ingreso en UCI: Enfermedad no cardiológica', 'Motivo de ingreso en UCI: IAM COMPLICADO', 'Motivo de ingreso en UCI: Insuficiencia cardiaca', 'Motivo de ingreso en UCI: PCR reanimada', 'Motivo de ingreso en UCI: Post-operatorio C.Cardiaca', 'Motivo de ingreso en UCI: Re-intervención precoz (en el mismo ingreso)', 'Motivo de ingreso en UCI: Re-intervención tardía (antes de 1 año)', 'Motivo de ingreso en UCI: TAVI']

cols2 = ['Talla', 'Peso', 'EDAD', 'Tiempo de CEC', 'Tiempo de clampaje', 'Tiempo de isquemia', 'Tiempo de parada circulatoria', 'SAPS-3' ,'Hombre', 'Mujer', 'cca_nyha_PF', 'Índice de Fragilidad', 'cca_fevi_PF_varab_pre-ingr', 'Tipo de cirugía', 'Pico de troponina', 'Sangrado en las primeras 24 horas', 'Cifra de plaquetas más baja', 'núm. concentrados hematies primeras 48 h', 'Factores de riesgo cardiovascular.HTA', 'Factores de riesgo cardiovascular.Enfermedad vascular cerebral', 'Factores de riesgo cardiovascular.Dislipemia', 'Factores de riesgo cardiovascular.Diabetes', 'Factores de riesgo cardiovascular.Enfermedad ileo-femoral', 'Factores de riesgo cardiovascular.Tabaco', 'Factores de riesgo cardiovascular.Enfermedad aorta abdominal', 'Enfermedad cardiológica.Coronaria', 'Enfermedad cardiológica.Intervencionismo coronario percutáneo previo', 'Enfermedad cardiológica.IAM', 'Enfermedad cardiológica.Aorta ascendente', 'Enfermedad cardiológica.Cirugía aórtica previa', 'Enfermedad cardiológica.F.A. crónica', 'Enfermedad cardiológica.Enfermedad valvular', 'Enfermedad cardiológica.Miocardiopatia dilatada crónica', 'Enfermedad cardiológica.Marcapasos tricameral/desfibrilador', 'Enfermedad cardiológica.Cirugía mitral previa', 'Enfermedad cardiológica.Marcapasos previo', 'Enfermedad cardiológica.Endocarditis protésica precoz', 'Enfermedad cardiológica.Endocarditis protésica tardía', 'Enfermedad cardiológica.Endocarditis v. nativa', 'Enfermedad cardiológica.Tumores cardiacos', 'Enfermedad cardiológica.Miocardiopatía Hipertrófica', 'Enfermedad cardiológica.Pericardio', 'Enfermedad cardiológica.Cirugía v. aórtica y raíz aórtica previa', 'Enfermedad cardiológica.no confirmada', 'Enfermedad cardiológica.BY-PASS coronario previo', 'Enfermedad cardiológica.Otro', 'Resultado último eco.Alteraciones segmentarias contractilidad', 'Resultado último eco.Patrón pseudonormal diastólico', 'Resultado último eco.V.I. hipertrófico', 'Resultado último eco.Estenosis aórtica', 'Resultado último eco.Insuficiencia mitral', 'Resultado último eco.Dilatación aurículas', 'Resultado último eco.Dilatación VI', 'Resultado último eco.TAPSE bajo', 'Resultado último eco.HTP mayor de 45 MMHG', 'Resultado último eco.Derrame pericárdico', 'Resultado último eco.Insuficiencia tricúspide', 'Resultado último eco.Insuficiencia aórtica', 'Resultado último eco.Disfunción protésica', 'Resultado último eco.Lesiones de endocarditis-imágenes vibrátiles velos valvulare', 'Resultado último eco.Disfunción anuloplastia', 'Resultado último eco.Mixoma auricular', 'Resultado último eco.Miocardiopatía hipertrófica con S.A.M.', 'Resultado último eco.Dilatación VD', 'Resultado último eco.Estenosis mitral', 'Resultado último eco.C.I.A.', 'Resultado último eco.Constricción pericárdica', 'Resultado último eco.Otro', 'cca_lesiones_coronariog_angioTC_PF_varia.DA', 'cca_lesiones_coronariog_angioTC_PF_varia.CD', 'cca_lesiones_coronariog_angioTC_PF_varia.Cx', 'cca_lesiones_coronariog_angioTC_PF_varia.TROCO', 'cca_lesiones_coronariog_angioTC_PF_varia.Dilatación aorta ascendente', 'Tratamiento anticoagulante.Antiagregantes', 'Tratamiento anticoagulante.A.C.O.', 'Intervención realizada.Bypass a DA', 'Intervención realizada.Bypass a OM', 'Intervención realizada.Cirugía de raiz Aórtica-Bental', 'Intervención realizada.Prótesis biológica aórtica', 'Intervención realizada.Bypass a CD', 'Intervención realizada.Reparación mitral', 'Intervención realizada.Bypass a CX', 'Intervención realizada.Bypass a diagonal', 'Intervención realizada.Trasplante cardiaco', 'Intervención realizada.Drenaje pericárdico', 'Intervención realizada.Cierre CIV', 'Intervención realizada.Implante asistencial', 'Intervención realizada.Prótesis mecánica aórtica', 'Intervención realizada.Explante marcapasos', 'Intervención realizada.Recambio prótesis disfuncionante', 'Intervención realizada.Prótesis biológica mitral', 'Intervención realizada.Tubo supra coronario', 'Intervención realizada.Prótesis mecánica mitral', 'Intervención realizada.Resección mixoma auricular', 'Intervención realizada.Reparación esternal', 'Intervención realizada.Reparación trucúspide', 'Intervención realizada.Miectomia septal', 'Intervención realizada.Cierre de CIA', 'Intervención realizada.Cirugía raiz aórtica-David/Jacoob', 'Intervención realizada.Limpieza o Desbridamiento', 'Intervención realizada.Resección pericardio', 'Intervención realizada.Otro', 'Medidas soporte a la salida de quirófano.Catecolaminas', 'Medidas soporte a la salida de quirófano.Balón I.A.', 'Medidas soporte a la salida de quirófano.Marcapasos', 'Medidas soporte a la salida de quirófano.Asistencia ventricular', 'Medidas soporte a la salida de quirófano.ECMO', 'Medidas soporte a la salida de quirófano.Otro', 'Complicaciones hemodinámicas.Fallo VI', 'Complicaciones hemodinámicas.FA/Flutter', 'Complicaciones hemodinámicas.Catecolaminas más de 12h', 'Complicaciones hemodinámicas.Hipotensión leve', 'Complicaciones hemodinámicas.Shock cardiogénico', 'Complicaciones hemodinámicas.Fallo VD', 'Complicaciones hemodinámicas.Bloqueos  AV', 'Complicaciones hemodinámicas.Shock séptico', 'Complicaciones hemodinámicas.Shock hemorrágico', 'Complicaciones hemodinámicas.TV', 'Complicaciones hemodinámicas.Parada cardiaca', 'Complicaciones hemodinámicas.Síndrome de bajo gasto - una o dos catecolaminas más de 12 h', 'Complicaciones hemodinámicas.Fallo VI- bajo gasto con PCE elevada', 'Complicaciones hemodinámicas.Fallo VD- bajo gasto con PVC elevada', 'Complicaciones hemodinámicas.Tormenta arrítmica (más de 3 episodios de TV sostenida en 24', 'Isquemia miocárdica peri-operatoria.Cambios en ECG', 'Isquemia miocárdica peri-operatoria.Cambios en ecocardio', 'Isquemia miocárdica peri-operatoria.Deterioro hemodinámico/arritmias ventriculares', 'Isquemia miocárdica peri-operatoria.Otro', 'Terapias en UCI.Ventilación mecánica más de 72 horas', 'Terapias en UCI.Catecolaminas más de 72 horas', 'Terapias en UCI.Cardioversión', 'Terapias en UCI.Ventilación mecánica invasiva menor de 24 horas', 'Terapias en UCI.Ventilación no invasiva', 'Terapias en UCI.Técnicas de sustitución renal', 'Terapias en UCI.Coronariografía indicada en UCI', 'Terapias en UCI.Balón I.A.', 'Terapias en UCI.Asistencia ventricular', 'Terapias en UCI.Ventilación no invasiva post-extubación', 'Terapias en UCI.Marcapasos endocavitario', 'Terapias en UCI.C. valvular durante ingreso en UCI', 'Terapias en UCI.ECMO VA', 'Terapias en UCI.TAC craneal', 'Terapias en UCI.Angiotac toraco-abdominal', 'Terapias en UCI.Óxido Nítrico', 'Terapias en UCI.Traqueotomía percutánea', 'Terapias en UCI.Otro', 'Complicaciones neurológicas.Delirio', 'Complicaciones neurológicas.Polineuropatía enfermo crítico', 'Complicaciones neurológicas.ACV', 'Complicaciones neurológicas.Convulsiones', 'Complicaciones neurológicas.Encefalopatía anoxica', 'Complicaciones respiratorias.Agudización EPOC', 'Complicaciones respiratorias.Derrame pleural', 'Complicaciones respiratorias.Dehiscencia esternal', 'Complicaciones respiratorias.Atelectasias', 'Complicaciones respiratorias.Neumotórax', 'Complicaciones respiratorias.H.T.P.', 'Complicaciones respiratorias.S. restrictivo (obesidad)', 'Complicaciones respiratorias.Traumatismo torácico post-resucitación', 'Complicaciones respiratorias.SDRA', 'Complicaciones respiratorias.Otro', 'Complicaciones renales.Inicio de tratamiento sustitutivo renal', 'Complicaciones renales.Creatinina mayor de 3 de nueva aparición', 'Complicaciones digestivas.Isquemia intestinal', 'Complicaciones digestivas.Hemorragia digestiva', 'Complicaciones digestivas.Ileo paralítico', 'Complicaciones digestivas.Otro', 'Complicaciones hemorrágicas.Sangrado por drenaje quirúrgico', 'Complicaciones hemorrágicas.Hematomas mediastinico', 'Complicaciones hemorrágicas.Parada cardiaca por rotura aorta/ventriculo', 'Complicaciones hemorrágicas.Otro', 'Tratamiento percutáneo tras el ingreso.STENT EN CD', 'Tratamiento percutáneo tras el ingreso.STENT EN DA', 'Tratamiento percutáneo tras el ingreso.STENT EN CX', 'Tratamiento percutáneo tras el ingreso.STENT EN TRONCO', 'Tratamiento percutáneo tras el ingreso.Otro', 'Infecciones.Traqueobronquitis', 'Infecciones.Infección herida quirúrgica', 'Infecciones.Infección urinaria', 'Infecciones.BGN resistentes', 'Infecciones.Mediastintis', 'Infecciones.Infección catéter', 'Infecciones.Neumonía', 'Infecciones.Bacteriemia', 'Infecciones.Infección inseción dispositivo', 'Infecciones.MARSA', 'Infecciones.Pericarditis', 'Infecciones.Otro', 'Reintervencion durante su estancia UCI.Acumulos mediastinicos pasadas 24 horas /Tamponade', 'Reintervencion durante su estancia UCI.Sangrado por drenajes en las primeras 24 horas', 'Reintervencion durante su estancia UCI.Trasplante cardíaco', 'Reintervencion durante su estancia UCI.Retirada de dispositivos', 'Reintervencion durante su estancia UCI.Otro', 'Situación funcional al ingreso Hospital.Fallo cardiaco con necesidad diuréticos IV', 'Situación funcional al ingreso Hospital.Estable', 'Situación funcional al ingreso Hospital.Ventilación no invasiva', 'Situación funcional al ingreso Hospital.Inestable necesita catecolaminas', 'Situación funcional al ingreso Hospital.En ventilación mecánica', 'Situación funcional al ingreso Hospital.Parada cardiaca recuperada', 'Situación funcional al ingreso Hospital.trasplante cardiaco', 'Situación funcional al ingreso Hospital.Otro', 'Ingreso en: Otra: HSA', 'Ingreso en: Otra: QUIROFANO', 'Ingreso en: Otra: URGENCIAS', 'Ingreso en: Otra: h Alvarez Buylla', 'Ingreso en: Otra: otro hospital', 'Ingreso en: Otra: unidad de ictus', 'Ingreso en: Planta', 'Ingreso en: Reanimación Quirúrgica', 'Ingreso en: UVI Cardiaca', 'Ingreso en: UVI Polivalente', 'Ingreso en: Unidad Coronaria', 'Motivo de ingreso en UCI: Complicaciones médicas del postoperatorio', 'Motivo de ingreso en UCI: Enfermedad no cardiológica', 'Motivo de ingreso en UCI: IAM COMPLICADO', 'Motivo de ingreso en UCI: Insuficiencia cardiaca', 'Motivo de ingreso en UCI: PCR reanimada', 'Motivo de ingreso en UCI: Post-operatorio C.Cardiaca', 'Motivo de ingreso en UCI: Re-intervención precoz (en el mismo ingreso)', 'Motivo de ingreso en UCI: Re-intervención tardía (antes de 1 año)', 'Motivo de ingreso en UCI: TAVI', 'Situación al alta de UCI.Control de fallo cardiaco', 'Situación al alta de UCI.Precisa ayuda respiratoria', 'Situación al alta de UCI.Ayuda movilización', 'Situación al alta de UCI.Estable no precisa cuidados especiales', 'Situación al alta de UCI.Precisa vigilancia herida quirúrgica', 'Situación al alta de UCI.Precisa telemetría', 'Situación al alta de UCI.Ayuda comer', 'Situación al alta de UCI.Vigilancia por delirio', 'Situación al alta de UCI.Vigilancia anemia/anticoagulantes', 'Situación al alta de UCI.Limitado esfuerzo terapéutico', 'Situación al alta de UCI.Cuidados especiales por vía aérea artificial', 'Situación al alta de UCI.Otro']

data = data[cols2]

#OUTLIERS
data['Peso'].loc[103] = 71.4
data['Peso'].loc[262] = 75.5
data['Peso'].loc[363] = 52.4

data.to_csv('../data/pacientes_ucic_v3.csv', sep = ';', na_rep = 'NA')