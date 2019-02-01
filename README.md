# TFG: Predicción del grado de dependencia de pacientes a la salida de la UCI

## Variables de salida

Situación al alta de UCI.Estable no precisa cuidados especiales:
45.12% - 54.88%

Situación al alta de UCI.Precisa telemetría:
74.63% - 25.37%

Situación al alta de UCI.Control de fallo cardiaco:
87.07% - 12.93%

Situación al alta de UCI.Vigilancia anemia/anticoagulantes:
90.98% - 9.02%

Situación al alta de UCI.Otro:
93.17% - 6.83%

Situación al alta de UCI.Ayuda movilización:
93.66% - 6.34%

Situación al alta de UCI.Precisa vigilancia herida quirúrgica:
93.9% - 6.1%

Situación al alta de UCI.Vigilancia por delirio:
95.61% - 4.39%

Situación al alta de UCI.Ayuda comer:
97.07% - 2.93%

Situación al alta de UCI.Precisa ayuda respiratoria:
97.56% - 2.44%

Situación al alta de UCI.Limitado esfuerzo terapéutico:
98.05% - 1.95%

Situación al alta de UCI.Cuidados especiales por vía aérea artificial:
99.76% - 0.24%

A continuación se muestra la proporción de los datos de salida, tanto de forma relativa como absoluta. Como vemos, todas las clases sufren un desbalanceo, en mayor o en menor medida.

| Situación al alta | False (%) | True (%) | False (abs) | True (abs) |
|-------------------|-----------|----------|-------------|------------|
| Estable no precisa cuidados especiales | 45.12% | 54.88% | 185 | 225 |
| Precisa telemetría | 74.63% | 25.37% | 306 | 104 |
| Control de fallo cardiaco | 87.07% | 12.93% | 357 | 53 |
| Vigilancia anemia/anticoagulantes | 90.98% | 9.02% | 373 | 37 |
| Otro | 93.17% | 6.83% | 382 | 28 |
| Ayuda movilización | 93.66% | 6.34% | 384 | 26 |
| Precisa vigilancia herida quirúrgica | 93.9% | 6.1% | 385 | 25 |
| Vigilancia por delirio | 95.61% | 4.39% | 392 | 18 |
| Ayuda comer | 97.07% | 2.93% | 398 | 12 |
| Precisa ayuda respiratoria | 97.56% | 2.44% | 400 | 10 |
| Limitado esfuerzo terapéutico | 98.05% | 1.95% | 402 | 8 |
| Cuidados especiales por vía aérea artificial | 99.76% | 0.24% | 409 | 1 |

## Preprocesado

A la hora de alimentar los modelos de aprendizaje automático con datos, estos deben ser inteligibles por el ordenador. Por esta misma razón, muchos de los datos de entrada recibidos por parte del hospital deben ser preprocesados de cierta forma. Dichos procesos se denominan *feature engineering*.

Habitualmente, se pueden clasificar los datos en *categóricos* y *numéricos*.
- Los datos categóricos son aquellos que toman un valor de un número limitado de posibles valores. [1] Estos, a su vez, pueden seguir un orden o no.  
Una variable categórica y ordinal podría ser la calificación asociada a las notas de un alumno. En el caso de la Universidad de Oviedo, estas calificaciones pueden ser *Suspenso*, *Aprobado*, *Notable*, *Sobresaliente* y *Matrícula de honor*, donde claramente existe un orden entre ellas, siendo *Suspenso* la peor y *Matrícula de honor* la mejor.  Por lo tanto, la forma correcta de procesar estos datos sería asignar un número a cada variable, respetando el orden entre ellas:
    - Suspenso: 0
    - Aprobado: 1
    - Notable: 2
    - Sobresaliente: 3
    - Matrícula de honor: 4
    
    Un claro ejemplo del caso contrario que, además, aparece en los datos proporcionados, es la variable *Sexo*. Dicha variable puede tomar dos valores: *Hombre* o *Mujer*. Sin embargo, en este caso no estamos ante una variable que tome un orden.  
    Si procesamos estos datos de forma que *Hombre* tome valor 0 y *Mujer*, valor 1, ciertos algoritmos podrían suponer que existe un orden entre los géneros. La forma correcta de procesarlos en este caso sería *binarizar* esta variable en dos nuevas:
    - Sexo. Hombre
    - Sexo. Mujer
    
    Que en este caso tomarán valor 0 o 1 según el ejemplo correspondiente. En otro tipo de variables, podría haber más de una que tome valor 1.

- Los datos numéricos son aquellos que toman valores continuos, como podría ser la altura o el peso de una persona.

A continuación se describe una lista de variables junto con el tipo de preprocesado realizado:

- cca_fevi_PF_varab_pre-ingr: variable categórica y ordinal que puede tomar los siguientes valores
    1. No cuantificada: 0
    2. Normal: 1
    3. Disfunción leve (>40): 2
    4. Disfunción moderada (30-40): 3
    5. Disfunción grave (<30): 4
- Cifra de plaquetas más baja: la cifra de plaquetas podría ser una variable numérica pero, a la hora de recoger los datos, se clasifican en categorías ordinales, que son las siguientes
    1. 20.000-49.999: 0
    2. 50.000-99.999: 1
    3. 100.000-149.999: 2
    4. &gt;150.000: 3
- Talla: variable numérica que indica la altura (en centímetros) del paciente
- Peso: variable numérica que indica la masa (en kilogramos) de un paciente
- FECHA NACIMIENTO: variable que indica el día de nacimiento del paciente en formato dd/mm/YY
- Fecha ingreso hospitalario: variable que indica el día en el que el paciente ingresó en la UCI en formato dd/mm/YYYY
- EDAD: variable numérica calculada a partir de las dos anteriores (*FECHA NACIMIENTO* y *Fecha ingreso hospitalario*) que indica la edad del paciente en años.
- Índice de Fragilidad: variable categórica y ordinal que puede tomar los siguientes valores
    1. nan: 0
    2. Muy saludable. Por encima de lo esperado para su edad: 1
    3. Sano: Paciente asintomático, con actividad normal: 2
    4. Controlado: Paciente con sintomatología tratada, con activid: 3
    5. Vulnerable:Independiente, pero con actividad limitada por su: 4
    6. Fragilidad leve: Necesita ayuda para tareas difíciles: 5
    7. Fragilidad moderada: Necesita ayuda dentro y fuera de la cas: 6
    8. Fragilidad grave: Totalmente dependiente para las actividade: 7
    9. Fragilidad muy grave: Totalmente dependiente, no tolera ni l: 8
    10. Enfermedad terminal: Esperanza de vida menor de 6 meses: 9
- Ingreso en: variable categórica que no indica ningún tipo de orden. Puede tomar los siguientes valores
    - Otra: HSA
    - Otra: QUIROFANO
    - Otra: URGENCIAS
    - Otra: h Alvarez Buylla
    - Otra: otro hospital
    - Otra: unidad de ictus
    - Planta
    - Reanimación Quirúrgica
    - UVI Cardiaca
    - UVI Polivalente
    - Unidad Coronaria
- Motivo de ingreso en UCI: variable categórica que no indica ningún orden. Puede tomar los siguientes valores
    - Complicaciones médicas del postoperatorio
    - Enfermedad no cardiológica
    - IAM COMPLICADO
    - Insuficiencia cardiaca
    - PCR reanimada
    - Post-operatorio C.Cardiaca
    - Re-intervención precoz (en el mismo ingreso)
    - Re-intervención tardía (antes de 1 año)
    - TAVI
- núm. concentrados hematies primeras 48 h: variable que, al igual que la cifra de plaquetas, es numérica, pero sus datos se recogen como categóricos de la siguiente forma
    - Menor de 3: 0
    - Mayor o igual a 3: 1
- Otros factores: variable categórica que no indica ningún orden. Además, en este caso, más de una variable podrá ser positiva. Los valores son los siguientes
    - Apertura pleuras
    - Transfusión de 1 a 3 CH intraoperatorios
    - Transfusión de plasma o plaquetas
    - Politrasfusión (más de 3 CH) intraoperatorios
    - Administración de factores de coagulación
- Pico de troponina: variable numérica expresada en categorías ordenadas, que pueden tomar los siguientes valores
    1. Menor de 500: 0
    2. 500-999: 1
    3. 1000-3999: 2
    4. 4000 o más: 3
- Sangrado en las primeras 24 horas: variable numérica expresada en categorías ordenadas, tomando los siguientes valores
    1. Menor de 500 ml: 0
    2. 500-1000 ml: 1
    3. Mayor de 1000 ml: 2
- SAPS-3: variable numérica
- SEXO: variable categórica que no indica orden. Sus valores posibles son
    - Mujer
    - Hombre
- Tiempo de CEC: variable numérica
- Tiempo de clampaje: variable numérica
- Tiempo de isquemia: variable numérica
- Tiempo de parada circulatoria: variable numérica
- Tipo de cirugía: variable categórica que indica un orden según la urgencia de la cirugía
    1. nan: 0
    2. Programada: 1
    3. Emergente (intervención en menos de 48 horas del diagnóstico: 2
    4. Urgente (se interviene durante el ingreso de una descompensa: 3
    
## Problema a resolver

El problema a resolver se trata de un caso de clasificación mediante aprendizaje supervisado. El conjunto de datos proporcionado por el hospital consiste en un archivo de 410 filas o ejemplos, donde cada una se corresponde con un paciente distinto.  
Tras el procesado realizado previamente, se obtienen 232 columnas o variables de entrada, de las cuales 12 serán las variables de salida a predecir. Son las siguientes:
- Situación al alta de UCI.Estable no precisa cuidados especiales
- Situación al alta de UCI.Precisa telemetría
- Situación al alta de UCI.Control de fallo cardiaco
- Situación al alta de UCI.Vigilancia anemia/anticoagulantes
- Situación al alta de UCI.Ayuda movilización
- Situación al alta de UCI.Precisa vigilancia herida quirúrgica
- Situación al alta de UCI.Vigilancia por delirio
- Situación al alta de UCI.Ayuda comer
- Situación al alta de UCI.Precisa ayuda respiratoria
- Situación al alta de UCI.Limitado esfuerzo terapéutico
- Situación al alta de UCI.Cuidados especiales por vía aérea artificial

Un paciente podrá estar clasificado en más de una etiqueta en el momento de su salida de la UCI, por lo que el problema de clasificación se denomina *multietiqueta*.

Inicialmente, se ha decidido tratar cada etiqueta posible como un problema independiente, es decir, entrenando un modelo distinto para cada salida. Para ello, se ha utilizado *Support Vector Machines* (SVM) como algoritmo.

Para estimar un error inicial, se ha empleado la técnica de validación cruzada. Dicha técnica es comúnmente utilizada para validar modelos de aprendizaje automático. Consiste en dividir el conjunto de datos en K particiones o *folds* y realizar K iteraciones, utilizando K-1 *folds* como conjunto de entrenamiento y 1 como conjunto de test.  
Además, es frecuente que dichas particiones sean estratificadas, es decir, que cada una de ellas siga la misma distribución que el conjunto total de datos. Esto es especialmente útil en casos como el presente, en el que los datos no siguen una distribución balanceada. Si, por ejemplo, se dispone de un conjunto de datos de 100 ejemplos, donde 95 etiquetas de salida son 0 y 5 son 1, sería probable encontrarse con una iteración en la que los 5 ejemplos positivos se encontrasen en el conjunto de entrenamiento, no pudiendo realizar las pruebas adecuadamente. También podría darse el caso contrario, donde los 5 casos positivos recayesen en el conjunto de test, de forma que el modelo sería incapaz de predecirlos. En el caso de una validación cruzada estratificada de 5 particiones, cada una de ellas contendría un ejemplo positivo, estando de esta forma siempre presente tanto en el conjunto de entrenamiento como en el de test. 

![Validacion cruzada](./Images/crossvalidation5.png)

Por norma general, los algoritmos disponen de ciertos hiperparámetros. [2]  
Una forma de ajustarlos es a base de prueba y error. Para ello, es común utilizar técnicas como *grid search* o *random search*.  
Para el caso de *grid search*, se selecciona una serie de valores para cada uno de los hiperparámetros y se realizan pruebas con todas las combinaciones posibles. Previamente se debe seleccionar una métrica y, tras realizar las pruebas, los mejores hiperparámetros serán los que ofrezcan unos resultados cuya métrica sea la más alta o la más baja, según el caso.  
En el caso de *random search*, se selecciona un abanico de valores para cada uno de los hiperparámetros y, de forma aleatoria, se realizan pruebas con distintas combinaciones un número prefijado de iteraciones.  
Además, es frecuente aplicar dichas técnicas a su vez con otra validación cruzada. De cara a comparar modelos entre sí y, por tanto, a seleccionar los mejores hiperparámetros, una forma adecuada de hacerlo es repitiendo una validación cruzada de 2 particiones 5 veces. [3]  

Dado que inicialmente se ha decidido aplicar el algoritmo SVM con un *kernel* lineal, el único hiperparámetro a optimizar es el de penalización *C*. Cuanto menor sea dicho hiperparámetro, más penalización habrá sobre el modelo, evitando posibles casos de *overfitting* o *sobreajuste*.

Aplicando las dos técnicas a la vez, el resultado es una partición inicial del conjunto de datos en 5 *folds*.  
En la primera iteración, se seleccionarán 4 particiones como conjunto de entrenamiento y 1 como conjunto de test. Dichos conjuntos seguirán la misma distribución. A su vez, el conjunto de entrenamiento resultante se dividirá en 2 *folds* aleatorios en 5 ocasiones, por lo que se obtendrán 10 resultados distintos para cada una de las combinaciones de hiperparámetros.  
Se calculará la media de estos vectores de 10 componentes (en este caso cada uno correspondiente a un valor de *C*) y, tras comparar los resultados, se obtendrá la mejor combinación de hiperparámetros de entre todas las probadas. Por último, se entrenará el conjunto de 4 *folds* inicial con este conjunto de hiperparámetros y se comprobarán los resultados en la partición restante.

[1]: https://es.wikipedia.org/wiki/Variable_categ%C3%B3rica

[2]: Aquellos que deben ser asignados antes de la ejecución

[3]: http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.37.3325&rep=rep1&type=pdf