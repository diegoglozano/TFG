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

A la hora de alimentar los modelos de aprendizaje automático con datos, estos deben ser inteligibles por el ordenador. Por esta misma razón, muchos de los datos de entrada recibidos por parte del hospital deben ser preprocesados de cierta forma.

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
    
    Que tomen valor 0 o 1 según el ejemplo correspondiente.

- Los datos numéricos son aquellos que toman valores continuos, como podría ser la altura o el peso de una persona.

A continuación se describe una lista de variables junto con el tipo de preprocesado realizado:

- cca_fevi_PF_varab_pre-ingr: variable categórica y ordinal que puede tomar los siguientes valores
    0. No cuantificada: 0
    1. Normal: 1
    2. Disfunción leve (>40): 2
    3. Disfunción moderada (30-40): 3
    4. Disfunción grave (<30): 4
- Cifra de plaquetas más baja: la cifra de plaquetas podría ser una variable numérica pero, a la hora de recoger los datos, se clasifican en categorías ordinales, que son las siguientes
    0. 20.000-49.999: 0
    1. 50.000-99.999: 1
    2. 100.000-149.999: 2
    3. &gt;150.000: 3
    
[1]: https://es.wikipedia.org/wiki/Variable_categ%C3%B3rica
