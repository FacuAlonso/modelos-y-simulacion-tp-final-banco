# Mesa de ayuda bancaria — Simulador + dashboard

El proyecto se puede utilizar de tres maneras:

- **Motor independiente:** [simulacion_tickets_banco.py](simulacion_tickets_banco.py) contiene el modelo, la generación de tickets, la simulación, la verificación y la salida por consola.
- **Estudio multiobjetivo independiente:** [estudio_decisiones.py](estudio_decisiones.py) compara SLA, costos relativos, equilibrio y robustez, e incluye análisis de sensibilidad y evaluación sintética del gemelo.
- **Laboratorio web:** [dashboard_tickets_banco.py](dashboard_tickets_banco.py) sirve una interfaz para cambiar parámetros, formular hipótesis, comparar escenarios y descargar el experimento completo.

## Ejecución

```bash
pip install -r requirements.txt
python dashboard_tickets_banco.py
```

Abrí `http://127.0.0.1:8765` si no se abre automáticamente. El dashboard utiliza únicamente la biblioteca estándar de Python para servir la interfaz web; no requiere Streamlit ni servicios externos.

La página se adapta a computadora, tablet y celular. Un menú (barra lateral, o barra inferior en pantallas angostas) lleva a cada pantalla:

1. **Resumen:** las cuatro preguntas principales de un vistazo.
2. **Decisiones:** SLA, costos, equilibrio, primera respuesta y resiliencia, con análisis cuantitativo y evaluación sintética del gemelo.
3. **Alternativas:** qué alternativa reduce mejor los tiempos sin aumentar el backlog, con un gráfico que compara cada una contra la base.
4. **Puntos críticos:** un deslizable de demanda (×1,0 a ×1,8) muestra dónde aparece el primer cuello de botella, dónde se incumple el SLA y dónde se satura la mesa.
5. **Red neuronal:** si conviene sumarla para anticipar el escalamiento. Usa [estudio_redes.py](estudio_redes.py), que también se ejecuta solo con `python estudio_redes.py` (admite `--historico tickets.csv`).
6. **Supuestos** bajo los que valen esas respuestas.
7. **Laboratorio:** para probar otra hipótesis o cambiar parámetros.

La primera ejecución calcula todo en segundo plano (unos minutos) y muestra cada respuesta apenas está lista. El resultado queda en `resultados_tickets_banco/resumen_dashboard.json` y solo se recalcula si cambia el modelo o se pulsa **Recalcular resultados**. Si el puerto 8765 está ocupado, el servidor usa el siguiente libre y lo informa por consola.

El motor sigue funcionando de manera independiente:

```bash
python simulacion_tickets_banco.py --dias 5 --replicas 15
python simulacion_tickets_banco.py --probar
python simulacion_tickets_banco.py --guardar experimento.json
python simulacion_tickets_banco.py --dias 5 --replicas 15 --salida salida_graficos
```

La última variante genera `comparacion_escenarios.png`, `evolucion_colas.png` y
`sensibilidad.png` sin iniciar el servidor web.

## Funcionalidades

La pantalla **Laboratorio** permite probar otra hipótesis:

- **Hipótesis**: reducción de escaladas N1, más especialistas N2, redistribución de turnos N1, reducción de servicio, mayor demanda y prioridad con envejecimiento.
- **Parámetros**: tasas por hora, distribuciones de llegadas (Poisson o determinística), servicio (triangular o fijo), dotación/turnos, escalamiento, reapertura, horizonte, réplicas y semilla.
- **Salida**: P95, tickets pendientes, incumplimiento SLA y resueltos, con efectos secundarios y utilización por equipo.
- **Persistencia**: descarga un JSON con entrada sintética, resultados de cada ticket, réplicas, hipótesis, configuración y resultados.
- **Comparabilidad**: semillas emparejadas en escenarios generados; cada escenario recibe características aleatorias reproducibles. Importar datos fija exactamente el mismo lote para todas las réplicas y escenarios. En ese modo, los IC entre réplicas no miden incertidumbre de demanda y no se recomienda interpretar las 30 repeticiones como 30 observaciones independientes.

## Precauciones y alcance

Los datos de ejemplo son **sintéticos**. Guardarlos facilita futuras prácticas de predicción, pero entrenar con esos registros no prueba capacidad de generalización a incidentes reales. Para entrenar modelos sin fuga de información, utilizar como entradas únicamente atributos conocidos al abrir el ticket (prioridad, categoría y momento de llegada; no usar complejidad si se conoce después del diagnóstico), y como etiquetas el escalamiento o la resolución **posterior**. Separar entrenamiento y prueba temporalmente.

Los SLA y tiempos son supuestos; P95 de resolución solo considera tickets cerrados; las duraciones se expresan en minutos corridos. Complementar siempre con pendientes, utilización y SLA. Verificación de código ≠ validación con datos reales.

El dashboard informa si la hipótesis queda **sostenida**, **refutada** o
**no concluyente** bajo estos supuestos. Para datos sintéticos, el IC95% de la
diferencia pareada usa t de Student. Un lote importado es una comparación
descriptiva puntual: no se presenta como evidencia independiente ni recibe un
IC artificial.

También se puede elegir el P95 de resolución incluyendo tickets abiertos. Es
una medida conservadora: para un ticket todavía abierto se usa el tiempo
transcurrido hasta el final del horizonte.

## Verificación del motor

Las invariantes del modelo se comprueban con `verificar()`, que permanece activa
incluso si Python se ejecuta con optimización. Para ejecutar las pruebas:

```bash
python simulacion_tickets_banco.py --probar
python -O simulacion_tickets_banco.py --probar
```

## NUEVO: estudio de decisiones y evaluación sintética del gemelo

Además de las hipótesis individuales, `estudio_decisiones.py` compara **nueve
políticas** y responde cinco objetivos: cumplimiento SLA, costo de dotación,
equilibrio, primera respuesta y resiliencia bajo +30 % de demanda.

```bash
python estudio_decisiones.py --dias 3 --replicas 12 --salida resultados_decisiones.json
```

- Diseño experimental: una modificación por escenario, réplicas Monte Carlo, misma
  lista de semillas y comparación de diferencias con IC95 % de t de Student.
- Se estudian SLA, pendientes, P95 de resolución (también con abiertos), primera
  respuesta, horas de dotación y costo relativo.
- **Costos ficticios**: N1 = 1, N2 = 1,6 y N3 = 2,2 unidades relativas por hora.
  No son salarios ni incluye formación, tecnología o eficiencia organizacional.
- El criterio de ahorro exige que el límite superior del IC95 % no supere
  **+1 punto porcentual de incumplimiento SLA** ni **+5 tickets pendientes**.
  Son tolerancias didácticas, no acuerdos establecidos por el banco.
- El equilibrio pondera explícitamente SLA (50 %), backlog (30 %) y costo (20 %),
  con escalas publicadas en el código. Los pesos son supuestos para discutir.
- Hay un barrido de sensibilidad separado (demanda, servicio y escalamiento,
  cada uno ±20 %) y un segundo conjunto con +30 % de demanda.
- La evaluación del gemelo ejecuta verificaciones automáticas y una calibración
  con referencia **sintética independiente**: 15 días para calibrar, 5 reservados
  para medir la coherencia del procedimiento. **No es validación de un banco real**.

En el dashboard se incorporó la sección **Decisiones**, con resultados completos y
un botón para repetir el estudio con los parámetros definidos en Laboratorio. Al
arrancar, el resumen general genera una batería predeterminada de 12 réplicas ×
3 días para decisiones, distinta de la comparación histórica P95 del panel, que
usa la configuración general. Esa distinción se muestra en pantalla.

### Verificación y reproducibilidad

```bash
python simulacion_tickets_banco.py --probar
python estudio_decisiones.py --replicas 12 --dias 3 --salida resultados_decisiones.json
python dashboard_tickets_banco.py
```

Se recomienda ejecutar `estudio_decisiones.py` por separado durante la defensa
si no se quiere esperar el estudio neuronal de la pantalla Resumen. También
permite abrir directamente el dashboard en `#decisiones`.

**Cautela:** las semillas comunes ayudan a reproducir comparaciones, pero si se
cambia la tasa de llegadas, el generador no necesariamente produce los mismos
tickets. El conjunto de políticas es exploratorio y los IC95 % no incluyen
corrección por comparación múltiple. Los resultados no son pronósticos del banco.
