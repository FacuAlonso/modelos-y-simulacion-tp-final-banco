# Mesa de ayuda bancaria — Simulador + dashboard

El proyecto tiene dos formas equivalentes de uso:

- **Entrega independiente:** [simulacion_tickets_banco.py](simulacion_tickets_banco.py) contiene el modelo, la generación de tickets, la simulación, la verificación y la salida por consola. Se puede entregar y ejecutar sin el servidor web.
- **Laboratorio web:** [dashboard_tickets_banco.py](dashboard_tickets_banco.py) sirve una interfaz para cambiar parámetros, formular hipótesis, comparar escenarios y descargar el experimento completo.

## Ejecución

```bash
pip install -r requirements.txt
python dashboard_tickets_banco.py
```

Abrí `http://127.0.0.1:8765` si no se abre automáticamente. El dashboard utiliza únicamente la biblioteca estándar de Python para servir la interfaz web; no requiere Streamlit ni servicios externos.

La página se adapta a computadora, tablet y celular. Un menú (barra lateral, o barra inferior en pantallas angostas) lleva a cada pantalla:

1. **Resumen:** la pregunta principal y las tres respuestas de un vistazo.
2. **Alternativas:** qué alternativa reduce mejor los tiempos sin aumentar el backlog, con un gráfico que compara cada una contra la base.
3. **Puntos críticos:** un deslizable de demanda (×1,0 a ×1,8) muestra dónde aparece el primer cuello de botella, dónde se incumple el SLA y dónde se satura la mesa.
4. **Red neuronal:** si conviene sumarla para anticipar el escalamiento. Usa [estudio_redes.py](estudio_redes.py), que también se ejecuta solo con `python estudio_redes.py` (admite `--historico tickets.csv`).
5. **Supuestos** bajo los que valen esas respuestas.
6. **Laboratorio:** para probar otra hipótesis o cambiar parámetros.

La primera ejecución calcula todo en segundo plano (unos minutos) y muestra cada respuesta apenas está lista. El resultado queda en `resultados_tickets_banco/resumen_dashboard.json` y solo se recalcula si cambia el modelo o se pulsa **Recalcular resultados**. Si el puerto 8765 está ocupado, el servidor usa el siguiente libre y lo informa por consola.

El motor sigue funcionando de manera independiente:

```bash
python simulacion_tickets_banco.py --dias 5 --replicas 15
python simulacion_tickets_banco.py --probar
python simulacion_tickets_banco.py --guardar experimento.json
python simulacion_tickets_banco.py --dias 5 --replicas 15 --graficos salida_graficos
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

Los datos de ejemplo son **sintéticos**. Guardarlos facilita futuras prácticas de predicción, pero entrenar con esos registros no prueba capacidad de generalización a incidentes reales. Para entrenar modelos sin fuga de información, utilizar como entradas únicamente atributos conocidos al abrir el ticket (prioridad, categoría, complejidad y llegada), y como etiquetas el escalamiento o la resolución **posterior**. Separar entrenamiento y prueba temporalmente.

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

Estas pruebas cubren prioridad, FIFO, escalamiento, tiempos de atención y
continuidad del trabajo entre turnos. Una verificación correcta indica que el
programa respeta sus reglas internas; no demuestra que los supuestos representen
la operación real del banco.
