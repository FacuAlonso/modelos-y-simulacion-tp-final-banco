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

- **Hipótesis**: reducción de escaladas N1, más especialistas N2, redistribución de turnos N1, reducción de servicio, mayor demanda y prioridad con envejecimiento.
- **Parámetros**: tasas por hora, distribuciones de llegadas (Poisson o determinística), servicio (triangular o fijo), dotación/turnos, escalamiento, reapertura, horizonte, réplicas y semilla.
- **Salida principal**: P95, tickets pendientes, incumplimiento SLA y resueltos. Dos gráficos breves comparan el objetivo y efectos secundarios; utilización identifica cuellos de botella.
- **Persistencia**: Guardar JSON con entrada sintética, resultados de cada ticket, réplicas, hipótesis, configuración y resultados. Cargar JSON recupera los datos y un resultado anterior cuando esté disponible.
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

El dashboard ofrece casos predeterminados para comenzar rápidamente: operación
normal, pico de demanda, incidente masivo, saturación de N2 y prioridad crítica.
Se pueden cargar para inspeccionar sus parámetros o ejecutar directamente con
**Cargar y ejecutar**.

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
