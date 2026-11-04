# Mesa de ayuda bancaria — Simulador + dashboard

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
```

## Funcionalidades

- **Hipótesis**: reducción de escaladas N1, más especialistas N2, redistribución de turnos N1, reducción de servicio, mayor demanda y prioridad con envejecimiento.
- **Parámetros**: tasas por hora, distribuciones de llegadas (Poisson o determinística), servicio (triangular o fijo), dotación/turnos, escalamiento, reapertura, horizonte, réplicas y semilla.
- **Salida principal**: P95, tickets pendientes, incumplimiento SLA y resueltos. Dos gráficos breves comparan el objetivo y efectos secundarios; utilización identifica cuellos de botella.
- **Persistencia**: Guardar JSON con entrada sintética, resultados de cada ticket, réplicas, hipótesis, configuración y resultados. Cargar JSON recupera los datos y un resultado anterior cuando esté disponible.
- **Comparabilidad**: semillas emparejadas en escenarios generados; cada escenario recibe características aleatorias reproducibles. Importar datos fija exactamente el mismo lote para todas las réplicas y escenarios. En ese modo, los IC entre réplicas no miden incertidumbre de demanda y no se recomienda interpretar las 30 repeticiones como 30 observaciones independientes.

## Precauciones académicas

Los datos de ejemplo son **sintéticos**. Guardarlos facilita futuras prácticas de predicción, pero entrenar con esos registros no prueba capacidad de generalización a incidentes reales. Para entrenar modelos sin fuga de información, utilizar como entradas únicamente atributos conocidos al abrir el ticket (prioridad, categoría, complejidad y llegada), y como etiquetas el escalamiento o la resolución **posterior**. Separar entrenamiento y prueba temporalmente.

Los SLA y tiempos son supuestos; P95 de resolución solo considera tickets cerrados; las duraciones se expresan en minutos corridos. Complementar siempre con pendientes, utilización y SLA. Verificación de código ≠ validación con datos reales.

## Relación con la materia

Modelo conceptual → clases 1–3; eventos discretos → 4; colas/latencia/throughput → 5–6; Monte Carlo → 7; riesgo → 8; diseño de experimentos → 9; sensibilidad y validación → 10–11. Se dejan modelos predictivos y redes neuronales como etapa futura, sin usarlos en esta versión.
