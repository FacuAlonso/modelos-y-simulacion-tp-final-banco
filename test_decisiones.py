"""Pruebas reproducibles del módulo de decisiones (sin depender del dashboard)."""
from dataclasses import replace
from estudio_decisiones import (
    ejecutar_estudio, costo_relativo, alternativas, _factible, TARIFAS_RELATIVAS,
)
from simulacion_tickets_banco import Configuracion


def main():
    c = replace(Configuracion(), dias=2, replicas=2)
    casos = alternativas(c)
    assert len(casos) == 9
    assert costo_relativo(casos['Base']) == costo_relativo(casos['Capacitación N1 (−25 % escaladas)'])
    assert costo_relativo(casos['Reducción N1 (−1 técnico)']) < costo_relativo(casos['Base'])
    assert costo_relativo(casos['Refuerzo N2 (+1 técnico)']) > costo_relativo(casos['Base'])
    assert TARIFAS_RELATIVAS['N3'] > TARIFAS_RELATIVAS['N1']
    r = ejecutar_estudio(c, replicas=2, dias=2, incluir_gemelo=False)
    assert len(r['escenarios']) == 9 and len(r['estres_mas_30pct']) == 9
    assert len(r['sensibilidad']) == 6
    assert r['decisiones']['sla'] in [x['nombre'] for x in r['escenarios']]
    assert r['decisiones']['balance'] in [x['nombre'] for x in r['escenarios']]
    assert 'Base' in r['decisiones']['factibles']
    assert all(_factible(x) for x in r['escenarios'] if x['nombre'] != 'Base'
               and x['nombre'] in r['decisiones']['factibles'])
    # Modelo no altera configuración de entrada; escenarios siempre parten de la misma base.
    assert c.dias == 2 and c.replicas == 2
    # Misma semilla y parámetros => misma salida.
    r2 = ejecutar_estudio(c, replicas=2, dias=2, incluir_gemelo=False)
    assert r == r2
    print('Tests decisiones OK: 9 políticas, restricciones, sensibilidad y reproducibilidad.')

if __name__ == '__main__':
    main()
