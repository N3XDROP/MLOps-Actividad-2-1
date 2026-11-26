import json
import math
from pathlib import Path

CONTRATO = json.loads((Path(__file__).parent / 'contrato_datos.json').read_text(encoding='utf-8'))
CAMPOS = CONTRATO['campos']
OBJETIVO = CONTRATO['objetivo_entrenamiento']['nombre']


def validar_entrada(datos):
    errores = []
    if not isinstance(datos, dict):
        return ['La entrada debe ser un objeto JSON.']
    for nombre, regla in CAMPOS.items():
        if nombre not in datos or datos[nombre] is None:
            errores.append(f'{nombre}: campo obligatorio o valor nulo')
            continue
        valor = datos[nombre]
        tipo = regla['tipo']
        if tipo == 'string':
            if not isinstance(valor, str) or valor not in regla['permitidos']:
                errores.append(f'{nombre}: categoría o tipo no válido')
            continue
        tipos = (int,) if tipo == 'integer' else (int, float)
        if isinstance(valor, bool) or not isinstance(valor, tipos):
            errores.append(f'{nombre}: se requiere {tipo}')
            continue
        if not regla['min'] <= valor <= regla['max']:
            errores.append(f'{nombre}: fuera del rango permitido')
    for nombre in datos.keys() - CAMPOS.keys():
        errores.append(f'{nombre}: campo no permitido')
    return errores


def convertir_fila(fila):
    datos = {}
    for nombre, regla in CAMPOS.items():
        valor = fila.get(nombre)
        if valor is None or not valor.strip() or valor.strip().lower() in {'nan', 'null', 'none', 'na'}:
            datos[nombre] = None
            continue
        valor = valor.strip()
        if regla['tipo'] == 'string':
            datos[nombre] = valor
        else:
            try:
                numero = float(valor)
                datos[nombre] = int(numero) if regla['tipo'] == 'integer' and math.isfinite(numero) and numero.is_integer() else numero
            except (ValueError, OverflowError):
                datos[nombre] = valor
    return datos


if __name__ == '__main__':
    for ejemplo in ('ejemplo_valido', 'ejemplo_invalido'):
        errores = validar_entrada(CONTRATO[ejemplo])
        print(ejemplo, ':', errores or 'Entrada válida')
