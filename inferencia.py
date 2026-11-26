import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from validacion import CAMPOS, CONTRATO, validar_entrada

BASE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description='Validación e inferencia de una solicitud')
    parser.add_argument('--entrada', type=Path, help='Archivo JSON con una solicitud')
    parser.add_argument('--modelo', type=Path, default=BASE / 'resultados' / 'modelo.joblib')
    args = parser.parse_args()
    datos = json.loads(args.entrada.read_text(encoding='utf-8')) if args.entrada else CONTRATO['ejemplo_valido']
    errores = validar_entrada(datos)
    if errores:
        print(json.dumps({'entrada_valida': False, 'errores': errores}, ensure_ascii=False, indent=2))
        raise SystemExit(1)
    if not args.modelo.is_file():
        parser.error('Ejecute primero python pipeline.py')
    modelo = joblib.load(args.modelo)
    X = pd.DataFrame([datos], columns=list(CAMPOS))
    print(json.dumps({'entrada_valida': True, 'clase': int(modelo.predict(X)[0]),
                      'probabilidad_modelo_clase_1': float(modelo.predict_proba(X)[0, 1])}, indent=2))


if __name__ == '__main__':
    main()
