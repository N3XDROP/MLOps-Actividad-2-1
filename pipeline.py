import argparse
import subprocess
import sys
from pathlib import Path

from app import entrenar

BASE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description='Automatización del pipeline MLOps')
    parser.add_argument('--data', type=Path, default=BASE / 'data' / 'Loan_Default.csv')
    parser.add_argument('--output', type=Path, default=BASE / 'resultados')
    args = parser.parse_args()
    print('1. Validación del contrato y pruebas unitarias', flush=True)
    subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', str(BASE / 'tests'), '-v'], cwd=BASE, check=True)
    if not args.data.is_file():
        parser.error(f'No se encuentra {args.data}. Consulte README.md.')
    print('2. Validación de datos, entrenamiento y evaluación en prueba', flush=True)
    entrenar(args.data, args.output)
    print('3. Pipeline finalizado. Resultados:', args.output)


if __name__ == '__main__':
    main()
