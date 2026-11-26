import unittest
from validacion import CONTRATO, validar_entrada, convertir_fila


class TestValidacion(unittest.TestCase):
    def setUp(self):
        self.datos = CONTRATO['ejemplo_valido'].copy()

    def test_valida(self):
        self.assertEqual(validar_entrada(self.datos), [])

    def test_invalida(self):
        self.assertEqual(len(validar_entrada(CONTRATO['ejemplo_invalido'])), 5)

    def test_nulo(self):
        self.datos['income'] = None
        self.assertTrue(validar_entrada(self.datos))

    def test_tipo(self):
        self.datos['Credit_Score'] = '710'
        self.assertTrue(validar_entrada(self.datos))

    def test_booleano(self):
        self.datos['loan_amount'] = True
        self.assertTrue(validar_entrada(self.datos))

    def test_rango(self):
        self.datos['term'] = 481
        self.assertTrue(validar_entrada(self.datos))

    def test_nan(self):
        self.datos['income'] = float('nan')
        self.assertTrue(validar_entrada(self.datos))

    def test_faltante(self):
        del self.datos['term']
        self.assertTrue(validar_entrada(self.datos))

    def test_adicional(self):
        self.datos['Status'] = 0
        self.assertTrue(validar_entrada(self.datos))

    def test_conversion_csv(self):
        fila = {k: str(v) for k, v in self.datos.items()}
        fila['term'] = '360.0'
        self.assertEqual(validar_entrada(convertir_fila(fila)), [])

    def test_categoria(self):
        self.datos['loan_type'] = 'type9'
        self.assertTrue(validar_entrada(self.datos))
