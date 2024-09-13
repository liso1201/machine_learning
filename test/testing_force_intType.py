import sys
sys.path.append('.')

from searchs.bayesian import force_intType

print("Test 1: Escalar inteiro")
print(force_intType(5))  # Esperado: 5

print("\nTest 2: Escalar float (convertido para inteiro)")
print(force_intType(5.8))  # Esperado: 5

print("\nTest 3: Escalar string representando um número inteiro")
print(force_intType("10"))  # Esperado: 10

print("\nTest 4: Escalar string representando um número float (convertido para inteiro)")
print(force_intType("3.9"))  # Esperado: 3

print("\nTest 5: Lista de inteiros")
print(force_intType([1, 2, 3, 4]))  # Esperado: array([1, 2, 3, 4])

print("\nTest 6: Lista de floats (convertido para inteiro)")
print(force_intType([1.1, 2.5, 3.8]))  # Esperado: array([1, 2, 3])

print("\nTest 7: Lista de strings representando números inteiros")
print(force_intType(["1", "2", "3"]))  # Esperado: array([1, 2, 3])

print("\nTest 8: Lista de strings representando números floats (convertido para inteiro)")
print(force_intType(["4.5", "5.9", "6.3"]))  # Esperado: array([4, 5, 6])

print("\nTest 9: Tupla de complexos")
print(force_intType((7+5j, 8+5j, 9+5j)))  # Esperado: array([7, 8, 9])

print("\nTest 10: Set de floats (convertido para inteiro)")
print(force_intType({10.1, 11.9, 12.6}))  # Esperado: array([10, 11, 12])

print("\nTest 10: strings (não convertido para inteiro)")
print(force_intType(["mean_squared_error", "namename", "name.name", "name+jname", "name"]))  # Esperado: array([10, 11, 12])