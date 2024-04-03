import pandas as pd

filename = 'ocurrencias_parseado.csv'
chunksize = 1000

for chunk in pd.read_csv(filename, chunksize=chunksize):
    # chunk is a DataFrame. To "process" the rows in the chunk:
    for col in chunk.columns:
        print(col)
    pd.DataFrame(chunk).head().to_csv('fichero.csv')
    break



# ficheros = gb.encontrar_ficheros('.')

# print('Que fichero quieres leer?')
# i = 0
# for fichero in ficheros:
#     print(f'[ {i}] {fichero}')
#     i+=1
# numeros_fichero = input("Coloca el numero(1,2,3): ")

# for numero in numeros_fichero.split(','):
#     numero = int(numero)
#     try:
#         df = pd.read_csv(ficheros[numero])
#         print(df.head())
#     except:
#         print(f'{ficheros[numero]} no es un fichero')