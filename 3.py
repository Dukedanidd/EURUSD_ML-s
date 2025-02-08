# Lenguajes y Automatas 2 Practica 3

# Analisis Semantico

'''
    Considera la siguiente regla gramatical para una declaración de variable:

    declaración -> tipo identificador ;

    Proponga acciones semánticas que se podrían añadir a esta regla para:

    Verificar si el identificador no ha sido declarado previamente.
    Añadir el identificador y su tipo a una tabla de símbolos.


'''

tabla = {}

def verificar_declaracion(tipo, identificador):
    # Aqui voy a verificar si ya existe el identificador en la tabla
    if identificador in tabla:
        print(f'Error: {identificador} ya ha sido declarado previamente.')
    else:
        # Si no existe lo creamos
        tabla[identificador] = tipo
        print(f'{identificador} declarado con tipo {tipo}')
        

verificar_declaracion('int', 'x')
verificar_declaracion('float', 'y')
verificar_declaracion('int', 'x')

print('\nTabla de simbolos:')
for identificador, tipo in tabla.items():
    print(f'{identificador}:{tipo}')
