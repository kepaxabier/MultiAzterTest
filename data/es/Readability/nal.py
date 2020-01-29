# Original script: https://github.com/Axiacore/number-to-letters

# Changes from the original: just adapted to python 3.

# The MIT License (MIT)

# Copyright (c) 2014 AxiaCore

#Permission is hereby granted, free of charge, to any person obtaining a copy
#of this software and associated documentation files (the "Software"), to deal
#in the Software without restriction, including without limitation the rights
#to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#copies of the Software, and to permit persons to whom the Software is
#furnished to do so, subject to the following conditions:

#The above copyright notice and this permission notice shall be included in all
#copies or substantial portions of the Software.

#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#SOFTWARE.


UNIDADES = (
    '',
    'UN ',
    'DOS ',
    'TRES ',
    'CUATRO ',
    'CINCO ',
    'SEIS ',
    'SIETE ',
    'OCHO ',
    'NUEVE ',
    'DIEZ ',
    'ONCE ',
    'DOCE ',
    'TRECE ',
    'CATORCE ',
    'QUINCE ',
    'DIECISEIS ',
    'DIECISIETE ',
    'DIECIOCHO ',
    'DIECINUEVE ',
    'VEINTE '
)

DECENAS = (
    'VENTI',
    'TREINTA ',
    'CUARENTA ',
    'CINCUENTA ',
    'SESENTA ',
    'SETENTA ',
    'OCHENTA ',
    'NOVENTA ',
    'CIEN '
)

CENTENAS = (
    'CIENTO ',
    'DOSCIENTOS ',
    'TRESCIENTOS ',
    'CUATROCIENTOS ',
    'QUINIENTOS ',
    'SEISCIENTOS ',
    'SETECIENTOS ',
    'OCHOCIENTOS ',
    'NOVECIENTOS '
)

UNITS = (
        ('',''),
        ('MIL ','MIL '),
        ('MILLON ','MILLONES '),
        ('MIL MILLONES ','MIL MILLONES '),
        ('BILLON ','BILLONES '),
        ('MIL BILLONES ','MIL BILLONES '),
        ('TRILLON ','TRILLONES '),
        ('MIL TRILLONES','MIL TRILLONES'),
        ('CUATRILLON','CUATRILLONES'),
        ('MIL CUATRILLONES','MIL CUATRILLONES'),
        ('QUINTILLON','QUINTILLONES'),
        ('MIL QUINTILLONES','MIL QUINTILLONES'),
        ('SEXTILLON','SEXTILLONES'),
        ('MIL SEXTILLONES','MIL SEXTILLONES'),
        ('SEPTILLON','SEPTILLONES'),
        ('MIL SEPTILLONES','MIL SEPTILLONES'),
        ('OCTILLON','OCTILLONES'),
        ('MIL OCTILLONES','MIL OCTILLONES'),
        ('NONILLON','NONILLONES'),
        ('MIL NONILLONES','MIL NONILLONES'),
        ('DECILLON','DECILLONES'),
        ('MIL DECILLONES','MIL DECILLONES'),
        ('UNDECILLON','UNDECILLONES'),
        ('MIL UNDECILLONES','MIL UNDECILLONES'),
        ('DUODECILLON','DUODECILLONES'),
        ('MIL DUODECILLONES','MIL DUODECILLONES'),
)


MONEDAS = (
    {'country': 'Colombia', 'currency': 'COP', 'singular': 'PESO COLOMBIANO', 'plural': 'PESOS COLOMBIANOS', 'symbol': '$'},
    {'country': 'Estados Unidos', 'currency': 'USD', 'singular': 'DÓLAR', 'plural': 'DÓLARES', 'symbol': 'US$'},
    {'country': 'Europa', 'currency': 'EUR', 'singular': 'EURO', 'plural': 'EUROS', 'symbol': '€', 'decimalsingular':'Céntimo','decimalplural':'Céntimos'},
    {'country': 'México', 'currency': 'MXN', 'singular': 'PESO MEXICANO', 'plural': 'PESOS MEXICANOS', 'symbol': '$'},
    {'country': 'Perú', 'currency': 'PEN', 'singular': 'NUEVO SOL', 'plural': 'NUEVOS SOLES', 'symbol': 'S/.'},
    {'country': 'Reino Unido', 'currency': 'GBP', 'singular': 'LIBRA', 'plural': 'LIBRAS', 'symbol': '£'}
)
# Para definir la moneda me estoy basando en los código que establece el ISO 4217
# Decidí poner las variables en inglés, porque es más sencillo de ubicarlas sin importar el país
# Si, ya sé que Europa no es un país, pero no se me ocurrió un nombre mejor para la clave.


def hundreds_word(number):
    """Converts a positive number less than a thousand (1000) to words in Spanish
    Args:
        number (int): A positive number less than 1000
    Returns:
        A string in Spanish with first letters capitalized representing the number in letters
    Examples:
        >>> to_word(123)
        'Ciento Ventitres'
    """
    converted = ''
    if not (0 < number < 1000):
        return 'No es posible convertir el numero a letras'

    number_str = str(number).zfill(9)
    cientos = number_str[6:]


    if(cientos):
        if(cientos == '001'):
            converted += 'UN '
        elif(int(cientos) > 0):
            converted += '%s ' % __convert_group(cientos)


    return converted.title().strip()



def __convert_group(n):
    """Turn each group of numbers into letters"""
    output = ''

    if(n == '100'):
        output = "CIEN "
    elif(n[0] != '0'):
        output = CENTENAS[int(n[0]) - 1]

    k = int(n[1:])
    if(k <= 20):
        output += UNIDADES[k]
    else:
        if((k > 30) & (n[2] != '0')):
            output += '%sY %s' % (DECENAS[int(n[1]) - 2], UNIDADES[int(n[2])])
        else:
            output += '%s%s' % (DECENAS[int(n[1]) - 2], UNIDADES[int(n[2])])

    return output

def to_word(number, mi_moneda=None):

    """Converts a positive number less than:
    (999999999999999999999999999999999999999999999999999999999999999999999999)
    to words in Spanish
    Args:
        number (int): A positive number less than specified above
        mi_moneda(str,optional): A string in ISO 4217 short format
    Returns:
        A string in Spanish with first letters capitalized representing the number in letters
    Examples:
        >>> number_words(53625999567)
        'Cincuenta Y Tres Mil Seiscientos Venticinco Millones Novecientos Noventa Y Nueve Mil Quinientos Sesenta Y Siete'
    
        >>>> number_words(1481.01, 'EUR')
        'Mil Cuatrocientos Ochenta Y Un Euros con Un Céntimo'
    """
    if mi_moneda != None:
        try:
            moneda = filter(lambda x: x['currency'] == mi_moneda, MONEDAS).next()
            if int(number) == 1:
                entero = moneda['singular']
            else:
                entero = moneda['plural']
                if round(float(number) - int(number), 2) == float(0.01):
                    fraccion = moneda['decimalsingular']
                else:
                    fraccion = moneda['decimalplural']

        except:
            return "Tipo de moneda inválida"
    else:
        entero = ""
        fraccion = ""

    human_readable = []
    human_readable_decimals = []
    num_decimals ='{:,.2f}'.format(round(number,2)).split('.') #Sólo se aceptan 2 decimales
    num_units = num_decimals[0].split(',')
    num_decimals = num_decimals[1].split(',')
    #print num_units
    for i,n in enumerate(num_units):
        if int(n) != 0:
            words = hundreds_word(int(n))
            units = UNITS[len(num_units)-i-1][0 if int(n) == 1 else 1]
            human_readable.append([words,units])
    for i,n in enumerate(num_decimals):
        if int(n) != 0:
            words = hundreds_word(int(n))
            units = UNITS[len(num_decimals)-i-1][0 if int(n) == 1 else 1]
            human_readable_decimals.append([words,units])

    #filtrar MIL MILLONES - MILLONES -> MIL - MILLONES
    for i,item in enumerate(human_readable):
        try:
            if human_readable[i][1].find(human_readable[i+1][1]):
                human_readable[i][1] = human_readable[i][1].replace(human_readable[i+1][1],'')
        except IndexError:
            pass
    human_readable = [item for sublist in human_readable for item in sublist]
    human_readable.append(entero)
    for i,item in enumerate(human_readable_decimals):
        try:
            if human_readable_decimals[i][1].find(human_readable_decimals[i+1][1]):
                human_readable_decimals[i][1] = human_readable_decimals[i][1].replace(human_readable_decimals[i+1][1],'')
        except IndexError:
            pass
    human_readable_decimals = [item for sublist in human_readable_decimals for item in sublist]
    human_readable_decimals.append(fraccion)
    sentence = ' '.join(human_readable).replace('  ',' ').title().strip()
    if sentence[0:len('un mil')] == 'Un Mil':
        sentence = 'Mil' + sentence[len('Un Mil'):]
    if num_decimals != ['00']:
        sentence = sentence + ' con ' + ' '.join(human_readable_decimals).replace('  ',' ').title().strip()
    return sentence
