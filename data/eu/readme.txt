import codecs
d = {}
with codecs.open('LB2014Maiztasunak_zenbakiakKenduta.csv',encoding='utf-8') as f:
    next(f)
    for line in f:
        (val, key) = line.split(",")
        d[key] = val

def zipf_frequency_eu(lemma):
    if d.get(lemma):
        return (int(d.get(lemma)))
    else:
        return 1

lemafrequency = zipf_frequency_eu(entry.lemma)
if lemafrequency <= difficult:

Nivel del alumno en euskara
En euskara, se obtuvo una lista de frecuencias por lema  del corpus  ``Lexikoaren Behatokia'' (http://lexikoarenbehatokia.euskaltzaindia.net/aurkezpena.htm) que a finales del 2014 tenía 41.773.391 palabras de Euskaltzaindia (mayoritariamente formado por periódicos, revistas y noticias de radio y televisión, aunque también se han incluido textos sobre literatura y educación en estos últimos años) 

En función de frecuencia de los lemas. 3 niveles:

Palabras de nivel alto $<=$ 6 lemas por 41.773.391 palabras son raras

No son son raras:
Palabras de nivel medio $<=$ 34 lemas por 41.773.391 palabras
Palabras de nivel bajo $<=$ 100.000 lemas por 41.773.391 palabras

