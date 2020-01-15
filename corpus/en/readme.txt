El corpus se ha obtenido de:
Corpusa https://zenodo.org/record/1219041#.XP-gnib9Brk helbidetik deskagatuta 2019ko ekainaren 11n.
Ref para el corpus: Vajjala, Sowmya and Lucic, Ivana, "OneStopEnglish corpus: A new corpus for automatic readability assessment and text simplification"(2018).English Conference Papers, Posters and Proceedings. 

Estructura de las carpetas:
original -contiene los ficheros originales los nombres de ficheros no tienen espacios en blanco, la primera fila contiene el nivel:Elementary, Adv...
cleaned_and_withoutfirstrow- el nombre de los ficheros ha sido modificado sustituyendo los blancos por guiones, y además se ha eliminado la primera fila que indica el nivel 
Para dividir en train/test: 
Hemos tenido en cuentas los siguientes criterios
-Dividir train/test en 80/20
-Que Amazon-adv.txt,Amazon-ele.txt,Amazon-int.txt cae sobre la misma carpeta en este caso Test. Esto nos permite eliminar textos malos en 3 niveles. Si queremos 2 niveles solo hace falta quitar los int.txt
Poder comparar con otros sistemas, no dependiendo de la aleatoriedad.
-Como tenemos 189 textos en los 3 niveles, el 20% de 189 sería 37 aprox. Creamos 2 carpetas Test y Train en :


cleaned_and_withoutfirstrow-> Train- El resto 152 ficheros siguientes en Train.
                              Test -para que metas en Test los 37 primeros ficheros alfabéticos de los 3 niveles=los primeros 111 ficheros en orden alfabético 

De corpus/en/3levels/cleaned_and_withoutfirstrow/[Test|Train]/*.txt se copian a corpus/en/3levels/[Test|Train]/[Adv-Txt|Ele-Txt|Int-Txt]/*.txt 

