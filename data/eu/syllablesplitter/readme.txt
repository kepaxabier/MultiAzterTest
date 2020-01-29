Lehenengo gauza FOMA instalatzea:
> sudo apt install foma-bin
> Leyendo lista de paquetes... Hecho
> Creando árbol de dependencias
> Leyendo la información de estado... Hecho
> Se instalarán los siguientes paquetes adicionales:
>   libfoma0
> Se instalarán los siguientes paquetes NUEVOS:
>   foma-bin libfoma0
>
>
> whereis foma
> foma: /usr/bin/foma

cd /media/datos/Dropbox/ikerkuntza/metrix-env/multilingual/data/eu/syllablesplitter
#Carga scripta
foma
foma[0]: source silabaEus.script
Opening file 'silabaEus.script'.
defined Obs: 956 bytes. 4 states, 21 arcs, 21 paths.
defined LiqGli: 345 bytes. 3 states, 4 arcs, 4 paths.
defined LowV: 329 bytes. 2 states, 3 arcs, 3 paths.
defined HighV: 287 bytes. 2 states, 2 arcs, 2 paths.
defined V: 413 bytes. 2 states, 5 arcs, 5 paths.
defined Nucleus: 477 bytes. 5 states, 9 arcs, 13 paths.
defined Onset: 1.2 kB. 6 states, 34 arcs, 110 paths.
defined Coda: 1.5 kB. 6 states, 54 arcs, 110 paths.
defined Syllable: 2.5 kB. 17 states, 107 arcs, 157300 paths.
defined MarkNuclei: 1.1 kB. 12 states, 40 arcs, Cyclic.
11.6 kB. 36 states, 680 arcs, Cyclic.
Writing to file silabaEus.fst.

# silabaEus.fst sortu ostean
# irten
foma[1]: quit


Ireki duzuen fitxategi horrek hainbat arau gordetzen ditu eta, behin 
beharrezko transduktorea sortutakoan, dena silabaEus.fst izeneko 
fitxategi batean gordetzen du (ikusi azken lerroa).

Oso sinplea da hori erabiltzea orain. Terminalean hau idatziz gero 
zuzenean erabil dezakezue:

flookup -ibx silabaEus.fst
karikaturismoaren
ka.ri.ka.tu.ris.mo.a.ren

Horrela zuzenean, interprete gisa, erabil dezakezue. Hitz bat sartu, 
enter sakatu, eta emaitza ikusi.

Baina, hitz asko silabatan banatu nahi badituzue, honakoa egin behar 
duzue. Suposa dezagun hitz zerrenda bat duen fitxategi bat duzuela 
(lerro saltoez banatuta), hitzZerrenda.txt izenarekin. Honakoa exekutatu 
behar duzue:

  > cat hitz_zerrenda.txt | flookup -ibx silabaEus.fst



https://fomafst.github.io/

Hemen daude FOMA erabiltzeko tutorial batzuk:
https://github.com/mhulden/foma/blob/master/foma/docs/simpleintro.md
https://fomafst.github.io/morphtut.html


