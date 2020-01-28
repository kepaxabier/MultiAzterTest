#!/bin/bash
#hemen dagoz hautatuak
cd /media/datos/Dropbox/ikerkuntza/metrix-env/multilingual/corpus/eu/Garaterm
mkdir hautatuakAcademic
for fich in `ls Garaterm200testuakOndo/Academic_*.garbia.txt`
do
	fich2=`echo $fich | cut -f5- -d_`
	echo $fich2
        fich3=`echo $fich2 | cut -f1-2 -d.`
        echo $fich3
        cp dokumentuak_garaterm_apunteak_2019_06_DENAK/$fich3* hautatuakAcademic 
done
#Garaterm_23_AApraiz_5572_150_11._subdukzioa.doc.html.txt.garbia.txt
#OAbalde_4339_258_2.ikasgaia.doc.html.txt.garbia.txt
#OAbalde_4339_258_2.ikasgaia.doc.html
mkdir hautatuakGaraterm
for fich in `ls Garaterm200testuakOndo/Garaterm_*.garbia.txt`
do
	echo $fich
        fich2=`echo $fich | cut -f3- -d_`
	echo $fich2
        fich3=`echo $fich2 | cut -f1-2 -d.`
        echo $fich3
        cp dokumentuak_garaterm_apunteak_2019_06_DENAK/$fich3* hautatuakGaraterm
done
#he quitado estos 2, para que sean 200
rm /media/datos/Dropbox/ikerkuntza/metrix-env/multilingual/corpus/eu/Garaterm/Garaterm200testuakOndo/AGutierrez_2478_530_post_produkzioa__protools_.docx.html.txt
rm /media/datos/Dropbox/ikerkuntza/metrix-env/multilingual/corpus/eu/Garaterm/Garaterm200testuakOndo/ECompains_4291_858_zt_3_ikasgaia.docx.html.txt.garbia.txt
#estos 3 sueltos hautatuakbanakoak
mkdir hautatuakbanakoak
#/media/datos/Dropbox/ikerkuntza/metrix-env/multilingual/corpus/eu/Garaterm/Garaterm200testuakOndo/IOrdenana_4772_813_27._behin-behineko_exeskuzioa.doc.html.txt
cp dokumentuak_garaterm_apunteak_2019_06_DENAK/IOrdenana_4772_813_27._behin-behineko_exeskuzioa.doc.html hautatuakbanakoak
#/media/datos/Dropbox/ikerkuntza/metrix-env/multilingual/corpus/eu/Garaterm/Garaterm200testuakOndo/NIriondo_5830_244_kt-nahasketa_baten_dentsitatea_garbi.docx.html.txt
cp dokumentuak_garaterm_apunteak_2019_06_DENAK/NIriondo_5830_244_kt-nahasketa_baten_dentsitatea_garbi.docx.html hautatuakbanakoak
#/media/datos/Dropbox/ikerkuntza/metrix-env/multilingual/corpus/eu/Garaterm/Garaterm200testuakOndo/OAbalde_4360_996_17.ikasgaia.rtf.html.txt.garbia.txt
cp dokumentuak_garaterm_apunteak_2019_06_DENAK/OAbalde_4360_996_17.ikasgaia.rtf.html hautatuakbanakoak
mkdir hautatuak200
cp hautatuakAcademic/* hautatuak200
cp hautatuakGaraterm/* hautatuak200
cp hautatuakbanakoak/* hautatuak200
#Al realizar esto hay 198, es decir hay 2 ficheros repetidos en diferentes carpetas
#Aprobecho los otros 2, que había elininado paraq completar
cp dokumentuak_garaterm_apunteak_2019_06_DENAK/AGutierrez_2478_530_post_produkzioa__protools_.docx.html hautatuak200
cp dokumentuak_garaterm_apunteak_2019_06_DENAK/ECompains_4291_858_zt_3_ikasgaia.docx.html hautatuak200

#Convertir html en txt
#
cd hautatuak200
mkdir convertidoatxt
html2doc 



#ls
#dokumentuak_garaterm_apunteak_2019_06_DENAK/bcpsigai_3671_488_9._gaia-apunteak_ikt_azken_bertsioa_ondo.docx.html.txt.garbia.txt 
#dokumentuak_garaterm_apunteak_2019_06_DENAK/bcpsigai_3671_488_9._gaia-apunteak_ikt_azken_bertsioa_ondo.docx.html
#hemen dagoz html-ak
#cd dokumentuak_garaterm_apunteak_2019_06_DENAK

#bcpsigai_3671_488_9._gaia-apunteak_ikt_azken_bertsioa_ondo.docx.html
#for fich in `ls *.doc`
#do
	#For .doc use catdoc:
#	catdoc $fich > $fich.txt
mkdir convertidoadoc
mkdir convertidoatxt
htmldoc ECompains_4291_858_zt_3_ikasgaia.docx.html > convertidoadoc/prueba.doc
catdoc convertidoadoc/prueba.doc > convertidoatxt/prueba.txt
docx2txt convertidoadoc/prueba.doc > convertidoatxt/prueba.txt
	#For .docx use docx2txt:
	#docx2txt foo.docx
	#Convertir 2 enters en uno, y eliminar los enters únicos:
#	less $fich.txt | tr "\n" "@" | sed s/@@/\\n@/g > $fich.tmp
#        less $fich.tmp | tr "@" " " > $fich.txt
#        rm *.tmp
#done
#wc -w *.txt
#ls *.txt | wc -l
#file *.txt

