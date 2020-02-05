#!/bin/bash
#hemen dagoz hautatuak
cd /media/datos/Dropbox/ikerkuntza/metrix-env/multilingual/corpus/eu/Garaterm
rm -rf hautatuakAcademic
mkdir hautatuakAcademic
for fich in `ls Garaterm200testuakOndo/Academic_*.garbia.txt`
do
	fich2=`echo $fich | cut -f5- -d_`
	echo $fich2
        fich3=`echo $fich2 | cut -f1-2 -d.`
        echo $fich3
        cp dokumentuak_garaterm_apunteak_2019_06_DENAK/$fich3* hautatuakAcademic 
done

rm -rf hautatuakGaraterm
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

#Necesit 4 sueltos para hautatuakbanakoak
rm -rf hautatuakbanakoak
mkdir hautatuakbanakoak
cp dokumentuak_garaterm_apunteak_2019_06_DENAK/NIriondo_5830_244_kt-nahasketa_baten_dentsitatea_garbi.docx.html hautatuakbanakoak
cp dokumentuak_garaterm_apunteak_2019_06_DENAK/OAbalde_4360_996_17.ikasgaia.rtf.html hautatuakbanakoak
cp dokumentuak_garaterm_apunteak_2019_06_DENAK/AAldezabal_5075_57_landare-histologia.doc.html hautatuakbanakoak
cp dokumentuak_garaterm_apunteak_2019_06_DENAK/AApraiz_5079_686_2._gerriko_orogenikoak.doc.html hautatuakbanakoak

rm -rf hautatuak200
mkdir hautatuak200
cp hautatuakAcademic/* hautatuak200
cp hautatuakGaraterm/* hautatuak200
cp hautatuakbanakoak/* hautatuak200


#Convertir html en txt
#
#cd hautatuak200
#mkdir convertidoatxt
#html2doc 



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
#mkdir convertidoadoc
#mkdir convertidoatxt
#htmldoc ECompains_4291_858_zt_3_ikasgaia.docx.html > convertidoadoc/prueba.doc
#catdoc convertidoadoc/prueba.doc > convertidoatxt/prueba.txt
#docx2txt convertidoadoc/prueba.doc > convertidoatxt/prueba.txt
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

