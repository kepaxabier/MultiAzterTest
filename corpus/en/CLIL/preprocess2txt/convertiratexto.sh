#!/bin/bash
for fich in `ls *.doc`
do
	#For .doc use catdoc:
	catdoc $fich > $fich.txt
	#For .docx use docx2txt:
	#docx2txt foo.docx
	#Convertir 2 enters en uno, y eliminar los enters únicos:
	less $fich.txt | tr "\n" "@" | sed s/@@/\\n@/g > $fich.tmp
        less $fich.tmp | tr "@" " " > $fich.txt
        rm *.tmp
done
wc -w *.txt
ls *.txt | wc -l
file *.txt

