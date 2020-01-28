#!/bin/bash

FILES=/home/igonzalez010/Dokumentuak/Itziardena/IKERKETA/corpusak/dokumentuak_garaterm_apunteak_2019_06/*.html

for file in $FILES
do
    #echo $(html2text $file)
    html2text $file > $file.txt
done
