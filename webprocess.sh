#!/bin/bash
cd /var/www/html/aztertest/aztertest-virtenv
source bin/activate
echo -e $1 "\n" $2 "\n" $3 "\n" $4 "\n" $5 "\n" > parametros.txt
#1 /var/www/html/aztertest/uploads/3a7ff6b0c873b575f3b79d5b6365f876/english.doc.txt
#2 /var/www/html/aztertest/downloads/Aztertest_3a7ff6b0c873b575f3b79d5b6365f876.zip
#3 /var/www/html/aztertest/uploads/3a7ff6b0c873b575f3b79d5b6365f876
#4 -f
#5 basque
#python3 ./main.py $4 $1
#
#python3 ./multiaztertest.py -s -c -r -f /var/www/html/aztertest/aztertest-virtenv/textos/english.doc.txt -l english -m stanford -d /var/www/html/aztertest/aztertest-virtenv
python3 ./multiaztertest.py -s -c $4 -f $1 -l $5 -m stanford -d /var/www/html/aztertest/aztertest-virtenv
cd $3
zip -q -x ./*.txt -r $2 ./*
#Para salir
deactivate
cd ..
