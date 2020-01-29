#!/bin/bash
# lerrosaltoak kentzeko

for file in `ls *.txt`
do
    less $file | sed ':a;N;$!ba;s/\n/ /g' > kk.tmp
    less kk.tmp | sed  -e 's/\[.*\]/\ /' > $file.garbia.txt
done
rm kk.tmp

