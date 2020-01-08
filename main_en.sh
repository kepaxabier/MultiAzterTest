#con python
#cd /media/datos/Dropbox/ikerkuntza/metrix-env/
#source bin/activate
#cd /media/datos/Dropbox/ikerkuntza/metrix-env/sepln2020/aztertestnlpcube
#python3 Classifier.py

function wekainstalatuubuntun()
{
sudo apt-get update
sudo apt-get install weka
sudo apt-get install libsvm-java
}
function weka()
{
#con weka:https://machinelearningmastery.com/how-to-run-your-first-classifier-in-weka/
#/media/datos/Dropbox/ikerkuntza/UZ/programak/logisticregression_uc
#https://datascience.stackexchange.com/questions/40900/whats-the-difference-between-sklearn-f1-score-micro-and-weighted-for-a-mult/40904#40904
WEKA_PATH=/home/kepa/weka-3-8-3
export CLASSPATH=$CLASSPATH:/home/kepa/weka-3-8-3/weka.jar #:/usr/share/java/libsvm.jar:/usr/share/java
CP="$CLASSPATH:/usr/share/java/"
cd /media/datos/Dropbox/ikerkuntza/metrix-env/sepln2020/aztertestnlpcube/dataset_aztertest_full
java -cp $CP -Xmx1024m weka.gui.explorer.Explorer
}

function obtenerdatoscohmetrix()
{
#dir=/media/datos/Dropbox/ikerkuntza/metrix-env/sepln2020/corpus/en/3levels/AztertestTrainTest #/Train/train_aztertest.csv
dir=/media/datos/Dropbox/ikerkuntza/metrix-env/sepln2020/aztertestnlpcube
dirtrain=$dir/Train
dirtest=$dir/Test
cd $dir
python cvsDatasetcohmetrixtest.py
python cvsDatasetcohmetrixtrain.py
tail -n +2 "$dirtrain/results_cohmetrix.csv" > "$dirtrain/results_cohmetrix_cab.csv"
tail -n +2 "$dirtest/results_cohmetrix.csv" > "$dirtest/results_cohmetrix_cab.csv"
#Guardar como arff
WEKA_PATH=/home/kepa/weka-3-8-3
export CLASSPATH=$CLASSPATH:/home/kepa/weka-3-8-3/weka.jar #:/usr/share/java/libsvm.jar:/usr/share/java
CP="$CLASSPATH:/usr/share/java/"
java -cp $CP -Xmx1024m weka.core.converters.CSVLoader $dirtrain/results_cohmetrix_cab.csv > $dirtrain/results_cohmetrix_cab.arff
java -cp $CP -Xmx1024m weka.core.converters.CSVLoader $dirtest/results_cohmetrix_cab.csv > $dirtest/results_cohmetrix_cab.arff

#java -cp $CP -Xmx1024m weka.core.converters.CSVLoader $dirtrain/train_aztertestadv.csv > $dirtrain/train_aztertestadv.arff
#java -cp $CP -Xmx1024m weka.core.converters.CSVLoader $dirtrain/train_aztertestele.csv > $dirtrain/train_aztertestele.arff
#java -cp $CP -Xmx1024m weka.core.converters.CSVLoader $dirtrain/train_aztertestint.csv > $dirtrain/train_aztertestint.arff
#java -cp $CP -Xmx1024m weka.core.converters.CSVLoader $dirtest/test_aztertest.csv > $dirtest/test_aztertest.arff
}
function pruebaobtenerdatosmultiaztertest()
{
cd /media/datos/Dropbox/ikerkuntza/metrix-env
source bin/activate
cd /media/datos/Dropbox/ikerkuntza/metrix-env/multilingual
python3 multiaztertest.py -s -c -r -f  Loterry-adv.txt -l english -m stanford
}
function obtenerdatosmultiaztertest()
{
modelo=$1
dir="/media/datos/Dropbox/ikerkuntza/metrix-env/multilingual/corpus/en" #/[Test|Train]/[Adv-Txt|Ele-Txt|Int-Txt]/*.txt "
#LEE README.TXT DE corpus/en
#Sarrera:corpus/en/cleaned_and_withoutfirstrow/[Test|Train]/*.txt -> contiene los ficheros cuyo nombre ha sido modificado, los blancos por guiones, y se ha eliminado la primera fila que indica el nivel 
#voy a dividir los ficheros en 5 en 5
cd $dir/Train/Adv-Txt/
i=1
j=0
mkdir $j
for fitx in `ls *.txt`
do
    d=$(echo " scale=0; $i%5" | bc -l)
    if [ $d -eq 0 ]
    then
           j=$i
           mkdir $j
    fi
    cp $fitx $j
    i=$(echo "$i+1" | bc -l)
done
cd $dir/Train/Ele-Txt/
i=1
j=0
mkdir $j
for fitx in `ls *.txt`
do
   d=$(echo " scale=0; $i%5" | bc -l)
   if [ $d -eq 0 ]
   then
       j=$i
       mkdir $j
   fi
   cp $fitx $j
   i=$(echo "$i+1" | bc -l)
done
cd $dir/Train/Int-Txt/
i=1
j=0
mkdir $j
for fitx in `ls *.txt`
do
   d=$(echo " scale=0; $i%5" | bc -l)
   if [ $d -eq 0 ]
   then
       j=$i
       mkdir $j
   fi
   cp $fitx $j
   i=$(echo "$i+1" | bc -l)
done
cd $dir/Test/Adv-Txt/
i=1
j=0
mkdir $j
for fitx in `ls *.txt`
do
    d=$(echo " scale=0; $i%5" | bc -l)
    if [ $d -eq 0 ]
    then
        j=$i
        mkdir $j
    fi
       cp $fitx $j
       i=$(echo "$i+1" | bc -l)
done
cd $dir/Test/Ele-Txt/
i=1
j=0
mkdir $j
for fitx in `ls *.txt`
do
    d=$(echo " scale=0; $i%5" | bc -l)
    if [ $d -eq 0 ]
    then
      j=$i
      mkdir $j
    fi
    cp $fitx $j
    i=$(echo "$i+1" | bc -l)
done
cd $dir/Test/Int-Txt/
i=1
j=0
mkdir $j
for fitx in `ls *.txt`
do
    d=$(echo " scale=0; $i%5" | bc -l)
    if [ $d -eq 0 ]
    then
        j=$i
        mkdir $j
    fi
    cp $fitx $j
    i=$(echo "$i+1" | bc -l)
done
cd /media/datos/Dropbox/ikerkuntza/metrix-env
source bin/activate
cd /media/datos/Dropbox/ikerkuntza/metrix-env/multilingual
for i in `seq 0 5 150`
do
   python3 multiaztertest.py -s -c -r -f  $dir/Train/Adv-Txt/$i/*.txt -l english -m $modelo
   python3 multiaztertest.py -s -c -r -f  $dir/Train/Int-Txt/$i/*.txt -l english -m $modelo
   python3 multiaztertest.py -s -c -r -f  $dir/Train/Ele-Txt/$i/*.txt -l english -m $modelo	
done
for i in `seq 0 5 35`
do
   python3 multiaztertest.py -s -c -r -f  $dir/Test/Adv-Txt/$i/*.txt -l english -m $modelo
   python3 multiaztertest.py -s -c -r -f  $dir/Test/Int-Txt/$i/*.txt -l english -m $modelo
   python3 multiaztertest.py -s -c -r -f  $dir/Test/Ele-Txt/$i/*.txt -l english -m $modelo	
done
#input:$dir/Train/[Adv-Txt|Ele-Txt|Int-Txt]/$i/results/full_results_aztertest.csv 
#output:$dir/Train/[Adv-Txt|Ele-Txt|Int-Txt]/results/full_results_aztertest.csv
#coge la primera fila, cabecera
mkdir -p $dir/Train/Adv-Txt/results
head -1 $dir/Train/Adv-Txt/0/results/full_results_aztertest.csv > $dir/Train/Adv-Txt/results/full_results_aztertest.csv
#Quita la cabecera
for i in `seq 0 5 150`
do
	tail -n +2 $dir/Train/Adv-Txt/$i/results/full_results_aztertest.csv >> $dir/Train/Adv-Txt/results/full_results_aztertest.csv
done
mkdir -p $dir/Train/Int-Txt/results
head -1 $dir/Train/Int-Txt/0/results/full_results_aztertest.csv > $dir/Train/Int-Txt/results/full_results_aztertest.csv
for i in `seq 0 5 150`
do
     	tail -n +2 $dir/Train/Int-Txt/$i/results/full_results_aztertest.csv >> $dir/Train/Int-Txt/results/full_results_aztertest.csv
done
mkdir -p $dir/Train/Ele-Txt/results
head -1 $dir/Train/Ele-Txt/0/results/full_results_aztertest.csv > $dir/Train/Ele-Txt/results/full_results_aztertest.csv
for i in `seq 0 5 150`
do
     	tail -n +2 $dir/Train/Ele-Txt/$i/results/full_results_aztertest.csv >> $dir/Train/Ele-Txt/results/full_results_aztertest.csv
done
mkdir -p $dir/Test/Adv-Txt/results
head -1 $dir/Test/Adv-Txt/0/results/full_results_aztertest.csv > $dir/Test/Adv-Txt/results/full_results_aztertest.csv
for i in `seq 0 5 35`
do
     	tail -n +2 $dir/Test/Adv-Txt/$i/results/full_results_aztertest.csv >> $dir/Test/Adv-Txt/results/full_results_aztertest.csv
done
mkdir -p $dir/Test/Int-Txt/results
head -1 $dir/Test/Int-Txt/0/results/full_results_aztertest.csv > $dir/Test/Int-Txt/results/full_results_aztertest.csv
for i in `seq 0 5 35`
do
     	tail -n +2 $dir/Test/Int-Txt/$i/results/full_results_aztertest.csv >> $dir/Test/Int-Txt/results/full_results_aztertest.csv
done
mkdir -p $dir/Test/Ele-Txt/results
head -1 $dir/Test/Ele-Txt/0/results/full_results_aztertest.csv > $dir/Test/Ele-Txt/results/full_results_aztertest.csv
for i in `seq 0 5 35`
do
     	tail -n +2 $dir/Test/Ele-Txt/$i/results/full_results_aztertest.csv >> $dir/Test/Ele-Txt/results/full_results_aztertest.csv
done

cd /media/datos/Dropbox/ikerkuntza/metrix-env/multilingual/corpus/en
#menu:  1: Anadir label de clase a train y test dataset : output:dataset_aztertest_full/train_aztertest.csv y test_aztertest.csv
python3 /media/datos/Dropbox/ikerkuntza/metrix-env/multilingual/corpus/en/aztertest_dataset.py
}
function wekarekinprobatu()
{
#Vete al menu 6 Weka y convierte  dataset_aztertest_full/train_aztertest.csv y test_aztertest.csv a arff-s
WEKA_PATH=/home/kepa/weka-3-8-3
export CLASSPATH=$CLASSPATH:/home/kepa/weka-3-8-3/weka.jar #:/usr/share/java/libsvm.jar:/usr/share/java
CP="$CLASSPATH:/usr/share/java/"
#arrancar weka online
#java -cp $CP -Xmx1024m weka.gui.explorer.Explorer
#convertir csv en arff
dir="/media/datos/Dropbox/ikerkuntza/metrix-env/multilingual/corpus/en/dataset_aztertest_full"
java -cp $CP -Xmx1024m --add-opens=java.base/java.lang=ALL-UNNAMED weka.core.converters.CSVLoader $dir/train_aztertest.csv > $dir/train_aztertest.arff
java -cp $CP -Xmx1024m --add-opens=java.base/java.lang=ALL-UNNAMED weka.core.converters.CSVLoader $dir/test_aztertest.csv > $dir/test_aztertest.arff
echo "Emaitza:$dir/train_aztertest.arff eta $dir/test_aztertest.arff"
}
function generarmodelopython()
{
cd /media/datos/Dropbox/ikerkuntza/metrix-env
source bin/activate
cd /media/datos/Dropbox/ikerkuntza/metrix-env/multilingual/corpus/en
echo "2: Realizar 10-Fold CV con feature selection y busqueda del mejor algoritmo y sus meta-parametros, finalmente guardar el mejor modelo y el mejor selector de características"
python3 /media/datos/Dropbox/ikerkuntza/metrix-env/multilingual/corpus/en/aztertest_dataset.py
}

function testearmodelopython()
{
cd /media/datos/Dropbox/ikerkuntza/metrix-env
source bin/activate
cd /media/datos/Dropbox/ikerkuntza/metrix-env/multilingual/corpus/en
echo "3: Se carga el modelo y el selector mediante el método load(), se aplica el selector de atributos a los datos mediante el método transform() y se realiza la predicción utilizando el método predict"
python3 /media/datos/Dropbox/ikerkuntza/metrix-env/multilingual/corpus/en/aztertest_dataset.py
}
function fin()
{
	echo -e "¿Quieres salir del programa?(S/N)\n"
        read respuesta
	if [ $respuesta == "N" ] 
		then
			opcionmenuppal=0
		fi	
}
### Main ###
modelo=$1
opcionmenuppal=0
while test $opcionmenuppal -ne 8
do
	#Muestra el menu
       	echo -e "1 Obtener datos cohmetrix \n"
	echo -e "2 Obtener datos multiaztertest \n"
        echo -e "3 pruebaobtenerdatosmultiaztertest \n"
	echo -e "4 wekarekinprobatu \n"
	echo -e "5 Python: Realizar 10-Fold CV con feature selection y busqueda del mejor algoritmo y sus meta-parametros, finalmente guardar el mejor modelo y el mejor selector de características \n"
	echo -e "6 Python cargar el mejor modelo y selector de características, para probar con el test final\n"
        echo -e "7 Obtener datos multiaztertest y mostrar predicción del modelo \n"
        echo -e "8 Exit \n"
	read -p "Elige una opcion:" opcionmenuppal
	case $opcionmenuppal in
                       	1) obtenerdatoscohmetrix;;
			2) obtenerdatosmultiaztertest $modelo;;
                        3) pruebaobtenerdatosmultiaztertest;;
			4) wekarekinprobatu;;
			5) generarmodelopython;;
			6) testearmodelopython;;
			7) weka;;
			8) fin;;
			*) ;;

	esac 
done 

echo "Fin del Programa" 
exit 0 

