

# MultiAztertest

You can test the online version of MultiAzterTest at the address http://ixa2.si.ehu.eus/aztertest

Multiaztertest is an upgrade to the Aztertest application meant to evaluate texts in various languages by calculating multiple metrics and indicators of the texts' content and analyzing those results to determine the complexity level of those texts.

If you want more information about the metrics analyzed in MultiAzterTest you can read the preprint that we have uploaded to arxiv:

https://arxiv.org/abs/2109.04870


## Install

1. Download `multiaztertest.py` and the `data`, `corpus` and `wordembeddings` folders into the same directory
2. Use the following commnads to install the necessary operative system packages in Ubuntu 18.04:
>
>**sudo apt install build-essential**
>
>**#dos2unix:to convert plain text files in DOS or Mac format to Unix format**
>
>**sudo apt install dos2unix**
>
>**#python-dev contains everything needed to compile python extension modules**
>
>**sudo apt-get install python3-dev**
>
>**#python3-pip: install pip3 for python3**
>
>**sudo apt install python3-pip**
>
>**#to install textract package in ubuntu 18.04 are necessary the following packages: libpulse-dev and swig**
>
>**sudo apt-get install libpulse-dev**
>
>**sudo apt-get install swig**

3. Use the following commands to install the necessary python packages:

>**pip3 install stanfordnlp**
>
>**pip3 install nlpcube**
>
>**pip3 install wordfreq**
>
>**pip3 install pandas**
>
>**pip3 install sklearn**
>
>**pip3 install --upgrade scikit-learn==0.22.1**
>
>**pip3 install --upgrade gensim**
>
>**apt-get install python-dev libxml2-dev libxslt1-dev antiword unrtf poppler-utils pstotext tesseract-ocr \ flac ffmpeg lame libmad0 libsox-fmt-mp3 sox libjpeg-dev swig**
>
>**pip install textract**

## Run

Once MultiAztertest has been installed, run it using the following parameters:
```
python3 multiaztertest.py -c -r -f $dir/*.txt -l language -m model -d /home/workdirectory
```
Currently available languages: english, spanish, basque

Currently available models: stanford, cube
