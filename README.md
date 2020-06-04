

# MultiAztertest

Multiaztertest is an upgrade to the Aztertest application meant to evaluate texts in various languages by calculating multiple metrics and indicators of the texts' content and analyzing those results to determine the complexity level of those texts.

## Install

1. Download `multiaztertest.py` and the `data`, `corpus` and `wordembeddings` folders into the same directory
2. Use the following commands to install the necessary python packages:

>**pip3 install stanfordnlp**
**pip3 install nlpcube**
**pip3 install wordfreq**
**pip3 install pandas**
**pip3 install sklearn**
**pip3 install --upgrade scikit-learn==0.22.1**
**pip3 install --upgrade gensim**
**apt-get install python-dev libxml2-dev libxslt1-dev antiword unrtf poppler-utils pstotext tesseract-ocr \ flac ffmpeg lame libmad0 libsox-fmt-mp3 sox libjpeg-dev swig**
**pip install textract**

## Run

Once MultiAztertest has been installed, run it using the following parameters:
```
python3 multiaztertest.py -c -r -f $dir/*.txt -l language -m model -d /home/workdirectory
```
Currently available languages: english, spanish, basque
Currently available models: stanford, cube
