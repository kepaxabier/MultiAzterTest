####Argumentos##################################
from collections import defaultdict


class FileLoader():
    #variables de clase que se comparte con todas las instancias
    irregular_verbs = []
    simple_words = []
    connectives = defaultdict(list)
    oxford_words = defaultdict(dict)
    files = []

    @staticmethod
    def load_files(args):
        FileLoader.files = args
        print("Parametros: " + str(FileLoader.files))

    # Se carga una lista de verbos irregulares utilizando una lista obtenida
    # de https://github.com/Bryan-Legend/babel-lang/blob/master/Babel.EnglishEmitter/Resources/Irregular%20Verbs.txt.
    @staticmethod
    def load_irregular_verbs_list():
        f = open('data/IrregularVerbs.txt', 'r')
        lineas = f.readlines()
        for linea in lineas:
            if not linea.startswith("//"):
                #carga el verbo en presente, dejando pasado y preterito
                FileLoader.irregular_verbs.append(linea.split()[0])
        f.close()

    # Se carga una lista de palabras/terminos simples de una lista obtenida
    # de http://www.readabilityformulas.com/articles/dale-chall-readability-word-list.php
    @staticmethod
    def load_dale_chall_list():
        f = open('data/dale-chall.txt', 'r')
        for line in f:
            for word in line.split():
                FileLoader.simple_words.append(word.lower())
        f.close()

    # Se carga una lista de conectores
    @staticmethod
    def load_connectives_list():
        f = open('data/connectives.txt', 'r')
        lineas = f.readlines()
        categoria = ''
        for linea in lineas:
            if linea.startswith("//"):
                categoria = linea.replace('//', '').replace('\n', '').lower()
            else:
                FileLoader.connectives[categoria].append(linea.replace('\n', ''))
        f.close()

    # Se carga la lista de palabras de Oxford
    @staticmethod
    def load_oxford_word_list():
        f = open('data/OxfordWordListByLevel.txt', 'r', encoding='utf-8')
        lines = f.readlines()
        level = ''
        for line in lines:
            if line.startswith("//"):
                level = line.replace('//', '').replace('\n', '')
            else:
                splitted_list = line.split()
                word = []
                pos = []
                for x in splitted_list:
                    if x.isupper():
                        pos.append(x)
                    else:
                        word.append(x)
                word = ' '.join(word)
                pos = ''.join(pos).split(",")
                FileLoader.oxford_words[level][word] = pos
        f.close()
