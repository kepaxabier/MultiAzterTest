####Argumentos##################################
from collections import defaultdict


class FileLoader():
    # variables de clase que se comparte con todas las instancias
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
        f = open('data/en/IrregularVerbs.txt', 'r')
        lineas = f.readlines()
        for linea in lineas:
            if not linea.startswith("//"):
                # carga el verbo en presente, dejando pasado y preterito
                FileLoader.irregular_verbs.append(linea.split()[0])
        f.close()

    # Se carga una lista de palabras/terminos simples de una lista obtenida
    # de http://www.readabilityformulas.com/articles/dale-chall-readability-word-list.php
    @staticmethod
    def load_dale_chall_list():
        f = open('data/en/dale-chall.txt', 'r')
        for line in f:
            for word in line.split():
                FileLoader.simple_words.append(word.lower())
        f.close()

    # Se carga una lista de conectores
    @staticmethod
    def load_connectives_list():
        f = open('data/en/connectives.txt', 'r')
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
        f = open('data/en/OxfordWordListByLevel.txt', 'r', encoding='utf-8')
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


import re
import math
#### Pronunciador- Utilizado para obtener silabas
from nltk.corpus import cmudict
####Tokenizador
from nltk.tokenize import sent_tokenize, word_tokenize
####Stopwords
from nltk.corpus import stopwords
####Wordnet
from nltk.corpus import wordnet as wn
####wordfreq###################
from wordfreq import zipf_frequency
####Numpy
import numpy as np
####Google Universal Encoder utiliza Tensorflow
## Importar tensorflow
import tensorflow as tf

## Desactivar mensajes de tensorflow
tf.logging.set_verbosity(tf.logging.ERROR)
# import tensorflow_hub as hub
####

####
import textract
####
from collections import defaultdict
####
import pandas as pd
from sklearn.externals import joblib
import pickle


class Analyzer:
    prondict = cmudict.dict()
    not_punctuation = lambda self, w: not (len(w) == 1 and (not w.isalpha()))
    syl_func = lambda self, w: self.allnum_syllables(w)

    def __init__(self, text, input, standford):
        self.indicators = defaultdict(int)
        self.aux_lists = defaultdict(list)
        self.words_freq = {}
        self.text = text
        self.standford = standford
        self.input = input

    # So it encodes a unicode string to ascii and ignores errors
    # filter(function, iterable)
    # function that tests if elements of an iterable returns true or false
    # get_word_count = lambda text: len(filter(not_punctuation, word_tokenize(text)))
    # (element for element in iterable if function(element))
    # filterwords=filter(not_punctuation, word_tokenize(text))
    def calculate_num_words(self):
        self.indicators['num_words'] = 0
        not_punctuation = lambda w: not (len(w) == 1 and (not w.isalpha()))
        filterwords = filter(not_punctuation, word_tokenize(self.text))
        for word in filterwords:
            self.indicators['num_words'] = self.indicators['num_words'] + 1

    #### Pronunciador- Utilizado para obtener silabas
    # cmudict is a pronouncing dictionary for north american english words.
    # it splits words into phonemes, which are shorter than syllables.
    # (e.g. the word 'cat' is split into three phonemes: K - AE - T).
    # but vowels also have a "stress marker":
    # either 0, 1, or 2, depending on the pronunciation of the word (so AE in 'cat' becomes AE1).
    # the code in the answer counts the stress markers and therefore the number of the vowels -
    # which effectively gives the number of syllables (notice how in OP's examples each syllable has exactly one vowel)

    def num_syllables(self, word):
        list = []
        max = 0
        for x in self.prondict[word.lower()]:
            tmp_list = []
            tmp_max = 0
            for y in x:
                if y[-1].isdigit():
                    tmp_max += 1
                    tmp_list.append(y)
            list.append(tmp_list)
            if tmp_max > max:
                max = tmp_max
        return (max)

    def syllables(self, word):
        """
        Calculate syllables of a word using a less accurate algorithm.
        Parse through the sentence, using common syllabic identifiers to count
        syllables.
        ADAPTED FROM:
        [http://stackoverflow.com/questions/14541303/count-the-number-of-syllables-in-a-word]
        """
        # initialize count
        count = 0
        # vowel list
        vowels = 'aeiouy'
        # take out punctuation
        word = word.lower()  # word.lower().strip(".:;?!")
        # various signifiers of syllabic up or down count
        if word[0] in vowels:
            count += 1
        for index in range(1, len(word)):
            if word[index] in vowels and word[index - 1] not in vowels:
                count += 1
        if word.endswith('e'):
            count -= 1
        if word.endswith('le') or word.endswith('a'):
            count += 1
        if count == 0:
            count += 1
        if "ooo" in word or "mm" in word:
            count = 1
        if word == 'll':
            count = 0
        if (word.startswith('x') and len(word) >= 2) and word[1].isdigit():
            count = 0
        if word == 'lmfao':
            count = 5
        if len(word) < 2 and word not in ['a', 'i', 'y', 'o']:
            count = 0
        return count

    def allnum_syllables(self, word):
        try:
            return self.num_syllables(word)
        except KeyError:
            # if word not found in cmudict
            return self.syllables(word)

    def text_statistics(self):
        get_sent_count = lambda text: len(sent_tokenize(text))
        sent_count = get_sent_count(self.text)
        syllable_count = sum(map(lambda w: self.allnum_syllables(w), word_tokenize(self.text)))
        return self.indicators['num_words'], sent_count, syllable_count

    # List of syllables of each word. This will be used to calculate mean/std dev of syllables.
    def get_syllable_list(self):
        filterwords = filter(self.not_punctuation, word_tokenize(self.text))
        list = []
        for word in filterwords:
            list.append(self.allnum_syllables(word))
        self.aux_lists['syllabes_list'] = list

    def flesch(self):
        flesch_formula = lambda word_count, sent_count, syllable_count: 206.835 - 1.015 * word_count / sent_count - 84.6 * syllable_count / word_count
        word_count, sent_count, syllable_count = self.text_statistics()
        flesch = flesch_formula(word_count, sent_count, syllable_count)
        if flesch >= 0: self.indicators['flesch'] = round(flesch, 4)

    def flesch_kincaid(self):
        fk_formula = lambda word_count, sent_count, syllable_count: 0.39 * word_count / sent_count + 11.8 * syllable_count / word_count - 15.59
        word_count, sent_count, syllable_count = self.text_statistics()
        fk = fk_formula(word_count, sent_count, syllable_count)
        if fk >= 0: self.indicators['flesch_kincaid'] = round(fk, 4)

    def sent2sequence(self, sent):
        # input:sent.words
        # [<Word index=1;text=;lemma=;>,
        #  <Word index=2;text=;lemma=;>,
        # ]
        sequence = []
        for word in sent.words:
            conllword = str(
                word.index) + "\t" + word.text + "\t" + word.lemma + "\t" + word.upos + "\t" + word.xpos + "\t" + word.feats + "\t" + str(
                word.governor) + "\t" + str(word.dependency_relation)
            sequence.append(conllword)
        # output:sequence:
        # [1 The the DET DT Feats 3 det _ _,
        #   2 Muslim .......................,
        #   3 .... ]
        return sequence

    # Este método devuelve true en caso de que la word pasada como parametro sea verbo. Para que una word sea
    # verbo se tiene que cumplir que sea VERB o que sea AUX y que su padre NO sea VERB.
    # def is_verb(self, word, frase):
    # return word.upos == 'VERB' or (word.upos == 'AUX' and frase[word.governor - 1].upos != 'VERB')
    def is_verb(self, word_upos, word_governor, frase):
        values = frase[int(word_governor) - 1].split("\t")
        if word_upos == 'VERB' or (word_upos == 'AUX' and values[3] != 'VERB'):
            # print (word.upos+"-"+values[3])
            return 1
        else:
            return 0

    # Este método devuelve true en caso de que la word pasada como parametro sea un verbo irregular. Se utiliza una lista
    # de verbos irregulares sacada de https://github.com/Bryan-Legend/babel-lang/blob/master/Babel.EnglishEmitter/Resources/Irregular%20Verbs.txt.
    def is_irregular(self, word):
        return True if word.lemma in FileLoader.irregular_verbs else False

    # Este método devuelve true en caso de que la word pasada como parametro sea una palabra simple. Se utiliza una lista
    # de palabras simples sacada de http://www.readabilityformulas.com/articles/dale-chall-readability-word-list.php
    def is_complex(self, word):
        return False if word.text.lower() in FileLoader.simple_words or word.lemma.lower() in FileLoader.simple_words else True

    # Este método devuelve true en caso de que la word pasada como parametro sea un verbo en pasado.
    def is_past(self, word):
        atributos = word.feats.split('|')
        return True if 'Tense=Past' in atributos else False

    # Este método devuelve true en caso de que la word pasada como parametro sea un verbo en presente.
    def is_present(self, word):
        atributos = word.feats.split('|')
        return True if 'Tense=Pres' in atributos else False

    # Este método devuelve true en caso de que la word pasada como parametro sea un verbo en futuro.
    def is_future(self, word, frase):
        values = frase[int(word.governor) - 1].split("\t")
        return word.upos == 'AUX' and word.lemma in ['will', 'shall'] and values[4] != 'VB'

    # Este método devuelve true en caso de que la word pasada como parametro sea un verbo en infinitivo.
    def is_infinitive(self, word, frase):
        atributos = word.feats.split('|')
        prev_word_index = int(word.index) - 1
        values = frase[prev_word_index - 1].split("\t")
        return 'VerbForm=Inf' in atributos and prev_word_index > 0 and values[1].lower() == 'to'

    # Este método devuelve true en caso de que la word pasada como parametro sea un verbo en gerundio.
    def is_gerund(self, word):
        atributos = word.feats.split('|')
        return True if 'VerbForm=Ger' in atributos else False

    # Este método devuelve true en caso de que la word pasada como parametro sea verbo un verbo en pasiva.
    def is_passive(self, word):
        atributos = word.feats.split('|')
        return True if 'Voice=Pass' in atributos else False

    # Este método devuelve true en caso de que la word pasada como parametro sea un verbo en modo indicativo.
    def is_indicative(self, word):
        atributos = word.feats.split('|')
        return True if 'Mood=Ind' in atributos else False

    # Este método devuelve true en caso de que la word pasada como parametro sea un verbo en modo imperativo.
    def is_imperative(self, word):
        atributos = word.feats.split('|')
        return True if 'Mood=Imp' in atributos else False

    def is_agentless(self, word, frase):
        # Si el siguiente indice esta dentro del rango de la lista
        if int(word.index) < len(frase):
            values = frase[int(word.index)].split("\t")
            siguiente_word = values[1].lower()
            if siguiente_word == 'by':
                return False
            else:
                return True

    def calculate_left_embeddedness(self, doc):
        list_left_embeddedness = []
        for sent in doc.sentences:
            verb_index = 0
            main_verb_found = False
            left_embeddedness = 0
            num_words = 0
            sequence = []
            # sent2sequence output:[1 The the DET DT Feats 3 det _ _,2 Muslim ......,3 .... ]
            sequence = self.sent2sequence(sent)
            for word in sent.words:
                if not len(word.text) == 1 or word.text.isalpha():
                    if not main_verb_found and int(word.governor) < len(sequence):
                        if self.is_verb(word.upos, int(word.governor), sequence):
                            verb_index += 1
                            wordgovernorvalues = sequence[int(word.governor) - 1].split("\t")
                            if (word.upos == 'VERB' and word.dependency_relation == 'root') or (
                                    word.upos == 'AUX' and wordgovernorvalues[7] == 'root' and wordgovernorvalues[
                                3] == 'VERB'):
                                main_verb_found = True
                                left_embeddedness = num_words
                            if verb_index == 1:
                                left_embeddedness = num_words
                    num_words += 1
            list_left_embeddedness.append(left_embeddedness)
        self.indicators['left_embeddedness'] = round(float(np.mean(list_left_embeddedness)), 4)

    def tree_depth(self, tree, root):
        if not tree[root]:
            return 1
        else:
            return 1 + max(self.tree_depth(tree, x) for x in tree[root])

    def calculate_simple_ttr(self, p_diff_forms=None, p_num_words=None):
        if (p_diff_forms and p_num_words) is not None:
            return (len(p_diff_forms)) / p_num_words
        else:
            self.indicators['simple_ttr'] = round(len(self.aux_lists['different_forms']) / self.indicators['num_words'],
                                                  4)

    def calculate_nttr(self):
        if self.indicators['num_noun'] > 0:
            self.indicators['nttr'] = round(len(self.aux_lists['different_nouns']) / self.indicators['num_noun'], 4)

    def calculate_vttr(self):
        if self.indicators['num_verb'] > 0:
            self.indicators['vttr'] = round(len(self.aux_lists['different_verbs']) / self.indicators['num_verb'], 4)

    def calculate_adj_ttr(self):
        if self.indicators['num_adj'] > 0:
            self.indicators['adj_ttr'] = round(len(self.aux_lists['different_adjs']) / self.indicators['num_adj'], 4)

    def calculate_adv_ttr(self):
        if self.indicators['num_adv'] > 0:
            self.indicators['adv_ttr'] = round(len(self.aux_lists['different_advs']) / self.indicators['num_adv'], 4)

    def calculate_content_ttr(self):
        nttr = self.indicators['nttr']
        vttr = self.indicators['vttr']
        adj_ttr = self.indicators['adj_ttr']
        adv_ttr = self.indicators['adv_ttr']
        self.indicators['content_ttr'] = round((nttr + vttr + adj_ttr + adv_ttr) / 4, 4)

    def calculate_all_ttr(self):
        self.calculate_simple_ttr()
        self.calculate_nttr()
        self.calculate_vttr()
        self.calculate_adj_ttr()
        self.calculate_adv_ttr()
        self.calculate_content_ttr()

    def calculate_lemma_ttr(self):
        self.indicators['lemma_ttr'] = round(len(self.aux_lists['different_lemmas']) / self.indicators['num_words'], 4)

    def calculate_lemma_nttr(self):
        if self.indicators['num_noun'] > 0:
            self.indicators['lemma_nttr'] = round(
                len(self.aux_lists['different_lemma_nouns']) / self.indicators['num_noun'], 4)

    def calculate_lemma_vttr(self):
        if self.indicators['num_verb'] > 0:
            self.indicators['lemma_vttr'] = round(
                len(self.aux_lists['different_lemma_verbs']) / self.indicators['num_verb'], 4)

    def calculate_lemma_adj_ttr(self):
        if self.indicators['num_adj'] > 0:
            self.indicators['lemma_adj_ttr'] = round(
                len(self.aux_lists['different_lemma_adjs']) / self.indicators['num_adj'], 4)

    def calculate_lemma_adv_ttr(self):
        if self.indicators['num_adv'] > 0:
            self.indicators['lemma_adv_ttr'] = round(
                len(self.aux_lists['different_lemma_advs']) / self.indicators['num_adv'], 4)

    def calculate_lemma_content_ttr(self):
        lnttr = self.indicators['lemma_nttr']
        lvttr = self.indicators['lemma_vttr']
        ladj_ttr = self.indicators['lemma_adj_ttr']
        ladv_ttr = self.indicators['lemma_adv_ttr']
        self.indicators['lemma_content_ttr'] = round((lnttr + lvttr + ladj_ttr + ladv_ttr) / 4, 4)

    def calculate_all_lemma_ttr(self):
        self.calculate_lemma_ttr()
        self.calculate_lemma_nttr()
        self.calculate_lemma_vttr()
        self.calculate_lemma_adj_ttr()
        self.calculate_lemma_adv_ttr()
        self.calculate_lemma_content_ttr()

    def get_num_hapax_legomena(self):
        num_hapax_legonema = 0
        for word, frecuencia in self.words_freq.items():
            if frecuencia == 1:
                num_hapax_legonema += 1
        return num_hapax_legonema

    def calculate_honore(self):
        n = self.indicators['num_words']
        v = len(self.aux_lists['different_forms'])
        v1 = self.get_num_hapax_legomena()
        self.indicators['honore'] = round(100 * ((np.log10(n)) / (1 - (v1 / v))), 4)

    def calculate_maas(self):
        n = self.indicators['num_words']
        v = len(self.aux_lists['different_forms'])
        self.indicators['maas'] = round((np.log10(n) - np.log10(v)) / (np.log10(v) ** 2), 4)

    def mtld(self, filtered_words):
        ttr_threshold = 0.72
        ttr = 1.0
        word_count = 0
        fragments = 0.0
        dif_words = []

        for i, word in enumerate(filtered_words):
            word = word.lower()
            word_count += 1
            if word not in dif_words:
                dif_words.append(word)
            ttr = self.calculate_simple_ttr(dif_words, word_count)
            if ttr <= ttr_threshold:
                # Se suma un fragmento y se deja preparado para comenzar otro
                fragments += 1
                word_count = 0
                dif_words.clear()
                ttr = 1.0
            elif i == len(filtered_words) - 1:
                # Si al final del texto queda un segmento sin alcanzar la TTR umbral este segmento no se desprecia sino
                # que se obtiene un número residual menor que uno (calculado proporcionalmente a la cantidad que le falta
                # a la TTR de este segmento para llegar a uno) que se suma al número de segmentos completos
                residual = (1.0 - ttr) / (1.0 - ttr_threshold)
                fragments += residual

        if fragments != 0:
            return len(filtered_words) / fragments
        else:
            return 0

    # MTLD
    def calculate_mtld(self):
        # Quitamos las marcas de puntuacion
        filtered_words = list(filter(self.not_punctuation, word_tokenize(self.text)))
        # El valor definitivo de MTLD se calcula haciendo la media de los dos valores obtenidos al repetir el
        # proceso de calculo dos veces, uno en sentido directo siguiendo el orden de lectura y otro en sentido inverso
        self.indicators['mtld'] = round((self.mtld(filtered_words) + self.mtld(filtered_words[::-1])) / 2, 4)

    # Dale-Chall Formula
    def calculate_dale_chall(self):
        ts = self.indicators['num_sentences']
        tc = self.indicators['num_complex_words']
        tw = self.indicators['num_words']
        percentage = (tc / tw) * 100
        if percentage >= 5.0:
            self.indicators['dale_chall'] = round(0.1579 * percentage + 0.0496 * (tw / ts) + 3.6365, 4)
        else:
            self.indicators['dale_chall'] = round(0.1579 * percentage + 0.0496 * (tw / ts), 4)

    def has_more_than_three_syllables(self, word):
        num_syl = 0
        try:
            num_syl = self.num_syllables(word)
        except KeyError:
            # if word not found in cmudict
            num_syl = self.syllables(word)
        return True if num_syl > 3 else False

    # SMOG=1,0430*SQRT(30*totalcomplex/totalsentences)+3,1291 (total polysyllables --> con mas de 3 silabas)
    def calculate_smog(self):
        ts = self.indicators['num_sentences']
        tps = self.indicators['num_words_more_3_syl']
        self.indicators['smog'] = round(1.0430 * math.sqrt(30 * tps / ts) + 3.1291, 4)

    def is_not_stopword(self, word):
        stop_words = stopwords.words('english')
        return word.lower() not in stop_words

    def calculate_levels_oxford_word_list(self, doc):
        i = self.indicators
        for sent in doc.sentences:
            sequence = []
            sequence = self.sent2sequence(sent)
            for entry in sent.words:
                if entry.text in FileLoader.oxford_words['A1']:
                    if entry.upos in FileLoader.oxford_words['A1'][entry.text]:
                        i['num_a1_words'] += 1
                elif entry.text in FileLoader.oxford_words['A2']:
                    if entry.upos in FileLoader.oxford_words['A2'][entry.text]:
                        i['num_a2_words'] += 1
                elif entry.text in FileLoader.oxford_words['B1']:
                    if entry.upos in FileLoader.oxford_words['B1'][entry.text]:
                        i['num_b1_words'] += 1
                elif entry.text in FileLoader.oxford_words['B2']:
                    if entry.upos in FileLoader.oxford_words['B2'][entry.text]:
                        i['num_b2_words'] += 1
                elif entry.text in FileLoader.oxford_words['C1']:
                    if entry.upos in FileLoader.oxford_words['C1'][entry.text]:
                        i['num_c1_words'] += 1
                elif self.is_lexic_word(entry.upos, int(entry.governor), sequence):
                    i['num_content_words_not_a1_c1_words'] += 1

    def get_ambiguity_level(self, word, FLAG):
        if FLAG == 'NOUN':
            ambiguity_level = len(wn.synsets(word, pos='n'))
        elif FLAG == 'ADJ':
            ambiguity_level = len(wn.synsets(word, pos='a'))
        elif FLAG == 'ADV':
            ambiguity_level = len(wn.synsets(word, pos='r'))
        else:
            ambiguity_level = len(wn.synsets(word, pos='v'))
        return ambiguity_level

    def get_abstraction_level(self, word, FLAG):
        abstraction_level = 0
        if len(wn.synsets(word, pos=FLAG)) > 0:
            abstraction_level = len(wn.synsets(word, pos=FLAG)[0].hypernym_paths()[0])
        return abstraction_level

    def calculate_polysemic_index(self, ambiguity_list):
        i = self.indicators
        i['polysemic_index'] = round(float(np.mean(ambiguity_list)), 4)

    def calculate_hypernymy_index(self, ambiguity_content_words_list, FLAG='VN'):
        i = self.indicators
        if FLAG == 'VN':
            i['hypernymy_index'] = round(float(np.mean(ambiguity_content_words_list)), 4)
        elif FLAG == 'V':
            i['hypernymy_verbs_index'] = round(float(np.mean(ambiguity_content_words_list)), 4)
        elif FLAG == 'N':
            i['hypernymy_nouns_index'] = round(float(np.mean(ambiguity_content_words_list)), 4)

    # def is_lexic_word(self, entry, sequence):
    #    return self.is_verb(entry, sequence) or entry.upos == 'NOUN' or entry.upos == 'ADJ' or entry.upos == 'ADV'

    def is_lexic_word(self, word_upos, word_governor, sequence):
        return self.is_verb(word_upos, word_governor,
                            sequence) or word_upos == 'NOUN' or word_upos == 'ADJ' or word_upos == 'ADV'

    def has_modifier(self, entry):
        # nominal head may be associated with different types of modifiers and function words
        return True if entry.dependency_relation in ['nmod', 'nmod:poss', 'appos', 'amod', 'nummod', 'acl', 'acl:relcl',
                                                     'det', 'clf', 'case'] else False

    def count_decendents(self, sent, list_np_indexes):
        num_modifiers = 0
        if len(list_np_indexes) == 0:
            return num_modifiers
        else:
            new_list_indexes = []
            for entry in sent.words:
                if int(entry.governor) in list_np_indexes and self.has_modifier(entry):
                    new_list_indexes.append(int(entry.index))
                    num_modifiers += 1
            return num_modifiers + self.count_decendents(sent, new_list_indexes)

    def count_modifiers(self, sent, list_np_indexes):
        num_modifiers_per_np = []
        # Por cada index que es un np, devuelve una lista con el número de modificadores por cada np
        for index in list_np_indexes:
            num_modifiers = 0
            # Por cada index me recorro toda la sentencia, buscando si ese index es padre nominal mediante una relación
            # de cabeza nominal:'nmod', 'nmod:poss', 'appos', 'amod', 'nummod', 'acl', 'acl:relcl', 'det', 'case', 'clf'
            for entry in sent.words:
                if int(entry.governor) == int(index) and self.has_modifier(entry):
                    num_modifiers += 1
            num_modifiers_per_np.append(num_modifiers)
        return num_modifiers_per_np

    def count_np_in_sentence(self, sent):
        list_np_indexes = []
        for entry in sent.words:
            # Si la palabra que vamos a tratar es nombre, pronombre o nombre propio
            if entry.upos == 'NOUN' or entry.upos == 'PRON' or entry.upos == 'PROPN':
                # si tiene una relación de multiword
                if entry.dependency_relation in ['fixed', 'flat', 'compound']:
                    # y su cabeza multiword no esta en la lista
                    if int(entry.governor) not in list_np_indexes:
                        # introducimos la cabeza en la lista
                        list_np_indexes.append(int(entry.governor))
                # si no tiene una relación multiword
                else:
                    # no esta en la lista, se añade a la lista
                    if int(entry.index) not in list_np_indexes:
                        list_np_indexes.append(int(entry.index))
        return list_np_indexes

    def count_vp_in_sentence(self, sent):
        num_np = 0
        sequence = []
        sequence = self.sent2sequence(sent)
        for entry in sent.words:
            if self.is_verb(entry.upos, int(entry.governor), sequence):
                num_np += 1
        return num_np

    # Noun overlap measure is binary (there either is or is not any overlap between a pair of adjacent sentences in a text ).
    # Noun overlap measures the proportion of sentences in a text for which there are overlapping nouns,
    # With no deviation in the morphological forms (e.g., table/tables)
    # (número pares de sentencias adjacentes que tienen al menos algún nombre en común)/(Número de pares de sentencias adjacentes)
    def calculate_noun_overlap_adjacent(self):
        i = self.indicators
        adjacent_noun_overlap_list = []
        # paragraph_list es una lista de doc.sentences donde doc.sentences es una "lista de obj sentencias" de un parrafo=[doc.sentence1,...]
        for paragraph in self.aux_lists['paragraphs_list']:
            # Por cada parrafo:paragraph es "lista de obj sentencias" de un parrafo=[doc.sentence1,...]
            if len(paragraph) > 1:
                # zip Python zip function takes iterable elements as input, and returns iterator que es un flujo de datos que
                # puede ser recorrido por for o map.
                # Si paragraph = [[sentence1], [sentence2], [sentence3]]
                # paragraph[1:] = [[sentence2], [sentence3]]
                test = zip(paragraph, paragraph[1:])  # zip the values
                # print(test) #-><zip object at 0x7eff7b354c08>=?[([sentence1],[sentence2]),([sentence2],[sentence3]),...]
                # for values in test:
                # print(values)  # print each tuples
                # ([sentence1],[sentence2])
                # ([sentence2],[sentence3])
                # map aplica la función list a todos los elementos de zip y como resultado se devuelve un iterable de tipo map
                # funcion list=The list() constructor returns a mutable (the object can be modified) sequence list of elements.
                # Por cada valor de test genera una lista
                testlist = map(list, test)
                # print(testlist) #<map object at 0x7eff7b3701d0>=?[[([sentence1],[sentence2])],[([sentence2],[sentence3])]]
                adjacents = list(map(list, test))
                # print(type(adjacents))
                # print(adjacents) ##Ejm: Parrafo1:[[[sent1], [sent2]], [[sent2], [sent3]]] donde sentenceX es conllword1,conllword2,...
                for x in adjacents:
                    sentence1 = []
                    sentence2 = []
                    for entry1 in x[0]:
                        values1 = entry1.split("\t")
                        if values1[3] == 'NOUN':
                            sentence1.append(values1[1].lower())
                    for entry2 in x[1]:
                        values2 = entry2.split("\t")
                        if values2[3] == 'NOUN':
                            sentence2.append(values2[1].lower())
                    # nombres en comun entre sentence1 y sentence2
                    in_common = list(set(sentence1).intersection(sentence2))
                    # si hay nombre en comun añado 1
                    if len(in_common) > 0:
                        adjacent_noun_overlap_list.append(1)
                    else:
                        adjacent_noun_overlap_list.append(0)
        if len(adjacent_noun_overlap_list) > 0:
            i['noun_overlap_adjacent'] = round(float(np.mean(adjacent_noun_overlap_list)), 4)

    # Noun overlap measures which is the average overlap between all pairs of sentences in the text for which there are overlapping nouns,
    # With no deviation in the morphological forms (e.g., table/tables)
    # (Sumatorio de todos pares de sentencias del texto que tienen alguna coincidencia en algún nombre)/(todos los pares de sentencias del texto)
    def calculate_noun_overlap_all(self):
        i = self.indicators
        all_noun_overlap_list = []
        for paragraph in self.aux_lists['paragraphs_list']:
            for index in range(len(paragraph)):
                similarity_tmp = paragraph[index + 1:]
                x = paragraph[index]
                for index2 in range(len(similarity_tmp)):
                    y = similarity_tmp[index2]
                    sentence1 = []
                    sentence2 = []
                    for entry1 in x:
                        values1 = entry1.split("\t")
                        if values1[3] == 'NOUN':
                            sentence1.append(values1[1].lower())
                    for entry2 in y:
                        values2 = entry2.split("\t")
                        if values2[3] == 'NOUN':
                            sentence2.append(values2[1].lower())
                    in_common = list(set(sentence1).intersection(sentence2))
                    if len(in_common) > 0:
                        all_noun_overlap_list.append(1)
                    else:
                        all_noun_overlap_list.append(0)
        if len(all_noun_overlap_list) > 0:
            i['noun_overlap_all'] = round(float(np.mean(all_noun_overlap_list)), 4)

    def is_personal_pronoun(self, word):
        values = word.split("\t")
        atributos = values[5].split('|')
        if "PronType=Prs" in atributos:
            return True
        else:
            return False

    # Argument overlap measure is binary (there either is or is not any overlap between a pair of adjacent
    # sentences in a text ). Argument overlap measures the proportion of sentences in a text for which there are overlapping the
    # between nouns (stem, e.g., “table”/”tables”) and personal pronouns (“he”/”he”)
    def calculate_argument_overlap_adjacent(self):
        i = self.indicators
        adjacent_argument_overlap_list = []
        for paragraph in self.aux_lists['paragraphs_list']:
            if len(paragraph) > 1:
                adjacents = list(map(list, zip(paragraph, paragraph[1:])))
                for x in adjacents:
                    sentence1 = []
                    sentence2 = []
                    for entry1 in x[0]:
                        values1 = entry1.split("\t")
                        if self.is_personal_pronoun(entry1) or values1[3] == 'NOUN':
                            sentence1.append(values1[2].lower())
                    for entry2 in x[1]:
                        values2 = entry2.split("\t")
                        if self.is_personal_pronoun(entry2) or values2[3] == 'NOUN':
                            sentence2.append(values2[2].lower())
                    in_common = list(set(sentence1).intersection(sentence2))
                    if len(in_common) > 0:
                        adjacent_argument_overlap_list.append(1)
                    else:
                        adjacent_argument_overlap_list.append(0)
        if len(adjacent_argument_overlap_list) > 0:
            i['argument_overlap_adjacent'] = round(float(np.mean(adjacent_argument_overlap_list)), 4)

    # Argument overlap measures which is the average overlap between all pairs of sentences in the
    # text for which there are overlapping stem nouns and personal pronouns.
    def calculate_argument_overlap_all(self):
        i = self.indicators
        all_argument_overlap_list = []
        for paragraph in self.aux_lists['paragraphs_list']:
            for index in range(len(paragraph)):
                similarity_tmp = paragraph[index + 1:]
                x = paragraph[index]
                for index2 in range(len(similarity_tmp)):
                    y = similarity_tmp[index2]
                    sentence1 = []
                    sentence2 = []
                    for entry1 in x:
                        values1 = entry1.split("\t")
                        if self.is_personal_pronoun(entry1) or values1[3] == 'NOUN':
                            sentence1.append(values1[2].lower())
                    for entry2 in y:
                        values2 = entry2.split("\t")
                        if self.is_personal_pronoun(entry2) or values2[3] == 'NOUN':
                            sentence2.append(values2[2].lower())
                    in_common = list(set(sentence1).intersection(sentence2))
                    if len(in_common) > 0:
                        all_argument_overlap_list.append(1)
                    else:
                        all_argument_overlap_list.append(0)
        if len(all_argument_overlap_list) > 0:
            i['argument_overlap_all'] = round(float(np.mean(all_argument_overlap_list)), 4)

    # Stem overlap measure is binary (there either is or is not any overlap between a pair of adjacent sentences in a text ).
    # Stem overlap measures the proportion of sentences in a text for which there are overlapping between a noun in one
    # sentence and a content word (i['e.,'] nouns,verbs, adjectives, adverbs) in a previous sentence
    # that shares a common lemma (e.g., “tree”/”treed”;”mouse”/”mousey”).
    def calculate_stem_overlap_adjacent(self):
        i = self.indicators
        adjacent_stem_overlap_list = []
        for paragraph in self.aux_lists['paragraphs_list']:
            if len(paragraph) > 1:
                adjacents = list(map(list, zip(paragraph, paragraph[1:])))
                for x in adjacents:
                    sentence1 = []
                    sentence2 = []
                    for entry1 in x[0]:
                        values1 = entry1.split("\t")
                        if self.is_lexic_word(values1[3], values1[6], x[0]):
                            sentence1.append(values1[2].lower())
                    for entry2 in x[1]:
                        values2 = entry2.split("\t")
                        if values2[3] == 'NOUN':
                            sentence2.append(values2[2].lower())
                    in_common = list(set(sentence1).intersection(sentence2))
                    if len(in_common) > 0:
                        adjacent_stem_overlap_list.append(1)
                    else:
                        adjacent_stem_overlap_list.append(0)
        if len(adjacent_stem_overlap_list) > 0:
            i['stem_overlap_adjacent'] = round(float(np.mean(adjacent_stem_overlap_list)), 4)

    # Global Stem overlap measures which is the average overlap between all pairs of sentences in
    # the text for which there are overlapping Between a noun in one sentence and a content word
    # (i['e.,'] nouns,verbs, adjectives, adverbs) in a previous sentence that shares a common
    # lemma (e.g., “tree”/”treed”;”mouse”/”mousey”).
    def calculate_stem_overlap_all(self):
        i = self.indicators
        all_stem_overlap_list = []
        for paragraph in self.aux_lists['paragraphs_list']:
            for index in range(len(paragraph)):
                similarity_tmp = paragraph[index + 1:]
                x = paragraph[index]
                for index2 in range(len(similarity_tmp)):
                    y = similarity_tmp[index2]
                    sentence1 = []
                    sentence2 = []
                    for entry1 in x:
                        values1 = entry1.split("\t")
                        if self.is_lexic_word(values1[3], values1[6], x):
                            sentence1.append(values1[2].lower())
                    for entry2 in y:
                        values2 = entry2.split("\t")
                        if values2[3] == 'NOUN':
                            sentence2.append(values2[2].lower())
                    in_common = list(set(sentence1).intersection(sentence2))
                    if len(in_common) > 0:
                        all_stem_overlap_list.append(1)
                    else:
                        all_stem_overlap_list.append(0)
        if len(all_stem_overlap_list) > 0:
            i['stem_overlap_all'] = round(float(np.mean(all_stem_overlap_list)), 4)

    # Metodo que calcula el numero de palabras de contenido en una frase. Counts number of content words in a sentence.
    def count_content_words_in(self, sent):
        num_words = 0
        for entry in sent:
            values = entry.split("\t")
            if self.is_verb(values[3], values[6], sent) or values[3] == 'NOUN' or values[3] == 'ADJ' or values[
                3] == 'ADV':
                num_words += 1
        return num_words

    # Content word overlap adjacent sentences proporcional mean refers to the proportion of content words
    # (nouns, verbs,adverbs,adjectives, pronouns) that shared Between pairs of sentences.For example, if
    # a sentence pair has fewer words and two words overlap, The proportion is greater than if a pair has
    # many words and two words overlap. This measure may be particulaly useful when the lenghts of the
    # sentences in the text are principal concern.
    def calculate_content_overlap_adjacent(self):
        i = self.indicators
        adjacent_content_overlap_list = []
        for paragraph in self.aux_lists['paragraphs_list']:
            if len(paragraph) > 1:
                adjacents = list(map(list, zip(paragraph, paragraph[1:])))
                for x in adjacents:
                    sentence1 = []
                    sentence2 = []
                    for entry1 in x[0]:
                        values1 = entry1.split("\t")
                        if self.is_lexic_word(values1[3], values1[6], x[0]):
                            sentence1.append(values1[1].lower())
                    for entry2 in x[1]:
                        values2 = entry2.split("\t")
                        if self.is_lexic_word(values2[3], values2[6], x[1]):
                            sentence2.append(values2[1].lower())
                    in_common = list(set(sentence1).intersection(sentence2))
                    n1 = self.count_content_words_in(x[0])
                    n2 = self.count_content_words_in(x[1])
                    if n1 + n2 > 0:
                        adjacent_content_overlap_list.append(len(in_common) / (n1 + n2))
                    else:
                        adjacent_content_overlap_list.append(0)
        if len(adjacent_content_overlap_list) > 0:
            i['content_overlap_adjacent_mean'] = round(float(np.mean(adjacent_content_overlap_list)), 4)
            i['content_overlap_adjacent_std'] = round(float(np.std(adjacent_content_overlap_list)), 4)

    # Content word overlap adjacent sentences proporcional mean refers to the proportion of content words
    # (nouns, verbs,adverbs,adjectives, pronouns) that shared Between pairs of sentences.For example, if
    # a sentence pair has fewer words and two words overlap, The proportion is greater than if a pair has
    # many words and two words overlap. This measure may be particulaly useful when the lenghts of the
    # sentences in the text are principal concern.
    def calculate_content_overlap_all(self):
        i = self.indicators
        all_content_overlap_list = []
        for paragraph in self.aux_lists['paragraphs_list']:
            for index in range(len(paragraph)):
                similarity_tmp = paragraph[index + 1:]
                x = paragraph[index]
                for index2 in range(len(similarity_tmp)):
                    y = similarity_tmp[index2]
                    sentence1 = []
                    sentence2 = []
                    for entry1 in x:
                        values1 = entry1.split("\t")
                        if self.is_lexic_word(values1[3], values1[6], x):
                            sentence1.append(values1[1].lower())
                    for entry2 in y:
                        values2 = entry2.split("\t")
                        if self.is_lexic_word(values2[3], values2[6], y):
                            sentence2.append(values2[2].lower())
                    in_common = list(set(sentence1).intersection(sentence2))
                    n1 = self.count_content_words_in(x)
                    n2 = self.count_content_words_in(y)
                    if n1 + n2 > 0:
                        all_content_overlap_list.append(len(in_common) / (n1 + n2))
                    else:
                        all_content_overlap_list.append(0)
        if len(all_content_overlap_list) > 0:
            i['content_overlap_all_mean'] = round(float(np.mean(all_content_overlap_list)), 4)
            i['content_overlap_all_std'] = round(float(np.std(all_content_overlap_list)), 4)

    # Este metodo devuelve la similitud calculada mediante Google Sentence Encoder
    # def calculate_similarity(self, sentence1, sentence2):
    #    return np.inner(sentence1, sentence2)

    # Este metodo calcula la similaridad local media de todas las similitudes en frases
    # def calculate_similarity_adjacent_sentences(self):
    #    i = self.indicators
    #    adjacent_similarity_list = []
    #    for sentence in self.aux_lists['sentences_in_paragraph_token_list']:
    #        if len(sentence) > 1:
    #            for x, y in zip(range(0, len(sentence) - 1), range(1, len(sentence))):
    #                adjacent_similarity_list.append(self.calculate_similarity(sentence[x], sentence[y]))
    #        else:
    #            adjacent_similarity_list.append(0)
    #    if len(adjacent_similarity_list) > 0:
    #        i['similarity_adjacent_mean'] = round(float(np.mean(adjacent_similarity_list)), 4)
    #        i['similarity_adjacent_std'] = round(float(np.std(adjacent_similarity_list)), 4)

    # Este metodo calcula la similaridad local media de todas las similitudes en parrafos
    # def calculate_similarity_adjacent_paragraphs(self):
    #    i = self.indicators
    #    adjacent_similarity_par_list = []
    #    if len(self.aux_lists['paragraph_token_list']) > 1:
    #        for x, y in zip(range(0, len(self.aux_lists['paragraph_token_list']) - 1), range(1, len(self.aux_lists['paragraph_token_list']))):
    #            adjacent_similarity_par_list.append(self.calculate_similarity(self.aux_lists['paragraph_token_list'][x], self.aux_lists['paragraph_token_list'][y]))
    #        if len(adjacent_similarity_par_list) > 0:
    #            i['similarity_adjacent_par_mean'] = round(float(np.mean(adjacent_similarity_par_list)), 4)
    #            i['similarity_adjacent_par_std'] = round(float(np.std(adjacent_similarity_par_list)), 4)

    # def calculate_similarity_pairs_in(self, paragraph):
    #    list_similarities_mean = []
    #    for index in range(len(paragraph)):
    #        similarity_tmp = paragraph[index+1:]
    #        x = paragraph[index]
    #        for index2 in range(len(similarity_tmp)):
    #            y = similarity_tmp[index2]
    #            list_similarities_mean.append(self.calculate_similarity(x, y))
    #    if len(list_similarities_mean) > 1:
    #        return round(float(np.mean(list_similarities_mean)), 4)
    #    else:
    #        return 0.0

    # Este metodo calcula la similaridad global media de todas las similitudes
    # (between all possible pairs of sentences in a paragraph)
    # def calculate_similarity_pairs_sentences(self):
    #    i = self.indicators
    #    similarity_pairs_list = []
    #    for paragraph in self.aux_lists['sentences_in_paragraph_token_list']:
    #        similarity_pairs_list.append(self.calculate_similarity_pairs_in(paragraph))
    #    i['similarity_pairs_par_mean'] = round(float(np.mean(similarity_pairs_list)), 4)
    #    i['similarity_pairs_par_std'] = round(float(np.std(similarity_pairs_list)), 4)

    # Este metodo cuenta el numero de pronombres clasificados segun el tipo
    def count_personal_pronoun(self, word):
        i = self.indicators
        atributos = word.feats.split('|')
        if "PronType=Prs" in atributos:
            i['num_personal_pronouns'] += 1
            if 'Person=1' in atributos:
                i['num_first_pers_pron'] += 1
                if 'Number=Sing' in atributos:
                    i['num_first_pers_sing_pron'] += 1
            elif 'Person=3' in atributos:
                i['num_third_pers_pron'] += 1

    def calculate_mean_depth_per_sentence(self, depth_list):
        i = self.indicators
        i['mean_depth_per_sentence'] = round(float(np.mean(depth_list)), 4)

    @staticmethod
    def get_incidence(indicador, num_words):
        return round(((1000 * indicador) / num_words), 4)

    def calculate_sentences_per_paragraph(self, similarity):
        # Si el fichero de entrada no tiene extension .txt
        if ".txt" not in self.input:
            # textract extrae el texto de todo tipo de formatos (odt, docx, doc ..)
            pre_text = textract.process(self.input)
            # decode(encoding='UTF-8',errors='strict') convierte a utf8 y si no puede lanza un error
            text = pre_text.decode()
            # Separador de parrafo '@'
            text = text.replace('\n', '@')
        else:
            # Si extensión .txt convierte texto a utf-8
            with open(self.input, encoding='utf-8') as f:
                text = f.read().replace('\n', '@')
        # Si varios enters consecutivos en uno
        text = re.sub(r'@+', '@', text)
        # lines es una lista de parrafos
        lines = text.split('@')
        paragraphs = []
        # Se puede comentar
        if similarity:
            # Preparar Google Universal Sentence Encoder
            module_url = "https://tfhub.dev/google/universal-sentence-encoder-large/2"
            # embed = hub.Module(module_url)
            # similarity_input_placeholder = tf.placeholder(tf.string, shape=(None))
            # similarity_sentences_encodings = embed(similarity_input_placeholder)
            # with tf.Session() as session:
            # session.run(tf.global_variables_initializer())
            # session.run(tf.tables_initializer())
            # for line in lines:
            # if not line.strip() == '':
            # paragraphs.append(line)
            # sentences_embeddings = session.run(similarity_sentences_encodings, feed_dict={similarity_input_placeholder: sent_tokenize(line)})
            # self.aux_lists['sentences_in_paragraph_token_list'].append(sentences_embeddings)
            # self.aux_lists['paragraph_token_list'] = session.run(similarity_sentences_encodings, feed_dict={similarity_input_placeholder: paragraphs})
        else:
            # Por cada parrafo:
            for line in lines:
                # si no es una linea vacia, añade en parrafos
                if not line.strip() == '':
                    # Lista de texto formado por parrafos
                    paragraphs.append(line)
        ###############Generar lista con el numero de sentencias en cada parrafo##############################################
        self.aux_lists['paragraphs_list'] = []
        # Por cada parrafo de texto
        for paragraph in paragraphs:
            # Parrafo a objeto doc Document object
            doc = self.standford(Analyzer.process_text(text=paragraph))
            sentences = []
            # doc.sentences es una lista de objectos sentence [doc.sentence1, doc.sentence2, ...]
            for sent in doc.sentences:
                # sent2sequence output:[1 The the DET DT Feats 3 det _ _,2 Muslim ......,3 .... ]
                sentence = []
                sentence = self.sent2sequence(sent)
                sentences.append(sentence)
            self.aux_lists['paragraphs_list'].append(sentences)  # [[[sent1]]] o [[[sent1], [sent2]]]
            self.aux_lists['sentences_per_paragraph'].append(len(sentences))  # [1,2,...]

        # Number of paragraphs
        self.indicators['num_paragraphs'] = len(paragraphs)

    def calculate_all_overlaps(self):
        self.calculate_noun_overlap_adjacent()
        self.calculate_noun_overlap_all()
        self.calculate_argument_overlap_adjacent()
        self.calculate_argument_overlap_all()
        self.calculate_stem_overlap_adjacent()
        self.calculate_stem_overlap_all()
        self.calculate_content_overlap_adjacent()
        self.calculate_content_overlap_all()

    def calculate_connectives_for(self, text, connective):
        list = FileLoader.connectives.get(connective)
        list_a = []
        list_b = []
        num_a = 0
        num_b = 0
        for x in list:
            if "*" in x:
                list_a.append(x)
            else:
                list_b.append(x)
        for a in list_a:
            split = a.split('*')
            matches_a = re.findall(r'\b%s\b[^.!?()]+\b%s\b' % (split[0], split[1]), text)
            num_a += len(matches_a)
        for b in list_b:
            matches_b = re.findall(r'\b%s\b' % b, text)
            num_b += len(matches_b)
        return num_a + num_b

    def calculate_connectives(self):
        i = self.indicators
        causal_connectives = 0
        logical_connectives = 0
        adversative_connectives = 0
        temporal_connectives = 0
        conditional_connectives = 0
        if ".txt" not in self.input:
            pre_text = textract.process(self.input)
            text = pre_text.decode()
            text = text.replace('\n', '@')
        else:
            with open(self.input, encoding='utf-8') as f:
                text = f.read().replace('\n', '@')
        text = re.sub(r'@+', '@', text)
        lines = text.split('@')
        for line in lines:
            causal_connectives += self.calculate_connectives_for(line, 'causal')
            logical_connectives += self.calculate_connectives_for(line, 'logical')
            adversative_connectives += self.calculate_connectives_for(line, 'adversative')
            temporal_connectives += self.calculate_connectives_for(line, 'temporal')
            conditional_connectives += self.calculate_connectives_for(line, 'conditional')
        total_connectives = causal_connectives + logical_connectives + adversative_connectives + temporal_connectives + conditional_connectives
        i['all_connectives_incidence'] = Analyzer.get_incidence(total_connectives, i['num_words'])
        i['causal_connectives_incidence'] = Analyzer.get_incidence(causal_connectives, i['num_words'])
        i['logical_connectives_incidence'] = Analyzer.get_incidence(logical_connectives, i['num_words'])
        i['adversative_connectives_incidence'] = Analyzer.get_incidence(adversative_connectives, i['num_words'])
        i['temporal_connectives_incidence'] = Analyzer.get_incidence(temporal_connectives, i['num_words'])
        i['conditional_connectives_incidence'] = Analyzer.get_incidence(conditional_connectives, i['num_words'])

    def calculate_all_incidence(self):
        i = self.indicators
        i['num_paragraphs_incidence'] = Analyzer.get_incidence(i['num_paragraphs'], i['num_words'])
        i['num_sentences_incidence'] = Analyzer.get_incidence(i['num_sentences'], i['num_words'])
        i['num_past_incidence'] = Analyzer.get_incidence(i['num_past'], i['num_words'])
        i['num_pres_incidence'] = Analyzer.get_incidence(i['num_pres'], i['num_words'])
        i['num_future_incidence'] = Analyzer.get_incidence(i['num_future'], i['num_words'])
        i['num_indic_incidence'] = Analyzer.get_incidence(i['num_indic'], i['num_words'])
        i['num_impera_incidence'] = Analyzer.get_incidence(i['num_impera'], i['num_words'])
        i['num_past_irregular_incidence'] = Analyzer.get_incidence(i['num_past_irregular'], i['num_words'])
        i['num_pass_incidence'] = Analyzer.get_incidence(i['num_pass'], i['num_words'])
        i['num_rare_nouns_4_incidence'] = Analyzer.get_incidence(i['num_rare_nouns_4'], i['num_words'])
        i['num_rare_adj_4_incidence'] = Analyzer.get_incidence(i['num_rare_adj_4'], i['num_words'])
        i['num_rare_verbs_4_incidence'] = Analyzer.get_incidence(i['num_rare_verbs_4'], i['num_words'])
        i['num_rare_advb_4_incidence'] = Analyzer.get_incidence(i['num_rare_advb_4'], i['num_words'])
        i['num_rare_words_4_incidence'] = Analyzer.get_incidence(i['num_rare_words_4'], i['num_words'])
        i['num_dif_rare_words_4_incidence'] = Analyzer.get_incidence(i['num_dif_rare_words_4'], i['num_words'])
        i['num_lexic_words_incidence'] = Analyzer.get_incidence(i['num_lexic_words'], i['num_words'])
        i['num_noun_incidence'] = Analyzer.get_incidence(i['num_noun'], i['num_words'])
        i['num_adj_incidence'] = Analyzer.get_incidence(i['num_adj'], i['num_words'])
        i['num_adv_incidence'] = Analyzer.get_incidence(i['num_adv'], i['num_words'])
        i['num_verb_incidence'] = Analyzer.get_incidence(i['num_verb'], i['num_words'])
        i['num_subord_incidence'] = Analyzer.get_incidence(i['num_subord'], i['num_words'])
        i['num_rel_subord_incidence'] = Analyzer.get_incidence(i['num_rel_subord'], i['num_words'])
        i['num_personal_pronouns_incidence'] = Analyzer.get_incidence(i['num_personal_pronouns'], i['num_words'])
        i['num_first_pers_pron_incidence'] = Analyzer.get_incidence(i['num_first_pers_pron'], i['num_words'])
        i['num_first_pers_sing_pron_incidence'] = Analyzer.get_incidence(i['num_first_pers_sing_pron'], i['num_words'])
        i['num_third_pers_pron_incidence'] = Analyzer.get_incidence(i['num_third_pers_pron'], i['num_words'])
        i['num_a1_words_incidence'] = Analyzer.get_incidence(i['num_a1_words'], i['num_words'])
        i['num_a2_words_incidence'] = Analyzer.get_incidence(i['num_a2_words'], i['num_words'])
        i['num_b1_words_incidence'] = Analyzer.get_incidence(i['num_b1_words'], i['num_words'])
        i['num_b2_words_incidence'] = Analyzer.get_incidence(i['num_b2_words'], i['num_words'])
        i['num_c1_words_incidence'] = Analyzer.get_incidence(i['num_c1_words'], i['num_words'])
        i['num_content_words_not_a1_c1_words_incidence'] = Analyzer.get_incidence(
            i['num_content_words_not_a1_c1_words'], i['num_words'])

    def calculate_all_means(self):
        i = self.indicators
        i['sentences_per_paragraph_mean'] = round(float(np.mean(self.aux_lists['sentences_per_paragraph'])), 4)
        i['sentences_length_mean'] = round(float(np.mean(self.aux_lists['sentences_length_list'])), 4)
        i['sentences_length_no_stopwords_mean'] = round(
            float(np.mean(self.aux_lists['sentences_length_no_stopwords_list'])), 4)
        i['num_syllables_words_mean'] = round(float(np.mean(self.aux_lists['syllabes_list'])), 4)
        i['words_length_mean'] = round(float(np.mean(self.aux_lists['words_length_list'])), 4)
        i['words_length_no_stopwords_mean'] = round(float(np.mean(self.aux_lists['words_length_no_stopwords_list'])), 4)
        i['lemmas_length_mean'] = round(float(np.mean(self.aux_lists['lemmas_length_list'])), 4)

    def calculate_all_std_deviations(self):
        i = self.indicators
        i['sentences_per_paragraph_std'] = round(float(np.std(self.aux_lists['sentences_per_paragraph'])), 4)
        i['sentences_length_std'] = round(float(np.std(self.aux_lists['sentences_length_list'])), 4)
        i['sentences_length_no_stopwords_std'] = round(
            float(np.std(self.aux_lists['sentences_length_no_stopwords_list'])), 4)
        i['num_syllables_words_std'] = round(float(np.std(self.aux_lists['syllabes_list'])), 4)
        i['words_length_std'] = round(float(np.std(self.aux_lists['words_length_list'])), 4)
        i['words_length_no_stopwords_std'] = round(float(np.std(self.aux_lists['words_length_no_stopwords_list'])), 4)
        i['lemmas_length_std'] = round(float(np.std(self.aux_lists['lemmas_length_list'])), 4)

    def calculate_density(self):
        i = self.indicators
        i['lexical_density'] = round(i['num_lexic_words'] / i['num_words'], 4)
        i['noun_density'] = round(i['num_noun'] / i['num_words'], 4)
        i['verb_density'] = round(i['num_verb'] / i['num_words'], 4)
        i['adj_density'] = round(i['num_adj'] / i['num_words'], 4)
        i['adv_density'] = round(i['num_adv'] / i['num_words'], 4)

    def calculate_syntactic_density(self):
        i = self.indicators
        i['agentless_passive_density_incidence'] = Analyzer.get_incidence(i['num_agentless'], i['num_words'])
        i['negation_density_incidence'] = Analyzer.get_incidence(i['num_neg'], i['num_words'])
        i['gerund_density_incidence'] = Analyzer.get_incidence(i['num_ger'], i['num_words'])
        i['infinitive_density_incidence'] = Analyzer.get_incidence(i['num_inf'], i['num_words'])

    def calculate_phrases(self, num_vp_list, num_np_list):
        i = self.indicators
        i['mean_vp_per_sentence'] = round(float(np.mean(num_vp_list)), 4)
        i['mean_np_per_sentence'] = round(float(np.mean(num_np_list)), 4)
        i['noun_phrase_density_incidence'] = Analyzer.get_incidence(sum(num_np_list), i['num_words'])
        i['verb_phrase_density_incidence'] = Analyzer.get_incidence(sum(num_vp_list), i['num_words'])

    def create_dataframe(self, coh_metrix=False):
        i = self.indicators
        indicators_dict = {}
        headers = []
        ignore_list = ['prop', 'num_complex_words',
                       'num_words_more_3_syl', 'num_lexic_words']
        similarity_list = ["similarity_adjacent_mean", "similarity_pairs_par_mean", "similarity_adjacent_par_mean",
                           "similarity_adjacent_std", "similarity_pairs_par_std", "similarity_adjacent_par_std"]
        for key, value in i.items():
            if key not in ignore_list and not coh_metrix:
                if key not in similarity_list:
                    indicators_dict[key] = i.get(key)
                    headers.append(key)
            else:
                coh_metrix_list = ['num_words', 'num_paragraphs', 'num_sentences', 'noun_overlap_adjacent',
                                   'noun_overlap_all', 'argument_overlap_adjacent', 'argument_overlap_all',
                                   'stem_overlap_adjacent', 'stem_overlap_all',
                                   'content_overlap_adjacent_mean', 'content_overlap_adjacent_std',
                                   'content_overlap_all_mean', 'content_overlap_all_std',
                                   'all_connectives_incidence', 'causal_connectives_incidence',
                                   'logical_connectives_incidence', 'adversative_connectives_incidence',
                                   'temporal_connectives_incidence', 'flesch_kincaid', 'flesch', 'simple_ttr',
                                   'lemma_content_ttr', 'mtld', 'sentences_per_paragraph_mean',
                                   'sentences_length_mean', 'num_syllables_words_mean', 'words_length_mean',
                                   'sentences_per_paragraph_std', 'sentences_length_std',
                                   'num_syllables_words_std', 'words_length_std', 'left_embeddedness',
                                   'noun_phrase_density_incidence', 'verb_phrase_density_incidence',
                                   'agentless_passive_density_incidence', 'negation_density_incidence',
                                   'gerund_density_incidence', 'infinitive_density_incidence',
                                   'num_modifiers_noun_phrase', 'polysemic_index', 'hypernymy_index',
                                   'hypernymy_verbs_index', 'hypernymy_nouns_index', 'num_noun_incidence',
                                   'num_adj_incidence', 'num_adv_incidence', 'num_verb_incidence',
                                   'num_personal_pronouns_incidence', 'num_first_pers_sing_pron_incidence']
                if key in coh_metrix_list:
                    indicators_dict[key] = i.get(key)
                    headers.append(key)
        return pd.DataFrame(indicators_dict, columns=headers, index=[0])

    def predict_dificulty(self, data):
        feature_names = data.columns.tolist()
        X_test = data[feature_names]

        # Para cargarlo, simplemente hacer lo siguiente:
        clf = joblib.load('./classifiers/classifier_aztertest.pkl')

        with open("./classifiers/selectorAztertest.pickle", "rb") as f:
            selector = pickle.load(f)

        X_test_new = selector.transform(X_test)
        return clf.predict(X_test_new)

        ###############Tratamiento de texto###############################################

    @staticmethod
    def process_text(input=None, text=None):
        # quitar todos los retornos \n si contiene
        if input is not None:
            if ".txt" not in input:
                pre_text = textract.process(input)
                text = pre_text.decode()
                text = text.replace('\n', '@')
            else:
                with open(input, encoding='utf-8') as f:
                    text = f.read().replace('\n', '@')
            text = re.sub(r'@+', '', text)
        # remove text inside parentheses
        # text = re.sub(r'\([^)]*\)', '', text)
        # separa , . ! ( ) ? ; del texto con espacios, teniendo en cuenta que los no son numeros en el caso de , y .
        text = re.sub(r"\_", " ", text)
        text = re.sub(r'[.]+(?![0-9])', r' . ', text)
        text = re.sub(r'[,]+(?![0-9])', r' , ', text)
        text = re.sub(r"!", " ! ", text)
        text = re.sub(r"\(", " ( ", text)
        text = re.sub(r"\)", " ) ", text)
        text = re.sub(r"\?", " ? ", text)
        text = re.sub(r";", " ; ", text)
        # sustituye 2 espacios seguidos por 1
        text = re.sub(r"\s{2,}", " ", text)
        return text

    def analyze(self, similarity):
        i = self.indicators
        text = self.text
        self.calculate_num_words()
        self.calculate_sentences_per_paragraph(similarity)
        doc = self.standford(text)
        subordinadas_labels = ['csubj', 'csubj:pass', 'ccomp', 'xcomp', 'advcl', 'acl', 'acl:relcl']
        ambiguity_content_words_list = []
        noun_abstraction_list = []
        verb_abstraction_list = []
        noun_verb_abstraction_list = []
        min_wordfreq_list = []
        num_np_list = []
        num_vp_list = []
        depth_list = []

        # lista de np vacia
        modifiers_per_np = []

        decendents_total = 0
        for sent in doc.sentences:
            root = 0
            wordfreq_list = []
            tokens_sentence = []
            dependency_tree = defaultdict(list)
            i['num_sentences'] += 1
            num_punct_marks_in_sentence = 0
            num_words_in_sentences = 0
            num_words_in_sentence_without_stopwords = 0

            # consigo todos los index de categoria nombre, pronombre o nombre propio
            np_indexes = self.count_np_in_sentence(sent)
            # Por cada index que es un np(nombre, pronombre o nombre propio), devuelve una lista con el número de modificadores por cada np
            modifiers_per_np += self.count_modifiers(sent, np_indexes)

            num_np_list.append(len(np_indexes))
            sequence = []
            sequence = self.sent2sequence(sent)
            num_vp_list.append(self.count_vp_in_sentence(sent))

            decendents_total += self.count_decendents(sent, np_indexes)
            for entry in sent.words:
                # print(str(entry.index)+"\t"+entry.word+"\t"+entry.lemma+"\t"+entry.upos+"\t"+entry.xpos+"\t"+entry.attrs+"\t"+str(entry.head)+"\t"+str(entry.label))
                if int(entry.governor) == 0:
                    root = int(entry.index)
                dependency_tree[int(entry.governor)].append(int(entry.index))
                if self.is_not_stopword(entry.lemma):
                    tokens_sentence.append(entry.lemma)
                i['num_words_with_punct'] += 1
                if int(entry.index) == 1:
                    i['prop'] = 1
                if entry.dependency_relation == 'conj' or entry.dependency_relation == 'csubj' or entry.dependency_relation == 'csubj:pass' or entry.dependency_relation == 'ccomp' or entry.dependency_relation == 'xcomp' or entry.dependency_relation == 'advcl' or entry.dependency_relation == 'acl' or entry.dependency_relation == 'acl:relcl':
                    i['prop'] += 1
                # Numero de sentencias subordinadas
                if entry.dependency_relation in subordinadas_labels:
                    i['num_subord'] += 1
                    # Numero de sentencias subordinadas relativas
                    if entry.dependency_relation == 'acl:relcl':
                        i['num_rel_subord'] += 1
                if entry.upos == 'PUNCT':
                    num_punct_marks_in_sentence += 1
                if entry.upos == 'PRON':
                    self.count_personal_pronoun(entry)
                if entry.upos == 'NOUN':
                    i['num_noun'] += 1
                    if entry.text.lower() not in self.aux_lists['different_nouns']:
                        self.aux_lists['different_nouns'].append(entry.text.lower())
                    if entry.lemma not in self.aux_lists['different_lemma_nouns']:
                        self.aux_lists['different_lemma_nouns'].append(entry.lemma)
                if entry.upos == 'ADJ':
                    i['num_adj'] += 1
                    if entry.text.lower() not in self.aux_lists['different_adjs']:
                        self.aux_lists['different_adjs'].append(entry.text.lower())
                    if entry.lemma not in self.aux_lists['different_lemma_adjs']:
                        self.aux_lists['different_lemma_adjs'].append(entry.lemma)
                if entry.upos == 'ADV':
                    i['num_adv'] += 1
                    if entry.text.lower() not in self.aux_lists['different_advs']:
                        self.aux_lists['different_advs'].append(entry.text.lower())
                    if entry.lemma not in self.aux_lists['different_lemma_advs']:
                        self.aux_lists['different_lemma_advs'].append(entry.lemma)
                if entry.lemma == 'not':
                    i['num_neg'] += 1
                if self.is_verb(entry.upos, int(entry.governor), sequence):
                    i['num_verb'] += 1
                    if self.is_passive(entry):
                        i['num_pass'] += 1
                        if self.is_agentless(entry, sequence):
                            i['num_agentless'] += 1
                    if self.is_past(entry):
                        i['num_past'] += 1
                        if self.is_irregular(entry):
                            i['num_past_irregular'] += 1
                    if self.is_present(entry):
                        i['num_pres'] += 1
                    if self.is_infinitive(entry, sequence):
                        i['num_inf'] += 1
                    if self.is_gerund(entry):
                        i['num_ger'] += 1
                    if self.is_indicative(entry):
                        i['num_indic'] += 1
                    if self.is_imperative(entry):
                        i['num_impera'] += 1
                    if entry.text.lower() not in self.aux_lists['different_verbs']:
                        self.aux_lists['different_verbs'].append(entry.text.lower())
                    if entry.lemma not in self.aux_lists['different_lemma_verbs']:
                        self.aux_lists['different_lemma_verbs'].append(entry.lemma)
                if self.is_future(entry, sequence):
                    i['num_future'] += 1
                if self.is_not_stopword(entry.text):
                    num_words_in_sentence_without_stopwords += 1
                if self.has_more_than_three_syllables(entry.text):
                    i['num_words_more_3_syl'] += 1
                ######wordfreq###########################################
                if (not len(entry.text) == 1 or entry.text.isalpha()) and entry.upos != "NUM":
                    wordfrequency = zipf_frequency(entry.text, 'en')
                    wordfreq_list.append(wordfrequency)
                    num_words_in_sentences += 1
                    if (self.is_lexic_word(entry.upos, int(entry.governor), sequence)):
                        if wordfrequency <= 4:
                            i['num_rare_words_4'] += 1
                            if entry.upos == 'NOUN':
                                i['num_rare_nouns_4'] += 1
                            elif entry.upos == 'ADJ':
                                i['num_rare_adj_4'] += 1
                            elif entry.upos == 'ADV':
                                i['num_rare_advb_4'] += 1
                            elif self.is_verb(entry.upos, int(entry.governor), sequence):
                                i['num_rare_verbs_4'] += 1
                        if entry.text.lower() not in self.aux_lists['different_lexic_words']:
                            self.aux_lists['different_lexic_words'].append(entry.text.lower())
                            if wordfrequency <= 4:
                                i['num_dif_rare_words_4'] += 1
                        if wn.synsets(entry.text):
                            if entry.upos == 'NOUN':
                                noun_abstraction_list.append(self.get_abstraction_level(entry.text, 'n'))
                                noun_verb_abstraction_list.append(self.get_abstraction_level(entry.text, 'n'))
                            elif self.is_verb(entry.upos, int(entry.governor), sequence):
                                verb_abstraction_list.append(self.get_abstraction_level(entry.text, 'v'))
                                noun_verb_abstraction_list.append(self.get_abstraction_level(entry.text, 'v'))
                            ambiguity_content_words_list.append(self.get_ambiguity_level(entry.text, entry.upos))
                        i['num_lexic_words'] += 1
                    # Numero de lemas distintos en el texto
                    if entry.lemma not in self.aux_lists['different_lemmas']:
                        self.aux_lists['different_lemmas'].append(entry.text.lower())
                    # Numero de formas distintas en el texto
                    if entry.text.lower() not in self.aux_lists['different_forms']:
                        self.aux_lists['different_forms'].append(entry.text.lower())
                    # Lista de words con sus frecuencias
                    if entry.text.lower() not in self.words_freq:
                        self.words_freq[entry.text.lower()] = 1
                    else:
                        self.words_freq[entry.text.lower()] = self.words_freq.get(entry.text.lower()) + 1
                    if self.is_not_stopword(entry.text):
                        self.aux_lists['words_length_no_stopwords_list'].append(len(entry.text))
                    if self.is_complex(entry):
                        i['num_complex_words'] += 1
                    self.aux_lists['words_length_list'].append(len(entry.text))
                    self.aux_lists['lemmas_length_list'].append(len(entry.lemma))
            i['num_total_prop'] = i['num_total_prop'] + i['prop']
            self.aux_lists['list_num_punct_marks'].append(num_punct_marks_in_sentence)
            self.aux_lists['sentences_length_list'].append(num_words_in_sentences)
            self.aux_lists['sentences_length_no_stopwords_list'].append(num_words_in_sentence_without_stopwords)
            if len(wordfreq_list) > 0:
                min_wordfreq_list.append(min(wordfreq_list))
            else:
                min_wordfreq_list.append(0)
            self.aux_lists['all_sentences_tokens'].append(tokens_sentence)
            depth_list.append(self.tree_depth(dependency_tree, root))
        # if similarity:
        # self.calculate_similarity_adjacent_sentences()
        # self.calculate_similarity_adjacent_paragraphs()
        # self.calculate_similarity_pairs_sentences()
        self.calculate_all_overlaps()
        # self.calculate_connectives()
        self.get_syllable_list()
        self.calculate_dale_chall()
        self.calculate_smog()
        self.flesch_kincaid()
        self.flesch()
        self.calculate_all_ttr()
        self.calculate_all_lemma_ttr()
        self.calculate_honore()
        self.calculate_maas()
        self.calculate_mtld()
        self.calculate_all_means()
        self.calculate_all_std_deviations()
        self.calculate_left_embeddedness(doc)
        self.calculate_levels_oxford_word_list(doc)
        self.calculate_mean_depth_per_sentence(depth_list)
        i['num_different_forms'] = len(self.aux_lists['different_forms'])
        i['num_pass_mean'] = round((i['num_pass']) / i['num_words'], 4)
        i['num_past_irregular_mean'] = round(((i['num_past_irregular']) / i['num_past']), 4) if i[
                                                                                                    'num_past'] != 0 else 0
        i['num_punct_marks_per_sentence'] = round(float(np.mean(self.aux_lists['list_num_punct_marks'])), 4)
        i['mean_propositions_per_sentence'] = round(i['num_total_prop'] / i['num_sentences'], 4)
        self.calculate_phrases(num_vp_list, num_np_list)
        self.calculate_density()
        self.calculate_syntactic_density()

        i['num_modifiers_noun_phrase'] = round(float(np.mean(modifiers_per_np)), 4)

        i['num_decendents_noun_phrase'] = round(decendents_total / sum(num_np_list), 4)
        i['mean_rare_4'] = round(((100 * i['num_rare_words_4']) / i['num_lexic_words']), 4)
        i['mean_distinct_rare_4'] = round(
            (100 * i['num_dif_rare_words_4']) / len(self.aux_lists['different_lexic_words']), 4)
        i['min_wf_per_sentence'] = round(float(np.mean(min_wordfreq_list)), 4)
        self.calculate_polysemic_index(ambiguity_content_words_list)
        self.calculate_hypernymy_index(noun_verb_abstraction_list)
        self.calculate_hypernymy_index(verb_abstraction_list, 'V')
        self.calculate_hypernymy_index(noun_abstraction_list, 'N')
        self.calculate_all_incidence()
        return self.indicators


import os
import sys
from pathlib import Path
import csv
import pandas as pd


class Printer:

    def __init__(self, input, indicators):
        self.input = input
        self.indicators = indicators

    def print_info(self, similarity, prediction, file_num, total):
        i = self.indicators
        kk = prediction
        print("------------------------------------------------------------------------------")
        print("\n Processing file " + str(file_num) + " of " + str(total) + ": " + os.path.basename(self.input) + "\n",
              end='\n', file=sys.stdout, flush=True)
        print("------------------------------------------------------------------------------")
        # print('Level of difficulty: ' + prediction[0].title())
        print("------------------------------------------------------------------------------")
        print('Number of words (total): ' + str(i['num_words']))
        # The number of distints lower and alfabetic words
        print("Number of distinct words (total): " + str(i['num_different_forms']))
        print('Number of words with punctuation (total): ' + str(i['num_words_with_punct']))

        print("Number of paragraphs (total): " + str(i['num_paragraphs']))
        print("Number of paragraphs (incidence per 1000 words): " + str(i['num_paragraphs_incidence']))
        print('Number of sentences (total): ' + str(i['num_sentences']))
        print('Number of sentences (incidence per 1000 words): ' + str(i['num_sentences_incidence']))

        # Numero de frases en un parrafo (media)
        print("Length of paragraphs (mean): " + str(i['sentences_per_paragraph_mean']))
        # Numero de frases en un parrafo (desv. Tipica)
        print("Standard deviation of length of paragraphs: " + str(i['sentences_per_paragraph_std']))

        print("Number of words (length) in sentences (mean): " + str(i['sentences_length_mean']))
        print("Number of words (length) in sentences (standard deviation): " + str(i['sentences_length_std']))

        print("Number of words (length) of sentences without stopwords (mean): " + str(
            i['sentences_length_no_stopwords_mean']))
        print("Number of words (length) of sentences without stopwords (standard deviation): " + str(
            i['sentences_length_no_stopwords_std']))

        print('Mean number of syllables (length) in words: ' + str(i['num_syllables_words_mean']))
        print('Standard deviation of the mean number of syllables in words: ' + str(i['num_syllables_words_std']))

        print("Mean number of letters (length) in words: " + str(i['words_length_mean']))
        print("Standard deviation of number of letters in words: " + str(i['words_length_std']))

        print("Mean number of letters (length) in words without stopwords: " + str(i['words_length_no_stopwords_mean']))
        print("Standard deviation of the mean number of letter in words without stopwords: " + str(
            i['words_length_no_stopwords_std']))

        print("Mean number of letters (length) in lemmas: " + str(i['lemmas_length_mean']))
        print("Standard deviation of letters (length) in lemmas: " + str(i['lemmas_length_std']))

        print('Lexical Density: ' + str(i['lexical_density']))
        print("Noun Density: " + str(i['noun_density']))
        print("Verb Density: " + str(i['verb_density']))
        print("Adjective Density: " + str(i['adj_density']))
        print("Adverb Density: " + str(i['adv_density']))

        # Simple TTR (Type-Token Ratio)
        print('STTR (Simple Type-Token Ratio) : ' + str(i['simple_ttr']))
        # Content TTR (Content Type-Token Ratio)
        print('CTTR (Content Type-Token Ratio): ' + str(i['content_ttr']))
        # NTTR (Noun Type-Token Ratio)
        print('NTTR (Noun Type-Token Ratio): ' + str(i['nttr']))
        # VTTR (Verb Type-Token Ratio)(incidence per 1000 words)
        print('VTTR (Verb Type-Token Ratio): ' + str(i['vttr']))

        # AdjTTR (Adj Type-Token Ratio)
        print('AdjTTR (Adj Type-Token Ratio): ' + str(i['adj_ttr']))
        # AdvTTR (Adv Type-Token Ratio)
        print('AdvTTR (Adv Type-Token Ratio): ' + str(i['adv_ttr']))

        # Lemma Simple TTR (Type-Token Ratio)
        print('LSTTR (Lemma Simple Type-Token Ratio): ' + str(i['lemma_ttr']))
        # Lemma Content TTR (Content Type-Token Ratio)
        print('LCTTR (Lemma Content Type-Token Ratio): ' + str(i['lemma_content_ttr']))
        # LNTTR (Lemma Noun Type-Token Ratio)
        print('LNTTR (Lemma Noun Type-Token Ratio) ' + str(i['lemma_nttr']))
        # LVTTR (Lemma Verb Type-Token Ratio)
        print('LVTTR (Lemma Verb Type-Token Ratio): ' + str(i['lemma_vttr']))
        # Lemma AdjTTR (Lemma Adj Type-Token Ratio)
        print('LAdjTTR (Lemma Adj Type-Token Ratio): ' + str(i['lemma_adj_ttr']))
        # Lemma AdvTTR (Lemma Adv Type-Token Ratio)
        print('LAdvTTR (Lemma Adv Type-Token Ratio): ' + str(i['lemma_adv_ttr']))

        # Honore
        print('Honore Lexical Density: ' + str(i['honore']))
        # Maas
        print('Maas Lexical Density: ' + str(i['maas']))
        # MTLD
        print('Measure of Textual Lexical Diversity (MTLD): ' + str(i['mtld']))

        # Flesch-Kincaid grade level =0.39 * (n.º de words/nº de frases) + 11.8 * (n.º de silabas/numero de words) – 15.59)
        print("Flesch-Kincaid Grade level: " + str(i['flesch_kincaid']))
        # Flesch readability ease=206.835-1.015(n.º de words/nº de frases)-84.6(n.º de silabas/numero de words)
        print("Flesch readability ease: " + str(i['flesch']))

        print("Dale-Chall readability formula: " + str(i['dale_chall']))
        print("Simple Measure Of Gobbledygook (SMOG) grade: " + str(i['smog']))

        print("Number of verbs in past tense: " + str(i['num_past']))
        print("Number of verbs in past tense (incidence per 1000 words): " + str(i['num_past_incidence']))
        print("Number of verbs in present tense: " + str(i['num_pres']))
        print("Number of verbs in present tense (incidence per 1000 words): " + str(i['num_pres_incidence']))
        print("Number of verbs in future tense: " + str(i['num_future']))
        print("Number of verbs in future tense (incidence per 1000 words): " + str(i['num_future_incidence']))

        # Numero de verbos en modo indicativo
        print("Number of verbs in indicative mood: " + str(i['num_indic']))
        print("Number of verbs in indicative mood (incidence per 1000 words): " + str(i['num_indic_incidence']))
        # Numero de verbos en modo imperativo
        print("Number of verbs in imperative mood: " + str(i['num_impera']))
        print("Number of verbs in imperative mood (incidence per 1000 words): " + str(i['num_impera_incidence']))
        # Numero de verbos en pasado que son irregulares (total)
        print("Number of irregular verbs in past tense: " + str(i['num_past_irregular']))
        # Numero de verbos en pasado que son irregulares (incidencia 1000 words)
        print("Number of irregular verbs in past tense (incidence per 1000 words): " + str(
            i['num_past_irregular_incidence']))
        # Porcentaje de verbos en pasado que son irregulares sobre total de verbos en pasado
        print("Mean of irregular verbs in past tense in relation to the number of verbs in past tense: " + str(
            i['num_past_irregular_mean']))
        # Number of personal pronouns
        print("Number of personal pronouns: " + str(i['num_personal_pronouns']))
        # Incidence score of pronouns (per 1000 words)
        print("Incidence score of pronouns (per 1000 words): " + str(i['num_personal_pronouns_incidence']))
        # Number of pronouns in first person
        print("Number of pronouns in first person: " + str(i['num_first_pers_pron']))
        # Incidence score of pronouns in first person  (per 1000 words)
        print(
            "Incidence score of pronouns in first person  (per 1000 words): " + str(i['num_first_pers_pron_incidence']))
        # Number of pronouns in first person singular
        print("Number of pronouns in first person singular: " + str(i['num_first_pers_sing_pron']))
        # Incidence score of pronouns in first person singular (per 1000 words)
        print("Incidence score of pronouns in first person singular (per 1000 words): " + str(
            i['num_first_pers_sing_pron_incidence']))
        # Number of pronouns in third person
        print("Number of pronouns in third person: " + str(i['num_third_pers_pron']))
        # Incidence score of pronouns in third person (per 1000 words)
        print(
            "Incidence score of pronouns in third person (per 1000 words): " + str(i['num_third_pers_pron_incidence']))

        print('Minimum word frequency per sentence (mean): ' + str(i['min_wf_per_sentence']))
        print('Number of rare nouns (wordfrecuency<=4): ' + str(i['num_rare_nouns_4']))
        print('Number of rare nouns (wordfrecuency<=4) (incidence per 1000 words): ' + str(
            i['num_rare_nouns_4_incidence']))
        print('Number of rare adjectives (wordfrecuency<=4): ' + str(i['num_rare_adj_4']))
        print('Number of rare adjectives (wordfrecuency<=4) (incidence per 1000 words): ' + str(
            i['num_rare_adj_4_incidence']))
        print('Number of rare verbs (wordfrecuency<=4): ' + str(i['num_rare_verbs_4']))
        print('Number of rare verbs (wordfrecuency<=4) (incidence per 1000 words): ' + str(
            i['num_rare_verbs_4_incidence']))
        print('Number of rare adverbs (wordfrecuency<=4): ' + str(i['num_rare_advb_4']))
        print('Number of rare adverbs (wordfrecuency<=4) (incidence per 1000 words): ' + str(
            i['num_rare_advb_4_incidence']))
        print('Number of rare content words (wordfrecuency<=4): ' + str(i['num_rare_words_4']))
        print('Number of rare content words (wordfrecuency<=4) (incidence per 1000 words): ' + str(
            i['num_rare_words_4_incidence']))
        print('Number of distinct rare content words (wordfrecuency<=4): ' + str(i['num_dif_rare_words_4']))
        print('Number of distinct rare content words (wordfrecuency<=4) (incidence per 1000 words): ' + str(
            i['num_dif_rare_words_4_incidence']))
        # The average of rare lexical words (whose word frequency value is less than 4) with respect to the total of lexical words
        print('Mean of rare lexical words (word frequency <= 4): ' + str(i['mean_rare_4']))
        # The average of distinct rare lexical words (whose word frequency value is less than 4) with respect to the total of distinct lexical words
        print('Mean of distinct rare lexical words (word frequency <= 4): ' + str(i['mean_distinct_rare_4']))

        print('Number of A1 vocabulary in the text: ' + str(i['num_a1_words']))
        print('Incidence score of A1 vocabulary  (per 1000 words): ' + str(i['num_a1_words_incidence']))
        print('Number of A2 vocabulary in the text: ' + str(i['num_a2_words']))
        print('Incidence score of A2 vocabulary  (per 1000 words): ' + str(i['num_a2_words_incidence']))
        print('Number of B1 vocabulary in the text: ' + str(i['num_b1_words']))
        print('Incidence score of B1 vocabulary  (per 1000 words): ' + str(i['num_b1_words_incidence']))
        print('Number of B2 vocabulary in the text: ' + str(i['num_b2_words']))
        print('Incidence score of B2 vocabulary  (per 1000 words): ' + str(i['num_b2_words_incidence']))
        print('Number of C1 vocabulary in the text: ' + str(i['num_c1_words']))
        print('Incidence score of C1 vocabulary  (per 1000 words): ' + str(i['num_c1_words_incidence']))
        print('Number of content words not in A1-C1 vocabulary: ' + str(i['num_content_words_not_a1_c1_words']))
        print('Incidence score of content words not in A1-C1 vocabulary (per 1000 words): ' + str(
            i['num_content_words_not_a1_c1_words_incidence']))

        print('Number of content words: ' + str(i['num_lexic_words']))
        print('Number of content words (incidence per 1000 words): ' + str(i['num_lexic_words_incidence']))
        print("Number of nouns: " + str(i['num_noun']))
        print("Number of nouns (incidence per 1000 words): " + str(i['num_noun_incidence']))
        print("Number of adjectives: " + str(i['num_adj']))
        print("Number of adjectives (incidence per 1000 words): " + str(i['num_adj_incidence']))
        print("Number of adverbs: " + str(i['num_adv']))
        print("Number of adverbs (incidence per 1000 words): " + str(i['num_adv_incidence']))
        print("Number of verbs: " + str(i['num_verb']))
        print("Number of verbs (incidence per 1000 words): " + str(i['num_verb_incidence']))
        # Left-Embeddedness
        print(
            "Left embeddedness (Mean of number of words before the main verb) (SYNLE): " + str(i['left_embeddedness']))

        print("Number of decendents per noun phrase (mean): " + str(i['num_decendents_noun_phrase']))
        print("Number of modifiers per noun phrase (mean) (SYNNP): " + str(i['num_modifiers_noun_phrase']))
        print("Mean of the number of levels of dependency tree (Depth): " + str(i['mean_depth_per_sentence']))

        # Numero de sentencias subordinadas
        print("Number of subordinate clauses: " + str(i['num_subord']))
        # Numero de sentencias subordinadas (incidence per 1000 words)
        print("Number of subordinate clauses (incidence per 1000 words): " + str(i['num_subord_incidence']))
        # Numero de sentencias subordinadas relativas
        print("Number of relative subordinate clauses: " + str(i['num_rel_subord']))
        # Numero de sentencias subordinadas relativas (incidence per 1000 words)
        print(
            "Number of relative subordinate clauses (incidence per 1000 words): " + str(i['num_rel_subord_incidence']))
        # Marcas de puntuacion por sentencia (media)
        print("Punctuation marks per sentence (mean): " + str(i['num_punct_marks_per_sentence']))
        print('Number of propositions: ' + str(i['num_total_prop']))
        # Mean of the number of propositions per sentence
        print('Mean of the number of propositions per sentence: ' + str(i['mean_propositions_per_sentence']))

        print('Mean of the number of VPs per sentence: ' + str(i['mean_vp_per_sentence']))
        print('Mean of the number of NPs per sentence: ' + str(i['mean_np_per_sentence']))
        print('Noun phrase density, incidence (DRNP): ' + str(i['noun_phrase_density_incidence']))
        print('Verb phrase density, incidence (DRVP): ' + str(i['verb_phrase_density_incidence']))
        # Numero de verbos en pasiva (total)
        print("Number of passive voice verbs: " + str(i['num_pass']))
        # Numero de verbos en pasiva (incidence per 1000 words)
        print("Number of passive voice verbs (incidence per 1000 words): " + str(i['num_pass_incidence']))
        # Porcentaje de verbos en pasiva
        print("Mean of passive voice verbs: " + str(i['num_pass_mean']))
        # Numero de verbos en pasiva que no tienen agente
        print("Number of agentless passive voice verbs: " + str(i['num_agentless']))
        print('Agentless passive voice density, incidence (DRPVAL): ' + str(i['agentless_passive_density_incidence']))
        print("Number of negative words: " + str(i['num_neg']))
        print('Negation density, incidence (DRNEG): ' + str(i['negation_density_incidence']))
        print("Number of verbs in gerund form: " + str(i['num_ger']))
        print('Gerund density, incidence (DRGERUND): ' + str(i['gerund_density_incidence']))
        print("Number of verbs in infinitive form: " + str(i['num_inf']))
        print('Infinitive density, incidence (DRINF): ' + str(i['infinitive_density_incidence']))

        # Ambigüedad de una palabra (polysemy in WordNet)
        print('Mean values of polysemy in the WordNet lexicon: ' + str(i['polysemic_index']))
        # Nivel de abstracción (hypernym in WordNet)
        print('Mean hypernym values of verbs in the WordNet lexicon: ' + str(i['hypernymy_verbs_index']))
        print('Mean hypernym values of nouns in the WordNet lexicon: ' + str(i['hypernymy_nouns_index']))
        print('Mean hypernym values of nouns and verbs in the WordNet lexicon: ' + str(i['hypernymy_index']))

        # Textbase. Referential cohesion
        print('Noun overlap, adjacent sentences, binary, mean (CRFNOl): ' + str(i['noun_overlap_adjacent']))
        print('Noun overlap, all of the sentences in a paragraph or text, binary, mean (CRFNOa): ' + str(
            i['noun_overlap_all']))
        print('Argument overlap, adjacent sentences, binary, mean (CRFAOl): ' + str(i['argument_overlap_adjacent']))
        print('Argument overlap, all of the sentences in a paragraph or text, binary, mean (CRFAOa): ' + str(
            i['argument_overlap_all']))
        print('Stem overlap, adjacent sentences, binary, mean (CRFSOl): ' + str(i['stem_overlap_adjacent']))
        print('Stem overlap, all of the sentences in a paragraph or text, binary, mean (CRFSOa): ' + str(
            i['stem_overlap_all']))
        print('Content word overlap, adjacent sentences, proportional, mean (CRFCWO1): ' + str(
            i['content_overlap_adjacent_mean']))
        print('Content word overlap, adjacent sentences, proportional, standard deviation (CRFCWO1d): ' + str(
            i['content_overlap_adjacent_std']))
        print('Content word overlap, all of the sentences in a paragraph or text, proportional, mean (CRFCWOa): ' + str(
            i['content_overlap_all_mean']))
        print(
            'Content word overlap, all of the sentences in a paragraph or text, proportional, standard deviation (CRFCWOad): ' + str(
                i['content_overlap_all_std']))

        # Semantic Similarity
        if similarity:
            print('Semantic Similarity between adjacent sentences (mean): ' + str(i['similarity_adjacent_mean']))
            print('Semantic Similarity between all possible pairs of sentences in a paragraph (mean):  ' + str(
                i['similarity_pairs_par_mean']))
            print('Semantic Similarity between adjacent paragraphs (mean): ' + str(i['similarity_adjacent_par_mean']))
            print('Semantic Similarity between adjacent sentences (standard deviation): ' + str(
                i['similarity_adjacent_std']))
            print(
                'Semantic Similarity between all possible pairs of sentences in a paragraph (standard deviation):  ' + str(
                    i['similarity_pairs_par_std']))
            print('Semantic Similarity between adjacent paragraphs (standard deviation): ' + str(
                i['similarity_adjacent_par_std']))

        # Connectives
        print('Number of connectives (incidence per 1000 words): ' + str(i['all_connectives_incidence']))
        print('Causal connectives (incidence per 1000 words): ' + str(i['causal_connectives_incidence']))
        print('Logical connectives (incidence per 1000 words):  ' + str(i['logical_connectives_incidence']))
        print('Adversative/contrastive connectives (incidence per 1000 words): ' + str(
            i['adversative_connectives_incidence']))
        print('Temporal connectives (incidence per 1000 words):  ' + str(i['temporal_connectives_incidence']))
        print('Conditional connectives (incidence per 1000 words): ' + str(i['conditional_connectives_incidence']))

    def generate_csv(self, csv_path, prediction, similarity):
        i = self.indicators
        kk = prediction
        # estadisticos
        output = os.path.join(csv_path, os.path.basename(self.input) + ".out.csv")
        # Write all the information in the file
        estfile = open(output, "w")
        # estfile.write("%s" % 'Level of difficulty: ' + prediction[0].title())
        estfile.write("\n%s" % 'Shallow or descriptive measures')
        estfile.write("\n%s" % 'Number of words (total): ' + str(i['num_words']))
        estfile.write("\n%s" % 'Number of distinct words (total): ' + str(i['num_different_forms']))
        estfile.write("\n%s" % 'Number of words with punctuation (total): ' + str(i['num_words_with_punct']))
        estfile.write("\n%s" % 'Number of paragraphs (total): ' + str(i['num_paragraphs']))
        estfile.write("\n%s" % 'Number of paragraphs (incidence per 1000 words): ' + str(i['num_paragraphs_incidence']))
        estfile.write("\n%s" % 'Number of sentences (total): ' + str(i['num_sentences']))
        estfile.write("\n%s" % 'Number of sentences (incidence per 1000 words): ' + str(i['num_sentences_incidence']))
        estfile.write("\n%s" % 'Length of paragraphs (mean): ' + str(i['sentences_per_paragraph_mean']))
        estfile.write("\n%s" % 'Standard deviation of length of paragraphs: ' + str(i['sentences_per_paragraph_std']))
        estfile.write("\n%s" % 'Number of words (length) in sentences (mean): ' + str(i['sentences_length_mean']))
        estfile.write(
            "\n%s" % 'Number of words (length) in sentences (standard deviation): ' + str(i['sentences_length_std']))
        estfile.write("\n%s" % 'Number of words (length) of sentences without stopwords (mean): ' + str(
            i['sentences_length_no_stopwords_mean']))
        estfile.write("\n%s" % 'Number of words (length) of sentences without stopwords (standard deviation): ' + str(
            i['sentences_length_no_stopwords_std']))
        estfile.write("\n%s" % 'Mean number of syllables (length) in words: ' + str(i['num_syllables_words_mean']))
        estfile.write("\n%s" % 'Standard deviation of the mean number of syllables in words: ' + str(
            i['num_syllables_words_std']))
        estfile.write("\n%s" % 'Mean number of letters (length) in words: ' + str(i['words_length_mean']))
        estfile.write("\n%s" % 'Standard deviation of number of letters in words: ' + str(i['words_length_std']))
        estfile.write("\n%s" % 'Mean number of letters (length) in words without stopwords: ' + str(
            i['words_length_no_stopwords_mean']))
        estfile.write("\n%s" % 'Standard deviation of the mean number of letter in words without stopwords: ' + str(
            i['words_length_no_stopwords_std']))
        estfile.write("\n%s" % 'Mean number of letters (length) in lemmas: ' + str(i['lemmas_length_mean']))
        estfile.write("\n%s" % 'Standard deviation of letters (length) in lemmas: ' + str(i['lemmas_length_std']))
        estfile.write("\n%s" % 'Lexical Richness/Lexical Density')
        estfile.write("\n%s" % 'Lexical Density: ' + str(i['lexical_density']))
        estfile.write("\n%s" % 'Noun Density: ' + str(i['noun_density']))
        estfile.write("\n%s" % 'Verb Density: ' + str(i['verb_density']))
        estfile.write("\n%s" % 'Adjective Density: ' + str(i['adj_density']))
        estfile.write("\n%s" % 'Adverb Density: ' + str(i['adv_density']))
        estfile.write("\n%s" % 'STTR (Simple Type-Token Ratio): ' + str(i['simple_ttr']))
        estfile.write("\n%s" % 'CTTR (Content Type-Token Ratio): ' + str(i['content_ttr']))
        estfile.write("\n%s" % 'NTTR (Noun Type-Token Ratio): ' + str(i['nttr']))
        estfile.write("\n%s" % 'VTTR (Verb Type-Token Ratio): ' + str(i['vttr']))
        estfile.write("\n%s" % 'AdjTTR (Adj Type-Token Ratio): ' + str(i['adj_ttr']))
        estfile.write("\n%s" % 'AdvTTR (Adv Type-Token Ratio): ' + str(i['adv_ttr']))
        estfile.write("\n%s" % 'LSTTR (Lemma Simple Type-Token Ratio): ' + str(i['lemma_ttr']))
        estfile.write("\n%s" % 'LCTTR (Lemma Content Type-Token Ratio): ' + str(i['lemma_content_ttr']))
        estfile.write("\n%s" % 'LNTTR (Lemma Noun Type-Token Ratio): ' + str(i['lemma_nttr']))
        estfile.write("\n%s" % 'LVTTR (Lemma Verb Type-Token Ratio): ' + str(i['lemma_vttr']))
        estfile.write("\n%s" % 'LAdjTTR (Lemma Adj Type-Token Ratio): ' + str(i['lemma_adj_ttr']))
        estfile.write("\n%s" % 'LAdvTTR (Lemma Adv Type-Token Ratio): ' + str(i['lemma_adv_ttr']))
        estfile.write("\n%s" % 'Honore Lexical Density: ' + str(i['honore']))
        estfile.write("\n%s" % 'Maas Lexical Density: ' + str(i['maas']))
        estfile.write("\n%s" % 'Measure of Textual Lexical Diversity (MTLD): ' + str(i['mtld']))
        estfile.write("\n%s" % 'Readability/Text Dimension/Grade Level')
        estfile.write("\n%s" % 'Flesch-Kincaid Grade level: ' + str(i['flesch_kincaid']))
        estfile.write("\n%s" % 'Flesch readability ease: ' + str(i['flesch']))
        estfile.write("\n%s" % 'Dale-Chall readability formula: ' + str(i['dale_chall']))
        estfile.write("\n%s" % 'Simple Measure Of Gobbledygook (SMOG) grade: ' + str(i['smog']))
        estfile.write("\n%s" % 'Morphological features')
        estfile.write("\n%s" % 'Number of verbs in past tense: ' + str(i['num_past']))
        estfile.write(
            "\n%s" % 'Number of verbs in past tense (incidence per 1000 words): ' + str(i['num_past_incidence']))
        estfile.write("\n%s" % 'Number of verbs in present tense: ' + str(i['num_pres']))
        estfile.write(
            "\n%s" % 'Number of verbs in present tense (incidence per 1000 words): ' + str(i['num_pres_incidence']))
        estfile.write("\n%s" % 'Number of verbs in future tense: ' + str(i['num_future']))
        estfile.write(
            "\n%s" % 'Number of verbs in future tense (incidence per 1000 words): ' + str(i['num_future_incidence']))
        estfile.write("\n%s" % 'Number of verbs in indicative mood: ' + str(i['num_indic']))
        estfile.write(
            "\n%s" % 'Number of verbs in indicative mood (incidence per 1000 words): ' + str(i['num_indic_incidence']))
        estfile.write("\n%s" % 'Number of verbs in imperative mood: ' + str(i['num_impera']))
        estfile.write(
            "\n%s" % 'Number of verbs in imperative mood (incidence per 1000 words): ' + str(i['num_impera_incidence']))
        estfile.write("\n%s" % 'Number of irregular verbs in past tense: ' + str(i['num_past_irregular']))
        estfile.write("\n%s" % 'Number of irregular verbs in past tense (incidence per 1000 words): ' + str(
            i['num_past_irregular_incidence']))
        estfile.write(
            "\n%s" % 'Mean of irregular verbs in past tense in relation to the number of verbs in past tense: ' + str(
                i['num_past_irregular_mean']))
        estfile.write("\n%s" % 'Number of personal pronouns: ' + str(i['num_personal_pronouns']))
        estfile.write(
            "\n%s" % 'Incidence score of pronouns (per 1000 words): ' + str(i['num_personal_pronouns_incidence']))
        estfile.write("\n%s" % 'Number of pronouns in first person: ' + str(i['num_first_pers_pron']))
        estfile.write("\n%s" % 'Incidence score of pronouns in first person  (per 1000 words): ' + str(
            i['num_first_pers_pron_incidence']))
        estfile.write("\n%s" % 'Number of pronouns in first person singular: ' + str(i['num_first_pers_sing_pron']))
        estfile.write("\n%s" % 'Incidence score of pronouns in first person singular (per 1000 words): ' + str(
            i['num_first_pers_sing_pron_incidence']))
        estfile.write("\n%s" % 'Number of pronouns in third person: ' + str(i['num_third_pers_pron']))
        estfile.write("\n%s" % 'Incidence score of pronouns in third person (per 1000 words): ' + str(
            i['num_third_pers_pron_incidence']))
        estfile.write("\n%s" % 'Word Frequency')
        estfile.write("\n%s" % 'Minimum word frequency per sentence (mean): ' + str(i['min_wf_per_sentence']))
        estfile.write("\n%s" % 'Number of rare nouns (wordfrecuency<=4): ' + str(i['num_rare_nouns_4']))
        estfile.write("\n%s" % 'Number of rare nouns (wordfrecuency<=4) (incidence per 1000 words): ' + str(
            i['num_rare_nouns_4_incidence']))
        estfile.write("\n%s" % 'Number of rare adjectives (wordfrecuency<=4): ' + str(i['num_rare_adj_4']))
        estfile.write("\n%s" % 'Number of rare adjectives (wordfrecuency<=4) (incidence per 1000 words): ' + str(
            i['num_rare_adj_4_incidence']))
        estfile.write("\n%s" % 'Number of rare verbs (wordfrecuency<=4): ' + str(i['num_rare_verbs_4']))
        estfile.write("\n%s" % 'Number of rare verbs (wordfrecuency<=4) (incidence per 1000 words): ' + str(
            i['num_rare_verbs_4_incidence']))
        estfile.write("\n%s" % 'Number of rare adverbs (wordfrecuency<=4): ' + str(i['num_rare_advb_4']))
        estfile.write("\n%s" % 'Number of rare adverbs (wordfrecuency<=4) (incidence per 1000 words): ' + str(
            i['num_rare_advb_4_incidence']))
        estfile.write("\n%s" % 'Number of rare content words (wordfrecuency<=4): ' + str(i['num_rare_words_4']))
        estfile.write("\n%s" % 'Number of rare content words (wordfrecuency<=4) (incidence per 1000 words): ' + str(
            i['num_rare_words_4_incidence']))
        estfile.write(
            "\n%s" % 'Number of distinct rare content words (wordfrecuency<=4): ' + str(i['num_dif_rare_words_4']))
        estfile.write(
            "\n%s" % 'Number of distinct rare content words (wordfrecuency<=4) (incidence per 1000 words): ' + str(
                i['num_dif_rare_words_4_incidence']))
        estfile.write("\n%s" % 'Mean of rare lexical words (word frequency <= 4): ' + str(i['mean_rare_4']))
        estfile.write(
            "\n%s" % 'Mean of distinct rare lexical words (word frequency <= 4): ' + str(i['mean_distinct_rare_4']))
        estfile.write("\n%s" % 'Vocabulary Knowledge')
        estfile.write("\n%s" % 'Number of A1 vocabulary in the text: ' + str(i['num_a1_words']))
        estfile.write(
            "\n%s" % 'Incidence score of A1 vocabulary  (per 1000 words): ' + str(i['num_a1_words_incidence']))
        estfile.write("\n%s" % 'Number of A2 vocabulary in the text: ' + str(i['num_a2_words']))
        estfile.write(
            "\n%s" % 'Incidence score of A2 vocabulary  (per 1000 words): ' + str(i['num_a2_words_incidence']))
        estfile.write("\n%s" % 'Number of B1 vocabulary in the text: ' + str(i['num_b1_words']))
        estfile.write(
            "\n%s" % 'Incidence score of B1 vocabulary  (per 1000 words): ' + str(i['num_b1_words_incidence']))
        estfile.write("\n%s" % 'Number of B2 vocabulary in the text: ' + str(i['num_b2_words']))
        estfile.write(
            "\n%s" % 'Incidence score of B2 vocabulary  (per 1000 words): ' + str(i['num_b2_words_incidence']))
        estfile.write("\n%s" % 'Number of C1 vocabulary in the text: ' + str(i['num_c1_words']))
        estfile.write(
            "\n%s" % 'Incidence score of C1 vocabulary  (per 1000 words): ' + str(i['num_c1_words_incidence']))
        estfile.write(
            "\n%s" % 'Number of content words not in A1-C1 vocabulary: ' + str(i['num_content_words_not_a1_c1_words']))
        estfile.write("\n%s" % 'Incidence score of content words not in A1-C1 vocabulary (per 1000 words): ' + str(
            i['num_content_words_not_a1_c1_words_incidence']))
        estfile.write("\n%s" % 'Syntactic Features / POS ratios')
        estfile.write("\n%s" % 'Number of content words: ' + str(i['num_lexic_words']))
        estfile.write(
            "\n%s" % 'Number of content words (incidence per 1000 words): ' + str(i['num_lexic_words_incidence']))
        estfile.write("\n%s" % 'Number of nouns: ' + str(i['num_noun']))
        estfile.write("\n%s" % 'Number of nouns (incidence per 1000 words): ' + str(i['num_noun_incidence']))
        estfile.write("\n%s" % 'Number of adjectives: ' + str(i['num_adj']))
        estfile.write("\n%s" % 'Number of adjectives (incidence per 1000 words): ' + str(i['num_adj_incidence']))
        estfile.write("\n%s" % 'Number of adverbs: ' + str(i['num_adv']))
        estfile.write("\n%s" % 'Number of adverbs (incidence per 1000 words): ' + str(i['num_adv_incidence']))
        estfile.write("\n%s" % 'Number of verbs: ' + str(i['num_verb']))
        estfile.write("\n%s" % 'Number of verbs (incidence per 1000 words): ' + str(i['num_verb_incidence']))
        estfile.write("\n%s" % 'Left embeddedness (Mean of number of words before the main verb) (SYNLE): ' + str(
            i['left_embeddedness']))
        estfile.write("\n%s" % 'Number of decendents per noun phrase (mean): ' + str(i['num_decendents_noun_phrase']))
        estfile.write(
            "\n%s" % 'Number of modifiers per noun phrase (mean) (SYNNP): ' + str(i['num_modifiers_noun_phrase']))
        estfile.write(
            "\n%s" % 'Mean of the number of levels of dependency tree (Depth): ' + str(i['mean_depth_per_sentence']))
        estfile.write("\n%s" % 'Number of subordinate clauses: ' + str(i['num_subord']))
        estfile.write(
            "\n%s" % 'Number of subordinate clauses (incidence per 1000 words): ' + str(i['num_subord_incidence']))
        estfile.write("\n%s" % 'Number of relative subordinate clauses: ' + str(i['num_rel_subord']))
        estfile.write("\n%s" % 'Number of relative subordinate clauses (incidence per 1000 words): ' + str(
            i['num_rel_subord_incidence']))
        estfile.write("\n%s" % 'Punctuation marks per sentence (mean): ' + str(i['num_punct_marks_per_sentence']))
        estfile.write("\n%s" % 'Number of propositions: ' + str(i['num_total_prop']))
        estfile.write(
            "\n%s" % 'Mean of the number of propositions per sentence: ' + str(i['mean_propositions_per_sentence']))
        estfile.write("\n%s" % 'Mean of the number of VPs per sentence: ' + str(i['mean_vp_per_sentence']))
        estfile.write("\n%s" % 'Mean of the number of NPs per sentence: ' + str(i['mean_np_per_sentence']))
        estfile.write("\n%s" % 'Noun phrase density, incidence (DRNP): ' + str(i['noun_phrase_density_incidence']))
        estfile.write("\n%s" % 'Verb phrase density, incidence (DRVP): ' + str(i['verb_phrase_density_incidence']))
        estfile.write("\n%s" % "Number of passive voice verbs: " + str(i['num_pass']))
        estfile.write(
            "\n%s" % "Number of passive voice verbs (incidence per 1000 words): " + str(i['num_pass_incidence']))
        estfile.write("\n%s" % "Mean of passive voice verbs: " + str(i['num_pass_mean']))
        estfile.write("\n%s" % "Number of agentless passive voice verbs: " + str(i['num_agentless']))
        estfile.write("\n%s" % 'Agentless passive voice density, incidence (DRPVAL): ' + str(
            i['agentless_passive_density_incidence']))
        estfile.write("\n%s" % "Number of negative words: " + str(i['num_neg']))
        estfile.write("\n%s" % 'Negation density, incidence (DRNEG): ' + str(i['negation_density_incidence']))
        estfile.write("\n%s" % "Number of verbs in gerund form: " + str(i['num_ger']))
        estfile.write("\n%s" % 'Gerund density, incidence (DRGERUND): ' + str(i['gerund_density_incidence']))
        estfile.write("\n%s" % "Number of verbs in infinitive form: " + str(i['num_inf']))
        estfile.write("\n%s" % 'Infinitive density, incidence (DRINF): ' + str(i['infinitive_density_incidence']))
        estfile.write("\n%s" % 'Semantics. Readability')
        estfile.write("\n%s" % 'Mean values of polysemy in the WordNet lexicon: ' + str(i['polysemic_index']))
        estfile.write(
            "\n%s" % 'Mean hypernym values of verbs in the WordNet lexicon: ' + str(i['hypernymy_verbs_index']))
        estfile.write(
            "\n%s" % 'Mean hypernym values of nouns in the WordNet lexicon: ' + str(i['hypernymy_nouns_index']))
        estfile.write(
            "\n%s" % 'Mean hypernym values of nouns and verbs in the WordNet lexicon: ' + str(i['hypernymy_index']))
        estfile.write("\n%s" % 'Referential cohesion')
        estfile.write(
            "\n%s" % 'Noun overlap, adjacent sentences, binary, mean (CRFNOl): ' + str(i['noun_overlap_adjacent']))
        estfile.write(
            "\n%s" % 'Noun overlap, all of the sentences in a paragraph or text, binary, mean (CRFNOa): ' + str(
                i['noun_overlap_all']))
        estfile.write("\n%s" % 'Argument overlap, adjacent sentences, binary, mean (CRFAOl): ' + str(
            i['argument_overlap_adjacent']))
        estfile.write(
            "\n%s" % 'Argument overlap, all of the sentences in a paragraph or text, binary, mean (CRFAOa): ' + str(
                i['argument_overlap_all']))
        estfile.write(
            "\n%s" % 'Stem overlap, adjacent sentences, binary, mean (CRFSOl): ' + str(i['stem_overlap_adjacent']))
        estfile.write(
            "\n%s" % 'Stem overlap, all of the sentences in a paragraph or text, binary, mean (CRFSOa): ' + str(
                i['stem_overlap_all']))
        estfile.write("\n%s" % 'Content word overlap, adjacent sentences, proportional, mean (CRFCWO1): ' + str(
            i['content_overlap_adjacent_mean']))
        estfile.write(
            "\n%s" % 'Content word overlap, adjacent sentences, proportional, standard deviation (CRFCWO1d): ' + str(
                i['content_overlap_adjacent_std']))
        estfile.write(
            "\n%s" % 'Content word overlap, all of the sentences in a paragraph or text, proportional, mean (CRFCWOa): ' + str(
                i['content_overlap_all_mean']))
        estfile.write(
            "\n%s" % 'Content word overlap, all of the sentences in a paragraph or text, proportional, standard deviation (CRFCWOad): ' + str(
                i['content_overlap_all_std']))
        if similarity:
            estfile.write("\n%s" % 'Semantic overlap')
            estfile.write(
                "\n%s" % 'Semantic Similarity between adjacent sentences (mean): ' + str(i['similarity_adjacent_mean']))
            estfile.write(
                "\n%s" % 'Semantic Similarity between all possible pairs of sentences in a paragraph (mean): ' + str(
                    i['similarity_pairs_par_mean']))
            estfile.write("\n%s" % 'Semantic Similarity between adjacent paragraphs (mean): ' + str(
                i['similarity_adjacent_par_mean']))
            estfile.write("\n%s" % 'Semantic Similarity between adjacent sentences (standard deviation): ' + str(
                i['similarity_adjacent_std']))
            estfile.write(
                "\n%s" % 'Semantic Similarity between all possible pairs of sentences in a paragraph (standard deviation): ' + str(
                    i['similarity_pairs_par_std']))
            estfile.write("\n%s" % 'Semantic Similarity between adjacent paragraphs (standard deviation): ' + str(
                i['similarity_adjacent_par_std']))
        estfile.write("\n%s" % 'Connectives')
        estfile.write(
            "\n%s" % 'Number of connectives (incidence per 1000 words): ' + str(i['all_connectives_incidence']))
        estfile.write(
            "\n%s" % 'Causal connectives (incidence per 1000 words): ' + str(i['causal_connectives_incidence']))
        estfile.write(
            "\n%s" % 'Logical connectives (incidence per 1000 words):  ' + str(i['logical_connectives_incidence']))
        estfile.write("\n%s" % 'Adversative/contrastive connectives (incidence per 1000 words): ' + str(
            i['adversative_connectives_incidence']))
        estfile.write(
            "\n%s" % 'Temporal connectives (incidence per 1000 words):  ' + str(i['temporal_connectives_incidence']))
        estfile.write("\n%s" % 'Conditional connectives (incidence per 1000 words): ' + str(
            i['conditional_connectives_incidence']))
        estfile.close()

    def write_in_full_csv(self, df, similarity):
        i = self.indicators
        indicators_dict = {}
        headers = []
        ignore_list = ['prop', 'num_complex_words', 'num_words_more_3_syl', 'num_lexic_words']
        similarity_list = ["similarity_adjacent_mean", "similarity_pairs_par_mean", "similarity_adjacent_par_mean",
                           "similarity_adjacent_std", "similarity_pairs_par_std", "similarity_adjacent_par_std"]
        for key, value in i.items():
            if key not in ignore_list:
                if similarity:
                    indicators_dict[key] = i.get(key)
                    headers.append(key)
                else:
                    if key not in similarity_list:
                        indicators_dict[key] = i.get(key)
                        headers.append(key)
        df_new = pd.DataFrame([indicators_dict], columns=indicators_dict.keys())
        df = pd.concat([df, df_new], sort=False)
        return df

    @staticmethod
    def create_directory(path):
        newPath = os.path.normpath(str(Path(path).parent.absolute()) + "/results")
        if not os.path.exists(newPath):
            os.makedirs(newPath)
        return newPath


import nltk
####Argumentos##################################
from argparse import ArgumentParser
####analizador sintactico#######################
import stanfordnlp
import os


def start():
    # from analyzer import Analyzer
    p = ArgumentParser(description="python3 ./main.py -f \"laginak/*.doc.txt\" ")
    optional = p._action_groups.pop()  # Edited this line
    required = p.add_argument_group('Required arguments')
    required.add_argument("-f", "--files", nargs='+', help="Files to analyze (in .txt, .odt, .doc or .docx format)")
    optional.add_argument('-a', '--all', action='store_true', help="Generate a CSV file with all the results")
    optional.add_argument('-s', '--similarity', action='store_true', help="Calculate similarity (max. 5 files)")
    p._action_groups.append(optional)
    opts = p.parse_args()
    FileLoader.load_files(opts.files)
    FileLoader.load_irregular_verbs_list()
    FileLoader.load_dale_chall_list()
    FileLoader.load_connectives_list()
    FileLoader.load_oxford_word_list()
    # Cargar el analizador
    # MODELS_DIR = '/home/kepa/stanfordnlp_resources/en_ewt_models'
    # stanfordnlp.download('en', MODELS_DIR)
    stanford = stanfordnlp.Pipeline()
    df_row = None
    ### Files will be created in this folder
    path = Printer.create_directory(FileLoader.files[0])
    file_num = 0
    total = len(FileLoader.files)
    FileLoader.files = ["Loterry-adv.txt"]
    for input in FileLoader.files:
        texto = Analyzer.process_text(input=input)
        # Analizar
        a = Analyzer(texto, input, stanford)
        i = a.analyze(opts.similarity)
        df = a.create_dataframe()
        # prediction = a.predict_dificulty(df)
        prediction = "kk"
        file_num += 1
        p = Printer(input, i)
        p.print_info(opts.similarity, prediction, file_num, total)
        if opts.all:
            df_row = p.write_in_full_csv(df_row, opts.similarity)
        p.generate_csv(path, prediction, opts.similarity)
    if opts.all:
        df_row.to_csv(os.path.join(path, "full_results_aztertest.csv"), encoding='utf-8', index=False)


nltk.download('cmudict')
nltk.download('punkt')  # ,download_dir='/home/lsi/metrix-env/nltk_data')
nltk.download('stopwords')
nltk.download('wordnet')
start()