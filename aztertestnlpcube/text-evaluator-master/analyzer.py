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
import tensorflow_hub as hub
####
from file_loader import FileLoader
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

    def __init__(self, text, input, cube):
        self.indicators = defaultdict(int)
        self.aux_lists = defaultdict(list)
        self.words_freq = {}
        self.text = text
        self.cube = cube
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


    # Este método devuelve true en caso de que la word pasada como parametro sea verbo. Para que una word sea
    # verbo se tiene que cumplir que sea VERB o que sea AUX y que su padre NO sea VERB.
    def is_verb(self, word, frase):
        return word.upos == 'VERB' or (word.upos == 'AUX' and frase[word.head - 1].upos != 'VERB')

    # Este método devuelve true en caso de que la word pasada como parametro sea un verbo irregular. Se utiliza una lista
    # de verbos irregulares sacada de https://github.com/Bryan-Legend/babel-lang/blob/master/Babel.EnglishEmitter/Resources/Irregular%20Verbs.txt.
    def is_irregular(self, word):
        return True if word.lemma in FileLoader.irregular_verbs else False

    # Este método devuelve true en caso de que la word pasada como parametro sea una palabra simple. Se utiliza una lista
    # de palabras simples sacada de http://www.readabilityformulas.com/articles/dale-chall-readability-word-list.php
    def is_complex(self, word):
        return False if word.word.lower() in FileLoader.simple_words or word.lemma.lower() in FileLoader.simple_words else True

    # Este método devuelve true en caso de que la word pasada como parametro sea un verbo en pasado.
    def is_past(self, word):
        atributos = word.attrs.split('|')
        return True if 'Tense=Past' in atributos else False

    # Este método devuelve true en caso de que la word pasada como parametro sea un verbo en presente.
    def is_present(self, word):
        atributos = word.attrs.split('|')
        return True if 'Tense=Pres' in atributos else False

    # Este método devuelve true en caso de que la word pasada como parametro sea un verbo en futuro.
    def is_future(self, word, frase):
        return word.upos == 'AUX' and word.lemma in ['will', 'shall'] and frase[word.head - 1].xpos == 'VB'

    # Este método devuelve true en caso de que la word pasada como parametro sea un verbo en infinitivo.
    def is_infinitive(self, word, frase):
        atributos = word.attrs.split('|')
        prev_word_index = word.index - 1
        return 'VerbForm=Inf' in atributos and prev_word_index > 0 and frase[prev_word_index - 1].word.lower() == 'to'

    # Este método devuelve true en caso de que la word pasada como parametro sea un verbo en gerundio.
    def is_gerund(self, word):
        atributos = word.attrs.split('|')
        return True if 'VerbForm=Ger' in atributos else False

    # Este método devuelve true en caso de que la word pasada como parametro sea verbo un verbo en pasiva.
    def is_passive(self, word):
        atributos = word.attrs.split('|')
        return True if 'Voice=Pass' in atributos else False

    # Este método devuelve true en caso de que la word pasada como parametro sea un verbo en modo indicativo.
    def is_indicative(self, word):
        atributos = word.attrs.split('|')
        return True if 'Mood=Ind' in atributos else False

    # Este método devuelve true en caso de que la word pasada como parametro sea un verbo en modo imperativo.
    def is_imperative(self, word):
        atributos = word.attrs.split('|')
        return True if 'Mood=Imp' in atributos else False

    def is_agentless(self, word, frase):
        # Si el siguiente indice esta dentro del rango de la lista
        if word.index < len(frase):
            siguiente_word = frase[word.index].word.lower()
            if siguiente_word == 'by':
                return False
            else:
                return True

    def calculate_left_embeddedness(self, sequences):
        list_left_embeddedness = []
        for sequence in sequences:
            verb_index = 0
            main_verb_found = False
            left_embeddedness = 0
            num_words = 0
            for word in sequence:
                if not len(word.word) == 1 or word.word.isalpha():
                    if not main_verb_found and word.head < len(sequence):
                        if self.is_verb(word, sequence):
                            verb_index += 1
                            if (word.upos == 'VERB' and word.label == 'root') or (word.upos == 'AUX' and sequence[word.head].label == 'root' and sequence[word.head].upos == 'VERB'):
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
            self.indicators['simple_ttr'] = round(len(self.aux_lists['different_forms']) / self.indicators['num_words'], 4)

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
            self.indicators['lemma_nttr'] = round(len(self.aux_lists['different_lemma_nouns']) / self.indicators['num_noun'], 4)

    def calculate_lemma_vttr(self):
        if self.indicators['num_verb'] > 0:
            self.indicators['lemma_vttr'] = round(len(self.aux_lists['different_lemma_verbs']) / self.indicators['num_verb'], 4)

    def calculate_lemma_adj_ttr(self):
        if self.indicators['num_adj'] > 0:
            self.indicators['lemma_adj_ttr'] = round(len(self.aux_lists['different_lemma_adjs']) / self.indicators['num_adj'], 4)

    def calculate_lemma_adv_ttr(self):
        if self.indicators['num_adv'] > 0:
            self.indicators['lemma_adv_ttr'] = round(len(self.aux_lists['different_lemma_advs']) / self.indicators['num_adv'], 4)

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
        percentage = (tc/tw) * 100
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
        self.indicators['smog'] = round(1.0430*math.sqrt(30*tps/ts)+3.1291, 4)

    def is_not_stopword(self, word):
        stop_words = stopwords.words('english')
        return word.lower() not in stop_words

    def calculate_levels_oxford_word_list(self, sequences):
        i = self.indicators
        for sequence in sequences:
            for entry in sequence:
                if entry.word in FileLoader.oxford_words['A1']:
                    if entry.upos in FileLoader.oxford_words['A1'][entry.word]:
                        i['num_a1_words'] += 1
                elif entry.word in FileLoader.oxford_words['A2']:
                    if entry.upos in FileLoader.oxford_words['A2'][entry.word]:
                        i['num_a2_words'] += 1
                elif entry.word in FileLoader.oxford_words['B1']:
                    if entry.upos in FileLoader.oxford_words['B1'][entry.word]:
                        i['num_b1_words'] += 1
                elif entry.word in FileLoader.oxford_words['B2']:
                    if entry.upos in FileLoader.oxford_words['B2'][entry.word]:
                        i['num_b2_words'] += 1
                elif entry.word in FileLoader.oxford_words['C1']:
                    if entry.upos in FileLoader.oxford_words['C1'][entry.word]:
                        i['num_c1_words'] += 1
                elif self.is_lexic_word(entry, sequence):
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

    def calculate_hypernymy_index(self, ambiguity_content_words_list, FLAG = 'VN'):
        i = self.indicators
        if FLAG == 'VN':
            i['hypernymy_index'] = round(float(np.mean(ambiguity_content_words_list)), 4)
        elif FLAG == 'V':
            i['hypernymy_verbs_index'] = round(float(np.mean(ambiguity_content_words_list)), 4)
        elif FLAG == 'N':
            i['hypernymy_nouns_index'] = round(float(np.mean(ambiguity_content_words_list)), 4)

    def is_lexic_word(self, entry, sequence):
        return self.is_verb(entry, sequence) or entry.upos == 'NOUN' or entry.upos == 'ADJ' or entry.upos == 'ADV'

    def has_modifier(self, entry):
        #nominal head may be associated with different types of modifiers and function words
        return True if entry.label in ['nmod', 'nmod:poss', 'appos', 'amod', 'nummod', 'acl', 'acl:relcl', 'det', 'clf', 'case'] else False

    def count_decendents(self, sentence, list_np_indexes):
        num_modifiers = 0
        if len(list_np_indexes) == 0:
            return num_modifiers
        else:
            new_list_indexes = []
            for entry in sentence:
                if entry.head in list_np_indexes and self.has_modifier(entry):
                    new_list_indexes.append(entry.index)
                    num_modifiers += 1
            return num_modifiers + self.count_decendents(sentence, new_list_indexes)

    def count_modifiers(self, sentence, list_np_indexes):
        num_modifiers_per_np = []
        for index in list_np_indexes:
            num_modifiers = 0
            for entry in sentence:
                if int(entry.head) == int(index) and self.has_modifier(entry):
                    num_modifiers += 1
            num_modifiers_per_np.append(num_modifiers)
        return num_modifiers_per_np

    def count_np_in_sentence(self, sentence):
        list_np_indexes = []
        for entry in sentence:
            if entry.upos == 'NOUN' or entry.upos == 'PRON' or entry.upos == 'PROPN':
                if entry.label in ['fixed', 'flat', 'compound']:
                    if entry.head not in list_np_indexes:
                        list_np_indexes.append(entry.head)
                else:
                    if entry.index not in list_np_indexes:
                        list_np_indexes.append(entry.index)
        return list_np_indexes

    def count_vp_in_sentence(self, sentence):
        num_np = 0
        for entry in sentence:
            if self.is_verb(entry, sentence):
                 num_np += 1
        return num_np

    # Noun overlap measure is binary (there either is or is not any overlap between a pair of adjacent sentences in a text ).
    # Noun overlap measures the proportion of sentences in a text for which there are overlapping nouns,
    # With no deviation in the morphological forms (e.g., table/tables)
    # (número pares de sentencias adjacentes que tienen al menos algún nombre en común)/(Número de pares de sentencias adjacentes)
    def calculate_noun_overlap_adjacent(self):
        i = self.indicators
        adjacent_noun_overlap_list = []
        for paragraph in self.aux_lists['paragraphs_list']:
            if len(paragraph) > 1:
                adjacents = list(map(list, zip(paragraph, paragraph[1:])))
                for x in adjacents:
                    sentence1 = []
                    sentence2 = []
                    for entry1 in x[0]:
                        if entry1.upos == 'NOUN':
                           sentence1.append(entry1.word.lower())
                    for entry2 in x[1]:
                        if entry2.upos == 'NOUN':
                           sentence2.append(entry2.word.lower())
                    in_common = list(set(sentence1).intersection(sentence2))
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
                        if entry1.upos == 'NOUN':
                            sentence1.append(entry1.word.lower())
                    for entry2 in y:
                        if entry2.upos == 'NOUN':
                            sentence2.append(entry2.word.lower())
                    in_common = list(set(sentence1).intersection(sentence2))
                    if len(in_common) > 0:
                        all_noun_overlap_list.append(1)
                    else:
                        all_noun_overlap_list.append(0)
        if len(all_noun_overlap_list) > 0:
            i['noun_overlap_all'] = round(float(np.mean(all_noun_overlap_list)), 4)

    def is_personal_pronoun(self, word):
        atributos = word.attrs.split('|')
        return "PronType=Prs" in atributos

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
                        if self.is_personal_pronoun(entry1) or entry1.upos == 'NOUN':
                            sentence1.append(entry1.lemma.lower())
                    for entry2 in x[1]:
                        if self.is_personal_pronoun(entry2) or entry2.upos == 'NOUN':
                            sentence2.append(entry2.lemma.lower())
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
                        if self.is_personal_pronoun(entry1) or entry1.upos == 'NOUN':
                            sentence1.append(entry1.lemma.lower())
                    for entry2 in y:
                        if self.is_personal_pronoun(entry2) or entry2.upos == 'NOUN':
                            sentence2.append(entry2.lemma.lower())
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
                        if self.is_lexic_word(entry1, x[0]):
                            sentence1.append(entry1.lemma.lower())
                    for entry2 in x[1]:
                        if entry2.upos == 'NOUN':
                            sentence2.append(entry2.lemma.lower())
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
                        if self.is_lexic_word(entry1, x):
                            sentence1.append(entry1.lemma.lower())
                    for entry2 in y:
                        if entry2.upos == 'NOUN':
                            sentence2.append(entry2.lemma.lower())
                    in_common = list(set(sentence1).intersection(sentence2))
                    if len(in_common) > 0:
                        all_stem_overlap_list.append(1)
                    else:
                        all_stem_overlap_list.append(0)
        if len(all_stem_overlap_list) > 0:
            i['stem_overlap_all'] = round(float(np.mean(all_stem_overlap_list)), 4)

    # Metodo que calcula el numero de palabras de contenido en una frase. Counts number of content words in a sentence.
    def count_content_words_in(self, sentence):
        num_words = 0
        for entry in sentence:
            if self.is_verb(entry, sentence) or entry.upos == 'NOUN' or entry.upos == 'ADJ' or entry.upos == 'ADV':
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
                        if self.is_lexic_word(entry1, x[0]):
                            sentence1.append(entry1.word.lower())
                    for entry2 in x[1]:
                        if self.is_lexic_word(entry2, x[1]):
                            sentence2.append(entry2.word.lower())
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
                        if self.is_lexic_word(entry1, x):
                            sentence1.append(entry1.word.lower())
                    for entry2 in y:
                        if self.is_lexic_word(entry2, y):
                            sentence2.append(entry2.word.lower())
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
    def calculate_similarity(self, sentence1, sentence2):
        return np.inner(sentence1, sentence2)


    # Este metodo calcula la similaridad local media de todas las similitudes en frases
    def calculate_similarity_adjacent_sentences(self):
        i = self.indicators
        adjacent_similarity_list = []
        for sentence in self.aux_lists['sentences_in_paragraph_token_list']:
            if len(sentence) > 1:
                for x, y in zip(range(0, len(sentence) - 1), range(1, len(sentence))):
                    adjacent_similarity_list.append(self.calculate_similarity(sentence[x], sentence[y]))
            else:
                adjacent_similarity_list.append(0)
        if len(adjacent_similarity_list) > 0:
            i['similarity_adjacent_mean'] = round(float(np.mean(adjacent_similarity_list)), 4)
            i['similarity_adjacent_std'] = round(float(np.std(adjacent_similarity_list)), 4)

    # Este metodo calcula la similaridad local media de todas las similitudes en parrafos
    def calculate_similarity_adjacent_paragraphs(self):
        i = self.indicators
        adjacent_similarity_par_list = []
        if len(self.aux_lists['paragraph_token_list']) > 1:
            for x, y in zip(range(0, len(self.aux_lists['paragraph_token_list']) - 1), range(1, len(self.aux_lists['paragraph_token_list']))):
                adjacent_similarity_par_list.append(self.calculate_similarity(self.aux_lists['paragraph_token_list'][x], self.aux_lists['paragraph_token_list'][y]))
            if len(adjacent_similarity_par_list) > 0:
                i['similarity_adjacent_par_mean'] = round(float(np.mean(adjacent_similarity_par_list)), 4)
                i['similarity_adjacent_par_std'] = round(float(np.std(adjacent_similarity_par_list)), 4)

    def calculate_similarity_pairs_in(self, paragraph):
        list_similarities_mean = []
        for index in range(len(paragraph)):
            similarity_tmp = paragraph[index+1:]
            x = paragraph[index]
            for index2 in range(len(similarity_tmp)):
                y = similarity_tmp[index2]
                list_similarities_mean.append(self.calculate_similarity(x, y))
        if len(list_similarities_mean) > 1:
            return round(float(np.mean(list_similarities_mean)), 4)
        else:
            return 0.0

    # Este metodo calcula la similaridad global media de todas las similitudes
    # (between all possible pairs of sentences in a paragraph)
    def calculate_similarity_pairs_sentences(self):
        i = self.indicators
        similarity_pairs_list = []
        for paragraph in self.aux_lists['sentences_in_paragraph_token_list']:
            similarity_pairs_list.append(self.calculate_similarity_pairs_in(paragraph))
        i['similarity_pairs_par_mean'] = round(float(np.mean(similarity_pairs_list)), 4)
        i['similarity_pairs_par_std'] = round(float(np.std(similarity_pairs_list)), 4)

    # Este metodo cuenta el numero de pronombres clasificados segun el tipo
    def count_personal_pronoun(self, word):
        i = self.indicators
        atributos = word.attrs.split('|')
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
        if ".txt" not in self.input:
            pre_text = textract.process(self.input)
            text = pre_text.decode()
            text = text.replace('\n', '@')
        else:
            with open(self.input, encoding='utf-8') as f:
                text = f.read().replace('\n', '@')
        text = re.sub(r'@+', '@', text)
        lines = text.split('@')
        paragraphs = []
        if similarity:
            # Preparar Google Universal Sentence Encoder
            module_url = "https://tfhub.dev/google/universal-sentence-encoder-large/2"
            embed = hub.Module(module_url)
            similarity_input_placeholder = tf.placeholder(tf.string, shape=(None))
            similarity_sentences_encodings = embed(similarity_input_placeholder)
            with tf.Session() as session:
                session.run(tf.global_variables_initializer())
                session.run(tf.tables_initializer())
                for line in lines:
                    if not line.strip() == '':
                        paragraphs.append(line)
                        sentences_embeddings = session.run(similarity_sentences_encodings, feed_dict={similarity_input_placeholder: sent_tokenize(line)})
                        self.aux_lists['sentences_in_paragraph_token_list'].append(sentences_embeddings)
                    self.aux_lists['paragraph_token_list'] = session.run(similarity_sentences_encodings, feed_dict={similarity_input_placeholder: paragraphs})
        else:
            for line in lines:
                if not line.strip() == '':
                    paragraphs.append(line)
        ###############Generar lista con el numero de sentencias en cada parrafo##############################################
        self.aux_lists['paragraphs_list'] = []
        for paragraph in paragraphs:
            sequences = self.cube(Analyzer.process_text(text=paragraph))
            sentences = []
            for sequence in sequences:
                words_in_one_sentence = []
                for entry in sequence:
                    words_in_one_sentence.append(entry)
                sentences.append(words_in_one_sentence)
            self.aux_lists['paragraphs_list'].append(sentences)
            self.aux_lists['sentences_per_paragraph'].append(len(sequences))
        ###############Number of paragraphs##############################################
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
        i['temporal_connectives_incidence'] = Analyzer.get_incidence(temporal_connectives,i['num_words'])
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
        i['num_content_words_not_a1_c1_words_incidence'] = Analyzer.get_incidence(i['num_content_words_not_a1_c1_words'], i['num_words'])

    def calculate_all_means(self):
        i = self.indicators
        i['sentences_per_paragraph_mean'] = round(float(np.mean(self.aux_lists['sentences_per_paragraph'])), 4)
        i['sentences_length_mean'] = round(float(np.mean(self.aux_lists['sentences_length_list'])), 4)
        i['sentences_length_no_stopwords_mean'] = round(float(np.mean(self.aux_lists['sentences_length_no_stopwords_list'])), 4)
        i['num_syllables_words_mean'] = round(float(np.mean(self.aux_lists['syllabes_list'])), 4)
        i['words_length_mean'] = round(float(np.mean(self.aux_lists['words_length_list'])), 4)
        i['words_length_no_stopwords_mean'] = round(float(np.mean(self.aux_lists['words_length_no_stopwords_list'])), 4)
        i['lemmas_length_mean'] = round(float(np.mean(self.aux_lists['lemmas_length_list'])), 4)

    def calculate_all_std_deviations(self):
        i = self.indicators
        i['sentences_per_paragraph_std'] = round(float(np.std(self.aux_lists['sentences_per_paragraph'])), 4)
        i['sentences_length_std'] = round(float(np.std(self.aux_lists['sentences_length_list'])), 4)
        i['sentences_length_no_stopwords_std'] = round(float(np.std(self.aux_lists['sentences_length_no_stopwords_list'])), 4)
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
        sequences = self.cube(text)
        subordinadas_labels = ['csubj', 'csubj:pass', 'ccomp', 'xcomp', 'advcl', 'acl', 'acl:relcl']
        ambiguity_content_words_list = []
        noun_abstraction_list = []
        verb_abstraction_list = []
        noun_verb_abstraction_list = []
        min_wordfreq_list = []
        num_np_list = []
        num_vp_list = []
        depth_list = []
        modifiers_per_np = []
        decendents_total = 0
        for sequence in sequences:
            root = 0
            wordfreq_list = []
            tokens_sentence = []
            dependency_tree = defaultdict(list)
            i['num_sentences'] += 1
            num_punct_marks_in_sentence = 0
            num_words_in_sentences = 0
            num_words_in_sentence_without_stopwords = 0
            vp_indexes = self.count_np_in_sentence(sequence)
            num_np_list.append(len(vp_indexes))
            num_vp_list.append(self.count_vp_in_sentence(sequence))
            modifiers_per_np += self.count_modifiers(sequence, vp_indexes)
            decendents_total += self.count_decendents(sequence, vp_indexes)
            for entry in sequence:
                if entry.head == 0:
                    root = entry.index
                dependency_tree[entry.head].append(entry.index)
                if self.is_not_stopword(entry.lemma):
                    tokens_sentence.append(entry.lemma)
                i['num_words_with_punct'] += 1
                if entry.index == '1':
                    i['prop'] = 1
                if entry.label == 'conj' or entry.label == 'csubj' or entry.label == 'csubj:pass' or entry.label == 'ccomp' or entry.label == 'xcomp' or entry.label == 'advcl' or entry.label == 'acl' or entry.label == 'acl:relcl':
                    i['prop'] += 1
                # Numero de sentencias subordinadas
                if entry.label in subordinadas_labels:
                    i['num_subord'] += 1
                    # Numero de sentencias subordinadas relativas
                    if entry.label == 'acl:relcl':
                        i['num_rel_subord'] += 1
                if entry.upos == 'PUNCT':
                    num_punct_marks_in_sentence += 1
                if entry.upos == 'PRON':
                    self.count_personal_pronoun(entry)
                if entry.upos == 'NOUN':
                    i['num_noun'] += 1
                    if entry.word.lower() not in self.aux_lists['different_nouns']:
                        self.aux_lists['different_nouns'].append(entry.word.lower())
                    if entry.lemma not in self.aux_lists['different_lemma_nouns']:
                        self.aux_lists['different_lemma_nouns'].append(entry.lemma)
                if entry.upos == 'ADJ':
                    i['num_adj'] += 1
                    if entry.word.lower() not in self.aux_lists['different_adjs']:
                        self.aux_lists['different_adjs'].append(entry.word.lower())
                    if entry.lemma not in self.aux_lists['different_lemma_adjs']:
                        self.aux_lists['different_lemma_adjs'].append(entry.lemma)
                if entry.upos == 'ADV':
                    i['num_adv'] += 1
                    if entry.word.lower() not in self.aux_lists['different_advs']:
                        self.aux_lists['different_advs'].append(entry.word.lower())
                    if entry.lemma not in self.aux_lists['different_lemma_advs']:
                        self.aux_lists['different_lemma_advs'].append(entry.lemma)
                if entry.lemma == 'not':
                    i['num_neg'] += 1
                if self.is_verb(entry, sequence):
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
                    if entry.word.lower() not in self.aux_lists['different_verbs']:
                        self.aux_lists['different_verbs'].append(entry.word.lower())
                    if entry.lemma not in self.aux_lists['different_lemma_verbs']:
                        self.aux_lists['different_lemma_verbs'].append(entry.lemma)
                if self.is_future(entry, sequence):
                    i['num_future'] += 1
                if self.is_not_stopword(entry.word):
                    num_words_in_sentence_without_stopwords += 1
                if self.has_more_than_three_syllables(entry.word):
                    i['num_words_more_3_syl'] += 1
                ######wordfreq###########################################
                if (not len(entry.word) == 1 or entry.word.isalpha()) and entry.upos != "NUM":
                    wordfrequency = zipf_frequency(entry.word, 'en')
                    wordfreq_list.append(wordfrequency)
                    num_words_in_sentences += 1
                    if (self.is_lexic_word(entry, sequence)):
                        if wordfrequency <= 4:
                            i['num_rare_words_4'] += 1
                            if entry.upos == 'NOUN':
                                i['num_rare_nouns_4'] += 1
                            elif entry.upos == 'ADJ':
                                i['num_rare_adj_4'] += 1
                            elif entry.upos == 'ADV':
                                i['num_rare_advb_4'] += 1
                            elif self.is_verb(entry, sequence):
                                i['num_rare_verbs_4'] += 1
                        if entry.word.lower() not in self.aux_lists['different_lexic_words']:
                            self.aux_lists['different_lexic_words'].append(entry.word.lower())
                            if wordfrequency <= 4:
                                i['num_dif_rare_words_4'] += 1
                        if wn.synsets(entry.word):
                            if entry.upos == 'NOUN':
                                noun_abstraction_list.append(self.get_abstraction_level(entry.word, 'n'))
                                noun_verb_abstraction_list.append(self.get_abstraction_level(entry.word, 'n'))
                            elif self.is_verb(entry, sequence):
                                verb_abstraction_list.append(self.get_abstraction_level(entry.word, 'v'))
                                noun_verb_abstraction_list.append(self.get_abstraction_level(entry.word, 'v'))
                            ambiguity_content_words_list.append(self.get_ambiguity_level(entry.word, entry.upos))
                        i['num_lexic_words'] += 1
                    # Numero de lemas distintos en el texto
                    if entry.lemma not in self.aux_lists['different_lemmas']:
                        self.aux_lists['different_lemmas'].append(entry.word.lower())
                    # Numero de formas distintas en el texto
                    if entry.word.lower() not in self.aux_lists['different_forms']:
                        self.aux_lists['different_forms'].append(entry.word.lower())
                    # Lista de words con sus frecuencias
                    if entry.word.lower() not in self.words_freq:
                        self.words_freq[entry.word.lower()] = 1
                    else:
                        self.words_freq[entry.word.lower()] = self.words_freq.get(entry.word.lower()) + 1
                    if self.is_not_stopword(entry.word):
                        self.aux_lists['words_length_no_stopwords_list'].append(len(entry.word))
                    if self.is_complex(entry):
                        i['num_complex_words'] += 1
                    self.aux_lists['words_length_list'].append(len(entry.word))
                    self.aux_lists['lemmas_length_list'].append(len(entry.lemma))
            self.aux_lists['list_num_punct_marks'].append(num_punct_marks_in_sentence)
            i['num_total_prop'] = i['num_total_prop'] + i['prop']
            self.aux_lists['sentences_length_list'].append(num_words_in_sentences)
            self.aux_lists['sentences_length_no_stopwords_list'].append(num_words_in_sentence_without_stopwords)
            if len(wordfreq_list) > 0:
                min_wordfreq_list.append(min(wordfreq_list))
            else:
                min_wordfreq_list.append(0)
            self.aux_lists['all_sentences_tokens'].append(tokens_sentence)
            depth_list.append(self.tree_depth(dependency_tree, root))
        if similarity:
            self.calculate_similarity_adjacent_sentences()
            self.calculate_similarity_adjacent_paragraphs()
            self.calculate_similarity_pairs_sentences()
        self.calculate_all_overlaps()
        self.calculate_connectives()
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
        self.calculate_left_embeddedness(sequences)
        self.calculate_levels_oxford_word_list(sequences)
        self.calculate_mean_depth_per_sentence(depth_list)
        i['num_different_forms'] = len(self.aux_lists['different_forms'])
        i['num_pass_mean'] = round((i['num_pass']) / i['num_words'], 4)
        i['num_past_irregular_mean'] = round(((i['num_past_irregular']) / i['num_past']), 4) if i['num_past'] != 0 else 0
        i['num_punct_marks_per_sentence'] = round(float(np.mean(self.aux_lists['list_num_punct_marks'])), 4)
        i['mean_propositions_per_sentence'] = round(i['num_total_prop'] / i['num_sentences'], 4)
        self.calculate_phrases(num_vp_list, num_np_list)
        self.calculate_density()
        self.calculate_syntactic_density()
        i['num_modifiers_noun_phrase'] = round(float(np.mean(modifiers_per_np)), 4)
        i['num_decendents_noun_phrase'] = round(decendents_total / sum(num_np_list), 4)
        i['mean_rare_4'] = round(((100 * i['num_rare_words_4']) / i['num_lexic_words']), 4)
        i['mean_distinct_rare_4'] = round((100 * i['num_dif_rare_words_4']) / len(self.aux_lists['different_lexic_words']), 4)
        i['min_wf_per_sentence'] = round(float(np.mean(min_wordfreq_list)), 4)
        self.calculate_polysemic_index(ambiguity_content_words_list)
        self.calculate_hypernymy_index(noun_verb_abstraction_list)
        self.calculate_hypernymy_index(verb_abstraction_list, 'V')
        self.calculate_hypernymy_index(noun_abstraction_list, 'N')
        self.calculate_all_incidence()
        return self.indicators

