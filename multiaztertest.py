#!/usr/bin/env python
# coding: utf-8

# In[29]:
import os
import sys
from pathlib import Path
import csv
import stanfordnlp
from cube.api import Cube
import numpy as np
from collections import defaultdict
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import wordnet as wn
from nltk.corpus import cmudict
#####Argumentos##################################
from argparse import ArgumentParser


class ModelAdapter:

    def __init__(self, model, lib):
        # parser
        self.model = model
        # model_name
        self.lib = lib

    def model_analysis(self, text):
        d = Document(text)  # ->data = []
        if self.lib.lower() == "stanford":
            lines = text.split('@')
            for line in lines:  # paragraph
                p = Paragraph()  # -> paragraph = []
                if not line.strip() == '':
                    doc = self.model(line)
                    for sent in doc.sentences:
                        s = Sentence()
                        sequence = self.sent2sequenceStanford(sent)
                        print(sequence)
                        s.text = sequence
                        for word in sent.words:
                            # Por cada palabra de cada sentencia, creamos un objeto Word que contendra los attrs
                            w = Word()
                            w.index = str(word.index)
                            w.text = word.text
                            w.lemma = word.lemma
                            w.upos = word.upos
                            w.xpos = word.xpos
                            w.feats = word.feats
                            w.governor = word.governor
                            w.dependency_relation = word.dependency_relation
                            s.word_list.append(w)
                            print(str(
                                w.index) + "\t" + w.text + "\t" + w.lemma + "\t" + w.upos + "\t" +
                                  w.xpos + "\t" + w.feats + "\t" + str(w.governor) + "\t" + str(w.dependency_relation) +
                                  "\t")
                        p.sentence_list.append(s)  # ->paragraph.append(s)
                    d.paragraph_list.append(p)  # ->data.append(paragraph)

        elif self.lib.lower() == "cube":
            d = Document(text)  # ->data = []
            lines = text.split('@')
            for line in lines:
                p = Paragraph()  # -> paragraph = []
                sequences = self.model(line)
                for seq in sequences:
                    s = Sentence()
                    sequence = self.sent2sequenceCube(seq)
                    s.text = sequence
                    for entry in seq:
                        # Por cada palabra de cada sentencia, creamos un objeto Word que contendra los attrs
                        w = Word()
                        w.index = str(entry.index)
                        w.text = entry.word
                        w.lemma = entry.lemma
                        w.upos = entry.upos
                        w.xpos = entry.xpos
                        w.feats = entry.attrs
                        w.governor = str(entry.head)
                        w.dependency_relation = str(entry.label)
                        s.word_list.append(w)
                    p.sentence_list.append(s)  # ->paragraph.append(s)
                d.paragraph_list.append(p)  # ->data.append(paragraph)
        return d

    def sent2sequenceStanford(self, sent):
        conllword = ""
        for word in sent.words:
            conllword = conllword + " " + str(word.text)
        return conllword

    def sent2sequenceCube(self, sent):
        conllword = ""
        for entry in sent:
            conllword = conllword + " " + str(entry.word)
        return conllword


class Document:
    def __init__(self, text):
        self._text = text
        self._paragraph_list = []
        self.words_freq = {}
        # Indicadores
        self.indicators = defaultdict(float)
        self.aux_lists = defaultdict(list)

    @property
    def text(self):
        """ Access text of this document. Example: 'This is a sentence.'"""
        return self._text

    @text.setter
    def text(self, value):
        """ Set the document's text value. Example: 'This is a sentence.'"""
        self._text = value

    @property
    def paragraph_list(self):
        """ Access list of sentences for this document. """
        return self._paragraph_list

    @paragraph_list.setter
    def paragraph_list(self, value):
        """ Set the list of tokens for this document. """
        self._paragraph_list = value

    def get_indicators(self):
        self.calculate_all_numbers()
        self.calculate_all_means()
        self.calculate_all_std_deviations()
        self.calculate_all_incidence()
        self.calculate_density()
        self.calculate_all_overlaps()
        return self.indicators

    # self.indicators['num_words'] = self.calculate_num_words()
    #     def calculate_num_words(self):
    #         num_words = 0
    #         not_punctuation = lambda w: not (len(w.text) == 1 and (not w.text.isalpha()))
    #         for paragraph in self._paragraph_list:
    #             self.aux_lists['sentences_per_paragraph'].append(len(paragraph.sentence_list))  # [1,2,1,...]
    #             for sentence in paragraph.sentence_list:
    #                 filterwords = filter(not_punctuation, sentence.word_list)
    #                 sum = 0
    #                 for word in filterwords:
    #                     num_words += 1
    #                     self.aux_lists['words_length_list'].append(len(word.text))
    #                     self.aux_lists['lemmas_length_list'].append(len(word.lemma))
    #                     sum += 1
    #                 self.aux_lists['sentences_length_mean'].append(sum)
    #         return num_words

    #     def calculate_num_paragraphs(self):
    #         return len(self._paragraph_list)
    #    self.indicators['num_sentences'] = self.calculate_num_sentences()
    #    self.indicators['num_paragraphs'] = self.calculate_num_paragraphs()
    #     def calculate_num_sentences(self):
    #         num_sentences = 0
    #         for paragraph in self._paragraph_list:
    #             for sentence in paragraph.sentence_list:
    #                 num_sentences += 1
    #         return num_sentences

    def calculate_simple_ttr(self, p_diff_forms=None, p_num_words=None):
        if (p_diff_forms and p_num_words) is not None:
            return (len(p_diff_forms)) / p_num_words
        else:
            self.indicators['simple_ttr'] = round(self.indicators['num_different_forms'] / self.indicators['num_words'],
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

    def calculate_mean_depth_per_sentence(self, depth_list):
        i = self.indicators
        i['mean_depth_per_sentence'] = round(float(np.mean(depth_list)), 4)

    def tree_depth(self, tree, root):
        if not tree[root]:
            return 1

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
                fragments += 1
                word_count = 0
                dif_words.clear()
                ttr = 1.0
            elif i == len(filtered_words) - 1:
                residual = (1.0 - ttr) / (1.0 - ttr_threshold)
                fragments += residual

        if fragments != 0:
            return len(filtered_words) / fragments
        else:
            return 0

    def calculate_mtld(self):
        not_punctuation = lambda w: not (len(w) == 1 and (not w.isalpha()))
        filtered_words = list(filter(not_punctuation, word_tokenize(self.text)))
        self.indicators['mtld'] = round((self.mtld(filtered_words) + self.mtld(filtered_words[::-1])) / 2, 4)

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

    # Noun overlap measure is binary (there either is or is not any overlap between a pair of adjacent sentences in a text ).
    # Noun overlap measures the proportion of sentences in a text for which there are overlapping nouns,
    # With no deviation in the morphological forms (e.g., table/tables)
    # (número pares de sentencias adjacentes que tienen al menos algún nombre en común)/(Número de pares de sentencias adjacentes)
    def calculate_noun_overlap_adjacent(self):
        i = self.indicators
        adjacent_noun_overlap_list = []
        # paragraph_list es una lista de doc.sentences donde doc.sentences es una "lista de obj sentencias" de un parrafo=[doc.sentence1,...]
        for paragraph in self.paragraph_list:
            # Por cada parrafo:paragraph es "lista de obj sentencias" de un parrafo=[doc.sentence1,...]
            if len(paragraph.sentence_list) > 1:
                # zip Python zip function takes iterable elements as input, and returns iterator que es un flujo de datos que
                # puede ser recorrido por for o map.
                # Si paragraph = [[sentence1], [sentence2], [sentence3]]
                # paragraph[1:] = [[sentence2], [sentence3]]
                test = zip(paragraph.sentence_list, paragraph.sentence_list[1:])  # zip the values
                # print(test) #-><zip object at 0x7eff7b354c08>=?[([sentence1],[sentence2]),([sentence2],[sentence3]),...]
                # for values in test:
                # print(values)  # print each tuples
                # ([sentence1],[sentence2])
                # ([sentence2],[sentence3])
                # map aplica la función list a todos los elementos de zip y como resultado se devuelve un iterable de tipo map
                # funcion list=The list() constructor returns a mutable (the object can be modified) sequence list of elements.
                # Por cada valor de test genera una lista
                # print(testlist) #<map object at 0x7eff7b3701d0>=?[[([sentence1],[sentence2])],[([sentence2],[sentence3])]]
                adjacents = list(map(list, test))
                # print(type(adjacents))
                # print(adjacents) ##Ejm: Parrafo1:[[[sent1], [sent2]], [[sent2], [sent3]]] donde sentenceX es conllword1,conllword2,...
                for x in adjacents:
                    sentence1 = []
                    sentence2 = []
                    for entry1 in x[0].word_list:
                        #values1 = entry1.split("\t")
                        if entry1.upos == 'NOUN':
                            sentence1.append(entry1.text.lower())
                    for entry2 in x[1].word_list:
                        #values2 = entry2.split("\t")
                        if entry2.upos == 'NOUN':
                            sentence2.append(entry2.text.lower())
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
        for paragraph in self.paragraph_list:
            for index in range(len(paragraph.sentence_list)):
                similarity_tmp = paragraph.sentence_list[index + 1:]
                x = paragraph.sentence_list[index]
                for index2 in range(len(similarity_tmp)):
                    y = similarity_tmp[index2]
                    sentence1 = []
                    sentence2 = []
                    for entry1 in x.word_list:
                        #values1 = entry1.split("\t")
                        if entry1.upos == 'NOUN':
                            sentence1.append(entry1.text.lower())
                    for entry2 in y.word_list:
                        #values2 = entry2.split("\t")
                        if entry2.upos == 'NOUN':
                            sentence2.append(entry2.text.lower())
                    in_common = list(set(sentence1).intersection(sentence2))
                    if len(in_common) > 0:
                        all_noun_overlap_list.append(1)
                    else:
                        all_noun_overlap_list.append(0)
        if len(all_noun_overlap_list) > 0:
            i['noun_overlap_all'] = round(float(np.mean(all_noun_overlap_list)), 4)


    # Argument overlap measure is binary (there either is or is not any overlap between a pair of adjacent
    # sentences in a text ). Argument overlap measures the proportion of sentences in a text for which there are overlapping the
    # between nouns (stem, e.g., “table”/”tables”) and personal pronouns (“he”/”he”)
    def calculate_argument_overlap_adjacent(self):
        i = self.indicators
        adjacent_argument_overlap_list = []
        for paragraph in self.paragraph_list:
            if len(paragraph.sentence_list) > 1:
                adjacents = list(map(list, zip(paragraph.sentence_list, paragraph.sentence_list[1:])))
                for x in adjacents:
                    sentence1 = []
                    sentence2 = []
                    for entry1 in x[0].word_list:
                        if entry1.is_personal_pronoun or entry1.upos == 'NOUN':
                            sentence1.append(entry1.text.lower())
                    for entry2 in x[1].word_list:
                        if entry2.is_personal_pronoun or entry2.upos == 'NOUN':
                            sentence2.append(entry1.text.lower())
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
        for paragraph in self.paragraph_list:
            for index in range(len(paragraph.sentence_list)):
                similarity_tmp = paragraph.sentence_list[index + 1:]
                x = paragraph.sentence_list[index]
                for index2 in range(len(similarity_tmp)):
                    y = similarity_tmp[index2]
                    sentence1 = []
                    sentence2 = []
                    for entry1 in x.word_list:
                        if entry1.is_personal_pronoun or entry1.upos == 'NOUN':
                            sentence1.append(entry1.text.lower())
                    for entry2 in y.word_list:
                        if entry2.is_personal_pronoun or entry2.upos == 'NOUN':
                            sentence2.append(entry2.text.lower())
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
        for paragraph in self.paragraph_list:
            if len(paragraph.sentence_list) > 1:
                adjacents = list(map(list, zip(paragraph.sentence_list, paragraph.sentence_list[1:])))
                for x in adjacents:
                    sentence1 = []
                    sentence2 = []
                    for entry1 in x[0].word_list:
                        if entry1.is_lexic_word(x[0]):
                            sentence1.append(entry1.text.lower())
                    for entry2 in x[1].word_list:
                        if entry2.upos == 'NOUN':
                            sentence2.append(entry2.text.lower())
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
        for paragraph in self.paragraph_list:
            for index in range(len(paragraph.sentence_list)):
                similarity_tmp = paragraph.sentence_list[index + 1:]
                x = paragraph.sentence_list[index]
                for index2 in range(len(similarity_tmp)):
                    y = similarity_tmp[index2]
                    sentence1 = []
                    sentence2 = []
                    for entry1 in x.word_list:
                        if entry1.is_lexic_word(x):
                            sentence1.append(entry1.text.lower())
                    for entry2 in y.word_list:
                        if entry2.upos == 'NOUN':
                            sentence2.append(entry2.text.lower())
                    in_common = list(set(sentence1).intersection(sentence2))
                    if len(in_common) > 0:
                        all_stem_overlap_list.append(1)
                    else:
                        all_stem_overlap_list.append(0)
        if len(all_stem_overlap_list) > 0:
            i['stem_overlap_all'] = round(float(np.mean(all_stem_overlap_list)), 4)

    # Content word overlap adjacent sentences proporcional mean refers to the proportion of content words
    # (nouns, verbs,adverbs,adjectives, pronouns) that shared Between pairs of sentences.For example, if
    # a sentence pair has fewer words and two words overlap, The proportion is greater than if a pair has
    # many words and two words overlap. This measure may be particulaly useful when the lenghts of the
    # sentences in the text are principal concern.
    def calculate_content_overlap_adjacent(self):
        i = self.indicators
        adjacent_content_overlap_list = []
        for paragraph in self.paragraph_list:
            if len(paragraph.sentence_list) > 1:
                adjacents = list(map(list, zip(paragraph.sentence_list, paragraph.sentence_list[1:])))
                for x in adjacents:
                    sentence1 = []
                    sentence2 = []
                    for entry1 in x[0].word_list:
                        if entry1.is_lexic_word(x[0]):
                            sentence1.append(entry1.text.lower())
                    for entry2 in x[1].word_list:
                        if entry2.is_lexic_word(x[1]):
                            sentence2.append(entry2.text.lower())
                    in_common = list(set(sentence1).intersection(sentence2))
                    n1 = x[0].count_content_words_in()
                    n2 = x[1].count_content_words_in()
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
        for paragraph in self.paragraph_list:
            for index in range(len(paragraph.sentence_list)):
                similarity_tmp = paragraph.sentence_list[index + 1:]
                x = paragraph.sentence_list[index]
                for index2 in range(len(similarity_tmp)):
                    y = similarity_tmp[index2]
                    sentence1 = []
                    sentence2 = []
                    for entry1 in x.word_list:
                        if entry1.is_lexic_word(x):
                            sentence1.append(entry1.text.lower())
                    for entry2 in y.word_list:
                        if entry2.is_lexic_word(y):
                            sentence2.append(entry2.text.lower())
                    in_common = list(set(sentence1).intersection(sentence2))
                    n1 = x.count_content_words_in()
                    n2 = y.count_content_words_in()
                    if n1 + n2 > 0:
                        all_content_overlap_list.append(len(in_common) / (n1 + n2))
                    else:
                        all_content_overlap_list.append(0)
        if len(all_content_overlap_list) > 0:
            i['content_overlap_all_mean'] = round(float(np.mean(all_content_overlap_list)), 4)
            i['content_overlap_all_std'] = round(float(np.std(all_content_overlap_list)), 4)

    def calculate_all_overlaps(self):
        self.calculate_noun_overlap_adjacent()
        self.calculate_noun_overlap_all()
        self.calculate_argument_overlap_adjacent()
        self.calculate_argument_overlap_all()
        self.calculate_stem_overlap_adjacent()
        self.calculate_stem_overlap_all()
        self.calculate_content_overlap_adjacent()
        self.calculate_content_overlap_all()



    def calculate_all_numbers(self):
        i = self.indicators
        i['num_paragraphs'] = len(self._paragraph_list)
        i['num_words'] = 0
        i['num_sentences'] = 0
        num_np_list = []
        num_vp_list = []
        modifiers_per_np = []
        depth_list = []
        subordinadas_labels = ['csubj', 'csubj:pass', 'ccomp', 'xcomp', 'advcl', 'acl', 'acl:relcl']
        not_punctuation = lambda w: not (len(w.text) == 1 and (not w.text.isalpha()))
        decendents_total = 0
        for p in self.paragraph_list:
            self.aux_lists['sentences_per_paragraph'].append(len(p.sentence_list))  # [1,2,1,...]
            for s in p.sentence_list:
                if not s.text == "":
                    num_words_in_sentence_without_stopwords = 0
                    i['num_sentences'] += 1
                    filterwords = filter(not_punctuation, s.word_list)
                    sum = 0
                    dependency_tree = defaultdict(list)
                    vp_indexes = s.count_np_in_sentence()
                    num_np_list.append(len(vp_indexes))
                    num_vp_list.append(s.count_vp_in_sentence())
                    decendents_total += s.count_decendents(vp_indexes)
                    modifiers_per_np += s.count_modifiers(vp_indexes)
                    self.aux_lists['left_embeddedness'].append(s.calculate_left_embeddedness())
                    i['prop'] = 0
                    numPunct = 0
                    for w in s.word_list:
                        if w.governor == 0:
                            root = w.index
                        dependency_tree[w.governor].append(w.index)
                        # words without punc
                        if w in filterwords:
                            i['num_words'] += 1
                            self.aux_lists['words_length_list'].append(len(w.text))
                            self.aux_lists['lemmas_length_list'].append(len(w.lemma))
                            sum += 1
                        # words with punc
                        i['num_words_with_punct'] += 1
                        # words not in stopwords
                        if w.text.lower() not in Stopwords.stop_words:
                            num_words_in_sentence_without_stopwords += 1
                        if w.is_lexic_word(s):
                            i['num_lexic_words'] += 1
                        if w.upos == 'NOUN':
                            i['num_noun'] += 1
                            if w.text.lower() not in self.aux_lists['different_nouns']:
                                self.aux_lists['different_nouns'].append(w.text.lower())
                            if w.lemma not in self.aux_lists['different_lemma_nouns']:
                                self.aux_lists['different_lemma_nouns'].append(w.lemma)
                        if w.upos == 'ADJ':
                            i['num_adj'] += 1
                            if w.text.lower() not in self.aux_lists['different_adjs']:
                                self.aux_lists['different_adjs'].append(w.text.lower())
                            if w.lemma not in self.aux_lists['different_lemma_adjs']:
                                self.aux_lists['different_lemma_adjs'].append(w.lemma)
                        if w.upos == 'ADV':
                            i['num_adv'] += 1
                            if w.text.lower() not in self.aux_lists['different_advs']:
                                self.aux_lists['different_advs'].append(w.text.lower())
                            if w.lemma not in self.aux_lists['different_lemma_advs']:
                                self.aux_lists['different_lemma_advs'].append(w.lemma)
                        if w.is_verb(s):
                            i['num_verb'] += 1
                            if w.text.lower() not in self.aux_lists['different_verbs']:
                                self.aux_lists['different_verbs'].append(w.text.lower())
                            if w.lemma not in self.aux_lists['different_lemma_verbs']:
                                self.aux_lists['different_lemma_verbs'].append(w.lemma)
                            if w.is_past():
                                i['num_past'] += 1
                                if w.is_irregular():
                                    i['num_past_irregular'] += 1
                            if w.is_present():
                                i['num_pres'] += 1
                            if w.is_indicative():
                                i['num_indic'] += 1
                            if w.is_gerund():
                                i['num_ger'] += 1
                            if w.is_infinitive():
                                i['num_inf'] += 1
                            if w.is_imperative():
                                i['num_impera'] += 1
                        if w.is_personal_pronoun():
                            i['num_personal_pronouns'] += 1
                            if w.is_first_person_pronoun():
                                i['num_first_pers_pron'] += 1
                                if w.is_first_personal_pronoun_sing():
                                    i['num_first_pers_sing_pron'] += 1
                            if w.is_third_personal_pronoun():
                                i['num_third_pers_pron'] += 1
                        if w.text.lower() not in self.aux_lists['different_forms']:
                            self.aux_lists['different_forms'].append(w.text.lower())
                        if w.text.lower() not in self.words_freq:
                            self.words_freq[w.text.lower()] = 1
                        else:
                            self.words_freq[w.text.lower()] = self.words_freq.get(w.text.lower()) + 1
                        if w.is_subordinate():
                            i['num_subord'] += 1
                            # Numero de sentencias subordinadas relativas
                            if w.is_subordinate_relative():
                                i['num_rel_subord'] += 1
                        if w.upos == 'PUNCT':
                            numPunct += 1
                        if w.is_proposition():
                           i['prop'] += 1
                        if (not len(w.text) == 1 or w.text.isalpha()) and w.upos != "NUM":
                            if (w.is_lexic_word(s)):
                                i['num_lexic_words'] += 1
                                if wn.synsets(w.text):
                                    if w.upos == 'NOUN':
                                        self.aux_lists['noun_abstraction_list'].append(
                                            self.get_abstraction_level(w.text, 'n'))
                                        self.aux_lists['noun_verb_abstraction_list'].append(
                                            self.get_abstraction_level(w.text, 'n'))
                                    elif w.is_verb(s):
                                        self.aux_lists['verb_abstraction_list'].append(
                                            self.get_abstraction_level(w.text, 'v'))
                                        self.aux_lists['noun_verb_abstraction_list'].append(
                                            self.get_abstraction_level(w.text, 'v'))
                                    self.aux_lists['ambiguity_content_words_list'].append(
                                        self.get_ambiguity_level(w.text, w.upos))
                            if w.lemma not in self.aux_lists['different_lemmas']:
                                self.aux_lists['different_lemmas'].append(w.text.lower())
                i['num_total_prop'] = i['num_total_prop'] + i['prop']
                self.aux_lists['prop_per_sentence'].append(i['prop'])
                self.aux_lists['punct_per_sentence'].append(numPunct)
                self.aux_lists['sentences_length_mean'].append(sum)
                self.aux_lists['sentences_length_no_stopwords_list'].append(num_words_in_sentence_without_stopwords)
                depth_list.append(self.tree_depth(dependency_tree, root))
        # i['num_decendents_noun_phrase'] = round(decendents_total / sum(num_np_list), 4)
        i['num_different_forms'] = len(self.aux_lists['different_forms'])
        self.indicators['left_embeddedness'] = round(float(np.mean(self.aux_lists['left_embeddedness'])), 4)
        self.calculate_honore()
        self.calculate_maas()
        # i['num_decendents_noun_phrase'] = round(decendents_total / sum(num_np_list), 4)
        # i['num_modifiers_noun_phrase'] = round(float(np.mean(modifiers_per_np)), 4)
        self.calculate_phrases(num_vp_list, num_np_list)
        self.calculate_mean_depth_per_sentence(depth_list)
        self.calculate_mtld()

    def calculate_all_means(self):
        i = self.indicators
        i['sentences_per_paragraph_mean'] = round(float(np.mean(self.aux_lists['sentences_per_paragraph'])), 4)
        i['sentences_length_mean'] = round(float(np.mean(self.aux_lists['sentences_length_mean'])), 4)
        i['words_length_mean'] = round(float(np.mean(self.aux_lists['words_length_list'])), 4)
        i['lemmas_length_mean'] = round(float(np.mean(self.aux_lists['lemmas_length_list'])), 4)
        i['mean_propositions_per_sentence'] = round(float(np.mean(self.aux_lists['prop_per_sentence'])), 4)
        i['num_punct_marks_per_sentence'] = round(float(np.mean(self.aux_lists['punct_per_sentence'])), 4)
        i['polysemic_index'] = round(float(np.mean(self.aux_lists['ambiguity_content_words_list'])), 4)
        i['hypernymy_index'] = round(float(np.mean(self.aux_lists['noun_verb_abstraction_list'])), 4)
        i['hypernymy_verbs_index'] = round(float(np.mean(self.aux_lists['verb_abstraction_list'])), 4)
        i['hypernymy_nouns_index'] = round(float(np.mean(self.aux_lists['noun_abstraction_list'])), 4)
        i['sentences_length_no_stopwords_mean'] = round(
            float(np.mean(self.aux_lists['sentences_length_no_stopwords_list'])), 4)

    def calculate_all_std_deviations(self):
        i = self.indicators
        i['sentences_per_paragraph_std'] = round(float(np.std(self.aux_lists['sentences_per_paragraph'])), 4)
        i['sentences_length_std'] = round(float(np.std(self.aux_lists['sentences_length_mean'])), 4)
        i['words_length_std'] = round(float(np.std(self.aux_lists['words_length_list'])), 4)
        i['lemmas_length_std'] = round(float(np.std(self.aux_lists['lemmas_length_list'])), 4)
        i['sentences_length_no_stopwords_std'] = round(
            float(np.std(self.aux_lists['sentences_length_no_stopwords_list'])), 4)

    @staticmethod
    def get_incidence(indicador, num_words):
        return round(((1000 * indicador) / num_words), 4)

    def calculate_all_incidence(self):
        i = self.indicators
        n = i['num_words']
        i['num_sentences_incidence'] = self.get_incidence(i['num_sentences'], n)
        i['num_paragraphs_incidence'] = self.get_incidence(i['num_paragraphs'], n)
        i['num_impera_incidence'] = self.get_incidence(i['num_impera'], n)
        i['num_personal_pronouns_incidence'] = self.get_incidence(i['num_personal_pronouns'], n)
        i['num_first_pers_pron_incidence'] = self.get_incidence(i['num_first_pers_pron'], n)
        i['num_first_pers_sing_pron_incidence'] = self.get_incidence(i['num_first_pers_sing_pron'], n)
        i['num_third_pers_pron_incidence'] = self.get_incidence(i['num_third_pers_pron'], n)
        i['gerund_density_incidence'] = self.get_incidence(i['num_ger'], n)
        i['infinitive_density_incidence'] = self.get_incidence(i['num_inf'], n)
        i['num_subord_incidence'] = self.get_incidence(i['num_subord'], n)
        i['num_rel_subord_incidence'] = self.get_incidence(i['num_rel_subord'], n)
        i['num_past_incidence'] = self.get_incidence(i['num_past'], n)
        i['num_pres_incidence'] = self.get_incidence(i['num_pres'], n)
        i['num_future_incidence'] = self.get_incidence(i['um_future'], n)
        i['num_indic_incidence'] = self.get_incidence(i['num_indic'], n)
        i['num_verb_incidence'] = self.get_incidence(i['num_verb'], n)
        i['num_noun_incidence'] = self.get_incidence(i['num_noun'], n)
        i['num_adj_incidence'] = self.get_incidence(i['num_adj'], n)
        i['num_adv_incidence'] = self.get_incidence(i['num_adv'], n)
        i['num_lexic_words_incidence'] = self.get_incidence(i['num_lexic_words'], n)

    def calculate_density(self):
        i = self.indicators
        i['lexical_density'] = round(i['num_lexic_words'] / i['num_words'], 4)
        i['noun_density'] = round(i['num_noun'] / i['num_words'], 4)
        i['verb_density'] = round(i['num_verb'] / i['num_words'], 4)
        i['adj_density'] = round(i['num_adj'] / i['num_words'], 4)
        i['adv_density'] = round(i['num_adv'] / i['num_words'], 4)
        self.calculate_all_ttr()
        self.calculate_all_lemma_ttr()

    def calculate_phrases(self, num_vp_list, num_np_list):
        i = self.indicators
        i['mean_vp_per_sentence'] = round(float(np.mean(num_vp_list)), 4)
        i['mean_np_per_sentence'] = round(float(np.mean(num_np_list)), 4)
        i['noun_phrase_density_incidence'] = self.get_incidence(sum(num_np_list), i['num_words'])
        i['verb_phrase_density_incidence'] = self.get_incidence(sum(num_vp_list), i['num_words'])


class Paragraph:

    def __init__(self):
        self._sentence_list = []

    @property
    def sentence_list(self):
        """ Access list of sentences for this document. """
        return self._sentence_list

    @sentence_list.setter
    def sentence_list(self, value):
        """ Set the list of tokens for this document. """
        self.sentence_list = value


class Sentence:

    def __init__(self):
        self._word_list = []
        self.text = None

    @property
    def word_list(self):
        """ Access list of words for this sentence. """
        return self._word_list

    @word_list.setter
    def word_list(self, value):
        """ Set the list of words for this sentence. """
        self._word_list = value

    def calculate_left_embeddedness(self):
        verb_index = 0
        main_verb_found = False
        left_embeddedness = 0
        num_words = 0
        for word in self.word_list:
            if not len(word.text) == 1 or word.text.isalpha():
                if not main_verb_found and word.governor < len(self.word_list):
                    if word.is_verb(self):
                        verb_index += 1
                        if (word.upos == 'VERB' and word.dependency_relation == 'root') or (
                                word.upos == 'AUX' and self.word_list[
                            word.governor].dependency_relation == 'root'
                                and self.word_list[word.governor].upos == 'VERB'):
                            main_verb_found = True
                            left_embeddedness = num_words
                        if verb_index == 1:
                            left_embeddedness = num_words
                num_words += 1
        return left_embeddedness

    def count_np_in_sentence(self):
        list_np_indexes = []
        for word in self.word_list:
            list_np_indexes = word.is_np(list_np_indexes)
        return list_np_indexes

    def count_vp_in_sentence(self):
        num_np = 0
        for entry in self.word_list:
            if entry.is_verb(self):
                num_np += 1
        return num_np

    def count_modifiers(self, list_np_indexes):
        num_modifiers_per_np = []
        for index in list_np_indexes:
            num_modifiers = 0
            for entry in self.word_list:
                if int(entry.governor) == int(index) and entry.has_modifier():
                    num_modifiers += 1
            num_modifiers_per_np.append(num_modifiers)
        return num_modifiers_per_np

    def count_decendents(self, list_np_indexes):
        num_modifiers = 0
        if len(list_np_indexes) == 0:
            return num_modifiers
        else:
            new_list_indexes = []
            for entry in self.word_list:
                if entry.governor in list_np_indexes and entry.has_modifier():
                    new_list_indexes.append(entry.index)
                    num_modifiers += 1
            return num_modifiers + self.count_decendents(new_list_indexes)

    # Metodo que calcula el numero de palabras de contenido en una frase. Counts number of content words in a sentence.
    def count_content_words_in(self):
        num_words = 0
        for entry in self.word_list:
            if entry.is_verb(self) or entry.upos == 'NOUN' or entry.upos == 'ADJ' or entry.upos == 'ADV':
                num_words += 1
        return num_words

    def print(self):
        for words in self.word_list:
            print(words.text)


class Word:
    def __init__(self):
        self._index = None
        self._text = None
        self._lemma = None
        self._upos = None
        self._xpos = None
        self._feats = None
        self._governor = None
        self._dependency_relation = None

    @property
    def dependency_relation(self):
        """ Access dependency relation of this word. Example: 'nmod'"""
        return self._dependency_relation

    @dependency_relation.setter
    def dependency_relation(self, value):
        """ Set the word's dependency relation value. Example: 'nmod'"""
        self._dependency_relation = value

    @property
    def lemma(self):
        """ Access lemma of this word. """
        return self._lemma

    @lemma.setter
    def lemma(self, value):
        """ Set the word's lemma value. """
        self._lemma = value

    @property
    def governor(self):
        """ Access governor of this word. """
        return self._governor

    @governor.setter
    def governor(self, value):
        """ Set the word's governor value. """
        self._governor = value

    @property
    def pos(self):
        """ Access (treebank-specific) part-of-speech of this word. Example: 'NNP'"""
        return self._xpos

    @pos.setter
    def pos(self, value):
        """ Set the word's (treebank-specific) part-of-speech value. Example: 'NNP'"""
        self._xpos = value

    @property
    def text(self):
        """ Access text of this word. Example: 'The'"""
        return self._text

    @text.setter
    def text(self, value):
        """ Set the word's text value. Example: 'The'"""
        self._text = value

    @property
    def xpos(self):
        """ Access treebank-specific part-of-speech of this word. Example: 'NNP'"""
        return self._xpos

    @xpos.setter
    def xpos(self, value):
        """ Set the word's treebank-specific part-of-speech value. Example: 'NNP'"""
        self._xpos = value

    @property
    def upos(self):
        """ Access universal part-of-speech of this word. Example: 'DET'"""
        return self._upos

    @upos.setter
    def upos(self, value):
        """ Set the word's universal part-of-speech value. Example: 'DET'"""
        self._upos = value

    @property
    def feats(self):
        """ Access morphological features of this word. Example: 'Gender=Fem'"""
        return self._feats

    @feats.setter
    def feats(self, value):
        """ Set this word's morphological features. Example: 'Gender=Fem'"""
        self._feats = value

    @property
    def parent_token(self):
        """ Access the parent token of this word. """
        return self._parent_token

    @parent_token.setter
    def parent_token(self, value):
        """ Set this word's parent token. """
        self._parent_token = value

    @property
    def index(self):
        """ Access index of this word. """
        return self._index

    @index.setter
    def index(self, value):
        """ Set the word's index value. """
        self._index = value

    def has_modifier(self):
        # nominal head may be associated with different types of modifiers and function words
        return True if self.dependency_relation in ['nmod', 'nmod:poss', 'appos', 'amod', 'nummod', 'acl', 'acl:relcl',
                                                    'det', 'clf',
                                                    'case'] else False

    def is_personal_pronoun(self):
        atributos =self.feats.split('|')
        if "PronType=Prs" in atributos:
            return True
        else:
            return False

    def is_first_person_pronoun(self):
        atributos = self.feats.split('|')
        if 'PronType=Prs' in atributos and 'Person=1' in atributos:
            return True
        else:
            return False

    def is_third_personal_pronoun(self):
        atributos = self.feats.split('|')
        if 'PronType=Prs' in atributos and 'Person=3' in atributos:
            return True
        else:
            return False

    def is_first_personal_pronoun_sing(self):
        atributos = self.feats.split('|')
        if 'PronType=Prs' in atributos and 'Person=1' in atributos and 'Number=Sing' in atributos:
            return True
        else:
            return False

    def num_syllables(self):
        list = []
        max = 0
        for x in Pronouncing.prondict[self.text.lower()]:
            tmp_list = []
            tmp_max = 0
            for y in x:
                if y[-1].isdigit():
                    tmp_max += 1
                    tmp_list.append(y)
            list.append(tmp_list)
            if tmp_max > max:
                max = tmp_max
        return max

    def syllables(self):
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
        word = self.text.lower()  # word.lower().strip(".:;?!")
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

    def allnum_syllables(self):
        try:
            return self.num_syllables()
        except KeyError:
            # if word not found in cmudict
            return self.syllables()

    def is_lexic_word(self, sequence):
        return self.is_verb(sequence) or self.upos == 'NOUN' or self.upos == 'ADJ' or self.upos == 'ADV'

    def is_verb(self, frase):
        return self.upos == 'VERB' or (self.upos == 'AUX' and frase.word_list[self.governor - 1].upos != 'VERB')

    def is_future(self, frase):
        return self.upos == 'AUX' and self.lemma in ['will', 'shall'] and frase.word_list[
            int(self.governor) - 1].xpos == 'VB'

    def is_past(self):
        atributos = self.xpos.split('|')
        if "Tense=Past" in atributos:
            return True
        else:
            return False

    def is_present(self):
        atributos = self.xpos.split('|')
        if "Tense=Pres" in atributos:
            return True
        else:
            return False

    def is_indicative(self):
        atributos = self.xpos.split('|')
        if "Mood=Ind" in atributos:
            return True
        else:
            return False

    def is_np(self, list_np_indexes):
        if self.upos == 'NOUN' or self.upos == 'PRON' or self.upos == 'PROPN':
            if self.dependency_relation in ['fixed', 'flat', 'compound']:
                if self.governor not in list_np_indexes:
                    list_np_indexes.append(self.governor)
            else:
                if self.index not in list_np_indexes:
                    ind = int(self.index)
                    list_np_indexes.append(ind)
        return list_np_indexes

    def is_gerund(self):
        atributos = self.feats.split('|')
        if 'VerbForm=Ger' in atributos:
            return True
        else:
            return False

    def is_infinitive(self):
        atributos = self.feats.split('|')
        if 'VerbForm=Inf' in atributos:
            return True
        else:
            return False

    def is_imperative(self):
        atributos = self.feats.split('|')
        if 'Mood=Imp' in atributos:
            return True
        else:
            return False

    def is_proposition(self):
        if self.dependency_relation == 'conj' or self.dependency_relation == 'csubj' or self.dependency_relation == 'csubj:pass' or\
                self.dependency_relation == 'ccomp' or self.dependency_relation == 'xcomp' or self.dependency_relation == 'advcl' or self.dependency_relation == 'acl' or self.dependency_relation == 'acl:relcl':
            return True
        else:
            return False

    def is_subordinate(self):
        subordinadas_labels = ['csubj', 'csubj:pass', 'ccomp', 'xcomp',
                                   'advcl', 'acl', 'acl:relcl']
        return True if self.dependency_relation in subordinadas_labels else False

    def is_subordinate_relative(self):
        subordinate_relative_labels = ['acl:relcl']

        return True if self.dependency_relation in subordinate_relative_labels else False

    def is_stopword(self):
        return True if self.text.lower() in Stopwords.stop_words else False

    def __repr__(self):
        features = ['index', 'text', 'lemma', 'upos', 'xpos', 'feats', 'governor', 'dependency_relation']
        feature_str = ";".join(["{}={}".format(k, getattr(self, k)) for k in features if getattr(self, k) is not None])

        return f"<{self.__class__.__name__} {feature_str}>"

    def is_irregular(self):
        return True if self.lemma in Irregularverbs.irregular_verbs else False


class Irregularverbs:

    irregular_verbs = []

    def load(self):
        f = open('data/en/IrregularVerbs.txt', 'r')
        lineas = f.readlines()
        for linea in lineas:
            if not linea.startswith("//"):
                #carga el verbo en presente, dejando pasado y preterito
                Irregularverbs.irregular_verbs.append(linea.split()[0])
        f.close()


class Printer:

    def __init__(self, indicators):
        self.indicators = indicators

    def print_info(self):
        i = self.indicators
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
        # Connectives
        print('Number of connectives (incidence per 1000 words): ' + str(i['all_connectives_incidence']))
        print('Causal connectives (incidence per 1000 words): ' + str(i['causal_connectives_incidence']))
        print('Logical connectives (incidence per 1000 words):  ' + str(i['logical_connectives_incidence']))
        print('Adversative/contrastive connectives (incidence per 1000 words): ' + str(
            i['adversative_connectives_incidence']))
        print('Temporal connectives (incidence per 1000 words):  ' + str(i['temporal_connectives_incidence']))
        print('Conditional connectives (incidence per 1000 words): ' + str(i['conditional_connectives_incidence']))


'''
The aim of this class is the charge of the model with the specific language and nlp library.
In addition, it is going to create a unified data structure to obtain the indicators independent of the library 
and language.
'''


class Stopwords:
    stop_words = []

    def __init__(self, language):
        self.lang = language

    def print(self):
        for stopword in Stopwords.stop_words:
            print(stopword)

    def download(self):
        nltk.download('stopwords')

    def load(self):
        if self.lang.lower() == "english":
            Stopwords.stop_words = stopwords.words('english')
        if self.lang.lower() == "spanish":
            Stopwords.stop_words = stopwords.words('spanish')
        if self.lang.lower() == "basque":
            # Stopwords.stop_words = set(line.strip() for line in open('data/eu/stopwords_formaketakonektoreak.txt'))
            # Stopwords.stop_words.add("gero")
            print("n")


class NLPCharger:

    def __init__(self, language, library):
        self.lang = language
        self.lib = library
        self.text = None
        self.textwithparagraphs = None
        self.parser = None

    '''
    Download the respective model depending of the library and language. 
    '''

    def download_model(self):
        if self.lib.lower() == "stanford":
            print("-----------You are going to use Stanford library-----------")
            if self.lang.lower() == "basque":
                print("-------------You are going to use Basque model-------------")
                # MODELS_DIR = '/home/ibon/eu'
                # MODELS_DIR = '/home/kepa/eu'
                MODELS_DIR = 'J:\TextSimilarity\eu'
                # MODELS_DIR = '/home/kepa/eu'
                stanfordnlp.download('eu', MODELS_DIR)  # Download the Basque models
            elif self.lang.lower() == "english":
                print("-------------You are going to use English model-------------")
                MODELS_DIR = '/home/ibon/en'
                print("-------------Downloading Stanford Basque model-------------")
                stanfordnlp.download('en', MODELS_DIR)  # Download the English models
            elif self.lang.lower() == "spanish":
                print("-------------You are going to use Spanish model-------------")
                MODELS_DIR = '/home/kepa/es'
                stanfordnlp.download('es', MODELS_DIR)  # Download the Spanish models
            else:
                print("........You cannot use this language...........")
        elif self.lib.lower() == "cube":
            print("-----------You are going to use Cube Library-----------")
        else:
            print("You cannot use this library. Introduce a valid library (Cube or Stanford)")

    '''
    load model in parser object 
    '''

    def load_model(self):
        if self.lib.lower() == "stanford":
            print("-----------You are going to use Stanford library-----------")
            if self.lang.lower() == "basque":
                print("-------------You are going to use Basque model-------------")
                MODELS_DIR = 'J:\TextSimilarity\eu'
                # MODELS_DIR = '/home/kepa/eu'
                #               config = {'processors': 'tokenize,pos,lemma,depparse',  # Comma-separated list of processors to use
                #                           'lang': 'eu',  # Language code for the language to build the Pipeline in
                #                           'tokenize_model_path': MODELS_DIR + '\eu_bdt_models\eu_bdt_tokenizer.pt',
                #                           # Processor-specific arguments are set with keys "{processor_name}_{argument_name}"
                #                           'pos_model_path': MODELS_DIR + '\eu_bdt_models\eu_bdt_tagger.pt',
                #                           'pos_pretrain_path': MODELS_DIR + '\eu_bdt_models\eu_bdt.pretrain.pt',
                #                           'lemma_model_path': MODELS_DIR + '\eu_bdt_models\eu_bdt_lemmatizer.pt',
                #                           'depparse_model_path': MODELS_DIR + '\eu_bdt_models\eu_bdt_parser.pt',
                #                           'depparse_pretrain_path': MODELS_DIR + '\eu_bdt_models\eu_bdt.pretrain.pt'
                #                          }
                config = {'processors': 'tokenize,pos,lemma,depparse',  # Comma-separated list of processors to use
                          'lang': 'eu',  # Language code for the language to build the Pipeline in
                          'tokenize_model_path': MODELS_DIR + '/eu_bdt_models/eu_bdt_tokenizer.pt',
                          # Processor-specific arguments are set with keys "{processor_name}_{argument_name}"
                          'pos_model_path': MODELS_DIR + '/eu_bdt_models/eu_bdt_tagger.pt',
                          'pos_pretrain_path': MODELS_DIR + '/eu_bdt_models/eu_bdt.pretrain.pt',
                          'lemma_model_path': MODELS_DIR + '/eu_bdt_models/eu_bdt_lemmatizer.pt',
                          'depparse_model_path': MODELS_DIR + '/eu_bdt_models/eu_bdt_parser.pt',
                          'depparse_pretrain_path': MODELS_DIR + '/eu_bdt_models/eu_bdt.pretrain.pt'
                          }
                self.parser = stanfordnlp.Pipeline(**config)

            elif self.lang.lower() == "english":
                print("-------------You are going to use English model-------------")
                MODELS_DIR = '/home/ibon/en'
                config = {'processors': 'tokenize,mwt,pos,lemma,depparse',  # Comma-separated list of processors to use
                          'lang': 'en',  # Language code for the language to build the Pipeline in
                          'tokenize_model_path': MODELS_DIR + '/en_ewt_models/en_ewt_tokenizer.pt',
                          'pos_model_path': MODELS_DIR + '/en_ewt_models/en_ewt_tagger.pt',
                          'pos_pretrain_path': MODELS_DIR + '/en_ewt_models/en_ewt.pretrain.pt',
                          'lemma_model_path': MODELS_DIR + '/en_ewt_models/en_ewt_lemmatizer.pt',
                          'depparse_model_path': MODELS_DIR + '/en_ewt_models/en_ewt_parser.pt',
                          'depparse_pretrain_path': MODELS_DIR + '/en_ewt_models/en_ewt.pretrain.pt'
                          }
                self.parser = stanfordnlp.Pipeline(**config)
            elif self.lang.lower() == "spanish":
                print("-------------You are going to use Spanish model-------------")
                MODELS_DIR = '/home/kepa/es'
                config = {'processors': 'tokenize,pos,lemma,depparse',  # Comma-separated list of processors to use
                          'lang': 'es',  # Language code for the language to build the Pipeline in
                          'tokenize_model_path': MODELS_DIR + '/es_ancora_models/es_ancora_tokenizer.pt',
                          # Processor-specific arguments are set with keys "{processor_name}_{argument_name}"
                          'pos_model_path': MODELS_DIR + '/es_ancora_models/es_ancora_tagger.pt',
                          'pos_pretrain_path': MODELS_DIR + '/es_ancora_models/es_ancora.pretrain.pt',
                          'lemma_model_path': MODELS_DIR + '/es_ancora_models/es_ancora_lemmatizer.pt',
                          'depparse_model_path': MODELS_DIR + '/es_ancora_models/es_ancora_parser.pt',
                          'depparse_pretrain_path': MODELS_DIR + '/es_ancora_models/es_ancora.pretrain.pt'
                          }
                self.parser = stanfordnlp.Pipeline(**config)
            else:
                print("........You cannot use this language...........")
        elif self.lib.lower() == "cube":
            print("-----------You are going to use Cube Library-----------")
            if self.lang.lower() == "basque":
                # initialize it
                cube = Cube(verbose=True)
                # load(self, language_code, version="latest",local_models_repository=None,
                # local_embeddings_file=None, tokenization=True, compound_word_expanding=False,
                # tagging=True, lemmatization=True, parsing=True).
                # Ejemplo:load("es",tokenization=False, parsing=False)
                ## select the desired language (it will auto-download the model on first run)
                cube.load("eu", "latest")
            elif self.lang.lower() == "english":
                cube = Cube(verbose=True)
                cube.load("en", "latest")
            elif self.lang.lower() == "spanish":
                cube = Cube(verbose=True)
                cube.load("es", "latest")
            else:
                print("........You cannot use this language...........")
        else:
            print("You cannot use this library. Introduce a valid library (Cube or Stanford)")

    def process_text(self, text):
        self.text = text.replace('\n', '@')
        self.text = re.sub(r'@+', '@', self.text)
        return self.text

    '''
    Transform data into a unified structure.
    '''

    def get_estructure(self, text):
        self.text = text
        # Loading a text with paragraphs
        self.textwithparagraphs = self.process_text(self.text)
        # Getting a unified structure [ [sentences], [sentences], ...]
        return self.adapt_nlp_model()

    def adapt_nlp_model(self):
        ma = ModelAdapter(self.parser, self.lib)
        return ma.model_analysis(self.textwithparagraphs)



class Pronouncing:
    # Pronunciador(the Carnegie Mellon Pronouncing Dictionary)- Utilizado para obtener silabas: pip install cmudict
        # cmudict is a pronouncing dictionary for north american english words.
        # it splits words into phonemes, which are shorter than syllables.
        # (e.g. the word 'cat' is split into three phonemes: K - AE - T).
        # but vowels also have a "stress marker":
        # either 0, 1, or 2, depending on the pronunciation of the word (so AE in 'cat' becomes AE1).
        # the code in the answer counts the stress markers and therefore the number of the vowels -
        # which effectively gives the number of syllables (notice how in OP's examples each syllable has exactly one vowel)
        #from nltk.corpus import cmudict
        #pronunciation dictionary
        #prondict = cmudict.dict()
    prondict = {}

    def __init__(self, language):
        self.lang = language

    def load(self):
        if self.lang.lower() == "english":
            Pronouncing.prondict = cmudict.dict()

"This is a Singleton class which is going to start necessary classes and methods."


# from packageDev.Charger import NLPCharger
# import re

class Main(object):
    __instance = None

    def __new__(cls):
        if Main.__instance is None:
            Main.__instance = object.__new__(cls)
        return Main.__instance

    def extract_text_from_file(self, input):
        # Si el fichero de entrada no tiene extension .txt
        if ".txt" not in input:
            # textract extrae el texto de todo tipo de formatos (odt, docx, doc ..)
            pre_text = textract.process(input)
            # decode(encoding='UTF-8',errors='strict') convierte a utf8 y si no puede lanza un error
            text = pre_text.decode()
        else:
            # Si extensión .txt convierte texto a utf-8
            with open(input, encoding='utf-8') as f:
                text = f.read()
        return text

    def start(self):
        #####Argumentos##################################
        from argparse import ArgumentParser
        # ArgumentParser con una descripción de la aplicación
        p = ArgumentParser(description="python3 ./main.py -f \"laginak/*.doc.txt\" ")
        # Grupo de argumentos requeridos
        required = p.add_argument_group('required arguments')
        required.add_argument('-f', '--files', nargs='+', help='Files to analyze (in .txt, .odt, .doc or .docx format)')
        # Grupo de argumentos opcionales
        optional = p.add_argument_group('optional arguments')
        optional.add_argument('-a', '--all', action='store_true', help="Generate a CSV file with all the results")
        optional.add_argument('-s', '--similarity', action='store_true', help="Calculate similarity (max. 5 files)")
        # Por último parsear los argumentos
        opts = p.parse_args()

        language = "basque"
        model = "stanford"

        # Carga StopWords
        stopw = Stopwords(language)
        stopw.download()
        stopw.load()
        # stopw.print()

        # Load Pronouncing Dictionary
        prondic = Pronouncing(language)
        prondic.load()

        # Carga del modelo Stanford/NLPCube
        cargador = NLPCharger(language, model)
        cargador.download_model()
        cargador.load_model()

        files = opts.files
        files = ["kk.txt", "kk.txt"]
        for input in files:
            # texto directamente de text
            if language == "basque":
                text = "ibon hondartzan egon da. Eguraldi oso ona egin zuen.\nHurrengo astean mendira joango da. "                "\n\nBere lagunak saskibaloi partidu bat antolatu dute 18etan, baina berak ez du jolastuko. \n "                "Etor zaitez etxera.\n Nik egin beharko nuke lan hori. \n Gizonak liburua galdu du. \n Irten hortik!"                    "\n Emadazu ur botila! \n Zu beti adarra jotzen."
            if language == "english":
                text = "ibon is going to the beach. I am ibon. \n"                 "Eder is going too. He is Eder."
            if language == "spanish":
                text = "ibon va ir a la playa. Yo soy ibon. \n"                 "Ibon tambien va a ir. El es Ibon."
            # texto directamente de fichero
            text = self.extract_text_from_file(input)

            # Get indicators
            document = cargador.get_estructure(text)
            indicators = document.get_indicators()
            printer = Printer(indicators)
            printer.print_info()


main = Main()
main.start()

# In[ ]:

