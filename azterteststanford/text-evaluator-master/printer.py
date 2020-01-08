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
        kk=prediction
        print("------------------------------------------------------------------------------")
        print("\n Processing file " + str(file_num) + " of " + str(total) + ": " + os.path.basename(self.input) + "\n", end='\n', file=sys.stdout, flush=True)
        print("------------------------------------------------------------------------------")
        #print('Level of difficulty: ' + prediction[0].title())
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

        print("Number of words (length) of sentences without stopwords (mean): " + str(i['sentences_length_no_stopwords_mean']))
        print("Number of words (length) of sentences without stopwords (standard deviation): " + str(i['sentences_length_no_stopwords_std']))

        print('Mean number of syllables (length) in words: ' + str(i['num_syllables_words_mean']))
        print('Standard deviation of the mean number of syllables in words: ' + str(i['num_syllables_words_std']))

        print("Mean number of letters (length) in words: " + str(i['words_length_mean']))
        print("Standard deviation of number of letters in words: " + str(i['words_length_std']))

        print("Mean number of letters (length) in words without stopwords: " + str(i['words_length_no_stopwords_mean']))
        print("Standard deviation of the mean number of letter in words without stopwords: " + str(i['words_length_no_stopwords_std']))

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
        print("Number of irregular verbs in past tense (incidence per 1000 words): " + str(i['num_past_irregular_incidence']))
        # Porcentaje de verbos en pasado que son irregulares sobre total de verbos en pasado
        print("Mean of irregular verbs in past tense in relation to the number of verbs in past tense: " + str(i['num_past_irregular_mean']))
        # Number of personal pronouns
        print("Number of personal pronouns: " + str(i['num_personal_pronouns']))
        # Incidence score of pronouns (per 1000 words)
        print("Incidence score of pronouns (per 1000 words): " + str(i['num_personal_pronouns_incidence']))
        # Number of pronouns in first person
        print("Number of pronouns in first person: " + str(i['num_first_pers_pron']))
        # Incidence score of pronouns in first person  (per 1000 words)
        print("Incidence score of pronouns in first person  (per 1000 words): " + str(i['num_first_pers_pron_incidence']))
        # Number of pronouns in first person singular
        print("Number of pronouns in first person singular: " + str(i['num_first_pers_sing_pron']))
        # Incidence score of pronouns in first person singular (per 1000 words)
        print("Incidence score of pronouns in first person singular (per 1000 words): " + str(i['num_first_pers_sing_pron_incidence']))
        # Number of pronouns in third person
        print("Number of pronouns in third person: " + str(i['num_third_pers_pron']))
        # Incidence score of pronouns in third person (per 1000 words)
        print("Incidence score of pronouns in third person (per 1000 words): " + str(i['num_third_pers_pron_incidence']))


        print('Minimum word frequency per sentence (mean): ' + str(i['min_wf_per_sentence']))
        print('Number of rare nouns (wordfrecuency<=4): ' + str(i['num_rare_nouns_4']))
        print('Number of rare nouns (wordfrecuency<=4) (incidence per 1000 words): ' + str(i['num_rare_nouns_4_incidence']))
        print('Number of rare adjectives (wordfrecuency<=4): ' + str(i['num_rare_adj_4']))
        print('Number of rare adjectives (wordfrecuency<=4) (incidence per 1000 words): ' + str(i['num_rare_adj_4_incidence']))
        print('Number of rare verbs (wordfrecuency<=4): ' + str(i['num_rare_verbs_4']))
        print('Number of rare verbs (wordfrecuency<=4) (incidence per 1000 words): ' + str(i['num_rare_verbs_4_incidence']))
        print('Number of rare adverbs (wordfrecuency<=4): ' + str(i['num_rare_advb_4']))
        print('Number of rare adverbs (wordfrecuency<=4) (incidence per 1000 words): ' + str(i['num_rare_advb_4_incidence']))
        print('Number of rare content words (wordfrecuency<=4): ' + str(i['num_rare_words_4']))
        print('Number of rare content words (wordfrecuency<=4) (incidence per 1000 words): ' + str(i['num_rare_words_4_incidence']))
        print('Number of distinct rare content words (wordfrecuency<=4): ' + str(i['num_dif_rare_words_4']))
        print('Number of distinct rare content words (wordfrecuency<=4) (incidence per 1000 words): ' + str(i['num_dif_rare_words_4_incidence']))
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
        print('Incidence score of content words not in A1-C1 vocabulary (per 1000 words): ' + str(i['num_content_words_not_a1_c1_words_incidence']))

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
        print("Left embeddedness (Mean of number of words before the main verb) (SYNLE): " + str(i['left_embeddedness']))

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
        print("Number of relative subordinate clauses (incidence per 1000 words): " + str(i['num_rel_subord_incidence']))
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
        print('Noun overlap, all of the sentences in a paragraph or text, binary, mean (CRFNOa): ' + str(i['noun_overlap_all']))
        print('Argument overlap, adjacent sentences, binary, mean (CRFAOl): ' + str(i['argument_overlap_adjacent']))
        print('Argument overlap, all of the sentences in a paragraph or text, binary, mean (CRFAOa): ' + str(i['argument_overlap_all']))
        print('Stem overlap, adjacent sentences, binary, mean (CRFSOl): ' + str(i['stem_overlap_adjacent']))
        print('Stem overlap, all of the sentences in a paragraph or text, binary, mean (CRFSOa): ' + str(i['stem_overlap_all']))
        print('Content word overlap, adjacent sentences, proportional, mean (CRFCWO1): ' + str(i['content_overlap_adjacent_mean']))
        print('Content word overlap, adjacent sentences, proportional, standard deviation (CRFCWO1d): ' + str(i['content_overlap_adjacent_std']))
        print('Content word overlap, all of the sentences in a paragraph or text, proportional, mean (CRFCWOa): ' + str(i['content_overlap_all_mean']))
        print('Content word overlap, all of the sentences in a paragraph or text, proportional, standard deviation (CRFCWOad): ' + str(i['content_overlap_all_std']))

        # Semantic Similarity
        if similarity:
            print('Semantic Similarity between adjacent sentences (mean): ' + str(i['similarity_adjacent_mean']))
            print('Semantic Similarity between all possible pairs of sentences in a paragraph (mean):  ' + str(i['similarity_pairs_par_mean']))
            print('Semantic Similarity between adjacent paragraphs (mean): ' + str(i['similarity_adjacent_par_mean']))
            print('Semantic Similarity between adjacent sentences (standard deviation): ' + str(i['similarity_adjacent_std']))
            print('Semantic Similarity between all possible pairs of sentences in a paragraph (standard deviation):  ' + str(i['similarity_pairs_par_std']))
            print('Semantic Similarity between adjacent paragraphs (standard deviation): ' + str(i['similarity_adjacent_par_std']))

        # Connectives
        print('Number of connectives (incidence per 1000 words): ' + str(i['all_connectives_incidence']))
        print('Causal connectives (incidence per 1000 words): ' + str(i['causal_connectives_incidence']))
        print('Logical connectives (incidence per 1000 words):  ' + str(i['logical_connectives_incidence']))
        print('Adversative/contrastive connectives (incidence per 1000 words): ' + str(i['adversative_connectives_incidence']))
        print('Temporal connectives (incidence per 1000 words):  ' + str(i['temporal_connectives_incidence']))
        print('Conditional connectives (incidence per 1000 words): ' + str(i['conditional_connectives_incidence']))



    def generate_csv(self, csv_path, prediction, similarity):
        i = self.indicators
        kk=prediction
        # estadisticos
        output = os.path.join(csv_path, os.path.basename(self.input) + ".out.csv")
        # Write all the information in the file
        estfile = open(output, "w")
        #estfile.write("%s" % 'Level of difficulty: ' + prediction[0].title())
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
        estfile.write("\n%s" % 'Number of words (length) in sentences (standard deviation): ' + str(i['sentences_length_std']))
        estfile.write("\n%s" % 'Number of words (length) of sentences without stopwords (mean): ' + str(i['sentences_length_no_stopwords_mean']))
        estfile.write("\n%s" % 'Number of words (length) of sentences without stopwords (standard deviation): ' + str(i['sentences_length_no_stopwords_std']))
        estfile.write("\n%s" % 'Mean number of syllables (length) in words: ' + str(i['num_syllables_words_mean']))
        estfile.write("\n%s" % 'Standard deviation of the mean number of syllables in words: ' + str(i['num_syllables_words_std']))
        estfile.write("\n%s" % 'Mean number of letters (length) in words: ' + str(i['words_length_mean']))
        estfile.write("\n%s" % 'Standard deviation of number of letters in words: ' + str(i['words_length_std']))
        estfile.write("\n%s" % 'Mean number of letters (length) in words without stopwords: ' + str(i['words_length_no_stopwords_mean']))
        estfile.write("\n%s" % 'Standard deviation of the mean number of letter in words without stopwords: ' + str(i['words_length_no_stopwords_std']))
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
        estfile.write("\n%s" % 'Number of verbs in past tense (incidence per 1000 words): ' + str(i['num_past_incidence']))
        estfile.write("\n%s" % 'Number of verbs in present tense: ' + str(i['num_pres']))
        estfile.write("\n%s" % 'Number of verbs in present tense (incidence per 1000 words): ' + str(i['num_pres_incidence']))
        estfile.write("\n%s" % 'Number of verbs in future tense: ' + str(i['num_future']))
        estfile.write("\n%s" % 'Number of verbs in future tense (incidence per 1000 words): ' + str(i['num_future_incidence']))
        estfile.write("\n%s" % 'Number of verbs in indicative mood: ' + str(i['num_indic']))
        estfile.write("\n%s" % 'Number of verbs in indicative mood (incidence per 1000 words): ' + str(i['num_indic_incidence']))
        estfile.write("\n%s" % 'Number of verbs in imperative mood: ' + str(i['num_impera']))
        estfile.write("\n%s" % 'Number of verbs in imperative mood (incidence per 1000 words): ' + str(i['num_impera_incidence']))
        estfile.write("\n%s" % 'Number of irregular verbs in past tense: ' + str(i['num_past_irregular']))
        estfile.write("\n%s" % 'Number of irregular verbs in past tense (incidence per 1000 words): ' + str(i['num_past_irregular_incidence']))
        estfile.write("\n%s" % 'Mean of irregular verbs in past tense in relation to the number of verbs in past tense: ' + str(i['num_past_irregular_mean']))
        estfile.write("\n%s" % 'Number of personal pronouns: ' + str(i['num_personal_pronouns']))
        estfile.write("\n%s" % 'Incidence score of pronouns (per 1000 words): ' + str(i['num_personal_pronouns_incidence']))
        estfile.write("\n%s" % 'Number of pronouns in first person: ' + str(i['num_first_pers_pron']))
        estfile.write("\n%s" % 'Incidence score of pronouns in first person  (per 1000 words): ' + str(i['num_first_pers_pron_incidence']))
        estfile.write("\n%s" % 'Number of pronouns in first person singular: ' + str(i['num_first_pers_sing_pron']))
        estfile.write("\n%s" % 'Incidence score of pronouns in first person singular (per 1000 words): ' + str(i['num_first_pers_sing_pron_incidence']))
        estfile.write("\n%s" % 'Number of pronouns in third person: ' + str(i['num_third_pers_pron']))
        estfile.write("\n%s" % 'Incidence score of pronouns in third person (per 1000 words): ' + str(i['num_third_pers_pron_incidence']))
        estfile.write("\n%s" % 'Word Frequency')
        estfile.write("\n%s" % 'Minimum word frequency per sentence (mean): ' + str(i['min_wf_per_sentence']))
        estfile.write("\n%s" % 'Number of rare nouns (wordfrecuency<=4): ' + str(i['num_rare_nouns_4']))
        estfile.write("\n%s" % 'Number of rare nouns (wordfrecuency<=4) (incidence per 1000 words): ' + str(i['num_rare_nouns_4_incidence']))
        estfile.write("\n%s" % 'Number of rare adjectives (wordfrecuency<=4): ' + str(i['num_rare_adj_4']))
        estfile.write("\n%s" % 'Number of rare adjectives (wordfrecuency<=4) (incidence per 1000 words): ' + str(i['num_rare_adj_4_incidence']))
        estfile.write("\n%s" % 'Number of rare verbs (wordfrecuency<=4): ' + str(i['num_rare_verbs_4']))
        estfile.write("\n%s" % 'Number of rare verbs (wordfrecuency<=4) (incidence per 1000 words): ' + str(i['num_rare_verbs_4_incidence']))
        estfile.write("\n%s" % 'Number of rare adverbs (wordfrecuency<=4): ' + str(i['num_rare_advb_4']))
        estfile.write("\n%s" % 'Number of rare adverbs (wordfrecuency<=4) (incidence per 1000 words): ' + str(i['num_rare_advb_4_incidence']))
        estfile.write("\n%s" % 'Number of rare content words (wordfrecuency<=4): ' + str(i['num_rare_words_4']))
        estfile.write("\n%s" % 'Number of rare content words (wordfrecuency<=4) (incidence per 1000 words): ' + str(i['num_rare_words_4_incidence']))
        estfile.write("\n%s" % 'Number of distinct rare content words (wordfrecuency<=4): ' + str(i['num_dif_rare_words_4']))
        estfile.write("\n%s" % 'Number of distinct rare content words (wordfrecuency<=4) (incidence per 1000 words): ' + str(i['num_dif_rare_words_4_incidence']))
        estfile.write("\n%s" % 'Mean of rare lexical words (word frequency <= 4): ' + str(i['mean_rare_4']))
        estfile.write("\n%s" % 'Mean of distinct rare lexical words (word frequency <= 4): ' + str(i['mean_distinct_rare_4']))
        estfile.write("\n%s" % 'Vocabulary Knowledge')
        estfile.write("\n%s" % 'Number of A1 vocabulary in the text: ' + str(i['num_a1_words']))
        estfile.write("\n%s" % 'Incidence score of A1 vocabulary  (per 1000 words): ' + str(i['num_a1_words_incidence']))
        estfile.write("\n%s" % 'Number of A2 vocabulary in the text: ' + str(i['num_a2_words']))
        estfile.write("\n%s" % 'Incidence score of A2 vocabulary  (per 1000 words): ' + str(i['num_a2_words_incidence']))
        estfile.write("\n%s" % 'Number of B1 vocabulary in the text: ' + str(i['num_b1_words']))
        estfile.write("\n%s" % 'Incidence score of B1 vocabulary  (per 1000 words): ' + str(i['num_b1_words_incidence']))
        estfile.write("\n%s" % 'Number of B2 vocabulary in the text: ' + str(i['num_b2_words']))
        estfile.write("\n%s" % 'Incidence score of B2 vocabulary  (per 1000 words): ' + str(i['num_b2_words_incidence']))
        estfile.write("\n%s" % 'Number of C1 vocabulary in the text: ' + str(i['num_c1_words']))
        estfile.write("\n%s" % 'Incidence score of C1 vocabulary  (per 1000 words): ' + str(i['num_c1_words_incidence']))
        estfile.write("\n%s" % 'Number of content words not in A1-C1 vocabulary: ' + str(i['num_content_words_not_a1_c1_words']))
        estfile.write("\n%s" % 'Incidence score of content words not in A1-C1 vocabulary (per 1000 words): ' + str(i['num_content_words_not_a1_c1_words_incidence']))
        estfile.write("\n%s" % 'Syntactic Features / POS ratios')
        estfile.write("\n%s" % 'Number of content words: ' + str(i['num_lexic_words']))
        estfile.write("\n%s" % 'Number of content words (incidence per 1000 words): ' + str(i['num_lexic_words_incidence']))
        estfile.write("\n%s" % 'Number of nouns: ' + str(i['num_noun']))
        estfile.write("\n%s" % 'Number of nouns (incidence per 1000 words): ' + str(i['num_noun_incidence']))
        estfile.write("\n%s" % 'Number of adjectives: ' + str(i['num_adj']))
        estfile.write("\n%s" % 'Number of adjectives (incidence per 1000 words): ' + str(i['num_adj_incidence']))
        estfile.write("\n%s" % 'Number of adverbs: ' + str(i['num_adv']))
        estfile.write("\n%s" % 'Number of adverbs (incidence per 1000 words): ' + str(i['num_adv_incidence']))
        estfile.write("\n%s" % 'Number of verbs: ' + str(i['num_verb']))
        estfile.write("\n%s" % 'Number of verbs (incidence per 1000 words): ' + str(i['num_verb_incidence']))
        estfile.write("\n%s" % 'Left embeddedness (Mean of number of words before the main verb) (SYNLE): ' + str(i['left_embeddedness']))
        estfile.write("\n%s" % 'Number of decendents per noun phrase (mean): ' + str(i['num_decendents_noun_phrase']))
        estfile.write("\n%s" % 'Number of modifiers per noun phrase (mean) (SYNNP): ' + str(i['num_modifiers_noun_phrase']))
        estfile.write("\n%s" % 'Mean of the number of levels of dependency tree (Depth): ' + str(i['mean_depth_per_sentence']))
        estfile.write("\n%s" % 'Number of subordinate clauses: ' + str(i['num_subord']))
        estfile.write("\n%s" % 'Number of subordinate clauses (incidence per 1000 words): ' + str(i['num_subord_incidence']))
        estfile.write("\n%s" % 'Number of relative subordinate clauses: ' + str(i['num_rel_subord']))
        estfile.write("\n%s" % 'Number of relative subordinate clauses (incidence per 1000 words): ' + str(i['num_rel_subord_incidence']))
        estfile.write("\n%s" % 'Punctuation marks per sentence (mean): ' + str(i['num_punct_marks_per_sentence']))
        estfile.write("\n%s" % 'Number of propositions: ' + str(i['num_total_prop']))
        estfile.write("\n%s" % 'Mean of the number of propositions per sentence: ' + str(i['mean_propositions_per_sentence']))
        estfile.write("\n%s" % 'Mean of the number of VPs per sentence: ' + str(i['mean_vp_per_sentence']))
        estfile.write("\n%s" % 'Mean of the number of NPs per sentence: ' + str(i['mean_np_per_sentence']))
        estfile.write("\n%s" % 'Noun phrase density, incidence (DRNP): ' + str(i['noun_phrase_density_incidence']))
        estfile.write("\n%s" % 'Verb phrase density, incidence (DRVP): ' + str(i['verb_phrase_density_incidence']))
        estfile.write("\n%s" % "Number of passive voice verbs: " + str(i['num_pass']))
        estfile.write("\n%s" % "Number of passive voice verbs (incidence per 1000 words): " + str(i['num_pass_incidence']))
        estfile.write("\n%s" % "Mean of passive voice verbs: " + str(i['num_pass_mean']))
        estfile.write("\n%s" % "Number of agentless passive voice verbs: " + str(i['num_agentless']))
        estfile.write("\n%s" % 'Agentless passive voice density, incidence (DRPVAL): ' + str(i['agentless_passive_density_incidence']))
        estfile.write("\n%s" % "Number of negative words: " + str(i['num_neg']))
        estfile.write("\n%s" % 'Negation density, incidence (DRNEG): ' + str(i['negation_density_incidence']))
        estfile.write("\n%s" % "Number of verbs in gerund form: " + str(i['num_ger']))
        estfile.write("\n%s" % 'Gerund density, incidence (DRGERUND): ' + str(i['gerund_density_incidence']))
        estfile.write("\n%s" % "Number of verbs in infinitive form: " + str(i['num_inf']))
        estfile.write("\n%s" % 'Infinitive density, incidence (DRINF): ' + str(i['infinitive_density_incidence']))
        estfile.write("\n%s" % 'Semantics. Readability')
        estfile.write("\n%s" % 'Mean values of polysemy in the WordNet lexicon: ' + str(i['polysemic_index']))
        estfile.write("\n%s" % 'Mean hypernym values of verbs in the WordNet lexicon: ' + str(i['hypernymy_verbs_index']))
        estfile.write("\n%s" % 'Mean hypernym values of nouns in the WordNet lexicon: ' + str(i['hypernymy_nouns_index']))
        estfile.write("\n%s" % 'Mean hypernym values of nouns and verbs in the WordNet lexicon: ' + str(i['hypernymy_index']))
        estfile.write("\n%s" % 'Referential cohesion')
        estfile.write("\n%s" % 'Noun overlap, adjacent sentences, binary, mean (CRFNOl): ' + str(i['noun_overlap_adjacent']))
        estfile.write("\n%s" % 'Noun overlap, all of the sentences in a paragraph or text, binary, mean (CRFNOa): ' + str(i['noun_overlap_all']))
        estfile.write("\n%s" % 'Argument overlap, adjacent sentences, binary, mean (CRFAOl): ' + str(i['argument_overlap_adjacent']))
        estfile.write("\n%s" % 'Argument overlap, all of the sentences in a paragraph or text, binary, mean (CRFAOa): ' + str(i['argument_overlap_all']))
        estfile.write("\n%s" % 'Stem overlap, adjacent sentences, binary, mean (CRFSOl): ' + str(i['stem_overlap_adjacent']))
        estfile.write("\n%s" % 'Stem overlap, all of the sentences in a paragraph or text, binary, mean (CRFSOa): ' + str(i['stem_overlap_all']))
        estfile.write("\n%s" % 'Content word overlap, adjacent sentences, proportional, mean (CRFCWO1): ' + str(i['content_overlap_adjacent_mean']))
        estfile.write("\n%s" % 'Content word overlap, adjacent sentences, proportional, standard deviation (CRFCWO1d): ' + str(i['content_overlap_adjacent_std']))
        estfile.write("\n%s" % 'Content word overlap, all of the sentences in a paragraph or text, proportional, mean (CRFCWOa): ' + str(i['content_overlap_all_mean']))
        estfile.write("\n%s" % 'Content word overlap, all of the sentences in a paragraph or text, proportional, standard deviation (CRFCWOad): ' + str(i['content_overlap_all_std']))
        if similarity:
            estfile.write("\n%s" % 'Semantic overlap')
            estfile.write("\n%s" % 'Semantic Similarity between adjacent sentences (mean): ' + str(i['similarity_adjacent_mean']))
            estfile.write("\n%s" % 'Semantic Similarity between all possible pairs of sentences in a paragraph (mean): ' + str(i['similarity_pairs_par_mean']))
            estfile.write("\n%s" % 'Semantic Similarity between adjacent paragraphs (mean): ' + str(i['similarity_adjacent_par_mean']))
            estfile.write("\n%s" % 'Semantic Similarity between adjacent sentences (standard deviation): ' + str(i['similarity_adjacent_std']))
            estfile.write("\n%s" % 'Semantic Similarity between all possible pairs of sentences in a paragraph (standard deviation): ' + str(i['similarity_pairs_par_std']))
            estfile.write("\n%s" % 'Semantic Similarity between adjacent paragraphs (standard deviation): ' + str(i['similarity_adjacent_par_std']))
        estfile.write("\n%s" % 'Connectives')
        estfile.write("\n%s" % 'Number of connectives (incidence per 1000 words): ' + str(i['all_connectives_incidence']))
        estfile.write("\n%s" % 'Causal connectives (incidence per 1000 words): ' + str(i['causal_connectives_incidence']))
        estfile.write("\n%s" % 'Logical connectives (incidence per 1000 words):  ' + str(i['logical_connectives_incidence']))
        estfile.write("\n%s" % 'Adversative/contrastive connectives (incidence per 1000 words): ' + str(i['adversative_connectives_incidence']))
        estfile.write("\n%s" % 'Temporal connectives (incidence per 1000 words):  ' + str(i['temporal_connectives_incidence']))
        estfile.write("\n%s" % 'Conditional connectives (incidence per 1000 words): ' + str(i['conditional_connectives_incidence']))
        estfile.close()

    def write_in_full_csv(self, df, similarity):
        i = self.indicators
        indicators_dict = {}
        headers = []
        ignore_list = ['prop', 'num_complex_words', 'num_words_more_3_syl', 'num_lexic_words']
        similarity_list = ["similarity_adjacent_mean", "similarity_pairs_par_mean", "similarity_adjacent_par_mean", "similarity_adjacent_std", "similarity_pairs_par_std", "similarity_adjacent_par_std"]
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
