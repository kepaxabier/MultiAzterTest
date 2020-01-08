#Gold Standard to evaluate syllabification algorithms

##Languages included

###Spanish
This corpus of approximately 1000 words has been extracted from some books in Spanish at Project Gutenberg. The words were syllabified using an automatic rule-based syllabifier and manually checked.

###English
The corpus in English has approximately 1000 words, and it was extracted from the "For Better For Verse" projects poetry corpus[1][2]. 

###Basque
The corpus in Basque has been extracted and sampled from a Basque newspaper corpus (Egunkaria, which nowadays is Berria[3]) . The words were syllabified using an automatic rule-based syllabifier and manually checked.

##Rule-based syllabifiers
These rule-based syllabifiers were used to automatically create a gold standard that has been manually checked. The preliminary results show that they are quite robust and accurate for Spanish and Basque. There is a rule-based syllabifier for English at the project AthenaRhythm[4], but unfortunately, it relies on phonemic forms of the words, and it returns the phonemic representation, and not graphemes, as we are interested now.

These rule-based syllabifiers are written in Foma[5], a finite state compiler and library.

##Notation

We used the following notation when creating this collection.

WORD<TAB>SYLLABIFIED_WORD

The syllables must be marked using dots. We used some special characters when revising the automatically sylabified system. Lines starting at *, mean:

!: The syllabification was not correct and had to be corrected.

\#: Doubting syllabification.

?: Manually checked till there

#References

[1] "For Better For Verse" Interactive website: http://prosody.lib.virginia.edu/

[2] GitHub page of the project: https://github.com/waynegraham/for_better_for_verse/

[3] Berria: http://www.berria.eus

[4] AthenaRhythm project (Assigning stress to Out-Of-Vocabulary words): https://github.com/manexagirrezabal/athenarhythm

[5] Foma: https://github.com/mhulden/foma // https://foma.googlecode.com

#Author:
Manex Agirrezabal (2016/06/08)