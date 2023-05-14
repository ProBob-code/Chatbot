from spellchecker import SpellChecker
from symspellpy import SymSpell

spell = SpellChecker()
symsp = SymSpell()
symsp.load_dictionary('freq_dictionay_symspellpy.txt',\
                      term_index=0, \
                      count_index=1, \
                      separator=' ')

spell.word_frequency.load_words(['Genio','very'])
def spelling(sentence):
    splits = sentence.split()
    for split in splits:
        terms = symsp.lookup_compound(sentence,max_edit_distance=1)
        DD = terms[0].term
        sentence=DD.replace(split,spell.correction(split))
        
    return (sentence)
    
    
# text= "anik is slow, sanjani pagal ho gayi hai"
# print(spelling(text))