from nli import compare_texts_nli_full
from paraphrase import compare_texts_paraphrase
from micro import transcribe

text1 = '''Валентина Владимировна Терешкова всегда мечтала о небе. Она занималась парашютным спортом, совершила около 100 прыжков.'''
text2 = transcribe("records/test/Anya/1.ogg")
#print(text1, text2, sep="\n")
print(compare_texts_nli_full(text1, text2))
print(compare_texts_paraphrase(text1, text2))
