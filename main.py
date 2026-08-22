from nli import compare_texts_nli
from paraphrase import compare_texts_paraphrase


source = "Книги Пушкин любил с детства. По словам его младшего брата, он, еще будучи мальчиком, проводил бессонные ночи, тайком забираясь в кабинет отца. И без разбора «пожирал» все книги, попадавшиеся ему под руку."
good_retelling = "По словам его младшего брата, он проводил ночи, тайком забираясь в кабинет, и без разбора читал все книги, попадавшиеся ему под руку."
if __name__ == "__main__":
    print("----------------------------------")
    print("NLI результат:", compare_texts_nli(source, good_retelling))
    print("----------------------------------")
    print("Paraphrase результат:", compare_texts_paraphrase(source, good_retelling))
    print("----------------------------------")
