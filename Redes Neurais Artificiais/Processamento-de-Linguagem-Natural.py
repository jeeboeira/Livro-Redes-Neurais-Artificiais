#Biblioteca Natural Language toolkit
import nltk
nltk.download('popular')
nltk.download('rslp')
#Importa o conteúdo da variável base, do pacote base.py
#import base
base = [('estou muito feliz','emoção positiva'),
        ('sou uma pessoa feliz','emoção positiva'),
        ('alegria é o meu lema','emoção positiva'),
        ('muito bom ser amado','emoção positiva'),
        ('estou empolgado em começar','emoção positiva'),
        ('fui elogiado por meu trabalho','emoção positiva'),
        ('vencemos a partida','emoção positiva'),
        ('recebi uma promoção de cargo','emoção positiva'),
        ('o dia está muito bonito','emoção positiva'),
        ('estou bem, obrigado','emoção positiva'),
        ('fui aprovado','emoção positiva'),
        ('de bem com a vida','emoção positiva'),
        ('fui bem recebido em casa','emoção positiva'),
        ('estou com medo','emoção negativa'),
        ('estou com muito medo','emoção negativa'),
        ('estou um pouco triste','emoção negativa'),
        ('isto me deixou com raiva','emoção negativa'),
        ('fui demitida','emoção negativa'),
        ('esta comida está horrível','emoção negativa'),
        ('tenho pavor disso','emoção negativa'),
        ('estou incomodado','emoção negativa'),
        ('fiquei desmotivada com o resultado','emoção negativa'),
        ('fui reprovado','emoção negativa')]


#Dicionário em portugues para palavras irrelevantes ou que atrapalham o processo de mineração.
stopwords = nltk.corpus.stopwords.words('portuguese')

print('base')
print(base)

# Função que remove as stopwords da minha base
def removestopwords(texto):
    frases = []
    for (palavras, emocao) in texto:
        removesw = [p for p in palavras.split() if p not in stopwords]
        frases.append((removesw, emocao))
    return frases
print("remove stopwords")
print(removestopwords(base))

# Função
def reduzpalavras(texto):
    steemer = nltk.stem.RSLPStemmer()
    frases_redux = []
    for (palavras, emocao) in texto:
        reduzidas = [str(steemer.stem(p)) for p in palavras.split() if p not in stopwords]
        frases_redux.append((reduzidas, emocao))
    return frases_redux
a = removestopwords(base)
b = reduzpalavras(base)
print(redux)
print(b)