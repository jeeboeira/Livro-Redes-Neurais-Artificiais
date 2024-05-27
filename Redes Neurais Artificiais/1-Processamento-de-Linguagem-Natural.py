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
        ('com muito medo','emoção negativa'),
        ('um pouco triste','emoção negativa'),
        ('isto me deixou com raiva','emoção negativa'),
        ('fui demitida','emoção negativa'),
        ('esta comida está horrível','emoção negativa'),
        ('tenho pavor disso','emoção negativa'),
        ('fiquei desmotivada com o resultado','emoção negativa'),
        ('fui reprovado', 'emoção negativa'),
        ('muito mal', 'emoção negativa'),
        ('mal', 'emoção negativa'),
        ('um pouco triste', 'emoção negativa'),
        ('isto me deixou com raiva', 'emoção negativa'),
        ('fui demitida', 'emoção negativa'),
        ('esta comida está horrível', 'emoção negativa'),
        ('tenho pavor disso', 'emoção negativa'),
        ('fiquei desmotivada com o resultado', 'emoção negativa'),
        ('fui reprovado', 'emoção negativa'),
        ('mal', 'emoção negativa'),
        ('muito mal', 'emoção negativa'),
        ('mal', 'emoção negativa'),
        ('estou radiante com essa notícia', 'emoção positiva'),
        ('com medo', 'emoção negativa'),
        ('que dia maravilhoso para um passeio', 'emoção positiva'),
        ('fiquei com muito medo', 'emoção negativa'),
        ('consegui realizar meu sonho', 'emoção positiva'),
        ('meu coração está transbordando de gratidão', 'emoção positiva'),
        ('isto me deixou com raiva', 'emoção negativa'),
        ('adoro quando tudo dá certo', 'emoção positiva'),
        ('fui demitida', 'emoção negativa'),
        ('recebi um abraço caloroso que me deixou feliz o dia todo', 'emoção positiva'),
        ('esta comida está horrível', 'emoção negativa'),
        ('sinto-me revigorado depois de uma boa noite de sono', 'emoção positiva'),
        ('tenho pavor disso', 'emoção negativa'),
        ('estou transbordando de felicidade por ver meus amigos', 'emoção positiva'),
        ('incomodado', 'emoção negativa'),
        ('que sensação incrível é ver um pôr do sol deslumbrante', 'emoção positiva'),
        ('fiquei desmotivada com o resultado', 'emoção negativa'),
        ('estou animado para explorar novos horizontes', 'emoção positiva'),
        ('fui reprovado', 'emoção negativa'),
        ('estou exultante com essa conquista', 'emoção positiva'),
        ('que mal', 'emoção negativa'),
        ('o sorriso no rosto das pessoas me contagia', 'emoção positiva'),
        ('muito mal', 'emoção negativa'),
        ('a sensação de paz interior é indescritível', 'emoção positiva'),
        ('mal', 'emoção negativa'),
        ('sinto-me como se estivesse nas nuvens de tanta alegria', 'emoção positiva')]


#Dicionário em portugues para palavras irrelevantes ou que atrapalham o processo de mineração.
stopwords = nltk.corpus.stopwords.words('portuguese')

#print(f"base:\n{base}")


# Função que remove as stopwords da minha base, não vai ser usado pq a próxima função faz o mesmo e ainda tira o radical
def removestopwords(texto):
    frases = []
    for (palavras, emocao) in texto:
        removesw = [p for p in palavras.split() if p not in stopwords]
        frases.append((removesw, emocao))
    return frases


sem_stop_words = removestopwords(base)
#print(f"remove stopwords:\n {sem_stop_words}")



# Função que deixa somente o radical(stem) da palavra e remove as stopwords
def reduzpalavras(texto):
    steemer = nltk.stem.RSLPStemmer()
    frases_redux = []
    for (palavras, emocao) in texto:
        reduzidas = [str(steemer.stem(p)) for p in palavras.split() if p not in stopwords]
        frases_redux.append((reduzidas, emocao))
    return frases_redux


frases_reduzidas = reduzpalavras(base)

#print(f"frases_reduzidas:\n {frases_reduzidas}")


# Função pré processamento de palavra para referência nas associações de mineração
def buscapalavras(frases):
    todaspalavras = []
    for (palavras, emoção) in frases:
        todaspalavras.extend(palavras)
    return todaspalavras


palavras = buscapalavras(frases_reduzidas)

#print(f"palavras:\n{palavras}")

#A incidência/frequência com que certas palavras são assimiladas 
    #influenciam na margem de acertos do meu interpretador

# Função que me retorna a frequência com que certas palavras aparecem
def buscafrequencia(palavras):
    freq_palavras = nltk.FreqDist(palavras)
    return freq_palavras

frequencia = buscafrequencia(palavras)

# Função que me retorna as palavras únicas nos meus dados
def buscapalavrasunicas(frequencia):
    freq = frequencia.keys()
    return freq

palavras_unicas = buscapalavrasunicas(frequencia)


# Função que compara cada elemento com as palavras já classificadas na minha base de dados,
    #marcando como true quando houver combinações. se true,
    #o interpretador associa com a emoção referente aquela palavra
def extrator(documento):
    doc = set(documento) # Um set é uma coleção que não possui objetos repetidos.
    caracteristicas = {} # {} - dicionário: É um conjunto não ordenado de chave:valor
    for palavra in palavras_unicas:
        caracteristicas['%s' % palavra] = (palavra in doc) 
    return caracteristicas

baseprocessada = nltk.classify.apply_features(extrator, frases_reduzidas)
#print(f"baseprocessada:\n {baseprocessada[0]}")

classificador = nltk.NaiveBayesClassifier.train(baseprocessada)

print(classificador)

#Função main
def __main__():
    teste = ""
    while teste != "sair":
        #str converte para string
        teste = str(input("Digite como você está se sentindo ou sair para encerrar: "))
        teste_redux = []
        redux = nltk.stem.RSLPStemmer()
        for (palavras_treino) in teste.split():
            reduzida = [p for p in palavras_treino.split()]
            teste_redux.append(str(redux.stem(reduzida[0])))

        resultado = extrator(teste_redux)
        print(classificador.classify(resultado))
__main__()