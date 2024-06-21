# POLIMENTO DA MINHA IMAGEM
imagem = mimg.imread(r'C:\Users\ti2\TestVsCode\Livro-Redes-Neurais-Artificiais-1\Redes_Neurais_convolucionais\Arquivos_Convolucionais\1Um\num3.jpg')
imagem = resize(imagem, (28, 28), anti_aliasing = True)
rgb_channels = imagem[...,:3]
imagem = np.dot(rgb_channels, [0.299, 0.587, 0.114])
imagem = (16 - (imagem * 16)).astype('float32')
imagem = np.expand_dims(imagem, axis=2)
imagem = np.expand_dims(imagem, axis=0)
imagem /= 255