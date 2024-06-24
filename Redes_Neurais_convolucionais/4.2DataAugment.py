import numpy as np
from keras.models import Sequential
from keras import layers, models, optimizers
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.layers import BatchNormalization
#Reprocessa imagens, a partir de pradrões encontrados
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array

import pandas as pd
import matplotlib.pyplot as plt

base_treino_path = r'C:\Users\ti2\TestVsCode\Livro-Redes-Neurais-Artificiais-1\Redes_Neurais_convolucionais\Arquivos_Convolucionais\4Quatro\training_set'
base_validation_path = r'C:\Users\ti2\TestVsCode\Livro-Redes-Neurais-Artificiais-1\Redes_Neurais_convolucionais\Arquivos_Convolucionais\4Quatro\test_set'
#=================
#DATA AUGMENTATION
#=================

# Specify how to randomly transform training image data
train_generator = ImageDataGenerator(
    rescale            = 1./255,    # Rescale
    rotation_range     = 40,        # Randomly rotate
    width_shift_range  = 0.2,       # Randomly shift left or rigth
    height_shift_range = 0.2,       # Randomly shift up or down
    zoom_range         = 0.3,       # Randomly zooming in
    horizontal_flip    = True,      # Randomly flipping horizontally
    fill_mode          = 'nearest', # filling in newly created pixels
    )

# Validation image data "intentionally" NOT augmented
validation_data = ImageDataGenerator(
    rescale = 1./255) #Only rescale ; No Data augmetation

# Train Data Augmentation Generator
base_treino = train_generator.flow_from_directory(
    directory   = base_treino_path, # path to training images
    target_size = (150,150),        # resize images to 150 x 150
    batch_size  = 20,               # 32 images in each batch
    class_mode  = 'binary')         # since there are 2 labels

# Validation Generator
base_validation = validation_data.flow_from_directory(
    directory   = base_validation_path, # path to validation images
    target_size = (150,150),        # resize images to 150 x 150
    batch_size  = 20,               # 32 images in each batch
    class_mode  = 'binary')             # since there are 2 labels

#train_generator = train_generator.repeat()
#validation_data = validation_data.repeat()

#Inspect Generator Flow
for batch_images, batch_labels in base_validation:
    print(f'Batch shape: {batch_images.shape}')
    print(f'Labels shape: {batch_labels.shape}')
    break  # Exit after first batch for inspection

#======================================
# CNN architecture with Data Augmention
#======================================

# build linear stack of layers sequentialy, using 'Sequential()'
classificador = models.Sequential()

# a stack of alternated Conv2D & MaxPooling2D layers
classificador.add(layers.Conv2D(filters     = 32,
                         kernel_size = 3,      # As matrizes com 64x64 são separadas em blocos de 3x3
                         activation  = 'relu',
                         input_shape = (150, 150 ,3)))  #Imagem é uma matriz 64x64 com 3 canais, no caso rgb
classificador.add(layers.MaxPooling2D(pool_size = (2,2))) # Transforma em um mapa 2x2 maximum value pixel from the respective region of interest.

classificador.add(layers.Conv2D(filters     = 64,
                         kernel_size = 3,
                         activation  = 'relu'))
classificador.add(layers.MaxPooling2D(pool_size = (2,2)))

classificador.add(layers.Conv2D(filters     = 128,
                         kernel_size = 3,
                         activation  = 'relu'))
classificador.add(layers.MaxPooling2D(pool_size = (2,2)))

classificador.add(layers.Conv2D(filters     = 128,
                         kernel_size = 3,
                         activation  = 'relu'))
classificador.add(layers.MaxPooling2D(pool_size = (2,2)))


# Flatten, Dropout, Dense
classificador.add(layers.Flatten())
classificador.add(layers.Dropout(rate = 0.3))
classificador.add(layers.Dense(units = 512,
                               activation = 'relu'))
classificador.add(layers.Dense(units      = 1,         # Camada de saida com 1 neurônio
                               activation = 'sigmoid'))

# Check architecture
print(classificador.summary())

#==============
# Compile & Fit
#==============

# compilation
classificador.compile(
    optimizer = optimizers.RMSprop(),
    loss      = 'binary_crossentropy',
    metrics   = ['acc'])

epochs = 5
# Fit
classificador_fit = classificador.fit(
    x                = base_treino,     # data provided by generator
    steps_per_epoch  = 400,            # Realiza o teste sobre cada amostra, se não hover o valor informado
    epochs           = epochs,               # executa a rede 5x
    validation_data  = base_validation, # data provided by generator
    validation_steps = 100
    )


#Visualizar resultados
plt.plot([i+1 for i in range(epochs)],
         classificador_fit.history['acc'],
         label = 'Training Acc')
plt.plot([i+1 for i in range(epochs)],
         classificador_fit.history['val_acc'],
         label = 'Validation Acc')
plt.legend(), plt.xlabel("Epochs"), plt.ylabel("Accuracy")


# TESTA UMA IMAGEM

def tratarTeste(imagem):
    imagem = load_img(imagem, target_size = (150,150))
    imagem = img_to_array(imagem)
    imagem = np.expand_dims(imagem, axis = 0)
    imagem = classificador.predict(imagem)
    if imagem[0][0] == 1:
        prediction = 'dog'
    else:
        prediction = 'cat'
    return prediction

frida = tratarTeste(r'C:\Users\ti2\TestVsCode\Livro-Redes-Neurais-Artificiais-1\Redes_Neurais_convolucionais\Arquivos_Convolucionais\4Quatro\Frida.jpg')
cachorro = tratarTeste(r'C:\Users\ti2\TestVsCode\Livro-Redes-Neurais-Artificiais-1\Redes_Neurais_convolucionais\Arquivos_Convolucionais\4Quatro\cachorro.jpg')
cachorrodoDoTreino = tratarTeste(r'C:\Users\ti2\TestVsCode\Livro-Redes-Neurais-Artificiais-1\Redes_Neurais_convolucionais\Arquivos_Convolucionais\4Quatro\training_set\dogs\dog.6.jpg')
gatoDoTreino = tratarTeste(r'C:\Users\ti2\TestVsCode\Livro-Redes-Neurais-Artificiais-1\Redes_Neurais_convolucionais\Arquivos_Convolucionais\4Quatro\training_set\cats\cat.1.jpg')
