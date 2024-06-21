import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.layers import BatchNormalization
#Reprocessa imagens, a partir de pradrões encontrados
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array

import pandas as pd
import matplotlib as plt

base_treino_path = r'C:\Users\ti2\TestVsCode\Livro-Redes-Neurais-Artificiais-1\Redes_Neurais_convolucionais\Arquivos_Convolucionais\4Quatro\training_set'
base_validation_path = r'C:\Users\ti2\TestVsCode\Livro-Redes-Neurais-Artificiais-1\Redes_Neurais_convolucionais\Arquivos_Convolucionais\4Quatro\test_set'
#=================
#DATA AUGMENTATION
#=================

# Specify how to randomly transform training image data
train_generator = ImageDataGenerator(
    rescale            = 1.0/255, # Rescale
    rotation_range    = 7,        # Randomly rotate
    horizontal_flip    = True,    # Randomly flipping horizontally
    shear_range        = 0.2,
    height_shift_range = 0.07,    # Randomly shift up or down
    zoom_range         = 0.2      # Randomly zooming in
    )

# Validation image data "intentionally" NOT augmented
validation_data = ImageDataGenerator(rescale = 1.0/255) #Only rescale ; No Data augmetation

# Train Data Augmentation Generator
base_treino = train_generator.flow_from_directory(
    directory   = base_treino_path, # path to training images
    target_size = (64,64),          # resize images to 64 x 64
    batch_size  = 32,               # 32 images in each batch
    class_mode  = 'binary')         # since there are 2 labels

# Validation Generator
base_validation = validation_data.flow_from_directory(
    directory   = base_validation_path,
    target_size = (64,64),
    batch_size  = 32,
    class_mode  = 'binary')

#train_generator = train_generator.repeat()
#validation_data = validation_data.repeat()

#Inspect Generator Flow
for batch_images, batch_labels in base_validation:
    print(f'Batch shape: {batch_images.shape}')
    print(f'Labels shape: {batch_labels.shape}')
    break  # Exit after first batch for inspection

# REDE NEURAL

classificador = Sequential()
classificador.add(Conv2D(32,
                         (3,3), # As matrizes com 64x64 são separadas em blocos de 3x3
                         input_shape = (64,64,3), #Imagem é uma matriz 64x64 com 3 canais, no caso rgb
                         activation = 'relu'))
classificador.add(BatchNormalization())#Faz uma limpeza de pixels borrados
classificador.add(MaxPooling2D(pool_size = (2,2))) # Transforma em um mapa 2x2 maximum value pixel from the respective region of interest.
classificador.add(Conv2D(32,
                         (3,3),
                         activation = 'relu'))
classificador.add(BatchNormalization())
classificador.add(MaxPooling2D(pool_size = (2,2))) 
#Transforma as matrizes para uma dimensão, para cada valor da linha atuar como um neurônio
classificador.add(Flatten())
#Cria camadas densas
classificador.add(Dense(units = 128,
                        activation = 'relu'))
classificador.add(Dropout(0.2))
classificador.add(Dense(units = 128,
                        activation = 'relu'))
classificador.add(Dropout(0.2))
classificador.add(Dense(units = 1, # Camada de saida com 1 neurônio
                        activation = 'sigmoid'))#Sigmoid pois estamos classificando de forma binária
                        #Se fossem usadas 3 ou mais amostras usaria softmax

# COMPILADOR

classificador.compile(loss = 'binary_crossentropy', #Uma boa função de custo retorna valores altos para previsões ruins e valores baixos para previsões boas.
                      optimizer = 'adam',
                      metrics = ['accuracy'])

steps_per_epoch = len(base_treino)
validation_steps = len(base_validation)

classificador.fit(base_treino, # Alimenta a rede
                  steps_per_epoch = steps_per_epoch, #4000, # Realiza o teste sobre cada amostra, se não hover o valor informado
                  #ele realiza o teste sobre amostras repetidas ou geradas anteriormente
                  epochs = 5, # executa a rede 5x
                  validation_data = base_validation,
                  validation_steps = validation_steps)#1000)


#Visualizar resultados

# Convert classificador.history to a DataFrame
history_df = pd.DataFrame(classificador.history)

# Plot training and validation accuracy
plt.plot(history_df.index + 1, history_df['accuracy'], label='Training Accuracy')
plt.plot(history_df.index + 1, history_df['val_accuracy'], label='Validation Accuracy')

plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.show()



# TESTA UMA IMAGEM

def tratarTeste(imagem):
    imagem = load_img(imagem, target_size = (64,64))
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
