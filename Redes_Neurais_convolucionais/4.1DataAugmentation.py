import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.layers import BatchNormalization
#Reprocessa imagens, a partir de pradrões encontrados
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing import image

import pandas as pd
import matplotlib.pyplot as plt

#=================
#DATA AUGMENTATION
#=================

# Specify how to randomly transform training image data
train_generator = ImageDataGenerator(
    rescale            = 1.0/255, # Rescale
    #rotation_range    = 7,       # Randomly rotate
    horizontal_flip    = True,    # Randomly flipping horizontally
    shear_range        = 0.2,
    height_shift_range = 0.07,    # Randomly shift up or down
    zoom_range         = 0.2      # Randomly zooming in
    )

# Validation image data "intentionally" NOT augmented
gerador_teste = ImageDataGenerator(rescale = 1.0/255) #Only rescale ; No Data augmetation

# Choose one image and see without transformation
one_image = r'C:\Users\ti2\TestVsCode\Livro-Redes-Neurais-Artificiais-1\Redes_Neurais_convolucionais\Arquivos_Convolucionais\4Quatro\cachorro.jpg'
img = image.load_img(one_image,               # read image
                     target_size = (150,150)) # resize to 150x150
plt.imshow(img)
plt.show()

# Convert the image to a Numpy array with shape (150, 150, 3)
x = image.img_to_array(img)
x.shape # (150, 150, 3)

# Expand dimensions
x = np.expand_dims(x, axis=0)
x.shape #(1, 150, 150, 3)

# Visualize 4 transformations
i = 0
# use train_generator from above to perform Data Augmentation
for batch in Train_generator.flow(x = x, batch_size = 1):
    plt.subplot(2, 2, i+1) # put 4 transformations together
    plt.imshow(image.array_to_img(batch[0])) #visualize
    i += 1
    if i % 4 == 0:
        break
'''
4 Randomly transformed / distirted images

As we can see, Data Augmentation generates "believeble=looking" images
to expose our model to more aspects of the data and generalize better!!
'''

