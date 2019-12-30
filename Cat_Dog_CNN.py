                                 #Cat OR Dog

#Building the CNN
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

#Initializing the CNN
classifier = Sequential()

#The Convolutional Layer
classifier.add(Convolution2D(32, (3,3) ,input_shape = (64,64,3), activation = "relu"))

#The Pooling Layer
classifier.add(MaxPooling2D(pool_size = (2,2) ))

#2nd Convolutional Layer
classifier.add(Convolution2D(32, (3,3) , activation = "relu"))

#2nd Convolutional Layer Max Pooling
classifier.add(MaxPooling2D(pool_size = (2,2) )) 

#The Flattening Layer
classifier.add(Flatten())

#Full Connection 
classifier.add(Dense(units= 128,activation = "relu"))
classifier.add(Dense(units= 1, activation = "sigmoid"))

#Compiling the CNN
classifier.compile(optimizer = "adam",loss = "binary_crossentropy" , metrics = ["accuracy"])

#Fitting the CNN to the images
import scipy.ndimage
from keras.preprocessing.image import ImageDataGenerator


train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

trainingset = train_datagen.flow_from_directory(
        'dataset/training_set',
        target_size=(64,64),
        batch_size=32,
        class_mode='binary')

testset = test_datagen.flow_from_directory(
        'dataset/test_set',
        target_size=(64,64),
        batch_size=32,
        class_mode='binary')

classifier.fit_generator(    
        trainingset,
        steps_per_epoch=8000,
        epochs=2, 
        validation_data=testset,
         validation_steps=2000)