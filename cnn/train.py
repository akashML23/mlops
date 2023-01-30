import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.losses import sparse_categorical_crossentropy
from keras.optimizers import Adam
from keras.models import save_model

img_width, img_height = 28, 28
batch_size = 250
no_epochs = 10
no_classes = 10
validation_split = 0.3
verbosity = 1
input_shape = (img_width, img_height, 1)


# Load MNIST dataset
def load_data():
    (input_train, target_train), (input_test, target_test) = tf.keras.datasets.mnist.load_data()
    return   (input_train, target_train), (input_test, target_test)




def process_data(input_train,input_test):
    # Reshape data
    input_train = input_train.reshape(input_train.shape[0], img_width, img_height, 1)
    input_test = input_test.reshape(input_test.shape[0], img_width, img_height, 1)
    # Scale data
    input_train = input_train / 255.0
    input_test = input_test / 255.0
    return input_train,input_test

# Create the models

def mnist_model():
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(no_classes, activation='softmax'))

    # Compile the models
    model.compile(loss=sparse_categorical_crossentropy,
                  optimizer=Adam(),
                  metrics=['accuracy'])
    return model



(input_train, target_train), (input_test, target_test) = load_data()
input_train,input_test = process_data(input_train,input_test)
model = mnist_model()
# Fit data to models
model.fit(input_train, target_train,
          batch_size=batch_size,
          epochs=no_epochs,
          verbose=verbosity,
          validation_split=validation_split)

# Generate generalization metrics
score = model.evaluate(input_test, target_test, verbose=0)
print(f'Test loss: {score[0]} / Test accuracy: {score[1]}')

filepath = 'models/cnn-model.h5'
save_model(model, filepath)