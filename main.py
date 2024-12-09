import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Decide if to load an existing model or to train a new one
train_new_model = True

if train_new_model:
    # Loading the MNIST data set with samples and splitting it
    mnist = tf.keras.datasets.mnist
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # Normalizing the data (making length = 1)
    X_train = tf.keras.utils.normalize(X_train, axis=1)
    X_test = tf.keras.utils.normalize(X_test, axis=1)

    # Create a neural network model
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(units=10, activation=tf.nn.softmax))

    # Compiling and optimizing model
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Training the model
    model.fit(X_train, y_train, epochs=3)

    # Evaluating the model
    val_loss, val_acc = model.evaluate(X_test, y_test)
    print(val_loss)
    print(val_acc)

    # Saving the model
    model.save('handwritten_digits.keras')
else:
    # Load the model
    model = tf.keras.models.load_model('handwritten_digits.keras')

# Load custom images and predict them
image_number = 1
images = []
predictions = []
while os.path.isfile('digits/digit{}.png'.format(image_number)):
    try:
        img = cv2.imread('digits/digit{}.png'.format(image_number))[:, :, 0]
        img = np.invert(np.array([img]))
        prediction = model.predict(img)
        print(f"The number is probably a {np.argmax(prediction)}")

        # Append the image and its prediction
        images.append(img[0])
        predictions.append(np.argmax(prediction))
        image_number += 1
    except:
        print("Error reading image! Proceeding with next image...")
        image_number += 1

# Interactive display of one image at a time


class ImageNavigator:
    def __init__(self, images, predictions):
        self.images = images
        self.predictions = predictions
        self.index = 0

        # Setup the figure and axis for the plot
        self.fig, self.ax = plt.subplots()
        self.ax.axis('off')
        self.update_image()

        # Connect click event to next image
        self.fig.canvas.mpl_connect('button_press_event', self.next_image)

    def update_image(self):
        self.ax.imshow(self.images[self.index], cmap=plt.cm.binary)
        self.ax.set_title(f"Prediction: {self.predictions[self.index]}")
        self.fig.canvas.draw()

    def next_image(self, event):
        self.index = (self.index + 1) % len(self.images)  # Loop through images
        self.update_image()


# Create the navigator and show the images
navigator = ImageNavigator(images, predictions)
plt.show()
