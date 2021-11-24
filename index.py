# TensorFlow and tf.keras
import tensorflow as tf

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

print('tensor flow image',tf.__version__)

# download dataset
fashion_mnist = tf.keras.datasets.fashion_mnist

# the dataset returnns numpy arrays assigned to these variables
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# store the images class names in this array since it is not part of the dataset
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# explore the dataset
print('train Images shape', train_images.shape)

print('train labels length:', len(train_labels))

print('train labels:',train_labels)

print('test image shape:',test_images.shape)

print('test labels length:',len(test_labels))

# Process the data:

  ## inspect first image : check pixel values
# plt.figure()
# plt.imshow(train_images[0])
# plt.colorbar()
# plt.grid(False)
# plt.show()

  ## Scale pixel values to a range of 0 to 1 before feeding them to the neural network model.
    # To do so, since pixels are in the range of 0 to 255, we divide each set by 255
    # It's important that the training set and the testing set be preprocessed in the same way
   
train_images = train_images / 255.0

test_images = test_images / 255.0

  ## Verify data by displaying first 25 images with their class labels.

# plt.figure(figsize=(10,10))
# for i in range(25):
#     plt.subplot(5,5,i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(train_images[i], cmap=plt.cm.binary)
#     plt.xlabel(class_names[train_labels[i]])
# plt.show()

# Build the Model
  ## Set up the layers of the model
    # Most of deep learning consists of chaining together simple layers.
    #  tf.keras.layers.Flatten, transforms the format of the images from a two-dimensional array (of 28 by 28 pixels) 
    # to a one-dimensional array (of 28 * 28 = 784 pixels)
    #  tf.keras.layers.Dense layers are densely connected, or fully connected, neural layers. The first Dense layer has 128 nodes (or neurons).
    #  The second (and last) layer returns a logits array with length of 10. Each node contains a score that indicates the current image belongs to one of the 10 classes.
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
])

  ## compile the model with some setting

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Train the model
    ## Feed the model
model.fit(train_images, train_labels, epochs=10)

    ## Evaluate accurac : check how model performs on the datasetS

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

print('\nTest accuracy:', test_acc)

# Make predictions
   ## The softmax layer converts the logits output to probabilities, which are easier to interpret.
probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])

   ## Predict the label for each image in the testing set
predictions = probability_model.predict(test_images)

  ## Let's take a look at the first prediction
predictions[0]

  ## Let's see which label has the highest confidence
print('first image class:',np.argmax(predictions[0]))

  ## Verify the output by examining the first label
print('firstiimage label class:', test_labels[0])

  ## Graph to look at the full set of 10 predictions
def plot_image(i, predictions_array, true_label, img):
  true_label, img = true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  plt.imshow(img, cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'

  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
  true_label = true_label[i]
  plt.grid(False)
  plt.xticks(range(10))
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)

  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')

# Verify predictions: Correct prediction are blue and weong predictions are red
  ## Plot image 1
# i = 0
# plt.figure(figsize=(6,3))
# plt.subplot(1,2,1)
# plot_image(i, predictions[i], test_labels, test_images)
# plt.subplot(1,2,2)
# plot_value_array(i, predictions[i],  test_labels)
# plt.show()

 ## Plot image 13
# i = 12
# plt.figure(figsize=(6,3))
# plt.subplot(1,2,1)
# plot_image(i, predictions[i], test_labels, test_images)
# plt.subplot(1,2,2)
# plot_value_array(i, predictions[i],  test_labels)
# plt.show()

 ## Plot several images
# Plot the first X test images, their predicted labels, and the true labels.
# Color correct predictions in blue and incorrect predictions in red.
# num_rows = 5
# num_cols = 3
# num_images = num_rows*num_cols
# plt.figure(figsize=(2*2*num_cols, 2*num_rows))
# for i in range(num_images):
#   plt.subplot(num_rows, 2*num_cols, 2*i+1)
#   plot_image(i, predictions[i], test_labels, test_images)
#   plt.subplot(num_rows, 2*num_cols, 2*i+2)
#   plot_value_array(i, predictions[i], test_labels)
# plt.tight_layout()
# plt.show()

# Use trained model on single image

# Grab an image from the test dataset.
img = test_images[1]

print('single image shape:',img.shape)

# Add the image to a batch where it's the only member.
img = (np.expand_dims(img,0))

print('single image batch shape:',img.shape)

# predict correct label for this image

predictions_single = probability_model.predict(img)

print('single image label prediction array:', predictions_single)

# Plot single prediction probability , index 0 since the image is the only image in the batch
# plot_value_array(1, predictions_single[0], test_labels)
# _ = plt.xticks(range(10), class_names, rotation=45)
# plt.show()

# Print the label with the highest confidence
print('single image label:', np.argmax(predictions_single[0]))
