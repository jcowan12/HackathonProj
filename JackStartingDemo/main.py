import tensorflow as tf
import matplotlib.pyplot as plt
import random
print(tf.__version__)

class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('accuracy')>0.99):
      print("\nReached 99% accuracy so cancelling training!")
      self.model.stop_training = True

callbacks = myCallback()
mnist = tf.keras.datasets.fashion_mnist
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()

training_images=training_images/255.0
test_images=test_images/255.0
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(512, activation=tf.nn.relu),
  tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(tf.expand_dims(training_images, axis =-1), training_labels, epochs=10, callbacks=[callbacks])
predictions = model.predict(test_images)
index1 = random.randint(0, 10000)
index2 = random.randint(0, 10000)
index3 = random.randint(0, 10000)
index4 = random.randint(0, 10000)
index5 = random.randint(0, 10000)
print(predictions[index1])
plt.imshow(test_images[index1])
print(test_labels[index1])
# print(predictions[index2])
# print(predictions[index3])
# print(predictions[index4])
# print(predictions[index5])

