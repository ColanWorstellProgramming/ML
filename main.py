import tensorflow as tf
from tensorflow import keras as ks
print("TensorFlow version:", tf.__version__)

#Import Dataset
mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train / 255.0
x_test = x_test / 255.0

#setup keras' model Squential, one input one output
model = ks.models.Sequential([

#little confused, makes input 1d maybe? instead of (28,28) its 28*28=784??
  tf.keras.layers.Flatten(input_shape=(28, 28)),

#normal fully connected layer
  tf.keras.layers.Dense(128, activation='relu'),

#ignores random neurons
  tf.keras.layers.Dropout(0.2),

#normal fully connected layer
  tf.keras.layers.Dense(10)
])


predictions = model(x_train[:1]).numpy()

#tf.nn.softmax turns 'logits' into probablilites
tf.nn.softmax(predictions).numpy()

#calculates loss
loss_fn = ks.losses.SparseCategoricalCrossentropy(from_logits=True)
loss_fn(y_train[:1], predictions).numpy()

#optimizes with 'adam'? uses our loss_fn function and accuracy metrics
model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])

#set epoch counts and to keep nums between 0-1
model.fit(x_train, y_train, epochs=5)

#checks actual preformance/ runs model
model.evaluate(x_test,  y_test, verbose=2)
