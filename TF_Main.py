import tensorflow as tf
print("TensorFlow version:", tf.__version__)
import numpy as np
import matplotlib.pyplot as plt

def visualizeResults(predictions, true_labels, images):
    num_rows = 5
    num_cols = 6
    num_images = num_rows * num_cols
    plt.figure(figsize=(2*2*num_cols, 2*num_rows))
    for i in range(num_images):
        plt.subplot(num_rows, 2*num_cols, 2*i+1)
        plt.imshow(images[i], cmap=plt.cm.binary)
        plt.xticks([])
        plt.yticks([])
        predicted_label = np.argmax(predictions[i])
        true_label = true_labels[i]
        if predicted_label == true_label:
            color = 'blue'
        else:
            color = 'red'
        plt.xlabel(f"{predicted_label} ({100*np.max(predictions[i]):.2f}%)\nTrue: {true_label}", color=color)

        plt.subplot(num_rows, 2*num_cols, 2*i+2)
        plt.bar(range(10), predictions[i], color="#777777")
        plt.ylim([0, 1])
        plt.xticks(range(10))
        plt.yticks([])
        thisplot = plt.bar(range(10), predictions[i], color="#777777")
        thisplot[predicted_label].set_color('red')
        thisplot[true_label].set_color('blue')
    plt.tight_layout()
    plt.show()

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(64, activation='relu'),
  tf.keras.layers.Dense(10)
])

# predictions = model(x_train[:1]).numpy()
# print(predictions)
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# loss_fn = tf.keras.losses.MeanSquaredError()

model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)

model.evaluate(x_test, y_test, verbose=2)

small_data = x_test[:30]
small_labels = y_test[:]
predictions = tf.nn.softmax(model(small_data))
predictions = predictions.numpy()
visualizeResults(predictions, small_labels, small_data)