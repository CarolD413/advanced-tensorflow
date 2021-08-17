import tensorflow as tf
from tensorflow.python.keras.utils.vis_utils import plot_model


def build_model_with_sequential():
    seq_model = tf.keras.models.Sequential(
        [tf.keras.layers.Flatten(input_shape=(28, 28)),
         tf.keras.layers.Dense(128, activation=tf.nn.relu),
         tf.keras.layers.Dense(10, activation=tf.nn.softmax)
         ])
    return seq_model


def build_model_with_functional():
    # step1 define input
    input_layer = tf.keras.layers.Input(shape=(28, 28))

    # step2 define layers
    flatten_layer = tf.keras.layers.Flatten()(input_layer)
    first_dense = tf.keras.layers.Dense(128, activation="relu")(flatten_layer)
    predictions = tf.keras.layers.Dense(10, activation="softmax")(first_dense)

    # step3 define the model
    func_model = tf.keras.models.Model(inputs=input, outputs=predictions)

    return func_model


def plot_models():
    # you can look at the graph to see that two methods are same
    model = build_model_with_functional()
    # model = build_model_with_functional()

    # plot model graph
    plot_model(model, show_shapes=Ture, show_layer_names=True, to_file="model.png")


def train_model():
    mnist = tf.keras.datasets.fashion_mnist
    (training_images, training_labels), (test_images, test_labels) = mnist.load_data()
    # 归一化：像素点数值范围为0-255
    training_images = training_images / 255.0
    test_images = test_images / 255.0

    model = build_model_with_functional()
    model.compile(optimizer=tf.optimizers.Adam(),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(training_images, training_labels, epochs=5)
    model.evaluate(test_images, test_labels)

