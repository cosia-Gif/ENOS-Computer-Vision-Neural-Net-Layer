from ENOSNet import *



def build_model(input_shape):
    model = NeuralNet()
    model.add(Conv2DLayer(num_filters=8, kernel_size=3, input_shape=input_shape, stride=1, padding=1))
    model.add(ReLU())
    model.add(MaxPoolLayer(pool_size=2, stride=2))
    model.add(ENOSLayer(threshold=0.5))
    model.add(Conv2DLayer(num_filters=16, kernel_size=3, input_shape=(14, 14, 32), stride=1, padding=1))
    model.add(ReLU())
    model.add(MaxPoolLayer(pool_size=2, stride=2))
    model.add(FlattenLayer())
    model.add(DenseLayer(input_dim=7 * 7 * 16, output_dim=10))
    model.add(Softmax())
    return model

def load_mnist_data(x=None):
    from tensorflow.keras.datasets import mnist
    from tensorflow.keras.utils import to_categorical
    (X_train, y_train_labels), (X_test, y_test_labels) = mnist.load_data()
    
    if x is not None:
        X_train, y_train_labels = X_train[:x], y_train_labels[:x]
        X_test, y_test_labels = X_test[:x], y_test_labels[:x]
        
    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0
    
    X_train = X_train.reshape(-1, 28, 28, 1)
    X_test = X_test.reshape(-1, 28, 28, 1)

    y_train = to_categorical(y_train_labels, num_classes=10)
    y_test = to_categorical(y_test_labels, num_classes=10)
    
    return X_train, y_train, X_test, y_test

# Build and train the model
X_train, y_train, X_test, y_test = load_mnist_data()
model = build_model(input_shape=(28, 28, 1))
model.train(X_train, y_train, epochs=10, learning_rate=0.05, batch_size=64)

# Predict and evaluate
predictions = model.predict(X_test)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = np.argmax(y_test, axis=1)
accuracy = np.mean(predicted_classes == true_classes)
print(f"Test Accuracy: {accuracy * 100:.2f}%")