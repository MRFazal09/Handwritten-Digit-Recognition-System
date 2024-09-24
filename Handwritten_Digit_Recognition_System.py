# Step 1: Import the necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical

# Step 2: Load and preprocess the data
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train / 255.0
X_test = X_test / 255.0
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Step 3: Build the neural network model
model = Sequential()
model.add(Flatten(input_shape=(28, 28)))
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))

# Step 4: Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Step 5: Train the model
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Step 6: Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc:.4f}")

# Step 7: Take user input for an index and predict the corresponding digit
try:
    # Ask user for an index number
    index = int(input(f"Enter an index number (0 to {len(X_test) - 1}): "))

    # Ensure the index is within bounds
    if index < 0 or index >= len(X_test):
        print("Invalid index! Please enter a number within the test set range.")
    else:
        # Reshape the selected image and make a prediction
        sample_image = X_test[index].reshape(1, 28, 28)
        prediction = np.argmax(model.predict(sample_image), axis=-1)

        # Display the image and the predicted label
        plt.imshow(X_test[index], cmap='gray')
        plt.title(f"Predicted: {prediction[0]}")
        plt.show()

except ValueError:
    print("Invalid input! Please enter a valid number.")