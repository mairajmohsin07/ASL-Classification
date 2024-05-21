from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import pickle

# Load the data
data_dict = pickle.load(open('./data.pickle', 'rb'))
data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# Create and train the Random Forest model
rf_model = RandomForestClassifier()
rf_model.fit(x_train, y_train)

# Save the Random Forest model
with open('model.p', 'wb') as f:
    pickle.dump({'model': rf_model}, f)

# Reshape the data for CNN input
data = data.reshape(-1, 21, 2, 1)  # Assuming the landmarks contain 21 points, 2 coordinates (x, y)

# Split the data again for CNN model
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# Create the CNN model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

cnn_model = Sequential()
cnn_model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(21, 2, 1)))
cnn_model.add(MaxPooling2D((2, 2)))
cnn_model.add(Conv2D(64, (3, 3), activation='relu'))
cnn_model.add(MaxPooling2D((2, 2)))
cnn_model.add(Flatten())
cnn_model.add(Dense(128, activation='relu'))
cnn_model.add(Dropout(0.5))
cnn_model.add(Dense(number_of_classes, activation='softmax'))

# Compile the model
cnn_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
cnn_model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_test, y_test))

# Evaluate the model
loss, accuracy = cnn_model.evaluate(x_test, y_test)
print('Accuracy: {:.2f}%'.format(accuracy * 100))

# Save the CNN model
cnn_model.save('asl_gesture_model.h5')

# Model comparison
# Load the Random Forest model
rf_model_dict = pickle.load(open('./model.p', 'rb'))
rf_model = rf_model_dict['model']

# Predict and evaluate the Random Forest model
y_predict = rf_model.predict(x_test)
rf_score = accuracy_score(y_predict, y_test)
print('Random Forest Accuracy: {:.2f}%'.format(rf_score * 100))
