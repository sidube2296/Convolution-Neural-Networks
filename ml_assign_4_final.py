import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing import image_dataset_from_directory
from sklearn.metrics import confusion_matrix
import time
import urllib.error
from keras import Input
from keras.models import Sequential, Model, load_model
from keras.layers import Rescaling, Conv2D, Dense, Flatten, MaxPooling2D, Dropout, GlobalAveragePooling2D
from keras.preprocessing import image as preprocessing
from keras.applications import VGG16

# Removing non-JFIF files
files = glob.glob("Monkey Species Data/*/*/*")
for file in files:
    with open(file, "rb") as f:
        if not b"JFIF" in f.peek(10):
            os.remove(file)

# Load the training and test datasets
training_set = image_dataset_from_directory(
    "Monkey Species Data/Training Data",
    label_mode="categorical", image_size=(100, 100))
test_set = image_dataset_from_directory(
    "Monkey Species Data/Prediction Data",
    label_mode="categorical", image_size=(100, 100), shuffle=False)

print("\n********************************** Beginning of Task 1: CNN Architectures **********************************\n")
# Model 1: Basic CNN
model1 = Sequential([
    Rescaling(1./255, input_shape=(100, 100, 3)),
    Conv2D(32, kernel_size=(3, 3), activation="relu"),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation="relu"),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation="relu"),
    Dropout(0.5),
    Dense(10, activation="softmax")
])

model1.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Model 2: Advanced CNN with more layers
model2 = Sequential([
    Rescaling(1./255, input_shape=(100, 100, 3)),
    Conv2D(32, kernel_size=(3, 3), activation="relu"),
    Conv2D(32, kernel_size=(3, 3), activation="relu"),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation="relu"),
    Conv2D(64, (3, 3), activation="relu"),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(256, activation="relu"),
    Dropout(0.4),
    Dense(10, activation="softmax")
])

model2.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Function to train and evaluate a model
def train_and_evaluate_model(model, model_name):
    print(f"Training {model_name}")
    history = model.fit(training_set, epochs=30, validation_data=test_set)
    test_loss, test_acc = model.evaluate(test_set)
    print(f"{model_name} Test accuracy: {test_acc}")
    return model, test_acc, history

model1, acc1, history1 = train_and_evaluate_model(model1, "Model 1")
model2, acc2, history2 = train_and_evaluate_model(model2, "Model 2")

# Compare accuracies and save the best model
if acc1 > acc2:
    best_model = model1
    best_model_name = "Model1"
    best_acc = acc1
else:
    best_model = model2
    best_model_name = "Model2"
    best_acc = acc2

best_model.save(f"{best_model_name}_best_Task_1.keras")

# Predicting on the test set and computing confusion matrix
predictions = best_model.predict(test_set)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = np.concatenate([y.numpy() for x, y in test_set])
conf_matrix = confusion_matrix(np.argmax(true_classes, axis=1), predicted_classes)
print("Confusion Matrix:\n", conf_matrix)

print("\n********************************** End of Task 1: CNN Architectures **********************************\n")

print("\n********************************** Beginning of Task 2: Fine-Tuning Pre-trained CNN **********************************\n")

# Compare accuracies and save the best model
best_model, best_acc = (model1, acc1) if acc1 > acc2 else (model2, acc2)
best_model_name = "Model1" if acc1 > acc2 else "Model2"

# Fine-tuning VGG16
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(100, 100, 3))
x = GlobalAveragePooling2D()(base_model.output)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.5)(x)
final_output = Dense(10, activation='softmax')(x)
fine_tuned_model = Model(inputs=base_model.input, outputs=final_output)
for layer in base_model.layers:
    layer.trainable = False
fine_tuned_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train and evaluate the fine-tuned model
fine_tuned_model, ft_acc, ft_history = train_and_evaluate_model(fine_tuned_model, "Fine-Tuned Model")

# Compare the fine-tuned model with the best previous model
if ft_acc > best_acc:
    fine_tuned_model.save("fine_tuned_best_Task_2.keras")
    best_model = fine_tuned_model
    best_model_name = "Fine-Tuned Model"
else:
    best_model.save("overall_best_Task_2.keras")

# Predicting on the test set and computing confusion matrix
predictions = best_model.predict(test_set)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = np.concatenate([y.numpy() for x, y in test_set])
conf_matrix = confusion_matrix(np.argmax(true_classes, axis=1), predicted_classes)
print(f"Confusion Matrix for {best_model_name}:\n", conf_matrix)

print("\n********************************** End of Task 2: Fine-Tuning Pre-trained CNN **********************************\n")


print("\n********************************** Beginning of Task 3: Error Analysis **********************************\n")

# Provide the correct path to your test dataset directory
test_data_directory = '/Users/siddhi2218/Desktop/ml_assign_4/Monkey Species Data/Prediction Data'

# Load the test dataset
test_set = image_dataset_from_directory(
    test_data_directory,
    label_mode='categorical',
    image_size=(100, 100),
    batch_size=32,
    shuffle=False
)

# Load the best model from Task 1 based on the accuracy comparison

if acc2 > acc1:
    best_model_path = '/Users/siddhi2218/Desktop/ml_assign_4/Model2_best_Task_1.keras'
else:
    best_model_path = '/Users/siddhi2218/Desktop/ml_assign_4/Model1_best_Task_1.keras'  # Default path
best_model = load_model(best_model_path)

# Load your fine-tuned model from Task 2
fine_tuned_model = load_model(
    '/Users/siddhi2218/Desktop/ml_assign_4/fine_tuned_best_Task_2.keras')  # Update path as needed

# Get predictions for the test set with both models
predictions = best_model.predict(test_set)
fine_tuned_predictions = fine_tuned_model.predict(test_set)
predicted_classes = np.argmax(predictions, axis=1)
fine_tuned_predicted_classes = np.argmax(fine_tuned_predictions, axis=1)

# Collect true classes
true_classes = []
for images, labels in test_set.unbatch():
    true_classes.append(np.argmax(labels.numpy()))
true_classes = np.array(true_classes)

# Find indices where predictions and true labels mismatch for the initial model
incorrect_indices = np.where(predicted_classes != true_classes)[0]

# Check if there are enough incorrect samples
if len(incorrect_indices) < 10:
    print("There are less than 10 incorrect predictions in the test set.")
else:
    # Pick 10 random incorrect predictions to display
    selected_indices = np.random.choice(incorrect_indices, size=10, replace=False)

    plt.figure(figsize=(20, 10))
    for i, index in enumerate(selected_indices):
        img, label = list(test_set.unbatch())[index]

        # Display the image with prediction from best model
        plt.subplot(2, 10, 2 * i + 1)
        plt.imshow(img.numpy().astype("uint8"))
        plt.title(f"Task 1 Pred: {predicted_classes[index]}\nTrue: {true_classes[index]}")
        plt.axis('off')

        # Display the image with prediction from fine-tuned model
        plt.subplot(2, 10, 2 * i + 2)
        plt.imshow(img.numpy().astype("uint8"))
        plt.title(f"Task 2 Pred: {fine_tuned_predicted_classes[index]}\nTrue: {true_classes[index]}")
        plt.axis('off')

        # Print analysis for each image
        print(f"Analysis of incorrect prediction {i + 1}:")
        print(f"Task 1 Predicted: {predicted_classes[index]}, Actual: {true_classes[index]}")
        print(f"Task 2 Predicted: {fine_tuned_predicted_classes[index]}, Actual: {true_classes[index]}")
        print("Possible reasons for incorrect predictions in Task 1 might include:")
        print("- Image quality or lighting issues.")
        print("- Background clutter or occlusions.")
        print("- Similar features shared with other classes leading to confusion.")
        print("- Mislabeling in the dataset.")
        print("Improvements in Task 2 predictions can suggest better feature extraction and generalization from fine-tuning.")

    plt.show()

print("\n********************************** End of Task 3: Error Analysis **********************************\n")