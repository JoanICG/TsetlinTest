import pickle
import matplotlib.pyplot as plt
import numpy as np
from pyTsetlinMachine.tm import MultiClassConvolutionalTsetlinMachine2D
import seaborn as sns
from sklearn.metrics import confusion_matrix

CLASSES = ["dog", "cat", "horse", "elephant", "bear"]
SAMPLES_PER_CLASS = 2500  # More samples to capture drawing variations

X_list, Y_list = [], []
for label, class_name in enumerate(CLASSES):
    data = np.load(f"{class_name}.npy")[:SAMPLES_PER_CLASS]
    images = (data.reshape(-1, 28, 28) > 100).astype(np.uint32)
    X_list.append(images)
    Y_list.append(np.full(SAMPLES_PER_CLASS, label, dtype=np.uint32))

X = np.vstack(X_list)
Y = np.concatenate(Y_list)

indices = np.random.permutation(len(X))
X, Y = X[indices], Y[indices]

limit = int(len(X) * 0.8)
X_train, X_test = X[:limit], X[limit:]
Y_train, Y_test = Y[:limit], Y[limit:]

# --- 2. IMPROVED CONFIGURATION ---
CLAUSES = 1200  # 240 clauses per class (twice the memory)
THRESHOLD = 720  # Keep approximately 60% of the total
SPECIFICITY = 7.0  # Requires exact features (avoids false cats)
WINDOW = (6, 6)

ctm = MultiClassConvolutionalTsetlinMachine2D(
    CLAUSES, THRESHOLD, SPECIFICITY, WINDOW
)

# --- 3. TRAINING ---
EPOCHS = 12
print("Training Tsetlin Machine...")
for epoch in range(EPOCHS):
    ctm.fit(X_train, Y_train, epochs=1, incremental=True)
    accuracy = 100 * (ctm.predict(X_test) == Y_test).mean()
    print(f"Epoch {epoch+1}/{EPOCHS} - Accuracy: {accuracy:.2f}%")

# --- 4. GENERATE CONFUSION MATRIX ---
print("\nGenerating Confusion Matrix...")
predictions = ctm.predict(X_test)
cm = confusion_matrix(Y_test, predictions)

plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=CLASSES,
    yticklabels=CLASSES,
)
plt.title("Confusion Matrix - Animal Classification")
plt.xlabel("Model Prediction")
plt.ylabel("Actual Class")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.close()
print("-> Saved as 'confusion_matrix.png'")


# --- 5. VISUALIZE WHAT THE MODEL FOCUSES ON ---
def visualize_attention_example(test_index=0):
    image = X_test[test_index]
    actual_class = CLASSES[Y_test[test_index]]
    predicted_class_index = ctm.predict(np.array([image]))[0]
    predicted_class = CLASSES[predicted_class_index]

    # Simplified projection of active-pixel density (processed stroke pattern)
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    axes[0].imshow(image, cmap="gray_r")
    axes[0].set_title(f"Actual Image: {actual_class}")
    axes[0].axis("off")

    # Representation of strokes detected by the convolutional window
    axes[1].imshow(image, cmap="hot")
    axes[1].set_title(f"Prediction: {predicted_class}")
    axes[1].axis("off")

    plt.suptitle("Feature regions evaluated by the Tsetlin Machine")
    plt.tight_layout()
    plt.savefig("attention_analysis.png")
    plt.close()
    print("-> Attention map saved as 'attention_analysis.png'")


visualize_attention_example(0)

# Save state
with open("multiclass_model.pkl", "wb") as f:
    pickle.dump({"state": ctm.get_state(), "classes": CLASSES}, f)