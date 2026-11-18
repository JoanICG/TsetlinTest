import pickle
import tkinter as tk
import numpy as np
from PIL import Image, ImageDraw
from pyTsetlinMachine.tm import MultiClassConvolutionalTsetlinMachine2D

# 1. LOAD MODEL AND METADATA
with open("multiclass_model.pkl", "rb") as f:
    model_data = pickle.load(f)

CLASSES = model_data["classes"]
model_state = model_data["state"]

ctm = MultiClassConvolutionalTsetlinMachine2D(1200, 720, 7.0, (6, 6))

# Dummy fit to initialize C memory
X_dummy = np.zeros((len(CLASSES), 28, 28), dtype=np.uint32)
Y_dummy = np.arange(len(CLASSES), dtype=np.uint32)
ctm.fit(X_dummy, Y_dummy, epochs=0)
ctm.set_state(model_state)

# 2. CONTINUOUS-DRAWING GRAPHICAL INTERFACE
WIDTH, HEIGHT = 280, 280
root = tk.Tk()
root.title("Tsetlin Machine Classifier")

pil_img = Image.new("L", (WIDTH, HEIGHT), 0)
draw = ImageDraw.Draw(pil_img)

canvas = tk.Canvas(root, width=WIDTH, height=HEIGHT, bg="black")
canvas.pack()

previous_position = None


def start_stroke(event):
    global previous_position
    previous_position = (event.x, event.y)


def draw(event):
    global previous_position
    if previous_position is None:
        previous_position = (event.x, event.y)
        return

    x1, y1 = previous_position
    x2, y2 = event.x, event.y
    LINE_WIDTH = 6  # Approximately 1-2 pixels at 28x28 resolution

    # Draw continuous joined lines with rounded endpoints
    canvas.create_line(
        x1,
        y1,
        x2,
        y2,
        fill="white",
        width=LINE_WIDTH,
        capstyle=tk.ROUND,
        smooth=True,
    )
    draw.line([x1, y1, x2, y2], fill=255, width=LINE_WIDTH, joint="round")
    previous_position = (x2, y2)


def release(event):
    global previous_position
    previous_position = None


def predict():
    # Resize to 28x28 with bilinear smoothing to preserve the strokes
    image_28x28 = pil_img.resize((28, 28), Image.Resampling.BILINEAR)
    image_array = np.array(image_28x28)

    # Binarize as uint32
    binary_image = (image_array > 40).astype(np.uint32)
    active_pixels = binary_image.sum()

    print("\n" + "=" * 32)
    print(f"Active pixels detected: {active_pixels}")

    # Render the drawing in the console with ASCII characters
    print("\n[28x28 drawing preview]:")
    for row in binary_image:
        print("".join(["█" if pixel == 1 else " " for pixel in row]))

    if active_pixels < 12:
        result_label.config(text="⚠️ Draw a more complete stroke")
        return

    prediction_index = ctm.predict(np.array([binary_image]))[0]
    result = CLASSES[prediction_index]

    result_label.config(text=f"Prediction: {result.upper()}")
    print(f"\n🎯 Result: {result.upper()}")
    print("=" * 32)


def clear():
    global previous_position
    previous_position = None
    canvas.delete("all")
    draw.rectangle([0, 0, WIDTH, HEIGHT], fill=0)
    result_label.config(text="Draw an animal...")


canvas.bind("<Button-1>", start_stroke)
canvas.bind("<B1-Motion>", draw)
canvas.bind("<ButtonRelease-1>", release)

btn_frame = tk.Frame(root)
btn_frame.pack()

tk.Button(
    btn_frame, text="Predict", command=predict, font=("Helvetica", 11)
).pack(side=tk.LEFT, padx=5, pady=5)
tk.Button(
    btn_frame, text="Clear", command=clear, font=("Helvetica", 11)
).pack(side=tk.LEFT, padx=5, pady=5)

result_label = tk.Label(
    root, text="Draw an animal...", font=("Helvetica", 14, "bold")
)
result_label.pack(pady=5)

root.mainloop()