# Tsetlin Machine Animal Classifier

A Python example that trains a convolutional Tsetlin Machine to classify 28 x 28
doodle images of dogs, cats, horses, elephants, and bears. It includes:

- Training and evaluation in `main.py`
- A Tkinter drawing application in `test.py`
- Confusion-matrix and attention visualizations
- Saving/loading of the trained model

## Requirements

- Python 3.10 or newer
- A desktop environment with Tkinter support

## Installation

Create a virtual environment and install the required packages:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install numpy matplotlib seaborn scikit-learn pillow pyTsetlinMachine
```

## Download the data and trained model

From the project root, download the release assets with GitHub CLI:

```bash
gh release download v1.0.0 --repo JoanICG/TsetlinTest
```

Or download all six assets from the
[v1.0.0 release](https://github.com/JoanICG/TsetlinTest/releases/tag/v1.0.0)
in your browser and save them in the project root:

- `dog.npy`
- `cat.npy`
- `horse.npy`
- `elephant.npy`
- `bear.npy`
- `multiclass_model.pkl` (pretrained model for the drawing app)

The `.npy` files contain 28 x 28 grayscale Quick Draw images. The training
script uses the first 2,500 samples from each class. The datasets and model are
release downloads; they are not stored in Git history.

After installing the requirements, run the drawing app with the included
pretrained model:

```bash
python test.py
```

To train a new model from the downloaded datasets instead:

```bash
python main.py
```

This trains the model, prints epoch accuracy, creates
`confusion_matrix.png` and `attention_analysis.png`, and saves
`multiclass_model.pkl`.

## Drawing Interface

After training, launch the drawing application:

```bash
python test.py
```

Draw an animal in the window, then select **Predict**. Select **Clear** to reset
the drawing.

## Project Notes

Generated data, virtual environments, trained models, plots, Python caches, and
IDE files should not be committed. See `.gitignore` for the default exclusions.
