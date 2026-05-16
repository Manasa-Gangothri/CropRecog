from tkinter import *
from tkinter import filedialog, messagebox
import os
import numpy as np
import joblib
import pandas as pd
import cv2
import matplotlib
matplotlib.use('TkAgg')
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix
)

import matplotlib.pyplot as plt
import seaborn as sns
from skimage.io import imread
from skimage.transform import resize
from PIL import Image, ImageTk
from skimage.feature import hog
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# ================= GLOBAL VARIABLES =================
filename = ""
X = None
Y = None
x_train = x_test = y_train = y_test = None
categories = []
model_folder = "model"

metrics_overall = []
class_metrics_storage = {}

# ================= TEXT CLEAR FUNCTION =================
def clearText():
    text.delete('1.0', END)

# ================= DATASET UPLOAD =================

def uploadDataset():
    clearText()
    global filename, categories

    filename = filedialog.askdirectory(initialdir="Dataset")
    if not filename:
        return

    text.insert(END, f"Folder Loaded:\n{filename}\n\n")

    categories = sorted([
    d for d in os.listdir(filename)
    if os.path.isdir(os.path.join(filename, d))
])

    text.insert(END, "Subfolders found:\n")
    for label in categories:
        text.insert(END, f"- {label}\n")

# ===================================================
# ============ CREATE DATASET =======================
# ===================================================

# ===================================================
# ============ CREATE DATASET WITH AUGMENTATION =====
# ===================================================

def Image_Preprocessing():
    clearText()
    global X, Y

    if not filename:
        messagebox.showwarning("Warning", "Upload dataset first!")
        return

    IMG_SIZE = (64, 64)
    X = []
    Y = []

    text.insert(END, "\nCreating Dataset using HOG Feature Extraction with Flip & Rotation Augmentation...\n")

    for label in categories:
        label_path = os.path.join(filename, label)

        for img_name in os.listdir(label_path):
            img_path = os.path.join(label_path, img_name)

            img = cv2.imread(img_path)
            if img is None:
                continue

            # Resize and grayscale
            img = cv2.resize(img, IMG_SIZE)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # ===== ORIGINAL HOG =====
            hog_features = hog(
                gray,
                orientations=9,
                pixels_per_cell=(8,8),
                cells_per_block=(2,2),
                block_norm='L2-Hys'
            )
            X.append(hog_features)
            Y.append(categories.index(label))

            # ===== HORIZONTAL FLIP =====
            flip = cv2.flip(gray, 1)
            hog_flip = hog(
                flip,
                orientations=9,
                pixels_per_cell=(8,8),
                cells_per_block=(2,2),
                block_norm='L2-Hys'
            )
            X.append(hog_flip)
            Y.append(categories.index(label))

            # ===== ROTATION AUGMENTATION (±15°) =====
            for angle in [-15, 15]:
                M = cv2.getRotationMatrix2D((IMG_SIZE[0]//2, IMG_SIZE[1]//2), angle, 1)
                rotated = cv2.warpAffine(gray, M, IMG_SIZE)

                hog_rot = hog(
                    rotated,
                    orientations=9,
                    pixels_per_cell=(8,8),
                    cells_per_block=(2,2),
                    block_norm='L2-Hys'
                )
                X.append(hog_rot)
                Y.append(categories.index(label))

    X = np.array(X)
    Y = np.array(Y)

    os.makedirs(model_folder, exist_ok=True)
    np.save(os.path.join(model_folder, "X.txt.npy"), X)
    np.save(os.path.join(model_folder, "Y.txt.npy"), Y)

    text.insert(END, "HOG Dataset with Flip & Rotation Augmentation created successfully!\n")
    text.insert(END, f"X shape: {X.shape}\n")
    text.insert(END, f"Y shape: {Y.shape}\n")

# ================= TRAIN TEST SPLIT =================

def trainTestSplit():
    clearText()
    global X, Y, x_train, x_test, y_train, y_test

    if X is None or Y is None:
        messagebox.showwarning("Warning","Please run Data Preprocessing first!")
        return

    text.insert(END, "\nPerforming Train-Test Split...\n")
    text.update()

    x_train, x_test, y_train, y_test = train_test_split(
        X, Y,
        test_size=0.20,
        random_state=77,
        stratify=Y
    )

    # ===== FEATURE SCALING =====
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    # ===== PCA FEATURE REDUCTION =====
    pca = PCA(n_components=200)
    x_train = pca.fit_transform(x_train)
    x_test = pca.transform(x_test)

    # save for prediction
    joblib.dump(scaler, os.path.join(model_folder, "scaler.pkl"))
    joblib.dump(pca, os.path.join(model_folder, "pca.pkl"))

    text.insert(END, "Train-Test Split Completed!\n")
    text.insert(END, f"x_train shape: {x_train.shape}\n")
    text.insert(END, f"x_test shape: {x_test.shape}\n")
    text.insert(END, f"y_train shape: {y_train.shape}\n")
    text.insert(END, f"y_test shape: {y_test.shape}\n")
# ================= METRICS =================

def calculateMetrics(model_name, predict, testY, labels):
    # ===== OVERALL METRICS =====
    acc = accuracy_score(testY, predict) * 100
    prec = precision_score(testY, predict, average='macro', zero_division=0) * 100
    rec = recall_score(testY, predict, average='macro', zero_division=0) * 100
    f1 = f1_score(testY, predict, average='macro', zero_division=0) * 100

    text.insert(END, f"\n{model_name} RESULTS\n")
    text.insert(END, "-" * 45 + "\n")
    text.insert(END, f"Accuracy  : {acc:.2f} %\n")
    text.insert(END, f"Precision : {prec:.2f} %\n")
    text.insert(END, f"Recall    : {rec:.2f} %\n")
    text.insert(END, f"F1-Score  : {f1:.2f} %\n\n")

    # Store overall metrics
    metrics_overall.append({
        "Model": model_name,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1-Score": f1
    })

    # ===== CLASS-WISE METRICS =====
    unique_labels = np.unique(testY)

    report = classification_report(
        testY,
        predict,
        labels=unique_labels,
        target_names=[labels[i] for i in unique_labels],
        output_dict=True,
        zero_division=0
    )

    text.insert(END, "Class-wise Metrics:\n")

    for i in unique_labels:
        cls = labels[i]
        cls_prec = report[cls]['precision'] * 100
        cls_rec = report[cls]['recall'] * 100
        cls_f1 = report[cls]['f1-score'] * 100

        text.insert(
            END,
            f"{cls:25s}  "
            f"P: {cls_prec:6.2f}%  "
            f"R: {cls_rec:6.2f}%  "
            f"F1: {cls_f1:6.2f}%\n"
        )

        if cls not in class_metrics_storage:
            class_metrics_storage[cls] = []

        class_metrics_storage[cls].append({
            "Model": model_name,
            "Precision": cls_prec,
            "Recall": cls_rec,
            "F1-Score": cls_f1
        })

    # ===== CONFUSION MATRIX =====
    cm = confusion_matrix(testY, predict)

    plt.figure(figsize=(5, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=labels, yticklabels=labels
    )
    plt.title(f"{model_name} – Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.show()

# ================= MODELS =================

def trainLDA():
    clearText()

    if x_train is None:
        messagebox.showwarning("Warning", "Perform Train-Test Split first!")
        return

    model_path = os.path.join(model_folder, "lda_model.pkl")
    os.makedirs(model_folder, exist_ok=True)

    text.insert(END, "\nLinear Discriminant Analysis Classifier...\n")

    if os.path.exists(model_path):
        model = joblib.load(model_path)
        text.insert(END, "Model loaded.\n")
    else:
        model = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
        model.fit(x_train, y_train)
        joblib.dump(model, model_path)
        text.insert(END, "Model trained & saved.\n")

    y_pred = model.predict(x_test)
    calculateMetrics("Linear Discriminant Analysis", y_pred, y_test, categories)
    
def trainQDA():
    clearText()

    if x_train is None:
        messagebox.showwarning("Warning", "Perform Train-Test Split first!")
        return

    model_path = os.path.join(model_folder, "qda_model.pkl")
    os.makedirs(model_folder, exist_ok=True)

    text.insert(END, "\nQuadratic Discriminant Analysis Classifier...\n")

    if os.path.exists(model_path):
        model = joblib.load(model_path)
        text.insert(END, "Model loaded.\n")
    else:
        model = QuadraticDiscriminantAnalysis(reg_param=0.5)
        model.fit(x_train, y_train)
        joblib.dump(model, model_path)
        text.insert(END, "Model trained & saved.\n")

    y_pred = model.predict(x_test)
    calculateMetrics("Quadratic Discriminant Analysis", y_pred, y_test, categories)

def trainAdaBoost():
    clearText()

    if x_train is None:
        messagebox.showwarning("Warning", "Perform Train-Test Split first!")
        return

    model_path = os.path.join(model_folder, "adaboost_model.pkl")
    os.makedirs(model_folder, exist_ok=True)

    text.insert(END, "\nAdaBoost Classifier...\n")

    if os.path.exists(model_path):
        model = joblib.load(model_path)
        text.insert(END, "Model loaded.\n")
    else:
        model = AdaBoostClassifier(
            n_estimators=200,
            learning_rate=0.8,
            random_state=77
        )
        model.fit(x_train, y_train)
        joblib.dump(model, model_path)
        text.insert(END, "Model trained & saved.\n")

    y_pred = model.predict(x_test)

    calculateMetrics(
        "AdaBoost Classifier",
        y_pred,
        y_test,
        categories
    )    

def trainBRFC():
    clearText()

    if x_train is None:
        messagebox.showwarning("Warning", "Perform Train-Test Split first!")
        return

    model_path = os.path.join(model_folder, "brfc_model.pkl")
    os.makedirs(model_folder, exist_ok=True)

    text.insert(END, "\nBalanced Random Forest Classifier...\n")

    if os.path.exists(model_path):
        model = joblib.load(model_path)
        text.insert(END, "Model loaded.\n")
    else:
        model = BalancedRandomForestClassifier(
          n_estimators=400,
          max_depth=20,
          min_samples_split=4,
          random_state=77,
          n_jobs=-1
     )
        model.fit(x_train, y_train)
        joblib.dump(model, model_path)
        text.insert(END, "Model trained & saved.\n")

    y_pred = model.predict(x_test)

    calculateMetrics(
        "Balanced Random Forest",
        y_pred,
        y_test,
        categories
    )

# ================= MODEL COMPARISON =================

def modelComparison():
    clearText()
    if not metrics_overall or not class_metrics_storage:
        messagebox.showwarning("Warning", "Run models first!")
        return

    overall_df = pd.DataFrame(metrics_overall)
    overall_df.rename(columns={
        "Model": "Algorithm",
        "Accuracy": "Accuracy (%)",
        "Precision": "Precision (%)",
        "Recall": "Recall (%)",
        "F1-Score": "F1 Score (%)"
    }, inplace=True)

    x = np.arange(len(overall_df['Algorithm']))
    width = 0.2

    plt.figure(figsize=(10, 6))

    bars1 = plt.bar(x - 1.5*width, overall_df['Accuracy (%)'], width, label='Accuracy')
    bars2 = plt.bar(x - 0.5*width, overall_df['Precision (%)'], width, label='Precision')
    bars3 = plt.bar(x + 0.5*width, overall_df['Recall (%)'], width, label='Recall')
    bars4 = plt.bar(x + 1.5*width, overall_df['F1 Score (%)'], width, label='F1 Score')

    plt.xticks(x, overall_df['Algorithm'], rotation=45)
    plt.xlabel("Algorithm")
    plt.ylabel("Score (%)")
    plt.title("Overall Performance Comparison of Algorithms")
    plt.legend()

    plt.tight_layout()
    plt.show()

    text.insert(END, "Comparison Graph created successfully!\n")
# ================= PREDICTION =================

def predictImage():
    clearText()
    global categories

    model_path = os.path.join(model_folder, "brfc_model.pkl")

    if not os.path.exists(model_path):
        messagebox.showwarning("Warning", "Train Balanced Random Forest first!")
        return

    # load trained model
    model = joblib.load(model_path)

    # reload categories safely
    if len(categories) == 0 and filename != "":
        categories = sorted([
            d for d in os.listdir(filename)
            if os.path.isdir(os.path.join(filename, d))
        ])

    path = filedialog.askopenfilename(
        title="Select Image",
        filetypes=[("Image Files", "*.jpg *.jpeg *.png")]
    )

    if not path:
        return

    # ===== IMAGE PREPROCESS =====
    img = cv2.imread(path)
    img = cv2.resize(img, (64, 64))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    hog_features = hog(
        gray,
        orientations=9,
        pixels_per_cell=(8,8),
        cells_per_block=(2,2),
        block_norm='L2-Hys'
    )

    img = hog_features.reshape(1, -1)

    # ===== LOAD SCALER + PCA =====
    scaler_path = os.path.join(model_folder, "scaler.pkl")
    pca_path = os.path.join(model_folder, "pca.pkl")

    if os.path.exists(scaler_path) and os.path.exists(pca_path):
        scaler = joblib.load(scaler_path)
        pca = joblib.load(pca_path)

        img = scaler.transform(img)
        img = pca.transform(img)
    else:
        messagebox.showwarning("Warning", "Run Train-Test Split again!")
        return

    # ===== PREDICTION =====
    pred = model.predict(img)[0]

    if pred < len(categories):
        label = categories[pred]
    else:
        label = "Unknown"

    text.insert(END, f"\nPrediction Result: {label}")

    # ===== SHOW IMAGE =====
    img_show = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

    plt.imshow(img_show)
    plt.title(f"Predicted: {label}")
    plt.axis('off')
    plt.show()
# ================= UI =================

main = Tk()
main.title("Agricultural Crops Recognition from Field Images for Farm Management")

# Get screen size
screen_width = main.winfo_screenwidth()
screen_height = main.winfo_screenheight()

# Set window to full screen size
main.geometry(f"{screen_width}x{screen_height}")

# Load and resize background image to screen size
# Create automatic gradient background (no image file needed)

# ===== IMAGE BACKGROUND =====
bg = Image.open(r"C:\Users\SB\Desktop\Mini project\Image Processing Froentend\bg.jpg")
bg = bg.resize((screen_width, screen_height))

bg_photo = ImageTk.PhotoImage(bg)

bg_label = Label(main, image=bg_photo)
bg_label.image = bg_photo
bg_label.place(x=0, y=0, relwidth=1, relheight=1)

bg_label.lower()

font = ('times', 15, 'bold')
font1 = ('times', 13, 'bold')
ff = ('times', 12, 'bold')

Label(main, text="Agricultural Crops Recognition from Field Images for Farm Management",
      bg="gold2", fg="black",
      font=font, height=3, width=120).place(x=0, y=5)

Button(main, text="Dataset", command=uploadDataset, font=ff,
       bg="#1ABC9C", fg="black", activebackground="#16A085").place(x=20, y=150)

Button(main, text="Image Preprocessing", command=Image_Preprocessing, font=ff,
       bg="#3498DB", fg="white", activebackground="#2E86C1").place(x=20, y=200)

Button(main, text="Train-Test Split", command=trainTestSplit, font=ff,
       bg="#F1C40F", fg="black", activebackground="#D4AC0D").place(x=20, y=250)

Button(main, text="Train LDA", command=trainLDA, font=ff,
       bg="#00B894", fg="black", activebackground="#00997A").place(x=20, y=300)

Button(main, text="Train QDA", command=trainQDA, font=ff,
       bg="#FF7675", fg="white", activebackground="#D63031").place(x=20, y=350)

Button(main, text="Train AdaBoost", command=trainAdaBoost, font=ff,
       bg="#E17055", fg="white", activebackground="#D35400").place(x=20, y=400)

Button(main, text="Train Balanced Random Forest", command=trainBRFC, font=ff,
       bg="#9B59B6", fg="white", activebackground="#8E44AD").place(x=20, y=450)

Button(main, text="Model Comparison", command=modelComparison, font=ff,
       bg="#2ECC71", fg="black", activebackground="#27AE60").place(x=20, y=500)

Button(main, text="Predict Image", command=predictImage, font=ff,
       bg="#34495E", fg="white", activebackground="#2C3E50").place(x=20, y=550)


text = Text(main, height=20, width=70, font=font1, bg="white")
scroll = Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=330, y=100)
main.protocol("WM_DELETE_WINDOW", main.destroy)
main.mainloop()
