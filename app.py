import json
import os
import numpy as np
import cv2
import pandas as pd
from flask import (
    Flask,
    Response,
    redirect,
    render_template,
    request,
    url_for,
)
from werkzeug.utils import secure_filename
from forms import CorrectForm, LabelForm, PhotoForm
import torch
from torchvision import transforms
from flask import Flask, Response, redirect, render_template, request, url_for
from werkzeug.utils import secure_filename
from forms import LabelForm, PhotoForm
from models import model_last, cnn_based_ml
from train_SVM import train_svm
import pickle


# Create Flask App
app = Flask(__name__)
app.config["SECRET_KEY"] = "sucess2G15"
app.config["UPLOAD_FOLDER"] = "static/files"

# ------------ Route to home page --------------------
@app.route("/", methods=["GET", "POST"])
def home():
    return render_template("home.html")


# ------------ Route to labeling page --------------------
# Upload images route
@app.route("/labeling", methods=["GET", "POST"])
def upload_images():
    # Load form class instance from forms.py
    form1 = PhotoForm()
    # Validate image files after submit
    if form1.validate_on_submit():
        uploaded_files = form1.photos.data

        # Error handling for too many images
        if len(uploaded_files) > 50:
            error = "You are only allowed to upload a maximum of 50 files!"
            return render_template("labeling.html", form1=form1, error=error)
        else:
            for file in form1.photos.data:
                filename = secure_filename(file.filename)
                # Save image into static folder
                file.save(os.path.join(app.config["UPLOAD_FOLDER"], filename))
            # Get image names to plot grid
            uploaded_images = os.listdir(app.config["UPLOAD_FOLDER"])
            return render_template("labeling.html", uploaded_images=uploaded_images)
            # redirect(url_for("view_label_images"))

    # Remove all files if "Labeling" page is clicked
    filelist = [f for f in os.listdir(app.config["UPLOAD_FOLDER"])]
    for f in filelist:
        os.remove(os.path.join(app.config["UPLOAD_FOLDER"], f))
    return render_template("labeling.html", form1=form1)


# View and label images route
@app.route("/labeling/start", methods=["GET", "POST"])
def view_label_images():
    # Load form class instance from forms.py
    form2 = LabelForm(request.form)
    labels = {"annotations": []}
    # Get the annotations list
    exports = labels["annotations"]
    # Get file paths list
    image_names = os.listdir(app.config["UPLOAD_FOLDER"])
    # Check if the export button is clicked
    if request.method == "POST":
        for i, curr_file_name in enumerate(image_names):
            # Get data from the label form
            label = form2.labels[i].data

            # Append export of each labeled image
            export = {
                "file_name": curr_file_name,
                "damage": label,
            }
            exports.append(export)
            # Clean labeled image from static folder
            os.remove(os.path.join(app.config["UPLOAD_FOLDER"], curr_file_name))
        # Download json file after all labels are filled
        return Response(
            json.dumps(labels),
            mimetype="application/json",
            headers={"Content-Disposition": "attachment;filename=labels.json"},
        )

    return render_template(
        "labeling.html",
        form2=form2,
        image_names=image_names,
    )


# ------------ Route to prediction page --------------------
# Upload images route
@app.route("/prediction", methods=["GET", "POST"])
def upload_image_pred():
    # Load form class instance from forms.py
    form1 = PhotoForm()
    # Validate image files after submit
    if form1.validate_on_submit():
        # Get uploaded files
        uploaded_files = form1.photos.data
        # Error handling for too many images
        if len(uploaded_files) > 50:
            error = "You are only allowed to upload a maximum of 50 files!"
            return render_template("prediction.html", form1=form1, error=error)
        else:
            # Save all images into static folder
            for file in form1.photos.data:
                filename = secure_filename(file.filename)
                file.save(os.path.join(app.config["UPLOAD_FOLDER"], filename))
            # Get file paths list
            grid_images = os.listdir(app.config["UPLOAD_FOLDER"])
            return render_template("prediction.html", grid_images=grid_images)

    # Remove all files if "Prediction" page is clicked
    filelist = [f for f in os.listdir(app.config["UPLOAD_FOLDER"])]
    for f in filelist:
        os.remove(os.path.join(app.config["UPLOAD_FOLDER"], f))
    corrlist = [c for c in os.listdir("static/corrections/")]
    for c in corrlist:
        os.remove(os.path.join("static/corrections/", c))
    return render_template("prediction.html", form1=form1)


# View predictions route
@app.route("/prediction/show", methods=["GET", "POST"])
def image_predictions_resnet_based_ml():
    pred_images = os.listdir(app.config["UPLOAD_FOLDER"])

    # Load new model if it exists
    if os.path.exists("model_svm_new.pkl"):
        with open("model_svm_new.pkl", "rb") as f:
            cnn_based_ml = pickle.load(f)
    else:
        # Load the ML model
        with open("model_svm.pkl", "rb") as f:
            cnn_based_ml = pickle.load(f)

    # Read images and convert to numpy
    predictions = []
    for name in pred_images:
        # Read image and convert to numpy
        img = cv2.imread(app.config["UPLOAD_FOLDER"] + "/" + name)

        # Preprocessing pipeline
        mean = [0.3518, 0.3335, 0.3398]
        std = [0.2715, 0.2676, 0.2685]
        preprocess = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((224, 224)),
                transforms.Normalize(mean, std),
            ]
        )

        # Define process pipeline from RGB image to gray_scale 256*1 features(distribution of gray intensity)
        gray = cv2.cvtColor(
            preprocess(img).detach().numpy().reshape(224, 224, 3), cv2.COLOR_BGR2GRAY
        )
        # Reshape tensor to include batch size
        torch_image = torch.reshape(preprocess(img), (1, 3, 224, 224))

        # Get features from feature extractors
        with torch.no_grad():
            model_last.eval()

            # Get features from feature extractors
            features_resnet = model_last(torch_image)
            features_resnet = features_resnet.detach().numpy()
            features_resnet_temp = features_resnet.flatten().tolist()

            # Get features from grayscale img
            img_after_preprocess = preprocess(img).detach().numpy()
            gray = cv2.cvtColor(
                img_after_preprocess.reshape(224, 224, 3), cv2.COLOR_BGR2GRAY
            )
            hist = cv2.calcHist([gray], [0], None, [256], [0, 1])
            hist_temp = (hist / 255).flatten()
            features_merged = list(np.array(hist_temp)) + list(
                np.array(features_resnet_temp)
            )
            predicted_output = cnn_based_ml.predict(
                np.array(features_merged).reshape(1, -1)
            )

            # Convert the indexes to label names
            if predicted_output == 0:
                prediction = "dent"
            elif predicted_output == 1:
                prediction = "scratch"
            elif predicted_output == 2:
                prediction = "rim"
            elif predicted_output == 3:
                prediction = "other"

            # Add to prediction list
            predictions.append(prediction)

    # Get all image names from checkboxes
    if request.method == "POST":
        # Get list of names
        corr_images = request.form.getlist("corr_checkbox")

        # Load features from csv
        features_merged_train_numpy = np.loadtxt(
            "features_merged_train.csv", delimiter=","
        ).reshape(-1, 768)

        for file_name in corr_images:
            img = cv2.imread(app.config["UPLOAD_FOLDER"] + "/" + file_name)
            # Save correction image into static folder
            dest = "static/corrections/" + file_name
            cv2.imwrite(dest, img)

            # Run feature extractor for correction images
            gray = cv2.cvtColor(
                preprocess(img).detach().numpy().reshape(224, 224, 3),
                cv2.COLOR_BGR2GRAY,
            )
            # Reshape tensor to include batch size
            torch_image = torch.reshape(preprocess(img), (1, 3, 224, 224))

            # Get features from feature extractors
            with torch.no_grad():
                model_last.eval()
                # Generate features from resnet
                features_resnet = model_last(torch_image)
                features_resnet = features_resnet.detach().numpy()
                features_resnet_temp = features_resnet.flatten().tolist()
                # Get features from grayscale img
                img_after_preprocess = preprocess(img).detach().numpy()
                gray = cv2.cvtColor(
                    img_after_preprocess.reshape(224, 224, 3), cv2.COLOR_BGR2GRAY
                )
                hist = cv2.calcHist([gray], [0], None, [256], [0, 1])
                hist_temp = (hist / 255).flatten()
                features_merged = list(np.array(hist_temp)) + list(
                    np.array(features_resnet_temp)
                )

                # Append new corrected csv features
                features_merged = np.array(features_merged).reshape(1, 768)
                features_merged_train_numpy = np.concatenate(
                    (features_merged_train_numpy, features_merged), axis=0
                )

        # Save features to new csv file
        features_merged_train_pd = pd.DataFrame(features_merged_train_numpy)
        features_merged_train_pd.to_csv(
            "features_merged_train_new.csv", index=False, header=False
        )

        return redirect(url_for("retrain"))

    return render_template(
        "prediction.html",
        pred_images=pred_images,
        predictions=predictions,
    )


# ------------ Correction and retrain  page ------------------
@app.route("/prediction/retrain", methods=["GET", "POST"])
def retrain():
    form3 = CorrectForm(request.form)
    corr_images = os.listdir("static/corrections")

    # Check if the retrain button is clicked
    if request.method == "POST":
        labels_train_corr = []
        for i, curr_file_name in enumerate(corr_images):
            # Get data from the label form
            label = form3.labels[i].data
            if label == "dent":
                labels_train_corr.append(0)
            elif label == "scratch":
                labels_train_corr.append(1)
            elif label == "rim":
                labels_train_corr.append(2)
            elif label == "other":
                labels_train_corr.append(3)
            trained = True
        labels_train_corr = np.array(labels_train_corr).reshape(-1, 1)

        # Append pictures and labels found in corrections folder
        labels_train_numpy = np.loadtxt("labels_train.csv", delimiter=",").reshape(
            -1, 1
        )
        labels_train_numpy = np.concatenate(
            (labels_train_numpy, labels_train_corr), axis=0
        )

        # Save labels to csv file
        labels_train_numpy = pd.DataFrame(labels_train_numpy)
        labels_train_numpy.to_csv("labels_train_new.csv", index=False, header=False)

        # Retrain function
        train_svm(
            train_csv="features_merged_train_new.csv", label_csv="labels_train_new.csv"
        )

        if trained:
            return render_template("correction.html", trained=trained)

    return render_template("correction.html", form3=form3, corr_images=corr_images)


@app.route("/prediction/show_retrained", methods=["GET", "POST"])
def show_retrained():
    pred_images = os.listdir(app.config["UPLOAD_FOLDER"])

    # Load new model
    with open("model_svm_new.pkl", "rb") as f:
        cnn_based_ml = pickle.load(f)

    # Read images and convert to numpy
    predictions = []
    for name in pred_images:
        # Read image and convert to numpy
        img = cv2.imread(app.config["UPLOAD_FOLDER"] + "/" + name)

        # Preprocessing pipeline
        mean = [0.3518, 0.3335, 0.3398]
        std = [0.2715, 0.2676, 0.2685]
        preprocess = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((224, 224)),
                transforms.Normalize(mean, std),
            ]
        )

        # Define process pipeline from RGB image to gray_scale 256*1 features(distribution of gray intensity)
        gray = cv2.cvtColor(
            preprocess(img).detach().numpy().reshape(224, 224, 3), cv2.COLOR_BGR2GRAY
        )
        # Reshape tensor to include batch size
        torch_image = torch.reshape(preprocess(img), (1, 3, 224, 224))

        # Get features from feature extractors
        with torch.no_grad():
            model_last.eval()

            # Get features from feature extractors
            features_resnet = model_last(torch_image)
            features_resnet = features_resnet.detach().numpy()
            features_resnet_temp = features_resnet.flatten().tolist()

            # Get features from grayscale img
            img_after_preprocess = preprocess(img).detach().numpy()
            gray = cv2.cvtColor(
                img_after_preprocess.reshape(224, 224, 3), cv2.COLOR_BGR2GRAY
            )
            hist = cv2.calcHist([gray], [0], None, [256], [0, 1])
            hist_temp = (hist / 255).flatten()
            features_merged = list(np.array(hist_temp)) + list(
                np.array(features_resnet_temp)
            )
            predicted_output = cnn_based_ml.predict(
                np.array(features_merged).reshape(1, -1)
            )

            # Convert the indexes to label names
            if predicted_output == 0:
                prediction = "dent"
            elif predicted_output == 1:
                prediction = "scratch"
            elif predicted_output == 2:
                prediction = "rim"
            elif predicted_output == 3:
                prediction = "other"

            # Add to prediction list
            predictions.append(prediction)

    return render_template(
        "prediction.html",
        pred_images=pred_images,
        predictions=predictions,
    )


# ------------ Run flask app ------------------
@app.route("/about", methods=["GET", "POST"])
def about():
    return render_template("about.html")


@app.route("/team", methods=["GET", "POST"])
def team():
    return render_template("team.html")


# ------------ Run flask app ------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True, port=8888)
