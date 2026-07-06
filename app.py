# RNN Practical (One to One)

# Iris Flower Classification using Simple RNN

# Dataset : iris.csv

import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import SimpleRNN, Dense, Input
from tensorflow.keras.utils import to_categorical

MODEL = "one_to_one_rnn.keras"
ENCODER = "label_encoder.pkl"

# ------------------------------------
# Load Dataset
# ------------------------------------

df = pd.read_csv("iris.csv")

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# ------------------------------------
# Encode Labels
# ------------------------------------

encoder = LabelEncoder()
y = encoder.fit_transform(y)

with open(ENCODER, "wb") as f:
    pickle.dump(encoder, f)

# Reshape input for RNN
X = X.reshape((X.shape[0], 1, X.shape[1]))

# ------------------------------------
# Train Model
# ------------------------------------

def train_model():

    st.write("Training One-to-One RNN...")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Convert labels to one-hot AFTER splitting
    y_train = to_categorical(y_train, num_classes=3)
    y_test = to_categorical(y_test, num_classes=3)

    model = Sequential()

    model.add(
        Input(shape=(1,4))
    )

    model.add(
        SimpleRNN(32)
    )

    model.add(
        Dense(
            16,
            activation="relu"
        )
    )

    model.add(
        Dense(
            3,
            activation="softmax"
        )
    )

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    model.summary()

    model.fit(
        X_train,
        y_train,
        epochs=100,
        batch_size=8,
        validation_data=(X_test, y_test),
        verbose=1
    )

    model.save(MODEL)

    loss, accuracy = model.evaluate(
        X_test,
        y_test,
        verbose=0
    )

    st.success(f"Model Accuracy : {accuracy*100:.2f}%")

# ------------------------------------
# Prediction Function
# ------------------------------------

def predict_flower(features):

    model = load_model(MODEL)

    with open(ENCODER, "rb") as f:
        encoder = pickle.load(f)

    sample = np.array(features).reshape((1,1,4))

    prediction = model.predict(
        sample,
        verbose=0
    )

    index = np.argmax(prediction)

    flower = encoder.inverse_transform([index])[0]

    confidence = prediction[0][index]

    return flower, confidence

# ------------------------------------
# Train Only Once
# ------------------------------------

if not os.path.exists(MODEL):
    train_model()

# ------------------------------------
# Streamlit UI
# ------------------------------------

st.title("One-to-One RNN")

st.write("Iris Flower Classification using Simple RNN")

sepal_length = st.number_input(
    "Sepal Length",
    min_value=0.0,
    value=5.1
)

sepal_width = st.number_input(
    "Sepal Width",
    min_value=0.0,
    value=3.5
)

petal_length = st.number_input(
    "Petal Length",
    min_value=0.0,
    value=1.4
)

petal_width = st.number_input(
    "Petal Width",
    min_value=0.0,
    value=0.2
)

if st.button("Predict"):

    flower, confidence = predict_flower([
        sepal_length,
        sepal_width,
        petal_length,
        petal_width
    ])

    st.success(f"Prediction : {flower}")

    st.write(
        "Confidence :",
        round(confidence * 100, 2),
        "%"
    )