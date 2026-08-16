import numpy as np
import tensorflow as tf

from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    classification_report,
    confusion_matrix
)


class LSTMClassifier:

    def __init__(
        self,
        input_shape,
        labels,
        learning_rate=0.001
    ):

        self.input_shape = input_shape
        self.labels = labels

        # -------------------------------------------------
        # Encode string labels -> integers
        # -------------------------------------------------

        self.label_encoder = LabelEncoder()

        self.label_encoder.fit(labels)

        self.num_classes = len(
            self.label_encoder.classes_
        )

        # -------------------------------------------------
        # Build model
        # -------------------------------------------------

        self.model = self._build_model(
            learning_rate
        )

    # =====================================================
    # BUILD MODEL
    # =====================================================

    def _build_model(
        self,
        learning_rate
    ):

        model = tf.keras.Sequential([

            tf.keras.layers.Input(
                shape=self.input_shape
            ),

            tf.keras.layers.LSTM(
                64,
                return_sequences=False
            ),

            tf.keras.layers.Dropout(
                0.3
            ),

            tf.keras.layers.Dense(
                32,
                activation="relu"
            ),

            tf.keras.layers.Dropout(
                0.2
            ),

            tf.keras.layers.Dense(
                self.num_classes,
                activation="softmax"
            )
        ])

        model.compile(

            optimizer=tf.keras.optimizers.Adam(
                learning_rate=learning_rate
            ),

            loss="sparse_categorical_crossentropy",

            metrics=[
                "accuracy"
            ]
        )

        return model

    # =====================================================
    # ENCODE LABELS
    # =====================================================

    def encode_labels(
        self,
        y_train,
        y_test
    ):

        y_train_encoded = (
            self.label_encoder
            .transform(y_train)
        )

        y_test_encoded = (
            self.label_encoder
            .transform(y_test)
        )

        return (
            y_train_encoded,
            y_test_encoded
        )

    # =====================================================
    # CLASS WEIGHTS
    # =====================================================

    def calculate_class_weights(
        self,
        y_train_encoded
    ):

        classes = np.unique(
            y_train_encoded
        )

        weights = compute_class_weight(
            class_weight="balanced",
            classes=classes,
            y=y_train_encoded
        )

        class_weights = {
            int(cls): float(weight)
            for cls, weight
            in zip(classes, weights)
        }

        return class_weights

    # =====================================================
    # TRAIN
    # =====================================================

    def train(
        self,
        X_train,
        y_train,
        class_weights,
        validation_split=0.15,
        epochs=30,
        batch_size=64
    ):

        early_stopping = (
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=5,
                restore_best_weights=True
            )
        )

        history = self.model.fit(

            X_train,

            y_train,

            validation_split=validation_split,

            epochs=epochs,

            batch_size=batch_size,

            class_weight=class_weights,

            callbacks=[
                early_stopping
            ],

            verbose=1
        )

        return history

    # =====================================================
    # PREDICT
    # =====================================================

    def predict(
        self,
        X
    ):

        probabilities = (
            self.model.predict(
                X,
                verbose=0
            )
        )

        predictions = np.argmax(
            probabilities,
            axis=1
        )

        return (
            predictions,
            probabilities
        )

    # =====================================================
    # EVALUATE
    # =====================================================

    def evaluate(
        self,
        X_test,
        y_test
    ):

        predictions, probabilities = (
            self.predict(X_test)
        )

        # -------------------------------------------------
        # Classification report
        # -------------------------------------------------

        print("\n")
        print("=" * 80)
        print("CLASSIFICATION REPORT")
        print("=" * 80)

        print(
            classification_report(
                y_test,
                predictions,
                labels=np.arange(
                    self.num_classes
                ),
                target_names=(
                    self.label_encoder
                    .classes_
                ),
                zero_division=0
            )
        )

        # -------------------------------------------------
        # Confusion matrix
        # -------------------------------------------------

        print("=" * 80)
        print("CONFUSION MATRIX")
        print("=" * 80)

        cm = confusion_matrix(
            y_test,
            predictions,
            labels=np.arange(
                self.num_classes
            )
        )

        print("\nLabels:")

        for i, label in enumerate(
            self.label_encoder.classes_
        ):

            print(
                i,
                "->",
                label
            )

        print("\nMatrix:")
        print(cm)

        return (
            predictions,
            probabilities,
            cm
        )

    # =====================================================
    # SUMMARY
    # =====================================================

    def summary(self):

        print("\n")
        print("=" * 80)
        print("LSTM MODEL")
        print("=" * 80)

        self.model.summary()

        print("\nClasses:")

        for i, label in enumerate(
            self.label_encoder.classes_
        ):

            print(
                i,
                "->",
                label
            )