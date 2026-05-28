"""
Task 2A - Face Authentication: Training / Model Preparation
============================================================
We use DeepFace with the pretrained Facenet model.
This file handles:
  1. Downloading / verifying the pretrained Facenet weights
  2. Running a quick self-test to confirm the model loads correctly
  3. (Optional) Fine-tuning on your own dataset with triplet loss

For the assignment scope, the "training" is the weight download
and verification of the embedding pipeline.
"""

import os
import numpy as np
import cv2
from deepface import DeepFace
from deepface.commons import functions


# ─────────────────────────────────────────────
# 1.  Download / verify pretrained Facenet weights
# ─────────────────────────────────────────────

def download_and_verify_model(model_name: str = "Facenet") -> None:
    """
    Forces DeepFace to download the selected pretrained model weights
    and verifies that the model can be built successfully.
    """
    print(f"[*] Loading / downloading pretrained model: {model_name}")
    try:
        model = DeepFace.build_model(model_name)
        print(f"[✓] Model '{model_name}' loaded successfully.")
        print(f"    Input shape  : {model.input_shape}")
        print(f"    Output shape : {model.output_shape}")
    except Exception as e:
        print(f"[✗] Failed to load model: {e}")
        raise


# ─────────────────────────────────────────────
# 2.  Self-test: generate a dummy embedding
# ─────────────────────────────────────────────

def self_test_embedding(model_name: str = "Facenet") -> None:
    """
    Create a synthetic face image, pass it through DeepFace,
    and print the embedding shape to confirm the pipeline works.
    """
    print("\n[*] Running self-test with a synthetic image …")

    # Create a blank 160×160 RGB image (Facenet input size)
    dummy_image = np.zeros((160, 160, 3), dtype=np.uint8)
    dummy_image[40:120, 40:120] = [200, 180, 160]   # rough skin-tone square

    try:
        result = DeepFace.represent(
            img_path=dummy_image,
            model_name=model_name,
            enforce_detection=False,
            detector_backend="opencv",
        )
        embedding = np.array(result[0]["embedding"])
        print(f"[✓] Embedding shape: {embedding.shape}")
        print(f"    Sample values : {embedding[:5].round(4)}")
    except Exception as e:
        print(f"[✗] Self-test failed: {e}")
        raise


# ─────────────────────────────────────────────
# 3.  Optional: fine-tune on your own dataset
# ─────────────────────────────────────────────
#
# If you have labelled face pairs you can fine-tune using
# a triplet-loss or contrastive-loss approach.
# Structure your dataset as:
#
#   dataset/
#     person_1/  img1.jpg  img2.jpg …
#     person_2/  img1.jpg  img2.jpg …
#     …
#
# Then use the snippet below (requires TensorFlow / Keras).

def fine_tune_example_stub():
    """
    Stub showing how you would fine-tune Facenet on a custom dataset.
    Not executed by default — requires your own labelled image dataset.
    """
    # from tensorflow.keras.optimizers import Adam
    # from tensorflow.keras.losses import TripletSemiHardLoss
    #
    # model = DeepFace.build_model("Facenet")
    # model.compile(
    #     optimizer=Adam(learning_rate=1e-4),
    #     loss=TripletSemiHardLoss(),
    # )
    # # Load your (anchor, positive, negative) triplets here
    # # model.fit(triplet_dataset, epochs=10)
    # model.save("facenet_finetuned.h5")
    print("[i] Fine-tuning stub — replace with real dataset to train.")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

if __name__ == "__main__":
    MODEL_NAME = "Facenet"

    download_and_verify_model(MODEL_NAME)
    self_test_embedding(MODEL_NAME)

    print("\n[✓] Training / model preparation complete.")
    print("    The pretrained weights are cached in ~/.deepface/weights/")
    print("    Run 'python test_predict.py' to test on real images.")
