import math
from sklearn.neural_network import MLPClassifier
import os
import os.path
import pickle
from PIL import Image, ImageDraw
import face_recognition
from face_recognition.face_recognition_cli import image_files_in_folder


def train(train_dir, model_save_path=None, verbose=False):
    X = []
    y = []

    # Loop through each person in the training set
    for class_dir in os.listdir(train_dir):
        print(class_dir)
        if not os.path.isdir(os.path.join(train_dir, class_dir)):
            continue
        print(os.path.join(train_dir, class_dir))

        # Loop through each training image for the current person
        for img_path in image_files_in_folder(os.path.join(train_dir, class_dir)):
            print(img_path)
            image = face_recognition.load_image_file(img_path)
            face_bounding_boxes = face_recognition.face_locations(image)

            if len(face_bounding_boxes) != 1:
                # If there are no people (or too many people) in a training image, skip the image.
                if verbose:
                    print("Image {} not suitable for training: {}".format(img_path, "Didn't find a face" if len(
                        face_bounding_boxes) < 1 else "Found more than one face"))
            else:
                # Add face encoding for current image to the training set
                X.append(face_recognition.face_encodings(
                    image, known_face_locations=face_bounding_boxes)[0])
                y.append(class_dir)
                print(class_dir + " Added to database")

    # Create and train the MLP classifier
    MLP_clf = MLPClassifier(max_iter=500)
    MLP_clf.fit(X, y)

    # Save the trained KNN classifier
    if model_save_path is not None:
        with open(model_save_path, 'wb') as f:
            pickle.dump(MLP_clf, f)

    return MLP_clf


if __name__ == "__main__":
    # STEP 1: Train the KNN classifier and save it to disk
    # Once the model is trained and saved, you can skip this step next time.
    print("Training MLP classifier...")
    classifier = train("TrainImages",
                       model_save_path="trained_MLP_model.clf", verbose=True)
    print("Training complete!")
