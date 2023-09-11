from tensorflow.keras.utils import to_categorical
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import LabelEncoder
import numpy as np

def prepare_labels_and_weights(labels, label_encoder=None):
    # Prepare labels for the detector and classifier
    detector_labels = ["noise" if label == "noise" else "not_noise" for label in labels]
    classifier_labels = ["earthquake" if label == "earthquake" else "explosion" for label in labels]

    # Create or use existing label encoders for the detector and classifier
    if label_encoder is None:
        label_encoder = {'detector': LabelEncoder(), 'classifier': LabelEncoder()}

    # Encode and one-hot the labels
    detector_labels_encoded = label_encoder['detector'].fit_transform(detector_labels)
    classifier_labels_encoded = label_encoder['classifier'].fit_transform([label for label in classifier_labels if label != "noise"])
    classifier_labels_encoded_full = label_encoder['classifier'].transform(classifier_labels)

    detector_labels_onehot = to_categorical(detector_labels_encoded)
    classifier_labels_onehot = to_categorical(classifier_labels_encoded_full)

    # Calculate class weights for detector and classifier
    detector_class_weights = compute_class_weight('balanced', classes=np.unique(detector_labels_encoded), y=detector_labels_encoded)
    classifier_class_weights = compute_class_weight('balanced', classes=np.unique(classifier_labels_encoded), y=classifier_labels_encoded)

    # Create nested dictionaries keyed by indexes, containing one-hot encoded labels for both tasks
    nested_label_dict = {}
    for i, (det_label, cls_label) in enumerate(zip(detector_labels_onehot, classifier_labels_onehot)):
        nested_label_dict[i] = {
            'detector': det_label,
            'classifier': cls_label
        }

    # Create a dictionary to map encoded labels back to original labels for detector and classifier
    detector_label_map = {index: label for index, label in enumerate(label_encoder['detector'].classes_)}
    classifier_label_map = {index: label for index, label in enumerate(label_encoder['classifier'].classes_)}

    return nested_label_dict, detector_class_weights, classifier_class_weights, label_encoder, detector_label_map, classifier_label_map
def swap_labels(labels):
    output = []
    for label in labels:
        if label == 'induced or triggered event':
            output.append('earthquake')
        else:
            output.append(label)
    return output

