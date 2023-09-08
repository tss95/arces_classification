from tensorflow.keras.utils import to_categorical
from sklearn.utils.class_weight import compute_class_weight

def prepare_labels(labels, label_encoder=None):
    detector_labels = ["noise" if label == "noise" else "not_noise" for label in labels]
    classifier_labels = ["earthquake" if label == "earthquake" else "explosion" for label in labels if label != "noise"]
    
    # If no label_encoder is provided, create a new one
    if label_encoder is None:
        label_encoder = {'detector': LabelEncoder(), 'classifier': LabelEncoder()}
    
    # Encoding detector labels
    detector_labels_encoded = label_encoder['detector'].fit_transform(detector_labels)
    detector_labels_onehot = to_categorical(detector_labels_encoded)
    
    # Encoding classifier labels
    classifier_labels_encoded = label_encoder['classifier'].fit_transform(classifier_labels)
    classifier_labels_onehot = to_categorical(classifier_labels_encoded)
    
    return detector_labels_onehot, classifier_labels_onehot, label_encoder

def swap_labels(labels):
    output = []
    for label in labels:
        if label == 'induced or triggered event':
            output.append('earthquake')
        else:
            output.append(label)
    return output

