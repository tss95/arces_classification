from global_config import cfg, logger
import matplotlib.pyplot as plt

from Classes.LoadData import LoadData
from Classes.Scaler import Scaler
from datetime import datetime


from tensorflow.keras.utils import to_categorical
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import LabelEncoder
import numpy as np
import tensorflow as tf

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


def get_final_labels(pred_probs, label_map):
    """
    Given the predicted probabilities from the model, return the final label.
    
    Parameters:
    pred_probs (dict): A dictionary containing the predicted probabilities for both 'detector' and 'classifier'.
    label_map (dict): A dictionary containing mappings from label indices to their string representations.
    
    Returns:
    list: A list of final labels.
    """
    final_labels = []
    pred_probs_detector = tf.sigmoid(tf.cast(pred_probs['detector'], tf.float32)).numpy()       
    pred_labels_detector = apply_threshold(pred_probs_detector)

    pred_probs_classifier = tf.sigmoid(tf.cast(pred_probs['classifier'], tf.float32)).numpy()
    pred_labels_classifier = apply_threshold(pred_probs_classifier)

    final_labels = translate_labels(pred_labels_detector, pred_labels_classifier, label_map)
    return final_labels, {"detector": pred_probs_detector, "classifier": pred_probs_classifier}
        

def translate_labels(labels_detector, labels_classifier, label_map):
    final_labels = []
    for det, cls in zip(labels_detector, labels_classifier):
        det = int(det[0]) if isinstance(det, np.ndarray) else det
        cls = int(cls[0]) if isinstance(cls, np.ndarray) else cls


        if label_map["detector"][det] == "noise":
            final_labels.append("noise")
        else:
            final_labels.append(label_map["classifier"][cls])
    
    return final_labels

def apply_threshold(pred_probs):
    out = []
    for prob in pred_probs:
        if prob <= cfg.data.model_threshold:
            out.append(0)
        else:
            out.append(1)
    return out

def get_y_and_ypred(model, val_gen, label_maps):
    final_pred_labels = []
    final_true_labels = []
    final_pred_probs = []
    
    for batch_data, batch_labels in val_gen:
        pred_probs = model.predict(batch_data, verbose=0)
        labels, pred_probs = get_final_labels(pred_probs, label_maps)
        final_pred_labels.extend(labels)
        final_pred_probs.extend(pred_probs)
        final_true_labels.extend(translate_labels(batch_labels["detector"].numpy().astype(int), 
                                                        batch_labels["classifier"].numpy().astype(int), label_maps))
    return final_true_labels, final_pred_labels, final_pred_probs


def plot_confusion_matrix(conf_matrix, conf_matrix_normalized, class_names):
    # Create a custom confusion matrix plot
    plt.figure(figsize=(8, 10))
    plt.imshow(conf_matrix_normalized, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Normalized Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    
    # Add text annotations
    for i in range(conf_matrix_normalized.shape[0]):
        for j in range(conf_matrix_normalized.shape[1]):
            plt.text(j, i, f"{conf_matrix_normalized[i, j]:.2f}\n({conf_matrix[i, j]})",
                     horizontalalignment="center", verticalalignment="center",
                     color="white" if conf_matrix_normalized[i, j] > 0.5 else "black")
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return plt



def prep_data():
    now = datetime.now()
    date_and_time = now.strftime("%Y%m%d_%H%M%S")

    loadData = LoadData()
    train_dataset = loadData.get_train_dataset()
    val_dataset = loadData.get_val_dataset()

    train_data, train_labels, train_meta = train_dataset[0],train_dataset[1],train_dataset[2]
    logger.info("Train data shape: " + str(train_data.shape))
    train_data = np.transpose(train_data, (0,2,1))
    logger.info("Train data shape after transpose: " + str(train_data.shape))
    val_data, val_labels, val_meta = val_dataset[0],val_dataset[1],val_dataset[2]
    val_data = np.transpose(val_data, (0,2,1))



    train_labels = swap_labels(train_labels)
    val_labels = swap_labels(val_labels)

    print(np.unique(train_labels))
    print(np.unique(val_labels))

    if cfg.data.debug:
        def downsample_data_labels(data, labels, p=10):
            data = np.array(data)
            labels = np.array(labels)
            n = len(data)
            size = int((p / 100) * n)
            indices = np.random.choice(np.arange(n), size=size, replace=False)
            print(f"Type of data: {type(data)}, Type of labels: {type(labels)}, Type of indices: {type(indices)}")  # Debug line
            
            downsampled_data = data[indices]
            downsampled_labels = labels[indices]
            
            return downsampled_data, downsampled_labels

        train_data, train_labels = downsample_data_labels(train_data, train_labels, 5)
        val_data, val_labels = downsample_data_labels(val_data, val_labels, 5)


    # Ensure train_data and train_labels are NumPy arrays
    train_data = np.array(train_data)
    train_labels = np.array(train_labels)

    # Your existing code to get label counts before oversampling
    pre_oversample = len(train_labels)
    logger.info(f"Label distribution pre-oversample: {np.unique(train_labels, return_counts=True)}")


    # Find the indices of the earthquake samples
    earthquake_indices = np.where(train_labels == 'earthquake')[0]

    # Repeat the earthquake samples 3 times
    oversampled_earthquake_indices = np.repeat(earthquake_indices, 3)

    # Combine the original indices with the oversampled earthquake indices
    all_indices = np.concatenate([np.arange(len(train_labels)), oversampled_earthquake_indices])

    # Perform the oversampling in both data and labels
    train_data = train_data[all_indices]
    train_labels = train_labels[all_indices]

    # Your existing code to get label counts after oversampling
    post_oversample = len(train_labels)
    logger.info(f"Before oversampling: {pre_oversample}, after oversampling: {post_oversample}")
    logger.info(f"Label distribution post-oversample: {np.unique(train_labels, return_counts=True)}")
    logger.info(f"Label distribution validation: {np.unique(val_labels, return_counts=True)}")

    logger.info(f"Before scaling training shape is {train_data.shape}, with (min, max) ({np.min(train_data)}, {np.max(train_data)})")
    logger.info(f"Before scaling validation shape is {val_data.shape}, with (min, max) ({np.min(val_data)}, {np.max(val_data)})")

    scaler = Scaler()
    scaler.fit(train_data)
    train_data = scaler.transform(train_data)
    val_data = scaler.transform(val_data)

    logger.info(f"After scaling training shape is {train_data.shape}, with (min, max) ({np.min(train_data)}, {np.max(train_data)})")
    logger.info(f"After scaling validation shape is {val_data.shape}, with (min, max) ({np.min(val_data)}, {np.max(val_data)})")

    # Prepare labels for training and validation data
    nested_label_dict_train, detector_class_weights_train, classifier_class_weights_train, label_encoder_train, detector_label_map, classifier_label_map = prepare_labels_and_weights(train_labels)
    nested_label_dict_val, _, _, _, _, _ = prepare_labels_and_weights(val_labels, label_encoder=label_encoder_train)

    logger.info(f"Label encoder:{label_encoder_train}")

    # Create a translational dictionary for label strings based on training data
    detector_label_map = {index: label for index, label in enumerate(label_encoder_train['detector'].classes_)}
    classifier_label_map = {index: label for index, label in enumerate(label_encoder_train['classifier'].classes_)}

    # Convert class weights to dictionaries
    detector_class_weight_dict = {i: detector_class_weights_train[i] for i in range(len(detector_class_weights_train))}
    classifier_class_weight_dict = {i: classifier_class_weights_train[i] for i in range(len(classifier_class_weights_train))}

    # Combine the detector and classifier labels into single nested dictionaries
    # This should already be done by prepare_labels_and_weights as it returns nested_label_dict_train and nested_label_dict_val

    logger.info(f"Detector label map: {detector_label_map}")
    logger.info(f"Classifier label map: {classifier_label_map}")
    logger.info(f"Detector class weights: {detector_class_weight_dict}")
    logger.info(f"Classifier class weights: {classifier_class_weight_dict}")



    # Convert nested dictionaries to NumPy arrays
    detector_labels_array_train = np.argmax(np.array([nested_label_dict_train[i]['detector'] for i in range(len(nested_label_dict_train))]), axis=1)
    classifier_labels_array_train = np.argmax(np.array([nested_label_dict_train[i]['classifier'] for i in range(len(nested_label_dict_train))]), axis=1)

    detector_labels_array_val = np.argmax(np.array([nested_label_dict_val[i]['detector'] for i in range(len(nested_label_dict_val))]), axis=1)
    classifier_labels_array_val = np.argmax(np.array([nested_label_dict_val[i]['classifier'] for i in range(len(nested_label_dict_val))]), axis=1)



    train_labels_dict = {'detector': detector_labels_array_train[:, np.newaxis], 'classifier': classifier_labels_array_train[:, np.newaxis]}
    val_labels_dict = {'detector': detector_labels_array_val[:, np.newaxis], 'classifier': classifier_labels_array_val[:, np.newaxis]}

    logger.info(f"3 first classifier labels: {val_labels_dict['classifier'][:3]}")
    logger.info(f"3 first detector labels: {val_labels_dict['detector'][:3]}")
    # Log the class distribution for training and validation sets
    logger.info(f"Training Detector Class Distribution: {np.unique(train_labels_dict['detector'], return_counts=True)}")
    logger.info(f"Training Classifier Class Distribution: {np.unique(train_labels_dict['classifier'], return_counts=True)}")
    logger.info(f"Validation Detector Class Distribution: {np.unique(val_labels_dict['detector'], return_counts=True)}")
    logger.info(f"Validation Classifier Class Distribution: {np.unique(val_labels_dict['classifier'], return_counts=True)}")

    label_map = {
        'detector': detector_label_map,
        'classifier': classifier_label_map
    }
    return train_data, train_labels_dict, val_data, val_labels_dict, label_map, detector_class_weight_dict, classifier_class_weight_dict, classifier_label_map, detector_label_map, date_and_time, train_meta, val_meta
