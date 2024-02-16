from global_config import cfg, logger
import matplotlib.pyplot as plt

from src.LoadData import LoadData
from src.Scaler_tf import Scaler
from datetime import datetime

import geopandas as gpd
import math
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import LabelEncoder
import numpy as np
import torch
from typing import Dict, List, Tuple, Any, Optional

def collate_fn_train(batch):
    # Separate inputs and targets
    inputs, targets = zip(*batch)

    # Stack inputs and targets
    inputs = torch.stack(inputs, 0)  # This results in a tensor of shape (batch_size, channel, timeseries)
    targets = torch.stack(targets, 0)

    # Apply augmentation pipeline
    inputs = augment.augment_pipeline(inputs, targets)

    return inputs, targets

def prepare_labels_and_weights(labels: List[str], label_encoder: Optional[Dict[str, LabelEncoder]] = None) -> Tuple[Dict[int, Dict[str, np.ndarray]], np.ndarray, np.ndarray, Dict[str, LabelEncoder], Dict[int, str], Dict[int, str]]:
    """
    Prepare labels and weights for a seismic data classification task.

    This function prepares the labels for both a detector and a classifier. It encodes and one-hot encodes the labels, computes class weights, and creates label maps.

    Args:
    labels: A list of string labels representing seismic events.
    label_encoder: An optional dictionary containing label encoders for the detector and classifier. If not provided, new encoders will be created.

    Returns:
    A tuple containing:
        - A dictionary mapping indices to dictionaries with one-hot encoded labels for both detector and classifier.
        - Numpy arrays of class weights for detector and classifier.
        - A dictionary containing the label encoder for both detector and classifier.
        - Two dictionaries mapping encoded labels back to original labels for detector and classifier.
    """
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

    detector_labels_onehot = torch.nn.functional.one_hot(torch.tensor(detector_labels_encoded)).numpy()
    classifier_labels_onehot = torch.nn.functional.one_hot(torch.tensor(classifier_labels_encoded_full)).numpy()

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


def swap_labels(labels: List[str]) -> List[str]:
    """
    Swap specific labels in the provided list.

    This function is designed to replace 'induced or triggered event' labels with 'earthquake'. It's used in the context of seismic data where such a substitution is relevant.

    Args:
    labels: A list of string labels.

    Returns:
    A list of string labels where 'induced or triggered event' has been replaced with 'earthquake'.
    """
    output = []
    for label in labels:
        if label == 'induced or triggered event':
            output.append('earthquake')
        else:
            output.append(label)
    return output


def get_final_labels(pred_probs: Dict[str, np.ndarray], label_map: Dict[str, Dict[int, str]]) -> Tuple[List[str], Dict[str, np.ndarray]]:
    """
    Determine the final labels based on the predicted probabilities and label map.

    This function applies a sigmoid function to the predicted probabilities, applies a threshold, and translates them into final labels using the provided label map.

    Args:
    pred_probs: A dictionary containing the predicted probabilities for both 'detector' and 'classifier'.
    label_map: A dictionary mapping label indices to their string representations for both 'detector' and 'classifier'.

    Returns:
    A tuple containing:
        - A list of final string labels.
        - A dictionary with updated predicted probabilities for both 'detector' and 'classifier'.
    """
    final_labels = []
    pred_probs_detector = torch.sigmoid(torch.tensor(pred_probs['detector'], dtype=torch.float32)).numpy()       
    pred_labels_detector = apply_threshold(pred_probs_detector)

    pred_probs_classifier = torch.sigmoid(torch.tensor(pred_probs['classifier'], dtype=torch.float32)).numpy()
    pred_labels_classifier = apply_threshold(pred_probs_classifier)

    final_labels = translate_labels(pred_labels_detector, pred_labels_classifier, label_map)
    return final_labels, {"detector": pred_probs_detector, "classifier": pred_probs_classifier}
        

def translate_labels(labels_detector: np.ndarray, labels_classifier: np.ndarray, label_map: Dict[str, Dict[int, str]]) -> List[str]:
    """
    Translate numeric labels to their string counterparts using a label map.

    This function translates the numeric labels from the detector and classifier into their corresponding string labels.

    Args:
    labels_detector: A numpy array of numeric labels from the detector.
    labels_classifier: A numpy array of numeric labels from the classifier.
    label_map: A dictionary mapping label indices to their string representations for both 'detector' and 'classifier'.

    Returns:
    A list of final string labels after translation.
    """
    final_labels = []
    for det, cls in zip(labels_detector, labels_classifier):
        det = int(det[0]) if isinstance(det, np.ndarray) else det
        cls = int(cls[0]) if isinstance(cls, np.ndarray) else cls


        if label_map["detector"][det] == "noise":
            final_labels.append("noise")
        else:
            final_labels.append(label_map["classifier"][cls])
    
    return final_labels

def apply_threshold(pred_probs: np.ndarray) -> List[int]:
    """
    Apply a threshold to predicted probabilities to classify them as 0 or 1.

    This function applies a threshold defined in the global configuration to the predicted probabilities.

    Args:
    pred_probs: A numpy array of predicted probabilities.

    Returns:
    A list of integers (0 or 1) after applying the threshold.
    """
    out = []
    for prob in pred_probs:
        if prob <= cfg.data.model_threshold:
            out.append(0)
        else:
            out.append(1)
    return out

def get_index_of_wrong_predictions(true_labels: List[str], pred_labels: List[str]) -> List[int]:
    """
    Identify indices where the true labels and predicted labels do not match.

    This function is useful for analyzing the performance of the model by identifying where it makes incorrect predictions.

    Args:
    true_labels: A list of true labels.
    pred_labels: A list of predicted labels.

    Returns:
    A list of indices where the true label does not match the predicted label.
    """

    wrong_indices = []
    for i, (true_label, pred_label) in enumerate(zip(true_labels, pred_labels)):
        if true_label != pred_label:
            wrong_indices.append(i)
    return wrong_indices

def get_y_and_ypred(model: Any, val_gen: Any, label_maps: Dict[str, Dict[int, str]]) -> Tuple[List[str], List[str], Dict[str, List[float]]]:
    """
    Generate true and predicted labels, along with predicted probabilities, from a validation generator.

    This function iterates over the validation generator, predicts labels and probabilities using the model, and translates these into final string labels.

    Args:
    model: The trained model used for prediction.
    val_gen: A generator that yields batches of data and labels for validation.
    label_maps: A dictionary containing label maps for translating numeric labels to string labels.

    Returns:
    A tuple containing:
        - A list of true string labels.
        - A list of predicted string labels.
        - A dictionary with lists of predicted probabilities for both 'detector' and 'classifier'.
    
    Note:
    Will work with any generator, not only validation generators.
    """
    final_pred_labels = []
    final_true_labels = []
    final_pred_probs = {"detector": [], "classifier": []}
    
    for batch_data, batch_labels in val_gen:
        pred_probs = model.predict(batch_data, verbose=0)
        labels, pred_probs = get_final_labels(pred_probs, label_maps)
        final_pred_labels.extend(labels)

        # Convert NumPy arrays to scalars and extend the list for 'detector'
        final_pred_probs["detector"].extend([x.item() for x in pred_probs["detector"]])

        # Convert NumPy arrays to scalars and extend the list for 'classifier'
        final_pred_probs["classifier"].extend([x.item() if x is not np.nan else np.nan for x in pred_probs["classifier"]])
                
        final_true_labels.extend(translate_labels(batch_labels["detector"].numpy().astype(int), 
                                                  batch_labels["classifier"].numpy().astype(int), label_maps))

    return final_true_labels, final_pred_labels, final_pred_probs


def one_prediction(model: Any, x: np.ndarray, label_maps: Dict[str, Dict[int, str]]) -> Tuple[List[str], Dict[str, np.ndarray]]:
    """
    Generate a prediction for a single data instance.

    This function is used to make a prediction on a single instance of data using the provided model. It processes the input, makes a prediction, and then translates it into a final label.

    Args:
    model: The trained model used for prediction.
    x: A numpy array representing a single instance of input data.
    label_maps: A dictionary containing label maps for translating numeric labels to string labels.

    Returns:
    A tuple containing:
        - A list with the final string label for the input data.
        - A dictionary with the predicted probabilities for both 'detector' and 'classifier'.
    """
    x = np.reshape(x, (1, *x.shape))
    pred_probs = model(x)
    labels, pred_probs = get_final_labels(pred_probs, label_maps)
    return labels, pred_probs


def plot_confusion_matrix(conf_matrix: np.ndarray, conf_matrix_normalized: np.ndarray, class_names: List[str]) -> plt.Figure:
    """
    Plot a confusion matrix.

    This function creates a visual representation of the confusion matrix, both in its raw and normalized forms.

    Args:
    conf_matrix: A numpy array representing the confusion matrix.
    conf_matrix_normalized: A numpy array representing the normalized confusion matrix.
    class_names: A list of class names corresponding to the confusion matrix.

    Returns:
    A matplotlib Figure object representing the plotted confusion matrix.
    """
    # Create a custom confusion matrix plot
    plt.figure(figsize=(10, 10))
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

def downsample_data_labels(data: np.ndarray, labels: np.ndarray, unswapped: Optional[np.ndarray] = None, p: int = 10) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Downsample the given data and labels.

    This function randomly selects a subset of the data and corresponding labels based on the specified percentage. 
    It is useful for creating a smaller, more manageable dataset for experimentation or debugging.

    Args:
    data: A numpy array of data to be downsampled.
    labels: A numpy array of labels corresponding to the data.
    unswapped: An optional numpy array of unswapped labels, used in some specific cases.
    p: The percentage of data to keep after downsampling (default is 10%).

    Returns:
    A tuple containing:
        - The downsampled data numpy array.
        - The downsampled labels numpy array.
        - Optionally, the downsampled unswapped labels numpy array, if provided.
    """
    data = np.array(data)
    labels = np.array(labels)
    n = len(data)
    size = int((p / 100) * n)
    indices = np.random.choice(np.arange(n), size=size, replace=False)
    print(f"Type of data: {type(data)}, Type of labels: {type(labels)}, Type of indices: {type(indices)}")  # Debug line
    
    downsampled_data = data[indices]
    downsampled_labels = labels[indices]

    if unswapped is not None:
        return downsampled_data, downsampled_labels, unswapped[indices]
    return downsampled_data, downsampled_labels


def prep_data() -> Tuple[Any, ...]:
    """
    Prepare and preprocess the seismic data for training and validation.

    This extensive function handles various steps such as loading data, data transformation, label swapping, 
    downsampling, scaling, oversampling, and preparing labels and weights for training and validation. 
    It also sets up label encoders and class weights, and prepares the data structures required for 
    training a machine learning model.

    Returns:
    A tuple containing various elements essential for training and validation, 
    including processed data, labels, label maps, class weights, and other metadata.

    Note:
    Used to reduce the complexity of the main training script. Not used in production.
    """
    # Section 1: Initial setup and loading data
    now = datetime.now()
    date_and_time = now.strftime("%Y%m%d_%H%M%S")

    loadData = LoadData()
    train_dataset = loadData.get_train_dataset()
    val_dataset = loadData.get_val_dataset()
    scaler = Scaler()

    # Section 2: Processing and preparing training data
    # This includes transposing data, label swapping, downsampling, and checking for debugging.
    

    if train_dataset is not None:
        train_data, train_labels, train_meta = train_dataset[0],train_dataset[1],train_dataset[2]
        logger.info("Train data shape: " + str(train_data.shape))
        train_data = np.transpose(train_data, (0,2,1))
        logger.info("Train data shape after transpose: " + str(train_data.shape))
        train_labels = swap_labels(train_labels)
        logger.info(f"Train unique labels: {np.unique(train_labels)}")
        if cfg.data.debug:
            train_data, train_labels = downsample_data_labels(train_data, train_labels, unswapped = None, p = 10)

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

        # Code to get label counts after oversampling
        post_oversample = len(train_labels)
        logger.info(f"Before oversampling: {pre_oversample}, after oversampling: {post_oversample}")
        logger.info(f"Label distribution post-oversample: {np.unique(train_labels, return_counts=True)}")
        logger.info(f"Before scaling training shape is {train_data.shape}, with (min, max) ({np.min(train_data)}, {np.max(train_data)})")
        scaler.fit(train_data)

    # Section 3: Processing and preparing validation data
    # Similar to training data, but with additional steps specific to validation.
    if val_dataset is not None:
        val_data, val_labels, val_meta = val_dataset[0],val_dataset[1],val_dataset[2]
        val_data = np.transpose(val_data, (0,2,1))
        unswapped_labels = val_labels
        val_labels = swap_labels(val_labels)
        logger.info(f"Val unique labels: {np.unique(val_labels)}")
        if cfg.data.debug:
            val_data, val_labels, unswapped_labels = downsample_data_labels(val_data, val_labels, unswapped = unswapped_labels, p = 10)


        logger.info(f"Label distribution validation: {np.unique(val_labels, return_counts=True)}")
        logger.info(f"Before scaling validation shape is {val_data.shape}, with (min, max) ({np.min(val_data)}, {np.max(val_data)})")
        if train_dataset is not None:
            if cfg.scaling.global_or_local != "local":
                raise NotImplementedError("Global scaler needs to be fitted. Loading prefitted scaler is not implemented yet.")

    #train_data = scaler.transform(train_data)
    #val_data = scaler.transform(val_data)

    #logger.info(f"After scaling training shape is {train_data.shape}, with (min, max) ({np.min(train_data)}, {np.max(train_data)})")
    #logger.info(f"After scaling validation shape is {val_data.shape}, with (min, max) ({np.min(val_data)}, {np.max(val_data)})")

    # Section 4: Scaling data and preparing labels and weights
    # This includes fitting the scaler, transforming data, and preparing nested label dictionaries.
    
    if train_dataset is not None:
        nested_label_dict_train, detector_class_weights_train, classifier_class_weights_train, label_encoder_train, detector_label_map, classifier_label_map = prepare_labels_and_weights(train_labels)
        if val_dataset is not None:
            nested_label_dict_val, _, _, _, _, _ = prepare_labels_and_weights(val_labels, label_encoder=label_encoder_train)
    if train_dataset is None and val_dataset is not None:
        nested_label_dict_train, detector_class_weights_train, classifier_class_weights_train, label_encoder_train, detector_label_map, classifier_label_map = prepare_labels_and_weights(val_labels)
        nested_label_dict_val, _, _, _, _, _ = prepare_labels_and_weights(val_labels, label_encoder=label_encoder_train)
        train_data = val_data
        train_labels = val_labels
        train_meta = val_meta
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

     # Section 5: Final preparations and return
    # Setting up final structures and logging information before returning all prepared data and metadata.

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
    return train_data, train_labels_dict, val_data, val_labels_dict, label_map, detector_class_weight_dict, classifier_class_weight_dict, classifier_label_map, detector_label_map, date_and_time, train_meta, val_meta, scaler, unswapped_labels
