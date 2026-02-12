import numpy as np
import torch
from typing import Dict, List, Tuple, Any


DEFAULT_SINGLE_LABEL_MAP = {0: "noise", 1: "earthquake", 2: "explosion"}


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


def translate_single_labels(labels_single: np.ndarray, label_map: Dict[str, Dict[int, str]]) -> List[str]:
    single_map = label_map.get("single", DEFAULT_SINGLE_LABEL_MAP)
    final_labels = []
    for label in labels_single:
        label = int(label[0]) if isinstance(label, np.ndarray) and label.ndim > 0 else int(label)
        final_labels.append(single_map[label])
    return final_labels


def apply_threshold(pred_probs: np.ndarray, cfg) -> List[int]:
    """
    Apply a threshold to predicted probabilities to classify them as 0 or 1.

    This function applies a threshold defined in the global configuration to the predicted probabilities.

    Args:
    pred_probs: A numpy array of predicted probabilities.
    cfg: Configuration object containing data.model_threshold.

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

def _to_cpu_numpy(t):
    if isinstance(t, torch.Tensor):
        return t.detach().cpu().numpy()
    return np.asarray(t)


def get_final_labels(pred_probs: Dict[str, np.ndarray], label_map: Dict[str, Dict[int, str]], cfg) -> Tuple[List[str], Dict[str, np.ndarray]]:
    """
    Determine the final labels based on the predicted probabilities and label map.

    This function applies a sigmoid function to the predicted probabilities, applies a threshold, and translates them into final labels using the provided label map.

    Args:
    pred_probs: A dictionary containing the predicted probabilities for both 'detector' and 'classifier'.
    label_map: A dictionary mapping label indices to their string representations for both 'detector' and 'classifier'.
    cfg: Configuration object.

    Returns:
    A tuple containing:
        - A list of final string labels.
        - A dictionary with updated predicted probabilities for both 'detector' and 'classifier'.
    """
    if "single" in pred_probs:
        pred_probs_single = torch.softmax(
            torch.as_tensor(pred_probs["single"], dtype=torch.float32), dim=-1
        ).detach().cpu().numpy()
        pred_labels_single = np.argmax(pred_probs_single, axis=-1)
        final_labels = translate_single_labels(pred_labels_single, label_map)
        return final_labels, {"single": pred_probs_single}

    pred_probs_detector = torch.sigmoid(torch.as_tensor(pred_probs['detector'], dtype=torch.float32)).detach().cpu().numpy()
    pred_labels_detector = apply_threshold(pred_probs_detector, cfg)

    pred_probs_classifier = torch.sigmoid(torch.as_tensor(pred_probs['classifier'], dtype=torch.float32)).detach().cpu().numpy()
    pred_labels_classifier = apply_threshold(pred_probs_classifier, cfg)

    final_labels = translate_labels(pred_labels_detector, pred_labels_classifier, label_map)
    return final_labels, {"detector": pred_probs_detector, "classifier": pred_probs_classifier}

def one_prediction(model: Any, x: np.ndarray, label_maps: Dict[str, Dict[int, str]], cfg, is_torch=False) -> Tuple[List[str], Dict[str, np.ndarray]]:
    """
    Generate a prediction for a single data instance.

    This function is used to make a prediction on a single instance of data using the provided model. It processes the input, makes a prediction, and then translates it into a final label.

    Args:
    model: The trained model used for prediction.
    x: A numpy array representing a single instance of input data.
    label_maps: A dictionary containing label maps for translating numeric labels to string labels.
    cfg: Configuration object.
    is_torch: Boolean flag indicating if the model is a PyTorch model.

    Returns:
    A tuple containing:
        - A list with the final string label for the input data.
        - A dictionary with the predicted probabilities for both 'detector' and 'classifier'.
    """
    x = np.reshape(x, (1, *x.shape))
    if is_torch:
        #x = x.reshape(x.shape[0], x.shape[2], x.shape[1])[:-1]
        x = torch.tensor(x, dtype=torch.float32, device=next(model.parameters()).device)
    pred_probs = model(x)
    labels, pred_probs = get_final_labels(pred_probs, label_maps, cfg)
    return labels, pred_probs
