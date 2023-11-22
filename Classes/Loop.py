import tensorflow as tf
import numpy as np
tf.get_logger().setLevel('ERROR')
import wandb
from typing import List, Dict, Any

class MetricsPipeline:
    def __init__(self, metrics_list: List[str]):
        """
        Initialize metrics based on a list of metric names.

        Args:
            metrics_list (List[str]): A list of metric names to initialize.
        """
        self.metrics = []
        for metric_name in metrics_list:
            if metric_name == "precision":
                self.metrics.append(tf.keras.metrics.Precision())
            elif metric_name == "recall":
                self.metrics.append(tf.keras.metrics.Recall())
            elif metric_name == "accuracy":
                self.metrics.append(tf.keras.metrics.BinaryAccuracy())

    def update_state(self, y_true: tf.Tensor, y_pred: tf.Tensor):
        """
        Update the state of each metric.

        Args:
            y_true (tf.Tensor): The ground truth labels.
            y_pred (tf.Tensor): The predicted labels.
        """
        for metric in self.metrics:
            metric.update_state(y_true, y_pred)

    def result(self) -> Dict[str, tf.Tensor]:
        """
        Get the current result of each metric.

        Returns:
            Dict[str, tf.Tensor]: A dictionary mapping metric names to their current result.
        """
        return {metric.name: metric.result() for metric in self.metrics}



class Loop(tf.keras.Model):
    def __init__(
        self, 
        label_map_detector: Dict[str, int], 
        label_map_classifier: Dict[str, int],
        detector_metrics: List[str], 
        classifier_metrics: List[str], 
        detector_class_weights: Dict[str, float], 
        classifier_class_weights: Dict[str, float]
    ):
        """
        Initializes the modelling loop with various parameters.

        Args:
            label_map_detector (Dict[str, int]): Mapping of labels for the detector.
            label_map_classifier (Dict[str, int]): Mapping of labels for the classifier.
            detector_metrics (List[str]): List of metric names for the detector.
            classifier_metrics (List[str]): List of metric names for the classifier.
            detector_class_weights (Dict[str, float]): Class weights for the detector.
            classifier_class_weights (Dict[str, float]): Class weights for the classifier.
        """
        super(Loop, self).__init__()
        # Reverse mapping for label maps
        self.label_map_detector = {v: k for k, v in label_map_detector.items()}
        self.label_map_classifier = {v: k for k, v in label_map_classifier.items()}
        
        # Class weights as TensorFlow constants
        self.detector_class_weights = tf.constant(
            [detector_class_weights[key] for key in sorted(detector_class_weights.keys())]
        )
        self.classifier_class_weights = tf.constant(
            [classifier_class_weights[key] for key in sorted(classifier_class_weights.keys())]
        )

        # Initialize metrics pipelines
        self.detector_metrics = MetricsPipeline(detector_metrics)
        self.classifier_metrics = MetricsPipeline(classifier_metrics)

    
    @tf.function
    def on_epoch_end(self, epoch: int, logs: Dict[str, Any] = None):
        """
        Actions to perform at the end of an epoch.

        Args:
            epoch (int): The index of the epoch.
            logs (Dict[str, Any], optional): Dictionary of logs and metrics from the training/validation process.
        """
        if logs is not None:
            # Extracting training and validation metrics from the logs
            train_metrics = {k: logs[k] for k in logs if not k.startswith('val_')}
            val_metrics = {k: logs[k] for k in logs if k.startswith('val_')}
            
            # Logging metrics to Weight & Biases (wandb) for tracking
            wandb.log({"epoch": epoch, **train_metrics, **val_metrics})
            
        # Ensuring base class functionality is maintained
        super().on_epoch_end(epoch, logs)


    @tf.function
    def _shared_step(self, y_true: Dict[str, tf.Tensor], y_pred: Dict[str, tf.Tensor]):
        """
        Common operations for updating metrics in both training and testing steps.

        Args:
            y_true (Dict[str, tf.Tensor]): Ground truth labels.
            y_pred (Dict[str, tf.Tensor]): Predicted labels from the model.
        """
        # Updating detector metrics
        self.detector_metrics.update_state(y_true['detector'], y_pred['detector'])

        # Filtering out 'noise' events for classifier metrics
        mask_not_noise = tf.math.not_equal(y_true['detector'], self.label_map_detector['noise'])
        y_true_classifier = tf.boolean_mask(y_true['classifier'], mask_not_noise)
        y_pred_classifier = tf.boolean_mask(y_pred['classifier'], mask_not_noise)
        
        # Updating classifier metrics
        self.classifier_metrics.update_state(y_true_classifier, y_pred_classifier)
    
    @tf.function
    def train_step(self, data: Tuple[tf.Tensor, tf.Tensor]) -> Dict[str, Any]:
        """
        Training step to process one batch of data.

        Args:
            data (Tuple[tf.Tensor, tf.Tensor]): Tuple containing features and labels.

        Returns:
            Dict[str, Any]: A dictionary of training metrics and losses.
        """
        x, y = data
        with tf.GradientTape() as tape:
            # Calculating losses
            total_loss, loss_detector, loss_classifier, y_pred = self.calculate_loss(x, y, training=True)

        # Gradient calculation and optimization
        gradients = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        # Metrics update
        self._shared_step(y, y_pred)

        # Preparing result dictionary
        results = {"train_total_loss": total_loss,
                   "train_detector_loss": loss_detector,
                   "train_classification_loss": loss_classifier}
        results.update({"train_detector_" + k: v for k, v in self.detector_metrics.result().items()})
        results.update({"train_classifier_" + k: v for k, v in self.classifier_metrics.result().items()})

        return results

    @tf.function
    def test_step(self, data: Tuple[tf.Tensor, tf.Tensor]) -> Dict[str, Any]:
        """
        Testing step to evaluate the model on a batch of data.

        Args:
            data (Tuple[tf.Tensor, tf.Tensor]): Tuple containing features and labels.

        Returns:
            Dict[str, Any]: A dictionary of validation metrics and losses.
        """
        x, y = data
        # Loss calculation for validation data
        total_loss, loss_detector, loss_classifier, y_pred = self.calculate_loss(x, y, training=False)
        
        # Metrics update
        self._shared_step(y, y_pred)

        # Preparing result dictionary
        results = {"val_total_loss": total_loss,
                   "val_detector_loss": loss_detector,
                   "val_classification_loss": loss_classifier}
        results.update({"val_detector_" + k: v for k, v in self.detector_metrics.result().items()})
        results.update({"val_classifier_" + k: v for k, v in self.classifier_metrics.result().items()})

        return results


    @tf.function
    def calculate_loss(self, x: tf.Tensor, y: Dict[str, tf.Tensor], training: bool = True) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        Calculate the loss for the current batch.

        Args:
            x (tf.Tensor): Input features.
            y (Dict[str, tf.Tensor]): Ground truth labels.
            training (bool, optional): Flag indicating whether the model is in training mode. Defaults to True.

        Returns:
            Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]: Tuple containing total loss, detector loss, classifier loss, and predicted labels.
        """
        y_pred = self(x, training=training)
        
        # Loss computation for detector with class weights
        loss_detector = tf.keras.losses.binary_crossentropy(y['detector'], y_pred['detector'], from_logits=True)
        detector_weights = tf.gather(self.detector_class_weights, y['detector'])
        loss_detector *= tf.cast(detector_weights, dtype=loss_detector.dtype)

        # Filtering for classifier loss computation
        mask_not_noise = tf.math.not_equal(y['detector'], self.label_map_detector['noise'])
        y_true_classifier = tf.boolean_mask(y['classifier'], mask_not_noise)
        y_pred_classifier = tf.boolean_mask(y_pred['classifier'], mask_not_noise)

        # Loss computation for classifier with class weights
        loss_classifier = tf.keras.losses.binary_crossentropy(y_true_classifier, y_pred_classifier, from_logits=True)
        classifier_weights = tf.gather(self.classifier_class_weights, y_true_classifier)
        loss_classifier *= tf.cast(classifier_weights, dtype=loss_classifier.dtype)

        # Total loss calculation
        loss_detector = tf.reduce_mean(loss_detector)
        loss_classifier = tf.reduce_mean(loss_classifier)
        total_loss = loss_detector + loss_classifier

        return total_loss, loss_detector, loss_classifier, y_pred

