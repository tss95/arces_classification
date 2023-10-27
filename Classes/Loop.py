import tensorflow as tf
import numpy as np
tf.get_logger().setLevel('ERROR')
import wandb
# Pseudo-code to illustrate handling class weights and flexible metrics within the Loop class

class MetricsPipeline:
    def __init__(self, metrics_list):
        """
        Initialize metrics based on the list of metric names.
        """
        self.metrics = []
        for metric_name in metrics_list:
            if metric_name == "precision":
                self.metrics.append(tf.keras.metrics.Precision())
            elif metric_name == "recall":
                self.metrics.append(tf.keras.metrics.Recall())
            elif metric_name == "accuracy":
                self.metrics.append(tf.keras.metrics.BinaryAccuracy())
            #elif metric_name == "f1_score":
            #    self.metrics.append(tf.keras.metrics.F1Score(num_classes=1, threshold=0.5))
            # Add more metrics as needed

   

    def update_state(self, y_true, y_pred):
        """
        Update the state of each metric.
        """
        for metric in self.metrics:
            metric.update_state(y_true, y_pred)

    def result(self):
        """
        Get the current result of each metric.
        """
        return {metric.name: metric.result() for metric in self.metrics}



class Loop(tf.keras.Model):
    def __init__(self, label_map_detector, label_map_classifier, 
                 detector_metrics, classifier_metrics, detector_class_weights, 
                 classifier_class_weights):
        super(Loop, self).__init__()
        self.label_map_detector = label_map_detector
        self.label_map_classifier = label_map_classifier
        self.label_map_detector = {v: k for k, v in label_map_detector.items()}
        self.label_map_classifier = {v: k for k, v in label_map_classifier.items()}
        self.detector_class_weights = tf.constant([detector_class_weights[key] for key in sorted(detector_class_weights.keys())])
        self.classifier_class_weights = tf.constant([classifier_class_weights[key] for key in sorted(classifier_class_weights.keys())])

        # Initialize metrics pipeline for detector and classifier
        self.detector_metrics = MetricsPipeline(detector_metrics)
        self.classifier_metrics = MetricsPipeline(classifier_metrics)
    
    @tf.function
    def on_epoch_end(self, epoch, logs=None):
        if logs is not None:
            # Filter training and validation metrics from logs
            train_metrics = {k: logs[k] for k in logs if not k.startswith('val_')}
            val_metrics = {k: logs[k] for k in logs if k.startswith('val_')}
            
            # Log metrics to wandb
            wandb.log({"epoch": epoch, **train_metrics, **val_metrics})
            
        # Call the superclass method to preserve any base class functionality,
        # including potentially resetting metric states.
        super().on_epoch_end(epoch, logs)


    @tf.function
    def _shared_step(self, y_true, y_pred):
        """
        Update metrics common to both training and test steps.
        :param y_true: Ground truth labels
        :param y_pred: Predicted labels
        """
        # Update detector metrics
        self.detector_metrics.update_state(y_true['detector'], y_pred['detector'])

        # Create a mask to isolate 'not_noise' events for classifier metrics
        mask_not_noise = tf.math.not_equal(y_true['detector'], self.label_map_detector['noise'])
        
        # Apply mask to pre-filter labels and predictions for classifier metrics
        y_true_classifier = tf.boolean_mask(y_true['classifier'], mask_not_noise)
        y_pred_classifier = tf.boolean_mask(y_pred['classifier'], mask_not_noise)
        
        # Update classifier metrics
        self.classifier_metrics.update_state(y_true_classifier, y_pred_classifier)

    @tf.function
    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            total_loss, loss_detector, loss_classifier, y_pred = self.calculate_loss(x, y, training=True)
        # Compute and apply gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(total_loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics
        self._shared_step(y, y_pred)

        results = {"train_total_loss": total_loss,
                   "train_detector_loss": loss_detector,
                   "train_classification_loss": loss_classifier}
        results.update({"train_detector_" + k: v for k, v in self.detector_metrics.result().items()})
        results.update({"train_classifier_" + k: v for k, v in self.classifier_metrics.result().items()})


        return results

    @tf.function
    def test_step(self, data):
        x, y = data  # Unpack data
        total_loss, loss_detector, loss_classifier, y_pred = self.calculate_loss(x, y, training=False)
        # Update metrics
        self._shared_step(y, y_pred)

        results = {"val_total_loss": total_loss,
                   "val_detector_loss": loss_detector,
                   "val_classification_loss": loss_classifier}
        results.update({"val_detector_" + k: v for k, v in self.detector_metrics.result().items()})
        results.update({"val_classifier_" + k: v for k, v in self.classifier_metrics.result().items()})


        return results


    @tf.function
    def calculate_loss(self, x, y, training=True):
        y_pred = self(x, training=training)
        # Compute loss for detector and apply class weights
        loss_detector = tf.keras.losses.binary_crossentropy(y['detector'], y_pred['detector'], from_logits = True)
        # Map the labels to their corresponding weights for the detector
        detector_weights = tf.gather(self.detector_class_weights, y['detector'])
        detector_weights = tf.cast(detector_weights, dtype=loss_detector.dtype)  # Casting dtype
        loss_detector *= detector_weights
        
        # Create a mask to isolate 'not_noise' events
        mask_not_noise = tf.math.not_equal(y['detector'], self.label_map_detector['noise'])
        
        # Apply mask to pre-filter labels and predictions for classifier
        y_true_classifier = tf.boolean_mask(y['classifier'], mask_not_noise)
        y_pred_classifier = tf.boolean_mask(y_pred['classifier'], mask_not_noise)
        
        # Compute loss for classifier and apply class weights
        loss_classifier = tf.keras.losses.binary_crossentropy(y_true_classifier, y_pred_classifier, from_logits = True)
        # Map the labels to their corresponding weights for the classifier
        classifier_weights = tf.gather(self.classifier_class_weights, y_true_classifier)
        classifier_weights = tf.cast(classifier_weights, dtype=loss_classifier.dtype)  # Casting dtype
        loss_classifier *= classifier_weights
        
        # Total loss

        loss_detector = tf.reduce_mean(loss_detector)
        loss_classifier = tf.reduce_mean(loss_classifier)

        total_loss = loss_detector + loss_classifier
        return total_loss, loss_detector, loss_classifier, y_pred

