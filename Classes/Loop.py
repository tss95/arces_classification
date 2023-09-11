import tensorflow as tf
import numpy as np
tf.get_logger().setLevel('ERROR')
import wandb
from tensorflow.keras.utils import Progbar
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

    def on_train_epoch_end(self, train_metrics_list):
        avg_train_metrics = {
        k: np.mean([m[k].numpy() for m in train_metrics_list]) 
        for k in train_metrics_list[0].keys()
        }
        wandb.log({k: v for k, v in avg_train_metrics.items()})
        for metric in self.detector_metrics.metrics:
            metric.reset_states()
        for metric in self.classifier_metrics.metrics:
            metric.reset_states()

    def on_val_epoch_end(self, val_metrics_list):
        avg_val_metrics = {
        k: np.mean([m[k].numpy() for m in val_metrics_list]) 
        for k in val_metrics_list[0].keys()
    }
        wandb.log({k: v for k, v in avg_val_metrics.items()})

    def fit(self, train_generator, val_generator=None, epochs=1, callbacks=None):
        callbacks = callbacks or []
        for callback in callbacks:
            callback.set_model(self)
        logs = {}
        for callback in callbacks:
            callback.on_train_begin(logs)
        for epoch in range(epochs):
            logs = {'epoch': epoch}
            for callback in callbacks:
                callback.on_epoch_begin(epoch, logs)
            print(f"Starting epoch {epoch+1}/{epochs}")
            progbar = Progbar(target=len(train_generator))
            train_metrics_list = []
            for i, (x_batch, y_batch) in enumerate(train_generator):
                logs = {'batch': i}
                for callback in callbacks:
                    callback.on_batch_begin(i, logs)
                train_metrics = self.train_step((x_batch, y_batch))
                train_metrics_list.append(train_metrics)
                progbar.update(i+1)
                logs = {'batch': i, **train_metrics}
                for callback in callbacks:
                    callback.on_batch_end(i, logs)
            self.on_train_epoch_end(train_metrics_list)
            print("Epoch completed")
            if val_generator is not None:
                val_metrics_list = []
                for x_val_batch, y_val_batch in val_generator:
                    val_metrics = self.test_step((x_val_batch, y_val_batch))
                    val_metrics_list.append(val_metrics)
                self.on_val_epoch_end(val_metrics_list)
            logs = {'epoch': epoch, **train_metrics}
            for callback in callbacks:
                epoch_logs = {k: logs[k] for k in logs if k != 'epoch'}  # Exclude 'epoch' from logs
                callback.on_epoch_end(epoch, epoch_logs)

        logs = {}
        for callback in callbacks:
            callback.on_train_end(logs)

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
        x, y = data  # Unpack data
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)

            # Compute loss for detector and apply class weights
            loss_detector = tf.keras.losses.binary_crossentropy(y['detector'], y_pred['detector'], from_logits = True)
            # Map the labels to their corresponding weights for the detector
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
            total_loss = tf.reduce_sum(loss_detector) + tf.reduce_sum(loss_classifier)

        # Compute and apply gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(total_loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Update metrics
        self._shared_step(y, y_pred)

        results = {}
        results.update({"train_detector_" + k: v for k, v in self.detector_metrics.result().items()})
        results.update({"train_classifier_" + k: v for k, v in self.classifier_metrics.result().items()})

        return results

    @tf.function
    def test_step(self, data):
        x, y = data  # Unpack data
        y_pred = self(x, training=False)  # Forward pass

        # Update metrics
        self._shared_step(y, y_pred)

        results = {}
        results.update({"test_detector_" + k: v for k, v in self.detector_metrics.result().items()})
        results.update({"test_classifier_" + k: v for k, v in self.classifier_metrics.result().items()})


        return results

