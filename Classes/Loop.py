import tensorflow as tf
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
                self.metrics.append(tf.keras.metrics.CategoricalAccuracy())
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
        self.detector_class_weights = detector_class_weights
        self.classifier_class_weights = classifier_class_weights

        # Initialize metrics pipeline for detector and classifier
        self.detector_metrics = MetricsPipeline(detector_metrics)
        self.classifier_metrics = MetricsPipeline(classifier_metrics)

    def fit(self, train_generator, val_generator=None, epochs=1, callbacks=None):
        # Initialize callbacks
        callbacks = callbacks or []
        for callback in callbacks:
            callback.set_model(self)
        # Call on_train_begin for callbacks
        logs = {}
        for callback in callbacks:
            callback.on_train_begin(logs)
        for epoch in range(epochs):
            # Call on_epoch_begin for callbacks
            logs = {'epoch': epoch}
            for callback in callbacks:
                callback.on_epoch_begin(epoch, logs)
            print(f"Starting epoch {epoch+1}/{epochs}")
            progbar = Progbar(target=len(train_generator))
            for i, (x_batch, y_batch) in enumerate(train_generator):
                # Call on_batch_begin for callbacks
                logs = {'batch': i}
                for callback in callbacks:
                    callback.on_batch_begin(i, logs)
                train_metrics = self.train_step((x_batch, y_batch))
                progbar.update(i+1)
                # Call on_batch_end for callbacks
                logs = {'batch': i, **train_metrics}
                for callback in callbacks:
                    callback.on_batch_end(i, logs)
            print("Epoch completed")
            # Validation loop
            if val_generator is not None:
                for x_val_batch, y_val_batch in val_generator:
                    val_metrics = self.test_step((x_val_batch, y_val_batch)) 
                    # Log metrics to wandb
                    wandb.log({"test_" + k: v for k, v in val_metrics.items()})
            # Log metrics
            print(f"Train Metrics: {train_metrics}")
            if val_generator is not None:
                print(f"Validation Metrics: {val_metrics}")
            # Call on_epoch_end for callbacks
            logs = {'epoch': epoch, **train_metrics}
            for callback in callbacks:
                callback.on_epoch_end(epoch, logs)
        # Call on_train_end for callbacks
        logs = {}
        for callback in callbacks:
            callback.on_train_end(logs)


    #@tf.function
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

    #@tf.function
    def train_step(self, data):
        x, y = data  # Unpack data
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)

            # Compute loss for detector and apply class weights
            loss_detector = tf.keras.losses.binary_crossentropy(y['detector'], y_pred['detector'], from_logits = True)
            loss_detector *= self.detector_class_weights
            
            # Create a mask to isolate 'not_noise' events
            mask_not_noise = tf.math.not_equal(y['detector'], self.label_map_detector['noise'])
            
            # Apply mask to pre-filter labels and predictions for classifier
            y_true_classifier = tf.boolean_mask(y['classifier'], mask_not_noise)
            y_pred_classifier = tf.boolean_mask(y_pred['classifier'], mask_not_noise)
            
            # Compute loss for classifier and apply class weights
            loss_classifier = tf.keras.losses.binary_crossentropy(y_true_classifier, y_pred_classifier, from_logits = True)
            loss_classifier *= self.classifier_class_weights
            
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

        wandb.log(results)
        return results

    #@tf.function
    def test_step(self, data):
        x, y = data  # Unpack data
        y_pred = self(x, training=False)  # Forward pass

        # Update metrics
        self._shared_step(y, y_pred)

        results = {}
        results.update({"test_detector_" + k: v for k, v in self.detector_metrics.result().items()})
        results.update({"test_classifier_" + k: v for k, v in self.classifier_metrics.result().items()})


        return results

