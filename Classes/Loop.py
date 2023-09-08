from global_config import logger, cfg, model_cfg
import tensorflow as tf
import wandb

class Loop(tf.keras.Model):
    def __init__(self, model, least_frequent_class, label_map):
        super(Loop, self).__init__()
        self.model = model
        self.label_map = label_map
        self.least_frequent_class = least_frequent_class
        self.num_classes = len(list(label_map.keys()))

        self.precision_metric = tf.keras.metrics.Precision()
        self.recall_metric = tf.keras.metrics.Recall()
        self.accuracy_metric = tf.keras.metrics.CategoricalAccuracy()

    def train_step(self, data):
        x, y = data

        with tf.GradientTape() as tape:
            y_pred = self.model(x, training=True)
            loss = self.compiled_loss(y, y_pred)

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        self.compiled_metrics.update_state(y, y_pred)

        # Custom metrics for least frequent class
        y_true_least = y[:, self.least_frequent_class]
        y_pred_least = tf.argmax(y_pred, axis=-1)
        y_pred_least = tf.cast(tf.equal(y_pred_least, self.least_frequent_class), dtype=tf.float32)
        
        self.precision_metric.update_state(y_true_least, y_pred_least)
        self.recall_metric.update_state(y_true_least, y_pred_least)
        self.accuracy_metric.update_state(y, y_pred)

        wandb.log({
            "precision": self.precision_metric.result().numpy(),
            "recall": self.recall_metric.result().numpy(),
            "accuracy": self.accuracy_metric.result().numpy()
        })

        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        x, y = data
        y_pred = self.model(x, training=False)

        self.compiled_metrics.update_state(y, y_pred)
        
        # Custom metrics for least frequent class could also be added here

        return {m.name: m.result() for m in self.metrics}