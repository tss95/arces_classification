from global_config import logger, cfg, model_cfg
import tensorflow as tf
import numpy as np

def get_least_frequent_class_metrics(class_id, class_name, metric_name):
    def metric_fn(y_true, y_pred):
        y_true = tf.cast(y_true[:, class_id], tf.float32)
        y_pred = tf.cast(tf.argmax(y_pred, axis=-1), tf.float32)
        y_pred = tf.cast(tf.equal(y_pred, class_id), tf.float32)
        
        if metric_name == 'precision':
            return tf.keras.metrics.Precision()(y_true, y_pred)
        
        if metric_name == 'recall':
            return tf.keras.metrics.Recall()(y_true, y_pred)
        
        if metric_name == 'f1_score':
            precision = tf.keras.metrics.Precision()(y_true, y_pred)
            recall = tf.keras.metrics.Recall()(y_true, y_pred)
            return 2 * ((precision * recall) / (precision + recall + 1e-5))
    
    metric_fn.__name__ = f"{metric_name}_{class_name}"
    return metric_fn

def get_metrics(one_hot_labels, label_map, metrics=['accuracy']):
    metric_objects = []
    
    # Identify the least numerous class
    class_counts = np.sum(one_hot_labels, axis=0)
    min_class = np.argmin(class_counts)
    min_class_name = label_map[min_class]
    
    # Add accuracy
    if "accuracy" in metrics:
        metric_objects.append(tf.keras.metrics.CategoricalAccuracy(name='accuracy'))
        
    # Add metrics for the least frequent class
    if "f1_score" in metrics:
        metric_objects.append(get_least_frequent_class_metrics(min_class, min_class_name, 'f1_score'))
    if "precision" in metrics:
        metric_objects.append(get_least_frequent_class_metrics(min_class, min_class_name, 'precision'))
    if "recall" in metrics:
        metric_objects.append(get_least_frequent_class_metrics(min_class, min_class_name, 'recall'))

    return metric_objects