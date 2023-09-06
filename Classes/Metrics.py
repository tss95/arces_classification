import tensorflow as tf
import numpy as np

# Define the function
def get_least_frequent_class_metrics(one_hot_labels, label_map=None, sample_weight = None, metrics_list=['f1_score', 'precision', 'recall']):
    class_counts = np.sum(one_hot_labels, axis=0)
    min_class_id = np.argmin(class_counts)
    
    if label_map:
        label_name = label_map.get(min_class_id, str(min_class_id)).split(' ')[0]
    else:
        label_name = str(min_class_id)

    metrics_dict = {}
    for metric_name in metrics_list:
        class LeastFrequentClassMetrics(tf.keras.metrics.Metric):
            def __init__(self, **kwargs):
                super(LeastFrequentClassMetrics, self).__init__(name=f"{metric_name}_{label_name}", **kwargs)
                self.class_id = min_class_id
                self.metric_name = metric_name
                if metric_name == 'precision':
                    self.precision = tf.keras.metrics.Precision(name=f"precision_{label_name}")
                elif metric_name == 'recall':
                    self.recall = tf.keras.metrics.Recall(name=f"recall_{label_name}")
                elif metric_name == 'f1_score':
                    self.precision = tf.keras.metrics.Precision()
                    self.recall = tf.keras.metrics.Recall()

            def update_state(self, y_true, y_pred, sample_weight=None):
                y_true = tf.cast(y_true[:, self.class_id], tf.float32)
                y_pred = tf.cast(tf.argmax(y_pred, axis=-1), tf.float32)
                y_pred = tf.cast(tf.equal(y_pred, self.class_id), tf.float32)

                if sample_weight is not None:
                    sample_weight = tf.cast(sample_weight, tf.float32)
                    sample_weight = sample_weight[:, self.class_id]
                
                if self.metric_name == 'precision':
                    self.precision.update_state(y_true, y_pred, sample_weight=sample_weight)
                
                if self.metric_name == 'recall':
                    self.recall.update_state(y_true, y_pred, sample_weight=sample_weight)
                
                if self.metric_name == 'f1_score':
                    self.precision.update_state(y_true, y_pred, sample_weight=sample_weight)
                    self.recall.update_state(y_true, y_pred, sample_weight=sample_weight)

            def result(self):
                if self.metric_name == 'precision':
                    return self.precision.result()

                if self.metric_name == 'recall':
                    return self.recall.result()

                if self.metric_name == 'f1_score':
                    precision = self.precision.result()
                    recall = self.recall.result()
                    return 2 * ((precision * recall) / (precision + recall + 1e-5))

            def reset_states(self):
                if self.metric_name in ['precision', 'f1_score']:
                    self.precision.reset_states()
                if self.metric_name in ['recall', 'f1_score']:
                    self.recall.reset_states()
                    
        metrics_dict[metric_name] = LeastFrequentClassMetrics()
    
    return metrics_dict

if __name__ == '__main__':
    print("Running main")
    # Simulated one-hot encoded labels and predictions
    y_true = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0], [0, 1, 0]])
    y_pred = np.array([[0.8, 0.1, 0.1], [0.2, 0.6, 0.2], [0.1, 0.2, 0.7], [0.9, 0.05, 0.05], [0.1, 0.8, 0.1]])

    # Label map
    label_map = {0: 'explosion', 1: 'earthquake', 2: 'noise'}

    # Generate metrics
    metrics_dict = get_least_frequent_class_metrics(y_true, label_map=label_map)

    # Update states and print results
    for metric_name, metric_obj in metrics_dict.items():
        metric_obj.update_state(y_true, y_pred)
        print(f"{metric_name}: {metric_obj.result().numpy()}")