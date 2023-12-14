from src.Live import ClassifyGBF, load_model, LiveClassifier
from obspy import UTCDateTime
from src.Scaler import Scaler
from global_config import cfg, logger


"""
Basic script for running the event classifier on GBF data.

The script loads the pretrained model, and then runs the classifier on the data from the specified time interval.
After each prediction, an MP4 video is saved to the specified output folder, showing the waveforms and predictions for each step.

The model performs 1 prediction per 29ms.

The final output is from an ensamble prediction, where the model predicts on a number of different windows of the data, 
and then takes the most common prediction.

TODO: Use real P velocities and S velocities where available. Currently using default values.
TODO: Weight the predictions based on centrality of the event in the waveform.
TODO: Regenerate training data with better velocities.
TODO: Time the process. Each step
TODO: Check memory requirements.
TODO: 
"""

#starttime = UTCDateTime('2023-09-30T14:01:00')
#endtime =   UTCDateTime('2023-09-30T16:45:00')

starttime = UTCDateTime('2023-10-30T12:01:00')
endtime =   UTCDateTime('2023-10-30T16:45:00')

# Loads the pretrained model
model, label_maps = load_model()

# Initializes the data loader
classify = ClassifyGBF()
# Tracedata np.arr, streams is a list of streams (obspy), starttimes and endtimes are lists of start and endtimes (obspy.UTDDateTime)
tracedata, streams, starttimes, endtimes = classify.get_data_to_predict(starttime, endtime)

# Initializes the live classifier pipeline
model = LiveClassifier(model, Scaler(), label_maps, cfg)
final_classifications = []
mean_probas = []
print(starttimes)
for idx, trace in enumerate(tracedata):
    # Ensamble prediction on the trace
    final_yhat, mean_proba, yhats, yprobas, intervals = model.predict(trace)
    final_classifications.append(final_yhat)
    mean_probas.append(mean_proba)
    # Plots the end result
    model.plot_predicted_event(intervals, starttimes[idx], yprobas, yhats, final_yhat, mean_proba)

logger.info(f"All classifications: {final_classifications}")
logger.info(f"Mean probas: {mean_probas}")

# TODO: Plan what needs