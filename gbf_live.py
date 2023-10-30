from Classes.Live import ClassifyGBF, load_model, LiveClassifier
from obspy import UTCDateTime
from Classes.Scaler import Scaler
from global_config import cfg


model, label_maps = load_model()

classify = ClassifyGBF()

starttime = UTCDateTime('2023-09-30T14:01:00')
endtime =   UTCDateTime('2023-09-30T16:45:00')
tracedata, streams, starttimes, endtimes = classify.get_data_to_predict(starttime, endtime)


print(len(tracedata[0]))
model = LiveClassifier(model, Scaler(), label_maps, cfg)
final_classifications = []
mean_probas = []
print(starttimes)
for idx, trace in enumerate(tracedata):
    final_yhat, mean_proba, yhats, yprobas, intervals = model.predict(trace)
    final_classifications.append(final_yhat)
    mean_probas.append(mean_proba)
    model.plot_predicted_event(intervals, starttimes[idx], yprobas, yhats, final_yhat)

print("All classifications:", final_classifications)
print("Mean probas:", mean_probas)