import numpy as np
from global_config import logger, cfg
from Classes.Models import get_model
import os
import math
from obspy import Trace, Stream, UTCDateTime
from Classes.Utils import one_prediction
from seismonpy.norsardb import Client
from seismonpy.utils import create_global_mongodb_object
from seismonpy.io.mongodb.eventdb import MongoEventDataBase
import warnings
import matplotlib.pyplot as plt
from skimage.transform import resize
from imageio import get_writer

from PIL import Image
from datetime import datetime


def load_model(model_name = None):
    detector_label_map = {0: "noise", 1: "event"}
    classifier_label_map = {0:"earthquake", 1:"exlposion"}
    label_maps = {"detector": detector_label_map, "classifier": classifier_label_map}


    input_shape = (cfg.live.length*cfg.live.sample_rate + 1, 3)
    detector_class_weight_dict = {"noise": 1, "event": 1}
    classifier_class_weight_dict = {"earthquake": 1, "explosion": 1}

    logger.info("Input shape to the model: " + str(input_shape))
    classifier_metrics = [None]
    detector_metrics = [None]
    model = get_model(detector_label_map, classifier_label_map, detector_metrics, classifier_metrics, 
                    detector_class_weight_dict, classifier_class_weight_dict)
    model.build(input_shape=(None, *input_shape))  # Explicitly building the model here
    if model_name is None:
        model_name = cfg.model_name
    model.load_weights(os.path.join(cfg.paths.model_save_folder, model_name))
    logger.info(f"Loaded model weights from {os.path.join(cfg.paths.model_save_folder, cfg.model_name)}")
    return model, label_maps

import re

def sanitize_filename(filename):
    return re.sub(r'[\\/:*?"<>|]', '_', str(filename))

def resize_image(image, target_width):
    current_height, current_width = image.shape[:2]
    target_height = int((target_width / current_width) * current_height)
    resized_image = resize(image, (target_height, target_width), anti_aliasing=True)
    return (resized_image * 255).astype(np.uint8)

class ClassifyGBF:

    def __init__(self):
        warnings.simplefilter(action='ignore', category=UserWarning)

    def get_beam(self, start, end, picks, inventory):
        p_vel = cfg.live.p_vel
        s_vel = cfg.live.s_vel
        edge = cfg.live.edge
        startt = start - edge
        endt = end + edge
        baz = self.average_bazimuth(picks)
        try:
            comp = 'BH*'
            zcomp = '*Z'
            tcomp = '*T'
            rcomp = '*R'
            if start < UTCDateTime('2014-09-19T00:00:00'):
                comp = 's*'
                zcomp = '*z'

            stream = Client().get_waveforms(
                'AR*', comp, starttime=startt - edge, endtime=endt + edge, sampling_rate_tolerance=0.5
            )

            stream = self.correct_trace_start_times(stream)

            # Check for masked data, NaN values in traces
            # Remove traces with more than 5 s masked
            masked_traces = []
            for tr in stream.traces:
                if isinstance(tr.data, np.ma.masked_array):
                    time_filled = tr.stats.delta* np.sum(tr.data.mask)
                    if time_filled > 5.0:
                        print(f'{time_filled:.4f} s of trace data masked, dropping trace - {tr.stats.starttime.__str__()}')
                        masked_traces.append(tr)
                    else:
                        tr.data = tr.data.filled(0.0)
                        print(f'{time_filled:.4f} s of trace data masked, filling with zeros - {tr.stats.starttime.__str__()}')

                num_nans = np.sum(np.isnan(tr.data))
                if num_nans > 0:
                    time_containing_nans = num_nans*tr.stats.delta
                    if time_containing_nans > 5.0:
                        print(f'{time_containing_nans:.4f} s of trace has NaNs, dropping trace - {tr.stats.starttime.__str__()}')
                        masked_traces.append(tr)
                    else:
                        tr.data = np.nan_to_num(tr.data)
                        print(f'{time_containing_nans:.4f} s of trace has NaNs, filling with zeros - {tr.stats.starttime.__str__()}')


            for tr in masked_traces:
                if tr in stream.traces: # May have been removed already
                    stream.remove(tr)
            
            if len(stream) == 0:
                raise RuntimeError('Stream has no remaining traces')

            stream.detrend('demean')
            stream.taper(max_percentage=None, max_length=edge, type='cosine', halfcosine=True)
            stream.filter('highpass', freq=1.5)
            stream.resample(cfg.live.sample_rate)
            #print("SAMPLE RATE:", cfg.live.sample_rate)
            stream.rotate('NE->RT', back_azimuth=baz, inventory=inventory)

            p_time_delays = inventory.beam_time_delays(baz, p_vel)
            p_beam_z = stream.select(channel=zcomp).create_beam(p_time_delays)
            p_beam_z.stats.channel = 'P-beam, Z'

            s_time_delays = inventory.beam_time_delays(baz, s_vel)
            s_beam_t = stream.select(channel=tcomp).create_beam(s_time_delays)
            s_beam_t.stats.channel = 'S-beam, T'
            s_beam_r = stream.select(channel=rcomp).create_beam(s_time_delays)
            s_beam_r.stats.channel = 'S-beam, R'

            p_beam_z.trim(start, end)
            s_beam_t.trim(start, end)
            s_beam_r.trim(start, end)
            
            filter_name = cfg.filters.highpass_or_bandpass
            if filter_name == "highpass":
                stream.filter('highpass', freq = cfg.filters.high_kwargs.high_freq)
            if filter_name == "bandpass":
                stream.filter('bandpass', freqmin=cfg.filters.band_kwargs.min, freqmax=cfg.filters.band_kwargs.max)

            stream = Stream([p_beam_z, s_beam_t, s_beam_r])
            tracedata = np.array([p_beam_z.data, s_beam_t.data, s_beam_r.data])

            return tracedata, stream

        except Exception as exc:
            print('ERROR: {} - {}'.format(start, exc))
            return str(type(exc)) + str(exc), None

    def correct_trace_start_times(self, stream, max_delta=0.15):
        """
        For old data the traces might have tiny offset in start time, which breaks
        beamforming. Adjust this manually.
        Remove traces with diff > max_delta
        """
        sts = [tr.stats.starttime for tr in stream.traces]
        most_common = np.unique(sts)[0]

        for tr in stream.traces:
            this_starttime = tr.stats.starttime
            if this_starttime != most_common:
                if abs(this_starttime - most_common) <= max_delta:
                    tr.stats.starttime = most_common
                else:
                    print('Removing trace:', tr)
                    stream.remove(tr)
        
        return stream
    
    def get_data_to_predict(self, starttime, endtime):
        filtered_events, inventory = self.get_array_picks(starttime, endtime, cfg.live.array)
        print("Number of filtered events: ", len(filtered_events))
        print(f"Inventory: {inventory}")
        starttimes, endtimes = self.transform_events_to_start_and_end_times(filtered_events)
        print(f"starttimes: {starttimes}")
        tracedata, streams = [], []
        for i, (starttime, endtime) in enumerate(zip(starttimes, endtimes)):
            traced, stream = self.get_beam(starttime, endtime, filtered_events[i], inventory)
            if traced is not isinstance(traced, str):
                tracedata.append(traced)
                streams.append(stream)
        return tracedata, streams, starttimes, endtimes

    
    def average_bazimuth(self, picks):
        bazimuths = [pick['backazimuth'] for pick in picks]
        # Convert each azimuth to radians
        radian_bazimuths = [math.radians(az) for az in bazimuths]
        
        # Calculate mean of sin and cos
        mean_sin = sum(math.sin(az) for az in radian_bazimuths) / len(radian_bazimuths)
        mean_cos = sum(math.cos(az) for az in radian_bazimuths) / len(radian_bazimuths)
        
        # Use atan2 to compute average azimuth in radians
        average_bazimuth_rad = math.atan2(mean_sin, mean_cos)
        
        # Convert back to degrees, ensuring the result is within [0, 360)
        average_bazimuth_deg = math.degrees(average_bazimuth_rad) % 360
        
        return average_bazimuth_deg
    
    def predict_gbf_event(self):
        # Wrapper for predict function that handles GBF picks.
        raise NotImplementedError
    
    def load_events(self, starttime: UTCDateTime, endtime: UTCDateTime, collection: str = "gbf1440_large", dbname: str = "auto",  
        mongourl: str = "mongo.norsar.no:27017", mongodb_user: str = "guest", mongodb_password: str = "guest", 
        mongodb_authsource: str = "test"):
        query = {"$and":
            [
                {"origins.time": {"$gt": starttime.isoformat()}},
                {"origins.time": {"$lt": endtime.isoformat()}},
                {"picks.waveform_id.station_code": "ARCES"}
            ]
            }
        obj = create_global_mongodb_object(mongourl.split(":")[0], int(mongourl.split(":")[1]), mongodb_user, mongodb_password, mongodb_authsource)
        db = MongoEventDataBase(obj[dbname], collection)
        events = db.find_events(query, decode_result=True)
        inventory = Client().get_array_inventory(cfg.live.array)
        return events, inventory

    def get_array_picks(self, starttime: UTCDateTime, endtime:UTCDateTime, station_code: str):
        events, inventory = self.load_events(starttime, endtime)
        # Filter events where ARCES made a detection
        relevant_events = [event for event in events if any(pick.waveform_id.station_code == station_code for pick in event.picks)]
        
        # Extract only the ARCES-related picks from those events
        nested_filtered_events = []
        for event in relevant_events:
            arces_picks = [pick for pick in event.picks if pick.waveform_id.station_code == station_code]
            nested_filtered_events.append(arces_picks)
            
        return nested_filtered_events, inventory

    def transform_events_to_start_and_end_times(self, filtered_events: list):
        # TODO Bias towards start of the event
        starttimes, endtimes = [], []
        for event in filtered_events:
            pick_times = [pick.time for pick in event]
            start = min(pick_times)
            end = max(pick_times)
            duration = end - start
            # Need to make sure the event is long enough to include the entire event + event buffer. 
            # Also 
            if duration < cfg.live.length:
                missing_length = cfg.live.length - duration
                start = start - missing_length/2
                end = end + missing_length/2
            # How to handle events that are too long for the model? 
            starttimes.append(start - cfg.live.event_buffer)
            endtimes.append(end + cfg.live.event_buffer)

        return starttimes, endtimes
    
class LiveClassifier:
    def __init__(self, model, scaler, label_maps, cfg):
        """
        Initialize the LiveClassifier object.

        Parameters:
        - model: Pre-trained machine learning model for classification.
        - scaler: Scaler object to normalize features.
        - label_maps: Dictionary mapping from labels to integers.
        - cfg: Configuration object containing various settings.
        """
        self.cfg = cfg
        self.model = model
        self.label_maps = label_maps
        self.scaler = scaler

    def predict(self, trace):
        """
        Perform event classification for a specified time range.

        Parameters:
        - trace: Numpy array containing the seismic trace.

        Returns:
        - final_yhat: Final ensemble prediction.
        - mean_proba: Mean prediction probabilities for each class.
        - yhats: Individual predictions for each trace segment.
        - yprobas: Individual prediction probabilities for each trace segment.
        - X: Preprocessed input features.
        """
        trace = trace.T
        X = self.prepare_multiple_intervals(trace)
        X = [self.local_minmax(x) for x in X]
        X = np.array(X)
        # TODO: Figure out the logic for this
        yhats, yprobas, final_yhat, mean_proba = self.ensamble_predict(self.model, X)
        logger.info(f"Mean proba: {mean_proba}")

        return final_yhat, mean_proba, yhats, yprobas, X
    
    def prepare_multiple_intervals(self, trace):
        """
        Prepare multiple intervals from a single trace for ensemble prediction.

        Parameters:
        - trace: Numpy array containing the seismic trace.

        Returns:
        - List of sub-traces.
        """
        traces = []
        # Creates equally sized intervals of the trace, using the user defined step size.
        for start in range(0, len(trace) - (cfg.live.length*cfg.live.sample_rate)+1, (cfg.live.step*cfg.live.sample_rate)+1):
            traces.append(trace[start:(start+cfg.live.length*cfg.live.sample_rate)+1])
        return traces

    def ensamble_predict(self, model, X):
        """
        Perform ensemble prediction on multiple intervals.

        Parameters:
        - model: Pre-trained machine learning model.
        - X: Numpy array of shape (n_samples, timestamps, channels).

        Returns:
        - yhats: List of predicted labels.
        - yprobas: Dictionary containing prediction probabilities for "detector" and "classifier".
        - final_yhat: Final ensemble prediction.
        - mean_proba: Mean prediction probabilities.
        """
        # Basic ensamble prediction for the input data.
        # TODO: Consider weighing predictions higher around the center (assuming thats where the pick is).
        yhats, probas = [], {"detector": [], "classifier": []}
        for x in X:
            yhat, proba = one_prediction(model, x, self.label_maps)
            yhats.append(yhat)
            probas["detector"].append(proba["detector"])
            probas["classifier"].append(proba["classifier"])
        unqiue, counts = np.unique(yhats, return_counts=True)
        final_yhat = unqiue[np.argmax(counts)]
        mean_proba = {"detector": np.mean(probas["detector"], axis=0), "classifier": np.mean(probas["classifier"], axis=0)}
        return yhats, probas, final_yhat, mean_proba
    
    def local_minmax(self, trace):
        """
        Normalize a trace using min-max scaling.

        Parameters:
        - trace: Numpy array containing a seismic trace.

        Returns:
        - Normalized trace.
        """
        mmax = np.max(trace)
        mmin = np.min(trace)
        return (trace - mmin) / (mmax - mmin)
    

    def plot_predicted_event(self, intervals, event_time, yprobas, yhats, final_yhat, mean_proba, station='AR*'):
        frames = []
        for i, interval in enumerate(intervals):
            channel_names = ['P-beam, Z', 'S-beam, T', 'S-beam, R']
            stream = Stream()
            for j in range(3):
                tr = Trace()
                tr.data = interval[:, j]
                tr.stats.network = cfg.live.array  
                tr.stats.station = station
                tr.stats.channel = channel_names[j]
                tr.stats.sampling_rate = cfg.live.sample_rate
                
                # Set the start time for each Trace
                if event_time:
                    tr.stats.starttime = event_time + i * cfg.live.step
                
                stream.append(tr)
            
            # Plot the traces using ObsPy in its own figure
            fig1 = plt.figure()
            stream.plot(fig=fig1)
            fig1.canvas.draw()
            obspy_image = np.frombuffer(fig1.canvas.tostring_rgb(), dtype='uint8')
            obspy_image = obspy_image.reshape(fig1.canvas.get_width_height()[::-1] + (3,))
            plt.close(fig1)
            
            detector_values = [np.squeeze(val) for val in yprobas['detector']]
            classifier_values = [np.squeeze(val) for val in yprobas['classifier']]
            model_output_image = self.plot_model_output(detector_values, classifier_values, i, final_yhat, yhats, mean_proba)
            target_width = obspy_image.shape[1]
            model_output_image_resized = resize_image(model_output_image, target_width)
            combined_image = np.vstack((obspy_image, model_output_image_resized))
            frames.append(combined_image)

        safe_event_time = sanitize_filename(str(event_time))
        output_path = os.path.join(cfg.paths.live_test_path, f'{safe_event_time}_{final_yhat}.mp4')

        with get_writer(output_path, mode='I', fps=1, codec='libx264', pixelformat='yuv420p', format='FFMPEG') as writer:
            for frame in frames:
                writer.append_data(frame)

    def plot_model_output(self, detector_values, classifier_values, current_step, final_prediction, yhats, mean_proba):
        fig, ax = plt.subplots(figsize=(10, 4))  # Adjusted height

        ax.plot(detector_values, label='Detector', color='r', marker='o')
        ax.plot(classifier_values, label='Classifier', color='b', marker='o')

        if current_step is not None:
            ax.axvspan(current_step - 0.5, current_step + 0.5, facecolor='gray', alpha=0.5)
        ax.axhline(0.5, color='gray', linestyle='--')
        ax.axhline(mean_proba['detector'][0], color='red', linestyle='--', label='Detector mean')
        ax.axhline(mean_proba['classifier'][0], color='blue', linestyle='--', label='Classifier mean')

        ax.text(0.1 * len(detector_values), 0.55, 'Event', color='red')
        ax.text(0.1 * len(detector_values), 0.45, 'Noise', color='red')
        ax.text(0.8 * len(classifier_values), 0.55, 'Explosion', color='blue')
        ax.text(0.8 * len(classifier_values), 0.45, 'Earthquake', color='blue')

        ax.set_ylim([-0.05, 1.05])
        ax.set_xlim([-0.05, len(detector_values) + 0.05])
        ax.set_title(f'Model Predictions - Final Prediction: {final_prediction}')

        ax2 = ax.twiny()
        ax2.set_xlim(ax.get_xlim())
        ax2.set_xticks(np.arange(len(yhats)))
        ax2.set_xticklabels([pred[0] for pred in yhats], ha='center', rotation=45, color='green')
        ax2.xaxis.tick_bottom()
        ax2.xaxis.set_label_position('bottom')
        ax2.spines['bottom'].set_position(('outward', 60))  # Adjusted distance
        ax2.set_frame_on(False)

        ax.legend(loc='upper left')
        fig.canvas.draw()

        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)
        
        return image





