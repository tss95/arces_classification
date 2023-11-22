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
import re
from typing import Any, List, Tuple, Union, Optional, Dict
from obspy.core.inventory import Inventory

from PIL import Image
from datetime import datetime

# TODO: Resolve pick type declaration

def load_model(model_name: Optional[str] = None) -> Tuple[Any, Dict[str, Dict[int, str]]]:
    """
    Load a pre-trained model with specified weights and configurations.

    This function initializes the model with the given label maps for the detector and classifier.
    It sets up the input shape based on the configuration, builds the model, and loads the weights.

    Args:
        model_name (Optional[str]): The name of the model to be loaded. If None, the default model name
                                    from the configuration will be used.

    Returns:
        Tuple[Any, Dict[str, Dict[int, str]]]: A tuple containing the loaded model and the label maps
                                               for the detector and classifier.

    The function utilizes global configuration settings (`cfg`) and a logger instance (`logger`).
    """
    # Define label maps for the detector and classifier
    detector_label_map = {0: "noise", 1: "event"}
    classifier_label_map = {0: "earthquake", 1: "explosion"}
    label_maps = {"detector": detector_label_map, "classifier": classifier_label_map}

    # Calculate the input shape based on configuration settings
    input_shape = (cfg.live.length * cfg.live.sample_rate + 1, 3)

    # Define class weights for both detector and classifier
    detector_class_weight_dict = {"noise": 1, "event": 1}
    classifier_class_weight_dict = {"earthquake": 1, "explosion": 1}

    # Log the input shape
    logger.info("Input shape to the model: " + str(input_shape))

    # Initialize metrics for both detector and classifier
    classifier_metrics = [None]
    detector_metrics = [None]

    # Create and build the model
    model = get_model(detector_label_map, classifier_label_map, detector_metrics, classifier_metrics, 
                      detector_class_weight_dict, classifier_class_weight_dict)
    model.build(input_shape=(None, *input_shape))  # Explicitly building the model

    # Load model weights
    if model_name is None:
        model_name = cfg.model_name
    model.load_weights(os.path.join(cfg.paths.model_save_folder, model_name))

    # Log the model loading information
    logger.info(f"Loaded model weights from {os.path.join(cfg.paths.model_save_folder, cfg.model_name)}")

    return model, label_maps


def sanitize_filename(filename: str) -> str:
    """
    Sanitize the filename by removing or replacing invalid characters.

    This function ensures that the filename is compatible with various file systems
    by removing or replacing characters that are typically not allowed in file names.

    Args:
        filename (str): The original filename that needs to be sanitized.

    Returns:
        str: The sanitized filename with invalid characters replaced by underscores.

    Note: This function uses regular expressions for sanitization.
    """
    # Replace invalid characters with underscores
    return re.sub(r'[\\/:*?"<>|]', '_', filename)


def resize_image(image: np.ndarray, target_width: int) -> np.ndarray:
    """
    Resize an image to a specified width while maintaining the aspect ratio.

    This function calculates the target height to maintain the aspect ratio of the image,
    resizes the image using anti-aliasing, and converts it back to the appropriate data type.

    Args:
        image (np.ndarray): The original image to be resized.
        target_width (int): The target width for the resized image.

    Returns:
        np.ndarray: The resized image with the specified width and adjusted height.

    Note: This function assumes that the input image is a NumPy array.
    """
    # Calculate the current dimensions of the image
    current_height, current_width = image.shape[:2]

    # Calculate the target height to maintain the aspect ratio
    target_height = int((target_width / current_width) * current_height)

    # Resize the image with anti-aliasing
    resized_image = resize(image, (target_height, target_width), anti_aliasing=True)

    # Convert the resized image to an 8-bit unsigned integer format
    return (resized_image * 255).astype(np.uint8)


class ClassifyGBF:
    """
    A class for processing and classifying seismic data using ground-based facilities.

    This class includes methods for retrieving seismic data, creating beams for P and S waves,
    and predicting seismic events using the ground-based facilities (GBF) approach.
    """

    def __init__(self):
        """
        Initialize the ClassifyGBF instance.

        Sets up a simple filter to ignore user warnings during execution.
        """
        warnings.simplefilter(action='ignore', category=UserWarning)

    def get_beam(self, start: UTCDateTime, end: UTCDateTime, picks: List[Dict], inventory: Inventory) -> Tuple[Union[np.ndarray, str], Optional[Stream]]:
        """
        Retrieve and process seismic data to create P and S wave beams.

        Args:
            start (UTCDateTime): The start time for the data retrieval.
            end (UTCDateTime): The end time for the data retrieval.
            picks (List[Dict]): A list of seismic event picks.
            inventory (Inventory): Seismic station inventory information.

        Returns:
            Tuple[Union[np.ndarray, str], Optional[Stream]]: A tuple containing either the processed tracedata as a
                                                             numpy array and a Stream object with the beam data,
                                                             or an error message and None in case of an exception.

        Raises:
            RuntimeError: If the stream has no remaining traces after processing.
        """
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

    def correct_trace_start_times(self, stream: Stream, max_delta: float = 0.15) -> Stream:
        """
        Corrects the start times of traces in a Stream to synchronize them.

        This method is used for older data where traces might have slight offsets in start times.
        It standardizes the start times or removes traces with a significant difference.

        Args:
            stream (Stream): The stream of seismic data to be corrected.
            max_delta (float, optional): The maximum allowed time difference in seconds for a trace
                                         to be adjusted rather than removed. Defaults to 0.15 seconds.

        Returns:
            Stream: The corrected stream with synchronized start times.

        Note:
            Traces with a start time difference greater than `max_delta` are removed from the stream.
        """
        # Code for correcting trace start times
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
    
    def get_data_to_predict(self, starttime: UTCDateTime, endtime: UTCDateTime) -> Tuple[List[np.ndarray], List[Stream], List[UTCDateTime], List[UTCDateTime]]:
        """
        Retrieve and process seismic data for a given time range for prediction purposes.

        Args:
            starttime (UTCDateTime): The start time for data retrieval.
            endtime (UTCDateTime): The end time for data retrieval.

        Returns:
            Tuple[List[np.ndarray], List[Stream], List[UTCDateTime], List[UTCDateTime]]: A tuple containing lists of
                                                                                         processed data arrays, corresponding streams,
                                                                                         start times, and end times.
        """
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

    
    def average_bazimuth(self, picks: List[Dict]) -> float:
        """
        Calculate the average backazimuth from a list of seismic picks.

        Args:
            picks (List[Dict]): A list of seismic picks.

        Returns:
            float: The average backazimuth calculated from the picks.
        """
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

    
    def load_events(self, starttime: UTCDateTime, endtime: UTCDateTime, collection: str = "gbf1440_large", dbname: str = "auto",  
                    mongourl: str = "mongo.norsar.no:27017", mongodb_user: str = "guest", mongodb_password: str = "guest", 
                    mongodb_authsource: str = "test") -> Tuple[List[Dict], Inventory]:
        """
        Load seismic events from a MongoDB database within a specified time range.

        Args:
            starttime (UTCDateTime): The start time for event retrieval.
            endtime (UTCDateTime): The end time for event retrieval.
            collection (str, optional): The MongoDB collection to query. Defaults to "gbf1440_large".
            dbname (str, optional): The database name. Defaults to "auto".
            mongourl (str, optional): The URL of the MongoDB server. Defaults to "mongo.norsar.no:27017".
            mongodb_user (str, optional): The MongoDB username. Defaults to "guest".
            mongodb_password (str, optional): The MongoDB password. Defaults to "guest".
            mongodb_authsource (str, optional): The authentication source database. Defaults to "test".

        Returns:
            Tuple[List[Dict], Inventory]: A tuple containing a list of events and the corresponding inventory.
        """
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

    def get_array_picks(self, starttime: UTCDateTime, endtime: UTCDateTime, station_code: str) -> Tuple[List[List[Dict]], Inventory]:
        """
        Retrieve picks from seismic events for a specific array within a time range.

        Args:
            starttime (UTCDateTime): The start time for retrieving picks.
            endtime (UTCDateTime): The end time for retrieving picks.
            station_code (str): The code of the station array to filter picks.

        Returns:
            Tuple[List[List[Dict]], Inventory]: A tuple containing a nested list of filtered picks and the inventory.
        """
        events, inventory = self.load_events(starttime, endtime)
        # Filter events where ARCES made a detection
        relevant_events = [event for event in events if any(pick.waveform_id.station_code == station_code for pick in event.picks)]
        
        # Extract only the ARCES-related picks from those events
        nested_filtered_events = []
        for event in relevant_events:
            arces_picks = [pick for pick in event.picks if pick.waveform_id.station_code == station_code]
            nested_filtered_events.append(arces_picks)
            
        return nested_filtered_events, inventory

    def transform_events_to_start_and_end_times(self, filtered_events: List[List[Dict]]) -> Tuple[List[UTCDateTime], List[UTCDateTime]]:
        """
        Transform events to their start and end times.

        Args:
            filtered_events (List[List[Dict]]): A list of lists containing filtered event picks.

        Returns:
            Tuple[List[UTCDateTime], List[UTCDateTime]]: A tuple of lists containing start and end times for each event.
        """
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
    def __init__(self, model: Any, scaler: Any, label_maps: dict, cfg: Any):
        """
        Initialize the LiveClassifier object.

        Args:
            model (Any): Pre-trained machine learning model for classification.
            scaler (Any): Scaler object to normalize features.
            label_maps (dict): Dictionary mapping from labels to integers.
            cfg (Any): Configuration object containing various settings.
        """
        self.cfg = cfg
        self.model = model
        self.label_maps = label_maps
        self.scaler = scaler

    def predict(self, trace: np.ndarray) -> Tuple[Any, np.ndarray, List[Any], dict, np.ndarray]:
        """
        Perform event classification for a specified seismic trace.

        Args:
            trace (np.ndarray): Numpy array containing the seismic trace.

        Returns:
            Tuple[Any, np.ndarray, List[Any], dict, np.ndarray]: A tuple containing the final ensemble prediction,
            mean prediction probabilities, individual predictions for each trace segment,
            individual prediction probabilities for each trace segment, and preprocessed input features.
        """
        trace = trace.T
        X = self.prepare_multiple_intervals(trace)
        X = [self.local_minmax(x) for x in X]
        X = np.array(X)
        yhats, yprobas, final_yhat, mean_proba = self.ensamble_predict(self.model, X)
        logger.info(f"Mean proba: {mean_proba}")

        return final_yhat, mean_proba, yhats, yprobas, X
    
    def prepare_multiple_intervals(self, trace: np.ndarray) -> List[np.ndarray]:
        """
        Prepare multiple intervals from a single trace for ensemble prediction.

        Args:
            trace (np.ndarray): Numpy array containing the seismic trace.

        Returns:
            List[np.ndarray]: List of sub-traces.
        """
        traces = []
        # Creates equally sized intervals of the trace, using the user defined step size.
        for start in range(0, len(trace) - (cfg.live.length*cfg.live.sample_rate)+1, (cfg.live.step*cfg.live.sample_rate)+1):
            traces.append(trace[start:(start+cfg.live.length*cfg.live.sample_rate)+1])
        return traces

    def ensamble_predict(self, model: Any, X: np.ndarray) -> Tuple[List[Any], dict, Any, np.ndarray]:
        """
        Perform ensemble prediction on multiple intervals.

        Args:
            model (Any): Pre-trained machine learning model.
            X (np.ndarray): Numpy array of shape (n_samples, timestamps, channels).

        Returns:
            Tuple[List[Any], dict, Any, np.ndarray]: A tuple containing lists of predicted labels,
            dictionary containing prediction probabilities for "detector" and "classifier",
            final ensemble prediction, and mean prediction probabilities.
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
    
    def local_minmax(self, trace: np.ndarray) -> np.ndarray:
        """
        Normalize a trace using min-max scaling.

        Args:
            trace (np.ndarray): Numpy array containing a seismic trace.

        Returns:
            np.ndarray: Normalized trace.
        """
        mmax = np.max(trace)
        mmin = np.min(trace)
        return (trace - mmin) / (mmax - mmin)
    

    def plot_predicted_event(self, intervals: List[np.ndarray], event_time: Optional[UTCDateTime], yprobas: dict, 
                             yhats: List[Any], final_yhat: Any, mean_proba: np.ndarray, station: str = 'AR*') -> None:
        """
        Plot the predicted event with model output.

        Args:
            intervals (List[np.ndarray]): List of intervals (sub-traces).
            event_time (Optional[UTCDateTime]): UTCDateTime of the event.
            yprobas (dict): Dictionary of prediction probabilities.
            yhats (List[Any]): List of individual predictions.
            final_yhat (Any): Final ensemble prediction.
            mean_proba (np.ndarray): Mean prediction probabilities.
            station (str, optional): Station code. Defaults to 'AR*'.
        """
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

    def plot_model_output(self, detector_values: List[float], classifier_values: List[float], current_step: Optional[int], 
                          final_prediction: Any, yhats: List[Any], mean_proba: np.ndarray) -> np.ndarray:
        """
        Create a plot of the model's output.

        Args:
            detector_values (List[float]): List of detector values.
            classifier_values (List[float]): List of classifier values.
            current_step (Optional[int]): Current step index.
            final_prediction (Any): Final ensemble prediction.
            yhats (List[Any]): List of individual predictions.
            mean_proba (np.ndarray): Mean prediction probabilities.

        Returns:
            np.ndarray: An array representing the plotted image.
        """
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





