from Classes.Live import ClassifyGBF, load_model, LiveClassifier
from obspy import UTCDateTime
from Classes.Scaler import Scaler
from global_config import cfg, logger
import argparse

# Initialize argument parser
parser = argparse.ArgumentParser(description="Run GBF model on specified time intervals.")
parser.add_argument("--plots", action="store_true", help="Generate plots if this flag is set")
args = parser.parse_args()

# Load model only once
model, label_maps = load_model()
model = LiveClassifier(model, Scaler(), label_maps, cfg)

def get_user_input():
    while True:
        try:
            start = input("Enter start time (YYYY-MM-DDTHH:MM:SS): ")
            end = input("Enter end time (YYYY-MM-DDTHH:MM:SS): ")
            return UTCDateTime(start), UTCDateTime(end)
        except Exception as e:
            print(f"Invalid input: {e}. Please try again.")

def process_data(starttime, endtime):
    try:
        classify = ClassifyGBF()
        tracedata, streams, starttimes, endtimes = classify.get_data_to_predict(starttime, endtime)

        final_classifications = []
        mean_probas = []
        for idx, trace in enumerate(tracedata):
            print(f"Processing data for interval starting at {starttimes[idx]}...")
            final_yhat, mean_proba, yhats, yprobas, intervals = model.predict(trace)
            final_classifications.append(final_yhat)
            mean_probas.append(mean_proba)
            if args.plots:
                model.plot_predicted_event(intervals, starttimes[idx], yprobas, yhats, final_yhat, mean_proba)

        logger.info(f"All classifications: {final_classifications}")
        logger.info(f"Mean probas: {mean_probas}")

    except Exception as e:
        logger.error(f"An error occurred during processing: {e}")

# Main loop
while True:
    starttime, endtime = get_user_input()
    process_data(starttime, endtime)

    if input("Do you want to process another period? (yes/no): ").lower() != "yes":
        break