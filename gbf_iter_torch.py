from src.Live import ClassifyGBF, LiveClassifier
from obspy import UTCDateTime
from src.Scaler_torch import Scaler
from global_config import cfg, logger
import torch
import argparse
from src.Models_torch import AlexNet1D
import os

# Initialize argument parser
parser = argparse.ArgumentParser(description="Run GBF model on specified time intervals.")
parser.add_argument("--plots", action="store_true", help="Generate plots if this flag is set", default=True)
args = parser.parse_args()

scaler = Scaler(cfg)

detector_label_map = {0: "noise", 1: "event"}
classifier_label_map = {0:"earthquake", 1:"explosion"}
label_maps = {"detector": detector_label_map, "classifier": classifier_label_map}


detector_class_weight_dict = {"noise": 1.0, "event": 1.0}
classifier_class_weight_dict = {"earthquake": 1.0, "explosion": 1.0}
detector_metrics_list = ["auroc","accuracy"]
classifier_metrics_list = ["auroc","accuracy"]
expected_timesteps = int(cfg.data.window_seconds * cfg.data.sample_rate)
input_data = torch.randn((3, expected_timesteps), device="cuda" if torch.cuda.is_available() else "cpu")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = AlexNet1D(input_data.shape, 
                  [],
                  [],
                  detector_label_map, 
                  classifier_label_map, 
                  detector_class_weight_dict, 
                  classifier_class_weight_dict, 
                  cfg, 
                  kernel_sizes=None,
                  filters=None,
                  pooling='max')
# Load the pretrained weights
pretrained_weights_path =cfg.pretrained_model_name
if os.path.exists(pretrained_weights_path):
    checkpoint = torch.load(pretrained_weights_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    if "scaler_state" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler_state"])
    elif scaler.requires_fit:
        logger.warning("Scaler requires fitted parameters but none were found in checkpoint; predictions may be inconsistent.")
    model.load_state_dict(checkpoint['state_dict'])
    logger.info(f"Loaded pretrained weights from {pretrained_weights_path}")
else:
    logger.warning(f"Pretrained weights file not found at {pretrained_weights_path}. Using randomly initialized weights.")

# Set the model to evaluation mode
model.eval()
model.to(device)

# Load model only once
model = LiveClassifier(model, scaler, label_maps, cfg)

def get_user_input():
    while True:
        try:
            #start = input("Enter start time (YYYY-MM-DDTHH:MM:SS): ")
            #end = input("Enter end time (YYYY-MM-DDTHH:MM:SS): ")
            start = "2024-01-24T00:00:00"
            end = "2024-01-24T02:00:00"
            
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
