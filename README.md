## Repository Structure
This repository has the following structure:

- `config/`: This directory contains configuration files for the project.
    - `models/`: This directory contains configuration files for the models. The files are named after the model name, and contain the configuration settings for the model. The settings relate specifically to the hyperparameters of the relevant model.
    - `data_config.yaml`: This file contains the configuration settings for the data. It includes the following sections:
        - `data`: This section contains the settings for the data. It includes the following main keys:
        - `model`: Type of model to be used. Currenlty, only `alexnet` and `cnn_dense` is supported.
        - `model_name`: Name of the pretrained model. This is used to load the model weights.
        - `seed` : Set seed for reproducibility.
        - `predict`: Whether or not the model should be used for prediction.
        - `paths`: The paths which the project utilizes. These are extensions of the `ROOT_DIR. More information can be found below regarding setting the environment variable.
        - `data`: Parameters regarding data.
        - `live`: Parameters regarding loading and handling of live data.
        - `filters`: Parameters regarding filtering of data.
        - `callbacks`: Parameters regarding callbacks.
        - `scaling`: Parameters regarding scaling of data.
        - `augment`: Parameters regarding data augmentation.
        - `optimizer`: Parameters regarding the optimizer.
    - `sweep_config`: Paremters for hyperparameter optimization using Weights & Biases (wandb).
    - `logging_config.yaml`: Configuration settings for logging.
- `saved_scalers/`: This directory contains saved scaler files. Not in use as scalers are ran locally and don't need fitting.
- `src/`: This directory contains the source code of the project. It includes the following files:
    - `Live.py`: The most important file in the project, which implements the system for GBF. It includes two main classes:
        - `ClassifyGBF`: A class for processing and classifying seismic data using ground-based facilities. This class includes methods for retrieving seismic data, creating beams for P and S waves, and predicting seismic events using the ground-based facilities (GBF) approach.
        - `LiveClassifier`: A class that handles the live classification of seismic data. It includes methods for loading the model, preprocessing the data, and making predictions in real-time.
    - `Scaler.py`: This file contains various scalers available to the model. It includes four main classes:
        - `RobustScaler`: A class that scales features using statistics that are robust to outliers. This Scaler removes the median and scales the data according to the Interquartile Range (IQR). The IQR is the range between the 1st quartile (25th quantile) and the 3rd quartile (75th quantile).
        - `LogScaler`: A class that scales the features using logarithmic scaling. This Scaler applies a logarithmic transformation to the data, which can be useful for reducing the impact of outliers and transforming the data to a more Gaussian-like distribution.
        - `MinMaxScaler`: A class that scales and translates each feature individually such that it is in the given range on the training set, e.g. between zero and one.
        - `StandardScaler`: A class that standardizes features by removing the mean and scaling to unit variance. The standard score of a sample x is calculated as z = (x - u) / s, where u is the mean of the training samples, and s is the standard deviation of the training samples.
    - `Models.py`: This file defines the architecture of neural network models for seismic event classification and detection. It includes key functions   like get_initializer for selecting weight initializers and get_model for creating model instances based on configuration settings. The file also contains custom model classes AlexNet and CNN_dense, tailored for handling seismic data. These classes extend from a base Loop class and implement specific structures and behaviors, including convolutional and dense layers, for effective seismic event analysis.
    - `Utils.py`: Contains functions used throughout the project.
    - `Loop.py`: Central to the model's training and evaluation processes, this file defines the Loop class, which extends tf.keras.Model. It implements custom training and testing steps, loss calculation, and metrics handling for both detector and classifier components of the model. The class also integrates with Weight & Biases (wandb) for metric tracking. Additionally, it includes the MetricsPipeline class, designed to initialize, update, and retrieve results for various metrics like precision, recall, and accuracy. The Loop class handles the complexities of training, including handling class weights, updating state for metrics, and customizing gradient updates and loss calculations specific to the project's needs.
    - `Analysis.py`: Defines the Analysis class for evaluating and visualizing model performance. It includes methods for plotting seismic event samples, generating confusion matrices, precision-recall curves, and geographical maps of predictions. The class also provides exploratory analysis tools for specific seismic event types and detailed error analysis functionalities.
    - `Augment.py`: Provides functions for data augmentation in seismic signal processing. This includes adding noise, tapering signals, creating gaps, and zeroing specific channels in the data. These augmentations are configurable through the `cfg` settings, allowing for flexible application to various datasets. The augmentation techniques enhance the robustness and generalizability of the model by introducing variability in the training data.
    - `Callbacks.py`: Defines various TensorFlow Keras callbacks for enhancing the model training and evaluation process. These include:
        - **`ValidationConfusionMatrixCallback`**: Calculates and logs confusion matrices and prediction distributions at the end of each epoch, useful for monitoring model performance and biases.
        - **`InPlaceProgressCallback`**: Provides in-place progress updates during training, displaying current epoch progress and average training loss.
        - **`WandbLoggingCallback`**: Integrates with Weights & Biases (wandb) for logging training and validation metrics, aiding in detailed monitoring and analysis of model performance over epochs.
        - **`CosineAnnealingLearningRateScheduler`**: Implements a cosine annealing schedule for the learning rate, which can help in stabilizing and improving the training process.
    
    - `Generator.py`: Defines `Generator`, `TrainGenerator`, and `ValGenerator` classes for efficient data handling and batch generation in TensorFlow. These classes extend `tf.keras.utils.Sequence` and are crucial for loading and preprocessing data during model training and validation.

        - **`Generator` Class**: Serves as a base class for creating data generators. It handles data shuffling, normalization (using a provided scaler), and batch-wise data retrieval. It also supports chunk-wise processing of large datasets to optimize memory usage.

        - **`TrainGenerator` Class**: Inherits from `Generator` and is tailored for training data. It includes data augmentation methods to introduce variability and robustness in the training process. It ensures that data is shuffled and augmented in each epoch.

        - **`ValGenerator` Class**: Also inheriting from `Generator`, this class is designed for validation data. It ensures that data is not shuffled or augmented, providing a consistent set of data for evaluating the model's performance.


    - `LoadData.py`: Implements the `LoadData` class for managing the loading, preprocessing, and filtering of seismic data for machine learning models. The class is designed to handle different datasets (training, validation, test) and applies necessary preprocessing steps, such as filtering, based on the configuration specified in `data_config.py`.

        - **Initialization and Data Processing**: The constructor initializes the class and decides which datasets (train, validation, test) to load based on the `cfg.data.what_to_load` setting. It loads the datasets and applies filtering as required.

        - **Dataset Retrieval Methods**: Methods like `get_train_dataset`, `get_val_dataset`, and `get_test_dataset` provide easy access to the processed datasets.

        - **Data Filtering**: The `filter_data` method applies the configured filters to the seismic data. This is important for preparing the data for model training and ensuring data quality.

        - **Loading and Saving Datasets**: Methods like `load_data` and `save_filtered_data` handle the reading and writing of datasets to and from the disk, respectively.

        - **Utility Functions**: Functions like `remove_induced_events` and `change_key_to_index` perform specific transformations on the data, aiding in data preparation and management.

    - `MetricsCallback.py`: Evaluate if this is no longer used.
    - `Metrics.py`: Evaluate if this is no longer used.
    - `UMAPCallback.py`: Defines the `UMAPCallback` class, a custom callback for TensorFlow Keras models. This callback integrates Uniform Manifold Approximation and Projection (UMAP) for dimensionality reduction and visualization, and logs the results to Weights & Biases (wandb). 

        - **Initialization**: During the initialization phase, the callback takes validation data and labels, a label map for human-readable label names, and an interval specifying how frequently the UMAP visualization should be generated.

        - **Epoch-End Action**: The `on_epoch_end` method, which is triggered at the end of each epoch, checks if the current epoch aligns with the specified interval. If it does, the method extracts outputs from the model's penultimate layer, applies UMAP to reduce the dimensionality of these outputs, and generates a scatter plot.

        - **Visualization and Logging**: The scatter plot, where each point represents a validation data point and is color-coded based on its true label, is saved to a file. This file is then logged to wandb, providing a visual representation of how the model's embeddings of the validation data evolve over epochs.
- `docker.dockerfile`: This is the Dockerfile for running the project on the GPU machine.
- `data_analysis.ipynb`: This Jupyter notebook contains analysis code for looking at the model predictions.
- `live.ipynb`: This Jupyter notebook contains scratch code for developing the live.py file.
- `README.md`: This is the README file for the project.
- `export_data.py`: Script used by Steffen Mæland to generate the original datset.
- `gbf_iter.py`: This script is an example used to run the system, using command line inputs and keeping the model in memory.
- `gbf_live.py`: This script is an example used to run the system on a specified time interval.
- `generate_data_csv.py`: This script generates a CSV file from the model predictions for analysis.
- `generate_datasets.py`: This script generates preloaded training, validation and test datasets. Used when a new dataset has been added. 
- `get_model_name.py`: This script gets the model name for the bash scripts.
- `global_config.py`: This script contains two lines of code to make the variables set in `project_setup.py` accessible across the project.
- `predict.py`: This script runs the model exlusively on the validation set.
- `project_setup.py`: This script is the setup file for the project. It loads the config files and addresses project file paths on different systems. It also loads the logger, and assigns a unique ID for the run.
- `train.py`: Training script for one model.
- `sweep_train.py`: This unfinished script is used to perform a sweep run, or hyperparamter optimization using WandB.
- `common.sh`: This shell script handles creating folders and transferring files to the GPU machine. It then also runs the relevant code on the GPU machine. This script is used by all other shell scripts.
- `run_live.sh`: This shell script is a wrapper for running the live version of the model on the GPU machine. May be deprecated.
- `run_predict.sh`: This shell script runs the `predict.py` script on the GPU machine.
- `run.sh`: This shell script runs the `train.py` script on the GPU machine. 
- `sweep_run.sh`: This shell script runs the `sweep_train.py` script on the GPU machine. May be deprecated.
- `requirements.txt`: This file lists the Python dependencies for the project. Not all modules are necessary for the live version. This needs to be addressed at some point.

## Setting Up the Environment Variable

The application uses an environment variable `ROOT_DIR` to determine the root directory of the project. You need to set this environment variable before running the application.

### On Unix/Linux/macOS

Open your terminal and enter the following command:

```bash
export ROOT_DIR=/path/to/your/root/directory
```	

Replace /path/to/your/root/directory with the actual path to your root directory.

Please note that these commands will only set the `ROOT_DIR` environment variable for the current session. If you open a new terminal or Command Prompt window, you will need to set the environment variable again.

To set the environment variable permanently, you can add the above command to your shell's startup file (like `~/.bashrc` or `~/.bash_profile` on Unix/Linux/macOS, or the Environment Variables on Windows).

### On Windows
Open Command Prompt and enter the following command:

Replace `C:\path\to\your\root\directory` with the actual path to your root directory.

## Training a model

### On GPU machine (recommended)
To train a model, specify the parameters in the `config/data_config.yaml` file. Then edit the run.sh file to specify the name of the model you will train (this name should be the same in `data_config.yaml` - model_name). Then run the following command:

```bash
cd ROOT_DIR
bash run.sh

```
If you added any new external models to the requirements.txt file, you need to add the -b flag to the above command to install them.


#### On your local machine:
To train a model, specify the parameters in the `config/data_config.yaml` file. Then run the following command:

```
cd ROOT_DIR
python train.py
```

#### For both cases:
When the training is finished, the model will be saved in the directory defined by cfg.paths.model_save_folder in `data_config.yaml`. The model will be given a name that is defined by: `f"{cfg.model}_{haikunator.haikunate()}"`. The haikunator package is used to generate a random name for the model.

## Predicting with a model on validation set:
This depends on what you want to predict on. To predict on the validation set, you need to run the `predict.py` file.

#### On GPU machine:
**Step 1:** Specify the model config parameter in `predict.sh` file. This file should be accessible through `config/models/` directory. For example, if you want to predict on a model named "model_1", then you should have a file named `model_1.yaml` in `config/models/` directory. This file should contain the parameters of the model you want to predict on.
**Step 2:** Specify the name of the model weights file in `data_config.yaml`. E.g.: model_name: `"best_alexnet_epoch_30.hdf5"`
**Step 3:** Run the following command:

```bash
cd ROOT_DIR
bash predict.sh
```
You can specify the -b flag to install any new external models you added to the requirements.txt file.

#### On your local machine:
**Step 1:** Specify the name of the model weights file in `data_config.yaml`. E.g.: model_name: `"best_alexnet_epoch_30.hdf5"`
**Step 2:** Run the following command:
    
```bash
cd ROOT_DIR
python predict.py
```

## Running the model on GBF:
There are two options to run the model on GBF. The first option is to specify the time period you're interested in through the `gbf_live.py` file. The second option is to run `gbf_iter.py `where the script will prompt your for time periods. Currently, the output of both of these processeses is the same: gifs of the predictions are saved in the directory defined by `cfg.paths.live_test_path` in `data_config.yaml`. I recommend running on local machine, as GPU is overkill for this task.

Run the command:
    
```bash
cd ROOT_DIR
python gbf_live.py
```
Alternatively, you can run:

```bash
cd ROOT_DIR
python gbf_iter.py
```





