## Setting Up the Environment Variable

The application uses an environment variable `ROOT_DIR` to determine the root directory of the project. You need to set this environment variable before running the application.

### On Unix/Linux/macOS

Open your terminal and enter the following command:

```bash
export ROOT_DIR=/path/to/your/root/directory
```	

Replace /path/to/your/root/directory with the actual path to your root directory.

Please note that these commands will only set the ROOT_DIR environment variable for the current session. If you open a new terminal or Command Prompt window, you will need to set the environment variable again.

To set the environment variable permanently, you can add the above command to your shell's startup file (like ~/.bashrc or ~/.bash_profile on Unix/Linux/macOS, or the Environment Variables on Windows).

### On Windows
Open Command Prompt and enter the following command:

Replace C:\path\to\your\root\directory with the actual path to your root directory.

## Training a model

### On GPU machine (recommended)
To train a model, specify the parameters in the config/data_config.yaml file. Then edit the run.sh file to specify the name of the model you will train (this name should be the same in data_config.yaml - model_name). Then run the following command:

```bash
cd ROOT_DIR
bash run.sh

```
If you added any new external models to the requirements.txt file, you need to add the -b flag to the above command to install them.


#### On your local machine:
To train a model, specify the parameters in the config/data_config.yaml file. Then run the following command:

```
cd ROOT_DIR
python train.py
```

#### For both cases:
When the training is finished, the model will be saved in the directory defined by cfg.paths.model_save_folder in data_config.yaml. The model will be given a name that is defined by: f"{cfg.model}_{haikunator.haikunate()}". The haikunator package is used to generate a random name for the model.

## Predicting with a model on validation set:
This depends on what you want to predict on. To predict on the validation set, you need to run the predict.py file.

#### On GPU machine:
**Step 1:** Specify the model config parameter in predict.sh file. This file should be accessible through config/models/ directory. For example, if you want to predict on a model named "model_1", then you should have a file named model_1.yaml in config/models/ directory. This file should contain the parameters of the model you want to predict on.
**Step 2:** Specify the name of the model weights file in data_config.yaml. E.g.: model_name: "best_alexnet_epoch_30.hdf5"
**Step 3:** Run the following command:

    ```bash
    cd ROOT_DIR
    bash predict.sh
    ```
You can specify the -b flag to install any new external models you added to the requirements.txt file.

#### On your local machine:
**Step 1:** Specify the name of the model weights file in data_config.yaml. E.g.: model_name: "best_alexnet_epoch_30.hdf5"
**Step 2:** Run the following command:
        
    ```bash
    cd ROOT_DIR
    python predict.py
    ```

## Running the model on GBF:
There are two options to run the model on GBF. The first option is to specify the time period you're interested in through the gbf_live.py file. The second option is to run gbf_iter.py where the script will prompt your for time periods. Currently, the output of both of these processeses is the same: gifs of the predictions are saved in the directory defined by cfg.paths.live_test_path in data_config.yaml. I recommend running on local machine, as GPU is overkill for this task.

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




