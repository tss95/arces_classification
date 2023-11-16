import yaml
import sys

def get_model_name(yaml_file):
    with open(yaml_file, 'r') as file:
        data = yaml.safe_load(file)
        print(data['model'])  # Adjust the key based on your YAML structure

if __name__ == "__main__":
    get_model_name(sys.argv[1])