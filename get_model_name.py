import yaml
import sys

def get_model_name(yaml_file):
    with open(yaml_file, 'r') as file:
        data = yaml.safe_load(file)
        print(data['model'])

if __name__ == "__main__":
    get_model_name(sys.argv[1])