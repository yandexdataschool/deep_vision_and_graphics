import json


def load_config(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    if 'class_mapping' in config.get('data', {}):
        config['data']['class_mapping'] = {
            int(k): v for k, v in config['data']['class_mapping'].items()
        }
    
    return config

