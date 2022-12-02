import os 
import yaml
import argparse 


def yaml_config_hook(config_file) : 
    with open(config_file) as f: 
        config = yaml.safe_load(f)
        for d in config.get("defaults", []) : 
            config_dir, cf = d.popitem()
            cf = os.path.join(os.path.dirname(config_file), config_dir, cf + '.yaml')
            with open(cf) as f : 
                l = yaml.safe_load(f)
                config.update(l)
                
    if "defaults" in config.keys():
        del config["defaults"]
        
    return config


def get_args(filename = 'config/mnist.yaml') : 
    parser = argparse.ArgumentParser()
    config = yaml_config_hook(filename)
    for k, v in config.items() : 
        parser.add_argument(f'--{k}', default=v, type = type(v))
    args = parser.parse_args()
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)
        
    return args