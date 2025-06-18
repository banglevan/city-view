import yaml
import os
from pathlib import Path


def get_config():
    """
    Read configuration from config.yaml file
    Returns:
        dict: Configuration dictionary containing inference settings
    """
    config_path = Path(__file__).parent / "config.yaml"
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
    
    return config


def get_inference_config():
    """
    Get inference configuration specifically
    Returns:
        dict: Inference configuration settings
    """
    config = get_config()
    return config.get('inference', {})


if __name__ == "__main__":
    # Test the configuration reading
    try:
        config = get_config()
        print("Full configuration:")
        print(yaml.dump(config, default_flow_style=False))
        
        print("\nInference configuration:")
        inference_config = get_inference_config()
        print(yaml.dump(inference_config, default_flow_style=False))
        
    except Exception as e:
        print(f"Error reading configuration: {e}")
