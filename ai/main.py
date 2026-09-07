import json
import os

def load_rust_pipeline_output(json_path="image-reader/filtered_images.json"):
    """Ingeests the high-speed JSON payload produced by Rust Stage 1."""
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Pipeline payload missing: {json_path}. Run the Rust engine first.")
    
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)

if __name__ == "__main__":
    image_records = load_rust_pipeline_output()
    print(image_records)