import os
import gdown
import zipfile
from pathlib import Path
from huggingface_hub import hf_hub_download

SADTALKER_URLS = {
    'base': 'https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2',
    'files': [
        'epoch_20.pth',
        'facevid2vid_00189-model.pth.tar',
        'mapping_00109-model.pth.tar',
        'mapping_00229-model.pth.tar',
        'shape_predictor_68_face_landmarks.dat'
    ]
}

async def download_model(model_name):
    """Download model files from various sources"""
    if model_name.lower() == 'sadtalker':
        checkpoint_path = Path('checkpoints')
        checkpoint_path.mkdir(exist_ok=True)
        
        # Download from HuggingFace
        try:
            for file in SADTALKER_URLS['files']:
                output_path = checkpoint_path / file
                if not output_path.exists():
                    print(f"Downloading {file}...")
                    url = f"{SADTALKER_URLS['base']}/{file}"
                    gdown.download(url, str(output_path), quiet=False)
                    
            print("Model files downloaded successfully!")
            return True
            
        except Exception as e:
            print(f"Error downloading model files: {str(e)}")
            return False
            
    else:
        raise ValueError(f"Unknown model: {model_name}")

def extract_zip(zip_path, extract_path):
    """Extract zip file to specified path"""
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    os.remove(zip_path)  # Clean up zip file after extraction