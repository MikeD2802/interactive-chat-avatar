import os
import subprocess
import sys

def install_huggingface_cli():
    print("Installing Hugging Face CLI...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-U", "huggingface_hub[cli]"])

def download_weights():
    print("Creating pretrained_weights directory...")
    os.makedirs("pretrained_weights", exist_ok=True)
    
    print("\nDownloading LivePortrait weights...")
    try:
        subprocess.check_call([
            "huggingface-cli", "download",
            "KwaiVGI/LivePortrait",
            "--local-dir", "pretrained_weights",
            "--exclude", "*.git*", "README.md", "docs"
        ])
        print("\nDownload completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"\nError downloading weights: {e}")
        print("\nTrying alternative download method using hf-mirror...")
        try:
            os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
            subprocess.check_call([
                "huggingface-cli", "download",
                "KwaiVGI/LivePortrait",
                "--local-dir", "pretrained_weights",
                "--exclude", "*.git*", "README.md", "docs"
            ])
            print("\nDownload completed successfully using hf-mirror!")
        except subprocess.CalledProcessError as e2:
            print(f"\nError downloading weights using hf-mirror: {e2}")
            print("\nPlease download manually from:")
            print("1. Google Drive: https://drive.google.com/drive/folders/1UtKgzKjFAOmZkhNK-OYT0caJ_w2XAnib")
            print("2. Baidu Yun: https://pan.baidu.com/s/1MGctWmNla_vZxDbEp2Dtzw?pwd=z5cn")

if __name__ == "__main__":
    install_huggingface_cli()
    download_weights()