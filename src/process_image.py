from PIL import Image
import sys

def process_image(input_path, output_path):
    # Open the image
    img = Image.open(input_path)
    
    # Convert to RGB if needed
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Calculate dimensions for square crop
    width, height = img.size
    size = min(width, height)
    
    # Calculate crop box (center crop)
    left = (width - size) // 2
    top = (height - size) // 2
    right = left + size
    bottom = top + size
    
    # Crop to square
    img = img.crop((left, top, right, bottom))
    
    # Resize to recommended size (512x512)
    img = img.resize((512, 512), Image.Resampling.LANCZOS)
    
    # Save processed image
    img.save(output_path, quality=95)
    print(f"Image processed and saved to {output_path}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python process_image.py <input_path> <output_path>")
        sys.exit(1)
    
    process_image(sys.argv[1], sys.argv[2])