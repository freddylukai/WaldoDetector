import os
from PIL import Image

INPUT_DIR = r"E:\COS 429\FinalProj\OriginalIms"
OUTPUT_DIR = r"E:\COS 429\FinalProj\OriginalJPG"

for file in os.listdir(INPUT_DIR):
    f = Image.open(INPUT_DIR+"/"+file)
    f = f.convert("RGB")
    f.save(OUTPUT_DIR+"/"+file+".jpg", "JPEG")