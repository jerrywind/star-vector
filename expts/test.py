from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor
from starvector.data.util import process_and_rasterize_svg
import torch
import time
import os

def remove_transparency(inImage, bgColor=(255, 255, 255)):
    # Only process if image has transparency (http://stackoverflow.com/a/1963146)
    if inImage.mode in ('RGBA', 'LA') or (inImage.mode == 'P' and 'transparency' in inImage.info):
        png = inImage.convert('RGBA')
        background = Image.new('RGBA', png.size, bgColor)
        alphaComposite = Image.alpha_composite(background, png)
        # if you check the matrix dimension, channel, it would be still 4.  
        rgbImage = alphaComposite.convert('RGB')
        return rgbImage
    else:
        return inImage


root_dir =  os.path.dirname(os.path.abspath(__file__))
img_dir = os.path.join(root_dir, "inputs")
img_suffix = ".png"
out_dir = os.path.join(root_dir, "outs")

model_name = "starvector/starvector-8b-im2svg"
t1 = time.time()
starvector = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, trust_remote_code=True)
processor = starvector.model.processor
tokenizer = starvector.model.svg_transformer.tokenizer
starvector.cuda()
starvector.eval()

t2 = time.time()

countTime, count = 0, 0

print(f"....Loading model cost : {(t2-t1):4.4f} secs !")

for i in  os.listdir(img_dir):
    if i.endswith(img_suffix):
        t1 = time.time()
        m = os.path.splitext(i)[0]
        img_path = os.path.join(img_dir, i)
        out_base_path =  os.path.join(out_dir, m)

        image_pil = Image.open(img_path)
        image_fixed = remove_transparency(image_pil)

        image = processor(image_fixed, return_tensors="pt")['pixel_values'].cuda()
        if not image.shape[0] == 1:
            image = image.squeeze(0)
        
        batch = {"image": image}

        raw_svg = starvector.generate_im2svg(batch, max_length=4000)[0]
        svg, raster_image = process_and_rasterize_svg(raw_svg)

        t2 = time.time()
        tt = t2 - t1
        countTime += tt
        count += 1
        print(f"==========={i}: {tt:4.4f}/s =============")        
        #print(svg)
        with open(f"{out_base_path}.svg", 'w') as svg_f:
            svg_f.write(svg)
        raster_image.save(out_base_path+".png")

"""

image_pil = Image.open('../assets/examples/sample-18.png')

image = processor(image_pil, return_tensors="pt")['pixel_values'].cuda()
if not image.shape[0] == 1:
    image = image.squeeze(0)
batch = {"image": image}

raw_svg = starvector.generate_im2svg(batch, max_length=4000)[0]
svg, raster_image = process_and_rasterize_svg(raw_svg)

"""
