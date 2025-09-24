from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor
from starvector.data.util import process_and_rasterize_svg
import torch
import time

import sys
import os

model_name = "starvector/starvector-8b-im2svg"

root_dir =  os.path.dirname(os.path.abspath(__file__))
img_dir = os.path.join(root_dir, "inputs")
img_suffix = ".png"
out_dir = os.path.join(root_dir, "outs")
t1 = time.time()
starvector = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, trust_remote_code=True)
processor = starvector.model.processor
tokenizer = starvector.model.svg_transformer.tokenizer
starvector.cuda()
starvector.eval()
t2 = time.time()
print(f"....Loading model cost : {t2-t1} secs !")

countTime,count = 0,0
for i in  os.listdir(img_dir):
    if i.endswith(img_suffix):
        t1 = time.time()
        m = os.path.splitext(i)[0]
        img_path = os.path.join(img_dir, i)
        out_base_path =  os.path.join(out_dir, m)
        #print(f"~~~~ Image : {img_path}")
        image_pil = Image.open(img_path).convert("RGB") 
        #import pdb;pdb.set_trace()
        image = processor(image_pil, return_tensors="pt")['pixel_values'].cuda()
        batch = {"image": image}

        raw_svg = starvector.generate_im2svg(batch, max_length=1000)[0]
        svg, raster_image = process_and_rasterize_svg(raw_svg)
        t2 = time.time()
        tt = t2 - t1
        countTime += tt
        count += 1
        print(f"==========={i}: {tt:4.4f}/s =============")        
        print(svg)
        with open(f"{out_base_path}.svg", 'w') as svg_f:
            svg_f.write(svg)
        raster_image.save(out_base_path+".png")

print(f"Total time: {countTime}, total images : {count}, avg : {countTime/count} /secs")
