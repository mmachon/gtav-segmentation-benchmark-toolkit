import os
from tqdm import tqdm

files = os.listdir("onnx_export/")

for file in tqdm(files):
    os.system(f"python3 -m onnxsim onnx_export/{file} onnx_simp/simp-{file} --input-shape \"1,384,384,3\"")