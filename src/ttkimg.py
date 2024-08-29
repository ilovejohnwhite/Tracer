import os

import model_management
import torch
import comfy.sd
import comfy.utils
import folder_paths
import comfy.samplers
from nodes import common_ksampler
from comfy_extras.chainner_models import model_loading

from PIL import Image, ImageOps, ImageFilter, ImageDraw
from PIL.PngImagePlugin import PngInfo
import numpy as np
from torchvision.transforms import ToPILImage
#import cv2
#from deepface import DeepFace

import re
import random
import latent_preview
from datetime import datetime
import json
import piexif
import piexif.helper


MAX_RESOLUTION=8192

# Tensor to PIL
def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))
            
# Convert PIL to Tensor
def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)


class TTKImageTracer:
    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        return {"required":
                    {
                        "verbose": (["true", "false"],),
                        "image": (sorted(files), {"image_upload": True}),
                    },
                }

    CATEGORY = "TatToolkit/Billy Goncho's Wild Ride"
    ''' Return order:
        width(int), height(int)
    '''
    RETURN_TYPES = ("IMAGE", "INT", "INT")
    RETURN_NAMES = ("image", "width", "height")
    FUNCTION = "get_image_data"

    def get_image_data(self, image, verbose):
        image_path = folder_paths.get_annotated_filepath(image)
        with open(image_path,'rb') as file:
            img = Image.open(file)
            extension = image_path.split('.')[-1]
            image = img.convert("RGB")
            image = np.array(image).astype(np.float32) / 255.0
            image = torch.from_numpy(image)[None,]
                
        if(verbose == "true"):
            print("gg")

        try:
            width,height = img.size
        except:
            width,height = 512,512
        
        ''' Return order:
            width(int), height(int)
        '''

        return (image, width, height)

    @classmethod
    def IS_CHANGED(s, image,verbose):
        image_path = folder_paths.get_annotated_filepath(image)
        m = hashlib.sha256()
        with open(image_path, 'rb') as f:
            m.update(f.read())
        return m.digest().hex()

    @classmethod
    def VALIDATE_INPUTS(s, image):
        if not folder_paths.exists_annotated_filepath(image):
            return "Invalid image file: {}".format(image)

        return True

NODE_CLASS_MAPPINGS = {
        "Image Load TTK": TTKImageTracer
}
