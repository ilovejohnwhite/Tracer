"""
This file contains a Processor that can be used to process images with controlnet aux processors
"""
import io
import logging
from typing import Dict, Optional, Union

from PIL import Image
from TatToolkit.SuckerPunch import SuckerPunchPro
from TatToolkit.LinkMaster import LinkMaster
from TatToolkit.BillyGoat import BillyGoat
from TatToolkit.VooDoo import VooDoo
LOGGER = logging.getLogger(__name__)


MODELS = {
    # checkpoint models
    'lineart_coarse': {'class': LineartDetector, 'checkpoint': True},
    'lineart_realistic': {'class': LineartDetector, 'checkpoint': True},
    'lineart_anime': {'class': LineartAnimeDetector, 'checkpoint': True},
    # instantiate
    'canny': {'class': CannyDetector, 'checkpoint': False},
    'tile': {'class': TileDetector, 'checkpoint': False},
}
MODELS.update({
    'SuckerPunch': {'class': SuckerPunchPro, 'checkpoint': False},
    'LinkMaster': {'class': LinkMaster, 'checkpoint': False},
    'BillyGoat': {'class': BillyGoat, 'checkpoint': False},
    'VooDoo': {'class': VooDoo, 'checkpoint': False},
})


MODEL_PARAMS = {
    'scribble_pidinet': {'safe': False, 'scribble': True},
    'lineart_realistic': {'coarse': False},
    'lineart_coarse': {'coarse': True},
    'lineart_anime': {},
    'canny': {},
}

MODEL_PARAMS.update({
    'SuckerPunch': {},
    'LinkMaster': {},
    'VooDoo': {},
    'BillyGoat': {},
})

CHOICES = f"Choices for the processor are {list(MODELS.keys())}"


class Processor:
    def __init__(self, processor_id: str, params: Optional[Dict] = None) -> None:
        """Processor that can be used to process images with controlnet aux processors

        Args:
            processor_id (str): processor name, options are 'hed, midas, mlsd, openpose,
                                pidinet, normalbae, lineart, lineart_coarse, lineart_anime,
                                canny, content_shuffle, zoe, mediapipe_face, tile'
            params (Optional[Dict]): parameters for the processor
        """
        LOGGER.info("Loading %s".format(processor_id))

        if processor_id not in MODELS:
            raise ValueError(f"{processor_id} is not a valid processor id. Please make sure to choose one of {', '.join(MODELS.keys())}")

        self.processor_id = processor_id
        self.processor = self.load_processor(self.processor_id)

        # load default params
        self.params = MODEL_PARAMS[self.processor_id]
        # update with user params
        if params:
            self.params.update(params)

    def load_processor(self, processor_id: str) -> 'Processor':
        """Load controlnet aux processors

        Args:
            processor_id (str): processor name

        Returns:
            Processor: controlnet aux processor
        """
        processor = MODELS[processor_id]['class']

        # check if the proecssor is a checkpoint model
        if MODELS[processor_id]['checkpoint']:
            processor = processor.from_pretrained("lllyasviel/Annotators")
        else:
            processor = processor()
        return processor

    def __call__(self, image: Union[Image.Image, bytes],
                 to_pil: bool = True) -> Union[Image.Image, bytes]:
        """processes an image with a controlnet aux processor

        Args:
            image (Union[Image.Image, bytes]): input image in bytes or PIL Image
            to_pil (bool): whether to return bytes or PIL Image

        Returns:
            Union[Image.Image, bytes]: processed image in bytes or PIL Image
        """
        # check if bytes or PIL Image
        if isinstance(image, bytes):
            image = Image.open(io.BytesIO(image)).convert("RGB")

        processed_image = self.processor(image, **self.params)

        if to_pil:
            return processed_image
        else:
            output_bytes = io.BytesIO()
            processed_image.save(output_bytes, format='JPEG')
            return output_bytes.getvalue()
