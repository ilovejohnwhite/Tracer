# Ecstatic.py
from ..utils import common_annotator_call, create_node_input_types
import comfy.model_management as model_management

class EcstaticNode:
    @classmethod
    def INPUT_TYPES(cls):
        return create_node_input_types()

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"

    CATEGORY = "TatToolkit/Billy Goncho's Wild Ride X"

    def execute(self, image, resolution=2048, **kwargs):
        from TatToolkit.Ecstatic import Ecstatic

        # Instantiate your LinkMaster
        lines_processor = Ecstatic()

        # Process the image
        out = common_annotator_call(lines_processor, image, resolution=resolution, **kwargs)
        del lines_processor
        return (out, )

NODE_CLASS_MAPPINGS = {
    "EcstaticNode": EcstaticNode
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "EcstaticNode": "Shit"
}
