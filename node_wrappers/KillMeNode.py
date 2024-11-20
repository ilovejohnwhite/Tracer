from ..utils import common_annotator_call, create_node_input_types
import comfy.model_management as model_management

class KillMeNode:
    @classmethod
    def INPUT_TYPES(cls):
        return create_node_input_types()

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"

    CATEGORY = "TatToolkit/Billy Goncho's Wild Ride X"

    def execute(self, image, resolution=2048, **kwargs):
        from TatToolkit.KillMe import KillMe

        lines_processor = KillMe()

        out = common_annotator_call(lines_processor, image, resolution=resolution, **kwargs)
        del lines_processor
        return (out, )

NODE_CLASS_MAPPINGS = {
    "KillMeNode": KillMeNode
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "KillMeNode": "Cap"
}
