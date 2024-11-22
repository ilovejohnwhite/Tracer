from ..utils import common_annotator_call, define_preprocessor_inputs, INPUT
import comfy.model_management as model_management

class OutlineStandardNode:
    @classmethod
    def INPUT_TYPES(s):
        return define_preprocessor_inputs(
            guassian_sigma=INPUT.FLOAT(default=1.0, max=100.0),
            intensity_threshold=INPUT.INT(default=2, max=16),
            resolution=INPUT.RESOLUTION()
        )

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"

    CATEGORY = "TatToolkit/Billy Goncho's Wild Ride X"

    def execute(self, image, guassian_sigma=1, intensity_threshold=2, resolution=2048, **kwargs):
        from TatToolkit.OutlineStandard import OutlineStandard
        return (common_annotator_call(OutlineStandard(), image, guassian_sigma=guassian_sigma, intensity_threshold=intensity_threshold, resolution=resolution), )

NODE_CLASS_MAPPINGS = {
    "OutlineStandardNode": OutlineStandardNode
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "OutlineStandardNode": "Standard outline"
}