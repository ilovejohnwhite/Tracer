from ..utils import common_annotator_call, define_preprocessor_inputs, INPUT
import comfy.model_management as model_management

class OutlineRealNode:
    @classmethod
    def INPUT_TYPES(s):
        return define_preprocessor_inputs(
            coarse=INPUT.COMBO((["disable", "enable"])),
            resolution=INPUT.RESOLUTION()
        )

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"

    CATEGORY = "TatToolkit/Billy Goncho's Wild Ride X"

    def execute(self, image, resolution=2048, **kwargs):
        from TatToolkit.OutlineReal import OutlineReal

        model = OutlineReal.from_pretrained().to(model_management.get_torch_device())
        out = common_annotator_call(model, image, resolution=resolution, coarse = kwargs["coarse"] == "disable")
        del model
        return (out, )

NODE_CLASS_MAPPINGS = {
    "OutlineRealNode": OutlineRealNode
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "OutlineRealNode": "Real Outline"
}