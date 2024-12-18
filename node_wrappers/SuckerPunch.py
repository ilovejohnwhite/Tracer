from ..utils import common_annotator_call, annotator_ckpts_path, HF_MODEL_NAME, create_node_input_types
import comfy.model_management as model_management

class SuckerPunch:
    @classmethod
    def INPUT_TYPES(cls):
        return create_node_input_types(
            n_clusters=(
                [("Light"), ("Normal"), ("Heavy"), ("Unhinged")],
                {"default": "Normal"}  # Assuming you can set defaults by label. Adjust as needed.
            )
        )

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"

    CATEGORY = "TatToolkit/Billy Goncho's Wild Ride X"

    def execute(self, image, resolution=512, **kwargs):
        from TatToolkit.SuckerPunch import SuckerPunchPro

        cluster_map = {"Light": 3, "Normal": 4, "Heavy": 5, "Unhinged": 8}
        label = kwargs.get("n_clusters", "Normal")
        n_clusters = cluster_map.get(label, 4)  

        model = SuckerPunchPro(n_clusters=n_clusters)

        out = common_annotator_call(model, image, resolution=resolution)
        del model
        return (out,)


NODE_CLASS_MAPPINGS = {
    "SuckerPunch": SuckerPunch
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "SuckerPunch": "Dumb"
}
