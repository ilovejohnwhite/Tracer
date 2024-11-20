from ..utils import common_annotator_call, annotator_ckpts_path, HF_MODEL_NAME, create_node_input_types
import comfy.model_management as model_management

class OkayBuddyNode:
    @classmethod
    def INPUT_TYPES(cls):
        return create_node_input_types(
            n_clusters=(
                [("Light"), ("Normal"), ("Heavy"), ("Unhinged")],
                {"default": "Unhinged"}  # Assuming you can set defaults by label. Adjust as needed.
            )
        )

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"

    CATEGORY = "TatToolkit/Billy Goncho's Wild Ride X"

    def execute(self, image, resolution=512, **kwargs):
        from TatToolkit.OkayBuddy import OkayBuddy

        # Map labels to cluster numbers if necessary
        cluster_map = {"Light": 3, "Normal": 4, "Heavy": 6, "Unhinged": 8}
        label = kwargs.get("n_clusters", "Normal")
        n_clusters = cluster_map.get(label, 4)  # Default to 4 if mapping fails

        model = OkayBuddy(n_clusters=n_clusters)

        out = common_annotator_call(model, image, resolution=resolution)
        del model
        return (out,)


NODE_CLASS_MAPPINGS = {
    "OkayBuddyNode": OkayBuddyNode
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "OkayBuddyNode": "No"
}
