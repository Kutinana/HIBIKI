class SyntaxRegionalPrompter:
    """
    Phase 1 skeleton node.
    Step 1.4 will replace this placeholder implementation.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True}),
                "clip": ("CLIP",),
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "build_conditioning"
    CATEGORY = "conditioning"

    def build_conditioning(self, text, clip):
        raise NotImplementedError(
            "SyntaxRegionalPrompter skeleton is loaded. Implement Step 1.4 to output CONDITIONING."
        )


NODE_CLASS_MAPPINGS = {
    "SyntaxRegionalPrompter": SyntaxRegionalPrompter,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SyntaxRegionalPrompter": "Syntax Regional Prompter",
}
