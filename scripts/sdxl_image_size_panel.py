import contextlib

import gradio as gr
from modules import scripts, shared, script_callbacks
from modules.ui_components import FormRow, FormColumn, FormGroup, ToolButton
import json
import os
import math
import random
import numpy as np
from PIL.Image import Image as PILImage

available_resolutions = {}


def read_sdxl_resolutions():
    json_path = os.path.join(scripts.basedir(), "resolutions.json")
    with open(json_path, 'rt', encoding="utf-8") as file:
        json_data = json.load(file)

        for resolution in json_data:
            width = int(resolution.split("x")[0])
            height = int(resolution.split("x")[1])
            ratio = width / height
            lcm = math.lcm(width, height)
            aspect_w = int(lcm / height)
            aspect_h = int(lcm / width)
            label = f"{aspect_w}:{aspect_h} ({width}x{height})"
            available_resolutions[label] = {
                "width": width,
                "height": height,
                "ratio": ratio
            }

    print("[i] Loaded " + str(len(available_resolutions)) + " resolutions!")


def find_best_resolution(*images):

    for img in images:

        if img is None:
            continue

        if type(img) is np.ndarray:
            width = img.shape[1]
            height = img.shape[0]
        elif type(img) is PILImage:
            width = img.width
            height = img.height
        else:
            # raise gr.Error("Unknown image type: " + str(img))
            continue

        target_ratio = width / height

        closest_name = None
        min_difference = float('inf')

        for name, config in available_resolutions.items():
            difference = abs(target_ratio - config["ratio"])
            if difference < min_difference:
                min_difference = difference
                closest_name = name

        gr.Info(f"Best resolution is {closest_name} with abs difference {min_difference}")

        return closest_name

    raise gr.Error("No image found/provided!")


def apply_resolution(name):
    if name is None or name not in available_resolutions:
        raise gr.Error("No resolution selected!")

    width = available_resolutions[name]["width"]
    height = available_resolutions[name]["height"]

    gr.Info(f"Set resolution to {width}x{height}")

    return [width, height]


class ImageSizeSelector(scripts.Script):
    def __init__(self) -> None:
        super().__init__()
        self.image = None
        self.i2i_h = None
        self.i2i_w = None
        self.t2i_h = None
        self.t2i_w = None

    # Cache all styles
    read_sdxl_resolutions()

    def title(self):
        return "SDXL Resolutions"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        with gr.Group():
            with gr.Accordion("SDXL Resolutions", open=False):
                dropdown = gr.Dropdown(choices=sorted(available_resolutions), multiselect=False, label="Presets",
                                       interactive=True)
                reference_image_selector = gr.Image(visible=not is_img2img)

                with FormRow():
                    read_from_img_button = gr.Button(value="Read from image")
                    apply_button = gr.Button(value="Apply")

        # Reading data from images
        image_components = []
        if is_img2img:
            image_components += self.image
        else:
            image_components = [reference_image_selector]
        read_from_img_button.click(find_best_resolution, inputs=image_components, outputs=[dropdown])

        # Applying selection
        if is_img2img:
            resolution = [self.i2i_w, self.i2i_h]
        else:
            resolution = [self.t2i_w, self.t2i_h]
        apply_button.click(apply_resolution, inputs=[dropdown], outputs=resolution)

        return [dropdown]

    def after_component(self, component, **kwargs):
        if kwargs.get("elem_id") == "txt2img_width":
            self.t2i_w = component
        if kwargs.get("elem_id") == "txt2img_height":
            self.t2i_h = component

        if kwargs.get("elem_id") == "img2img_width":
            self.i2i_w = component
        if kwargs.get("elem_id") == "img2img_height":
            self.i2i_h = component

        if kwargs.get("elem_id") == "img2img_image":
            self.image = [component]
        if kwargs.get("elem_id") == "img2img_sketch":
            self.image.append(component)
        if kwargs.get("elem_id") == "img2maskimg":
            self.image.append(component)
        if kwargs.get("elem_id") == "inpaint_sketch":
            self.image.append(component)
        if kwargs.get("elem_id") == "img_inpaint_base":
            self.image.append(component)
