from diffusers import StableDiffusionXLPipeline
import torch

pipeline = StableDiffusionXLPipeline.from_single_file(
	"juggernautXL_version6Rundiffusion.safetensors", torch_dtype=torch.float16, variant="fp16"
).to("cuda")

pipeline.load_lora_weights("tattoo_gen.safetensors")

prompt = "skull with butterfly"
negative_prompt= "(low quality, worst quality:1.4), text, error, extra digit, fewer digits, cropped, normal quality, jpeg artifacts, signature, watermark, username, blurry, multiple views, letterbox, realistic, human realistic"

image = pipeline(
	prompt,
    negative_prompt=negative_prompt,
    width=512,
    height=512,
).images[0]

image.save("test3.png")

print(image)