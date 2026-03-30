from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config
from transformers.models.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor

model_path = "mlx-community/Huihui-Qwen3.5-4B-Claude-4.6-Opus-abliterated-6bit"

model, processor = load(model_path)
config = load_config(model_path)

# Manually overwrite the fast processor with the slow one to fix PyTorch issue
processor.image_processor = Qwen2VLImageProcessor.from_pretrained(model_path)

image = ["http://images.cocodataset.org/val2017/000000039769.jpg"]

# THE FIX: Explicitly hardcode the Qwen image tokens into a simple string
prompt = "<|vision_start|><|image_pad|><|vision_end|>\nDescribe the image in Vietnamese"

# Tell the template engine 0 images so it just applies the text chat roles 
# (this prevents mlx_vlm from trying to double-inject the tokens)
formatted_prompt = apply_chat_template(
    processor, config, prompt, num_images=0
)

# Generate output
output = generate(model, processor, formatted_prompt, image)
print(output)