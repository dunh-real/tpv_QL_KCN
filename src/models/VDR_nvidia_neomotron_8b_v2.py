import requests
from PIL import Image
from io import BytesIO
import torch
from transformers import AutoModel
from transformers.image_utils import load_image

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(device)

# load model
model = AutoModel.from_pretrained(
    'nvidia/nemotron-colembed-vl-8b-v2',
    device_map = device,
    trust_remote_code = True,
    torch_dtype = torch.bfloat16,
    attn_implementation = 'sdpa'
).eval()
# model.to(device)

# queries
queries = [
    'How is AI improving the intelligence and capabilities of robots?',
    'Canary, a multilingual model that transcribes speech in Enghlish, Spanish, German, and French with punctuation and capitalization.',
    'Generative AI can generate DNA sequences that can be translated into proteins for bioengineering.'
]

image_urls = [
    "https://developer.download.nvidia.com/images/isaac/nvidia-isaac-lab-1920x1080.jpg",
    "https://developer-blogs.nvidia.com/wp-content/uploads/2024/03/asr-nemo-canary-featured.jpg",
    "https://blogs.nvidia.com/wp-content/uploads/2023/02/genome-sequencing-helix.jpg"
]

# load all images (load_image handles both local paths and URLs)
images = [load_image(img_path) for img_path in image_urls]

# encoding
query_embeddings = model.forward_queries(queries, batch_size = 8)
image_embeddings = model.forward_images(images, batch_size = 8)

scores = model.get_scores(
    query_embeddings,
    image_embeddings
)

# diagonal should have higher scores
print(scores)

# tensor([[21.1969, 20.9079, 20.5363],
#         [32.2324, 32.8058, 32.2075],
#         [25.7505, 25.8201, 26.2726]], device='cuda:0')