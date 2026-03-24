import torch
from PIL import Image
from colpali_engine.models import ColQwen3_5, ColQwen3_5Processor

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

model = ColQwen3_5.from_pretrained(
    'athrael-soju/colqwen3.5-v1',
    torch_stype = torch.bfloat16,
    device_map = device,
    attn_implementation = 'sdpa'
)

processor = ColQwen3_5Processor.from_pretrained("athrael-soju/colqwen3.5-v1")

# embed document images
images = [Image.open("page1.png"), Image.open("page2.png")]
batch = processor.process_images(images).to(model.device)
with torch.no_grad():
    doc_embeddings = model(**batch)

# embed queries
queries = ["What is the revenue for Q4?", "Show me the organizational chart"]
batch = processor.process_queries(queries).to(model.device)
with torch.no_grad():
    model.rope_deltas = None
    query_embeddings = model(**batch)

scores = processor.score(query_embeddings, doc_embeddings)