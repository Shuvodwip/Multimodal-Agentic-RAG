import glob

from PIL import Image
from sentence_transformers import SentenceTransformer

IMAGES_DIR = "data/images"

clip_model = SentenceTransformer("clip-ViT-B-32")


def embed_images(images_dir: str) -> list[dict]:
    image_paths = sorted(glob.glob(f"{images_dir}/*"))
    images = [Image.open(p) for p in image_paths]
    embeddings = clip_model.encode(images)

    return [
        {"path": path, "embedding": embedding}
        for path, embedding in zip(image_paths, embeddings)
    ]


if __name__ == "__main__":
    records = embed_images(IMAGES_DIR)
    print(f"Images embedded: {len(records)}")
    print(f"Embedding shape: {records[0]['embedding'].shape}")
    for r in records:
        print(f"  {r['path']}")
