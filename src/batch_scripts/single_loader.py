import json
import os
from pathlib import Path
from typing import Dict, List, Any


class SingleImageLoader:
    """
    Minimal single-image loader that mimics CoconutLoader.

    Expected annotation JSON format:
    {
      "images": [
        {"id": 1, "file_name": "my.jpg", "width": 1280, "height": 720}
      ],
      "annotations": [
        {
          "id": 1,
          "image_id": 1,
          "category_id": 1,
          "bbox": [x, y, w, h],
          "segmentation": [[x1,y1,x2,y2,...]]   # COCO polygon
        }
      ],
      "categories": [
        {"id": 1, "name": "chair"}
      ]
    }
    """

    def __init__(self, image_root: str, annotation_json: str):
        self.image_root = image_root
        self.annotation_json = annotation_json

        with open(annotation_json, "r") as f:
            data = json.load(f)

        self.images: List[Dict[str, Any]] = data["images"]
        self.categories: List[Dict[str, Any]] = data.get("categories", [])

        self.annotations_by_image: Dict[int, List[Dict[str, Any]]] = {}
        for anno in data.get("annotations", []):
            img_id = anno["image_id"]
            self.annotations_by_image.setdefault(img_id, []).append(anno)

        if len(self.images) != 1:
            print(f"[SingleImageLoader] Warning: found {len(self.images)} images. Usually expected 1.")

    def get_images(self) -> List[Dict]:
        return self.images

    def get_image_by_index(self, index: int) -> Dict:
        return self.images[index]

    def get_annotations(self, image_id: int) -> List[Dict]:
        return self.annotations_by_image.get(image_id, [])

    def get_categories(self) -> List[Dict]:
        return self.categories

    def __len__(self) -> int:
        return len(self.images)


def get_single_paths(image_root: str, annotation_json: str):
    """
    Return paths in the same spirit as get_dataset_paths().
    dataset_root: directory containing the image file
    annotations_dir: directory containing the annotation json
    """
    dataset_root = image_root
    annotations_dir = str(Path(annotation_json).parent)
    return dataset_root, annotations_dir