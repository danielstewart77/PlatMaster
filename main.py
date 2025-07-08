import cv2
import pytesseract
import numpy as np
import os
from services.llm import text_to_llm
from models.plat import Plat
import json
from pdf2image import convert_from_path
from sklearn.cluster import DBSCAN
from dotenv import load_dotenv
import layoutparser as lp
from PIL import Image
from layoutparser.models import Detectron2LayoutModel

load_dotenv()

# Set this if Tesseract is not in PATH
# pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"

def pdf_to_images(pdf_path):
    return convert_from_path(pdf_path, dpi=300)

def detect_text_regions_layoutparser(image_pil):
    # Load model (can cache globally for speed)
    model = Detectron2LayoutModel(
        config_path='lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config',
        extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.8],
        label_map={0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"}
    )

    layout = model.detect(image_pil)
    boxes = []

    for block in layout:
        if block.type in ["Text", "Table"]:  # you can also include 'Title' or 'List' if needed
            x1, y1, x2, y2 = map(int, block.coordinates)
            boxes.append((x1, y1, x2 - x1, y2 - y1))  # convert to (x, y, w, h)

    return boxes

def cluster_and_crop(image, boxes):
    centers = np.array([[x + w // 2, y + h // 2] for x, y, w, h in boxes])
    db = DBSCAN(eps=50, min_samples=2).fit(centers)
    crops = []

    for label in set(db.labels_):
        if label == -1:
            continue
        indices = np.where(db.labels_ == label)[0]
        xs, ys, ws, hs = zip(*[boxes[i] for i in indices])
        x1, y1 = min(xs), min(ys)
        x2, y2 = max([x + w for x, w in zip(xs, ws)]), max([y + h for y, h in zip(ys, hs)])
        crop = image[y1:y2, x1:x2]
        crops.append(crop)

    return crops

def run_ocr_on_blocks(crops):
    text_blocks = []
    for i, crop in enumerate(crops):
        text = pytesseract.image_to_string(crop)
        if len(text.strip()) > 0:
            text_blocks.append(text)
    return "\n".join(text_blocks)

def extract_plat_structured(text):
    # Use the Plat model and text_to_llm service to extract structured data
    llm_model = "gpt-4.1"  # or your preferred deployment name
    try:
        return text_to_llm(llm_model, Plat, text)
    except Exception as e:
        print(f"[LLM Extraction Error]: {e}")
        return {}

def main():
    plats_dir = "plats"
    output_dir = "output"
    for filename in os.listdir(plats_dir):
        if not filename.lower().endswith(".pdf"):
            continue
        pdf_path = os.path.join(plats_dir, filename)
        images = pdf_to_images(pdf_path)
        base_name = os.path.splitext(filename)[0]
        for idx, img_pil in enumerate(images):
            # Step 1: Save image for manual inspection
            img_save_path = os.path.join(output_dir, f"{base_name}_page{idx+1}_input.png")
            img_pil.save(img_save_path)

            img_cv = np.array(img_pil)
            boxes = detect_text_regions_layoutparser(img_pil)

            # Step 2: Draw boxes and save preview
            img_boxes = img_cv.copy()
            for (x, y, w, h) in boxes:
                cv2.rectangle(img_boxes, (x, y), (x + w, y + h), (0, 255, 0), 2)
            boxes_preview_path = os.path.join(output_dir, f"{base_name}_page{idx+1}_boxes.png")
            cv2.imwrite(boxes_preview_path, cv2.cvtColor(img_boxes, cv2.COLOR_RGB2BGR))

            # Step 3: Print DBSCAN cluster info
            centers = np.array([[x + w // 2, y + h // 2] for x, y, w, h in boxes])
            if len(centers) > 0:
                db = DBSCAN(eps=50, min_samples=2).fit(centers)
                labels = db.labels_
                print(f"[DEBUG] {filename} page {idx+1}: DBSCAN found {len(set(labels)) - (1 if -1 in labels else 0)} clusters, labels: {set(labels)}")
            else:
                labels = []
            # Use original cluster_and_crop for actual crops
            crops = cluster_and_crop(img_cv, boxes)

            # Step 4: Save each crop and its OCR result
            ocr_texts = []
            for i, crop in enumerate(crops):
                crop_path = os.path.join(output_dir, f"{base_name}_page{idx+1}_crop{i+1}.png")
                cv2.imwrite(crop_path, cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
                text = pytesseract.image_to_string(crop)
                ocr_texts.append(text)
                ocr_txt_path = os.path.join(output_dir, f"{base_name}_page{idx+1}_crop{i+1}.txt")
                with open(ocr_txt_path, "w") as f:
                    f.write(text)

            # Step 5: Dump merged OCR text for LLM sanity check
            all_text = "\n".join(ocr_texts)
            merged_txt_path = os.path.join(output_dir, f"{base_name}_page{idx+1}_ocr_merged.txt")
            with open(merged_txt_path, "w") as f:
                f.write(all_text)

            print(f"[OCR Result for {filename} page {idx+1}]\n", all_text)
            structured = extract_plat_structured(all_text)
            print(f"[Extracted JSON for {filename} page {idx+1}]\n", structured)
            # Write output JSON file
            output_file = os.path.join(output_dir, f"{base_name}_page{idx+1}.json")
            if isinstance(structured, dict):
                with open(output_file, "w") as f:
                    json.dump(structured, f, indent=2)
            elif isinstance(structured, str):
                try:
                    json_obj = json.loads(structured)
                    with open(output_file, "w") as f:
                        json.dump(json_obj, f, indent=2)
                except Exception:
                    with open(output_file, "w") as f:
                        f.write(structured)

if __name__ == "__main__":
    main()
