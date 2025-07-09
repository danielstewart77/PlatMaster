import cv2
import numpy as np
import os
import json
from services.llm import text_to_llm
from models.plat import Plat
from pdf2image import convert_from_path
from paddleocr import PaddleOCR
from dotenv import load_dotenv
from PIL import Image

load_dotenv()

# Initialize PaddleOCR with layout analysis (structure parsing)
OCR_MODEL = PaddleOCR(
    device="gpu",  # "gpu" for GPU, "cpu" for CPU
    lang="en",
    use_textline_orientation=False,
    use_doc_orientation_classify=False,
    use_doc_unwarping=False
)

def pdf_to_images(pdf_path):
    """Convert PDF pages to a list of PIL images at 300 DPI."""
    return convert_from_path(pdf_path, dpi=300)

def detect_and_ocr(image_pil):
    """
    Detect text regions and run OCR using PaddleOCR's predict method.
    Returns a list of recognized text blocks with their bounding boxes.
    """
    image_np = np.array(image_pil)
    result = OCR_MODEL.predict(image_np)

    text_blocks = []
    
    if len(result) > 0:
        ocr_result = result[0]  # Get the first OCRResult object
        json_result = ocr_result.json
        
        # Extract text results from the JSON structure
        res = json_result.get('res', {})
        rec_texts = res.get('rec_texts', [])
        rec_polys = res.get('rec_polys', [])
        rec_scores = res.get('rec_scores', [])
        
        # Process each detected text
        for i, (text, poly, score) in enumerate(zip(rec_texts, rec_polys, rec_scores)):
            if text and text.strip():
                text_blocks.append((poly, text))
                print(f"[DEBUG] Detected text: '{text}' with confidence {score:.2f}")

    print(f"[DEBUG] Total detected text blocks: {len(text_blocks)}")
    return text_blocks, result

def draw_ocr_boxes(image, ocr_result, output_path, font_scale=0.5, font_thickness=1):
    """
    Draw OCR bounding boxes and text labels on the image.
    :param image: PIL Image
    :param ocr_result: PaddleOCR result (list containing OCRResult objects)
    :param output_path: Path to save the image with boxes
    """
    image_np = np.array(image).copy()

    if len(ocr_result) > 0:
        ocr_data = ocr_result[0]  # Get the first OCRResult object
        json_result = ocr_data.json
        
        # Extract text results from the JSON structure
        res = json_result.get('res', {})
        rec_texts = res.get('rec_texts', [])
        rec_polys = res.get('rec_polys', [])
        rec_scores = res.get('rec_scores', [])
        
        # Process each detected text
        for text, poly, score in zip(rec_texts, rec_polys, rec_scores):
            if text and text.strip():
                points = np.array(poly, dtype=np.int32)  # Polygon coordinates
                
                # Draw bounding box
                cv2.polylines(image_np, [points.reshape((-1, 1, 2))], isClosed=True, color=(0, 255, 0), thickness=2)

                # Put text label near the box
                text_pos = tuple(points[0])
                cv2.putText(
                    image_np,
                    f"{text} ({score:.2f})",
                    text_pos,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    (255, 0, 0),
                    font_thickness,
                    lineType=cv2.LINE_AA
                )

    # Save the annotated image
    cv2.imwrite(output_path, cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
    print(f"[DEBUG] Saved annotated image with OCR boxes at {output_path}")

def merge_text_blocks(text_blocks):
    """Merge all OCR text blocks into a single string."""
    all_text = "\n".join([text for _, text in text_blocks])
    return all_text

def extract_plat_structured(text):
    """Use the Plat model and LLM service to extract structured JSON data."""
    llm_model = "gpt-4.1"  # or your preferred deployment name
    try:
        return text_to_llm(llm_model, Plat, text)
    except Exception as e:
        print(f"[LLM Extraction Error]: {e}")
        return {}

def main():
    plats_dir = "plats"
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(plats_dir):
        if not filename.lower().endswith(".pdf"):
            continue
        pdf_path = os.path.join(plats_dir, filename)
        images = pdf_to_images(pdf_path)
        base_name = os.path.splitext(filename)[0]

        for idx, img_pil in enumerate(images):
            img_save_path = os.path.join(output_dir, f"{base_name}_page{idx+1}_input.png")
            img_pil.save(img_save_path)

            text_blocks, ocr_result = detect_and_ocr(img_pil)

            # Save image with bounding boxes for debugging
            boxes_preview_path = os.path.join(output_dir, f"{base_name}_page{idx+1}_boxes.png")
            draw_ocr_boxes(img_pil, ocr_result, boxes_preview_path)

            # Merge OCR results
            all_text = merge_text_blocks(text_blocks)
            merged_txt_path = os.path.join(output_dir, f"{base_name}_page{idx+1}_ocr_merged.txt")
            with open(merged_txt_path, "w") as f:
                f.write(all_text)

            print(f"[OCR Result for {filename} page {idx+1}]\n", all_text)
            structured = extract_plat_structured(all_text)
            print(f"[Extracted JSON for {filename} page {idx+1}]\n", structured.model_dump_json() if isinstance(structured, Plat) else structured)

            # Write output JSON
            output_file = os.path.join(output_dir, f"{base_name}_page{idx+1}.json")
            with open(output_file, "w") as f:
                json.dump(structured.model_dump_json(), f, indent=2) if isinstance(structured, Plat) else f.write(str(structured))

if __name__ == "__main__":
    main()
