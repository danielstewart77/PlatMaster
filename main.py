"""
PlatMaster - OCR and AI-powered document processing for oil and gas well location plats.
Copyright (C) 2025 Daniel Stewart

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.
"""

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

def extract_plat(pdf_path, output_dir=None, save_debug_files=False):
    """
    Extract structured data from a single PDF plat document.
    
    Args:
        pdf_path (str): Path to the PDF file to process
        output_dir (str, optional): Directory to save debug files. If None, creates temp directory.
        save_debug_files (bool): Whether to save intermediate debug files
        
    Returns:
        dict: Extracted structured data from the plat document
    """
    # Create output directory if saving debug files
    if save_debug_files and output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Convert PDF to images
    images = pdf_to_images(pdf_path)
    base_name = os.path.splitext(os.path.basename(pdf_path))[0]
    
    # Process all pages and collect results
    all_results = []
    
    for idx, img_pil in enumerate(images):
        # Save input image if debug files requested
        if save_debug_files and output_dir:
            img_save_path = os.path.join(output_dir, f"{base_name}_page{idx+1}_input.png")
            img_pil.save(img_save_path)

        # Perform OCR
        text_blocks, ocr_result = detect_and_ocr(img_pil)

        # Save image with bounding boxes for debugging if requested
        if save_debug_files and output_dir:
            boxes_preview_path = os.path.join(output_dir, f"{base_name}_page{idx+1}_boxes.png")
            draw_ocr_boxes(img_pil, ocr_result, boxes_preview_path)

        # Merge OCR results
        all_text = merge_text_blocks(text_blocks)
        
        # Save merged text if debug files requested
        if save_debug_files and output_dir:
            merged_txt_path = os.path.join(output_dir, f"{base_name}_page{idx+1}_ocr_merged.txt")
            with open(merged_txt_path, "w") as f:
                f.write(all_text)

        print(f"[OCR Result for {base_name} page {idx+1}]\n", all_text)
        
        # Extract structured data
        structured = extract_plat_structured(all_text)
        print(f"[Extracted JSON for {base_name} page {idx+1}]\n", 
              structured.model_dump_json() if isinstance(structured, Plat) else structured)

        # Save JSON output if debug files requested
        if save_debug_files and output_dir:
            output_file = os.path.join(output_dir, f"{base_name}_page{idx+1}.json")
            with open(output_file, "w") as f:
                if isinstance(structured, Plat):
                    json.dump(structured.model_dump(), f, indent=2)
                else:
                    json.dump(structured, f, indent=2)
        
        # Add to results collection
        if isinstance(structured, Plat):
            all_results.append(structured.model_dump())
        else:
            all_results.append(structured)
    
    # Return the first page result (most common case) or all results if multiple pages
    return all_results[0] if len(all_results) == 1 else {"pages": all_results}

def main():
    """Batch process all PDFs in the plats directory (legacy function)."""
    plats_dir = "plats"
    output_dir = "output"
    
    for filename in os.listdir(plats_dir):
        if not filename.lower().endswith(".pdf"):
            continue
        pdf_path = os.path.join(plats_dir, filename)
        print(f"Processing {filename}...")
        result = extract_plat(pdf_path, output_dir, save_debug_files=True)
        print(f"Completed processing {filename}")
        print("=" * 50)

if __name__ == "__main__":
    main()
