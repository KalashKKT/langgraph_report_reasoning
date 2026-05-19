
# ============================================================
# STEP 2: Imports
# ============================================================
from pdf2image import convert_from_path
from PIL import Image
import torch
import os
from transformers import AutoModelForObjectDetection
from torchvision import transforms

# ============================================================
# STEP 3: Load detection model (skip if already loaded)
# ============================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModelForObjectDetection.from_pretrained("microsoft/table-transformer-detection", revision="no_timm")
model.to(device)
print("Model loaded on:", device)

# ============================================================
# STEP 4: Define helper transforms & functions
# ============================================================
class MaxResize(object):
    def __init__(self, max_size=800):
        self.max_size = max_size

    def __call__(self, image):
        width, height = image.size
        current_max_size = max(width, height)
        scale = self.max_size / current_max_size
        resized_image = image.resize((int(round(scale*width)), int(round(scale*height))))
        return resized_image

detection_transform = transforms.Compose([
    MaxResize(800),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b

id2label = model.config.id2label
id2label[len(model.config.id2label)] = "no object"

def outputs_to_objects(outputs, img_size, id2label):
    m = outputs.logits.softmax(-1).max(-1)
    pred_labels = list(m.indices.detach().cpu().numpy())[0]
    pred_scores = list(m.values.detach().cpu().numpy())[0]
    pred_bboxes = outputs['pred_boxes'].detach().cpu()[0]
    pred_bboxes = [elem.tolist() for elem in rescale_bboxes(pred_bboxes, img_size)]

    objects = []
    for label, score, bbox in zip(pred_labels, pred_scores, pred_bboxes):
        class_label = id2label[int(label)]
        if not class_label == 'no object':
            objects.append({
                'label': class_label,
                'score': float(score),
                'bbox': [float(elem) for elem in bbox]
            })
    return objects

def objects_to_crops(img, objects, class_thresholds, padding=10):
    table_crops = []
    for obj in objects:
        if obj['score'] < class_thresholds.get(obj['label'], 0.5):
            continue

        bbox = obj['bbox']
        bbox = [
            max(0, bbox[0] - padding),
            max(0, bbox[1] - padding),
            min(img.width, bbox[2] + padding),
            min(img.height, bbox[3] + padding)
        ]

        cropped_img = img.crop(bbox)

        # Handle rotated tables
        if obj['label'] == 'table rotated':
            cropped_img = cropped_img.rotate(270, expand=True)

        table_crops.append({
            'image': cropped_img,
            'label': obj['label'],
            'score': obj['score'],
            'bbox': bbox
        })

    return table_crops

# ============================================================
# STEP 5: Main PDF processing function
# ============================================================
def process_pdf_for_tables(pdf_path, output_folder, dpi=200, padding=10, confidence_threshold=0.5):
    """
    Takes a PDF, goes through each page, detects tables,
    crops them and saves them to output_folder.

    Args:
        pdf_path           : Full path to the input PDF file
        output_folder      : Folder where cropped tables will be saved
        dpi                : Resolution for PDF to image conversion (higher = better quality)
        padding            : Pixels of padding around each cropped table
        confidence_threshold: Minimum confidence score to accept a detection
    """

    # --- Create output folder if it doesn't exist ---
    os.makedirs(output_folder, exist_ok=True)
    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]

    print(f"\n{'='*60}")
    print(f"Processing PDF: {pdf_name}")
    print(f"Output folder : {output_folder}")
    print(f"{'='*60}\n")

    # --- Convert PDF pages to images ---
    print("Converting PDF pages to images...")
    pages = convert_from_path(pdf_path, 
                              dpi=dpi, 
                            poppler_path=r"C:\poppler\poppler-25.12.0\Library\bin"
)
    print(f"Total pages found: {len(pages)}\n")

    detection_class_thresholds = {
        "table": confidence_threshold,
        "table rotated": confidence_threshold,
        "no object": 10
    }

    # --- Summary tracker ---
    total_tables_found = 0
    summary = []

    # --- Loop through each page ---
    for page_num, page_image in enumerate(pages, start=1):
        print(f"Processing page {page_num}/{len(pages)}...")

        # Convert to RGB
        page_image = page_image.convert("RGB")

        # Prepare image for model
        pixel_values = detection_transform(page_image).unsqueeze(0).to(device)

        # Run detection
        with torch.no_grad():
            outputs = model(pixel_values)

        # Get detected objects
        objects = outputs_to_objects(outputs, page_image.size, id2label)

        # Filter only tables
        tables = [obj for obj in objects if obj['label'] in ['table', 'table rotated']
                  and obj['score'] >= confidence_threshold]

        print(f"  → Tables detected on page {page_num}: {len(tables)}")

        if len(tables) == 0:
            summary.append({'page': page_num, 'tables_found': 0})
            continue

        # Crop and save each table
        table_crops = objects_to_crops(page_image, tables, detection_class_thresholds, padding=padding)

        for table_idx, crop in enumerate(table_crops, start=1):
            filename = f"{pdf_name}_page_{page_num}_table_{table_idx}.jpg"
            save_path = os.path.join(output_folder, filename)
            crop['image'].save(save_path, quality=95)
            print(f"  → Saved: {filename}")
            total_tables_found += 1

        summary.append({'page': page_num, 'tables_found': len(table_crops)})

    # --- Print final summary ---
    print(f"\n{'='*60}")
    print(f"DONE! Summary for: {pdf_name}")
    print(f"{'='*60}")
    print(f"{'Page':<10} {'Tables Found':<15}")
    print(f"{'-'*25}")
    for entry in summary:
        print(f"{entry['page']:<10} {entry['tables_found']:<15}")
    print(f"{'-'*25}")
    print(f"Total tables cropped and saved: {total_tables_found}")
    print(f"All saved to: {output_folder}\n")

    return summary


# ============================================================
# STEP 6: RUN IT — just change these two paths
# ============================================================
PDF_PATH     = r"D:\Piratech\data_agents\reports\aum_report.pdf"   # <-- your PDF
OUTPUT_FOLDER =  r"D:\Piratech\data_agents\reports\tables"


summary = process_pdf_for_tables(
    pdf_path=PDF_PATH,
    output_folder=OUTPUT_FOLDER,
    dpi=300,               # increase to 300 for better OCR quality
    padding=10,            # pixels of padding around each table
    confidence_threshold=0.5,
    
)