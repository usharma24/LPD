import os
import shutil
import random
import xml.etree.ElementTree as ET

# ================= CONFIG =================
SOURCE_DIR = "D:/app/npdataset/archive"
DEST_DIR = "D:/app/npdataset/yolo_dataset"
TRAIN_RATIO = 0.8
IMAGE_EXTS = (".jpg", ".jpeg", ".png")

CLASS_ID = 0  # single class: license_plate
# ========================================

def make_dirs():
    for p in [
        f"{DEST_DIR}/images/train",
        f"{DEST_DIR}/images/val",
        f"{DEST_DIR}/labels/train",
        f"{DEST_DIR}/labels/val",
    ]:
        os.makedirs(p, exist_ok=True)

def convert_xml_to_yolo(xml_path):
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        size = root.find("size")
        img_w = int(size.find("width").text)
        img_h = int(size.find("height").text)

        yolo_lines = []

        for obj in root.findall("object"):
            box = obj.find("bndbox")
            if box is None:
                continue

            xmin = float(box.find("xmin").text)
            ymin = float(box.find("ymin").text)
            xmax = float(box.find("xmax").text)
            ymax = float(box.find("ymax").text)

            x_center = ((xmin + xmax) / 2) / img_w
            y_center = ((ymin + ymax) / 2) / img_h
            width = (xmax - xmin) / img_w
            height = (ymax - ymin) / img_h

            yolo_lines.append(
                f"{CLASS_ID} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
            )

        return yolo_lines

    except Exception as e:
        print(f"[ERROR] Failed to parse {xml_path}: {e}")
        return []

def collect_pairs():
    pairs = []

    for root, _, files in os.walk(SOURCE_DIR):
        for file in files:
            if file.lower().endswith(IMAGE_EXTS):
                img_path = os.path.join(root, file)
                xml_path = img_path.rsplit(".", 1)[0] + ".xml"

                if os.path.exists(xml_path):
                    pairs.append((img_path, xml_path, os.path.basename(root)))

    return pairs

def split_dataset(pairs):
    random.shuffle(pairs)
    split = int(len(pairs) * TRAIN_RATIO)
    return pairs[:split], pairs[split:]

def process_pair(img, xml, tag, img_dst, lbl_dst):
    img_name = f"{tag}_{os.path.basename(img)}"
    lbl_name = img_name.rsplit(".", 1)[0] + ".txt"

    yolo_data = convert_xml_to_yolo(xml)

    # IMPORTANT: only skip if absolutely no boxes
    if len(yolo_data) == 0:
        return False

    shutil.copy(img, os.path.join(img_dst, img_name))

    with open(os.path.join(lbl_dst, lbl_name), "w") as f:
        f.write("\n".join(yolo_data))

    return True

def main():
    print("🚀 Converting XML annotations to YOLO format...")

    pairs = collect_pairs()
    print(f"✅ Found {len(pairs)} image + XML pairs")

    if not pairs:
        print("❌ No XML labels found. Check SOURCE_DIR.")
        return

    make_dirs()
    train, val = split_dataset(pairs)

    train_count = 0
    val_count = 0

    for img, xml, tag in train:
        if process_pair(
            img, xml, tag,
            f"{DEST_DIR}/images/train",
            f"{DEST_DIR}/labels/train"
        ):
            train_count += 1

    for img, xml, tag in val:
        if process_pair(
            img, xml, tag,
            f"{DEST_DIR}/images/val",
            f"{DEST_DIR}/labels/val"
        ):
            val_count += 1

    print("🎉 Dataset preparation finished!")
    print(f"📊 Images written to train: {train_count}")
    print(f"📊 Images written to val: {val_count}")

if __name__ == "__main__":
    main()
