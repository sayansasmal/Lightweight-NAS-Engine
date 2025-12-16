# app.py
import streamlit as st
import pandas as pd
import os, zipfile, io, shutil, json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

from utils import default_transforms, ImageFolderCSV
from train import run_search, fine_tune_model
from models import get_model

# ---------------- Streamlit basic config ----------------
st.set_page_config(page_title="Lightweight NAS MVP", layout="centered")

st.title("Lightweight Neural Architecture Search — MVP (Image Only)")

# Simple "page" selection in the sidebar
page = st.sidebar.radio("Page", ["Train / Search", "Inference"])

DATA_TMP_DIR = "tmp_dataset"
OUTPUT_DIR = "outputs"
MODEL_PATH = Path(OUTPUT_DIR) / "best_model.pt"
META_PATH = Path(OUTPUT_DIR) / "meta.json"


# ---------------- Helper: evaluate on validation set ----------------
@torch.no_grad()
def eval_full(model, dataset, batch_size=16, device="cpu"):
    """
    Run model on entire dataset and return:
    - y_true, y_pred, y_prob (numpy arrays)
    """
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    model.eval()
    all_true, all_pred, all_prob = [], [], []
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        probs = F.softmax(logits, dim=1)
        preds = torch.argmax(logits, dim=1)

        all_true.extend(y.cpu().numpy())
        all_pred.extend(preds.cpu().numpy())
        all_prob.extend(probs.cpu().numpy())

    return np.array(all_true), np.array(all_pred), np.array(all_prob)


# ---------------- Dataset helpers: folder-per-class only ----------------
def auto_generate_labels_from_folders(root_dir, allowed_ext=None):
    """
    Auto-build labels from folder-per-class structure.

    Supports BOTH:
    1) Root has class folders:
         Class1/img1.jpg
         Class2/img2.jpg
    2) Root has a single wrapper folder (e.g. 'Image') that contains class folders:
         Image/Class1/img1.jpg
         Image/Class2/img2.jpg

    Returns:
      df (columns: fname,label) with paths relative to root_dir,
      class_id_to_name dict (e.g. {0: 'Class1', 1: 'Class2', ...}),
      total_images
    """
    if allowed_ext is None:
        allowed_ext = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}

    top_root = root_dir  # for relative paths

    # Helper to list immediate subdirectories
    def list_subdirs(d):
        return [
            name for name in sorted(os.listdir(d))
            if os.path.isdir(os.path.join(d, name)) and not name.startswith(".")
        ]

    # First-level subdirs under root_dir
    subdirs = list_subdirs(root_dir)

    # If there is exactly one subdirectory (e.g. 'Image') and inside it
    # there are multiple class folders, treat THAT as the class root.
    class_root = root_dir
    if len(subdirs) == 1:
        candidate = os.path.join(root_dir, subdirs[0])
        inner_subdirs = list_subdirs(candidate)
        if len(inner_subdirs) >= 1:
            class_root = candidate
            subdirs = inner_subdirs  # these are the class folders

    class_id_to_name = {}
    rows = []
    class_id = 0

    for cls_name in subdirs:
        cls_dir = os.path.join(class_root, cls_name)
        class_id_to_name[class_id] = cls_name

        for root, _, files in os.walk(cls_dir):
            for fname in files:
                ext = os.path.splitext(fname)[1].lower()
                if ext not in allowed_ext:
                    continue
                full_path = os.path.join(root, fname)
                # Path relative to top_root, so ImageFolderCSV(root=top_root, fname=...) can read it.
                rel_path = os.path.relpath(full_path, top_root)
                rows.append((rel_path.replace("\\", "/"), class_id))

        class_id += 1

    df = pd.DataFrame(rows, columns=["fname", "label"])
    total_images = len(df)
    return df, class_id_to_name, total_images


def check_images_exist_and_valid(root_dir, df, max_check=None):
    """
    Check:
      - missing files for paths in df['fname']
      - corrupted images (cannot be opened)
    """
    from PIL import Image

    missing_files = []
    corrupted_files = []

    paths = df["fname"].tolist()
    if max_check is not None:
        paths = paths[:max_check]

    for rel in paths:
        full_path = os.path.join(root_dir, rel)
        if not os.path.exists(full_path):
            missing_files.append(rel)
            continue
        try:
            with Image.open(full_path) as img:
                img.verify()
        except Exception:
            corrupted_files.append(rel)

    return missing_files, corrupted_files


# ---------------- PAGE 1: TRAIN / SEARCH ----------------
if page == "Train / Search":
    st.markdown("""
    ### Step 1: Upload dataset zip

    **Only one simple format is supported: folder-per-class**, with an optional
    top-level wrapper folder.

    Examples (both are OK):

    **Example A (no wrapper):**
    ```
    Class1/img1.jpg
    Class1/img2.jpg
    Class2/img3.jpg
    Class3/img4.jpg
    ```

    **Example B (with 'Image' wrapper):**
    ```
    Image/Class1/img1.jpg
    Image/Class1/img2.jpg
    Image/Class2/img3.jpg
    Image/Class3/img4.jpg
    ```

    - Each folder name is treated as a class name (Class1, Class2, ...).
    - The app automatically creates labels internally. No labels.csv needed.
    """)

    uploaded = st.file_uploader("Upload dataset zip", type=["zip"])
    if uploaded is not None:
        # Clean old tmp dir (but don't crash if Windows locks a file)
        if os.path.exists(DATA_TMP_DIR):
            try:
                shutil.rmtree(DATA_TMP_DIR)
            except PermissionError as e:
                st.warning(
                    f"Could not fully clear previous temporary dataset folder because "
                    f"some files are in use (Windows lock). Error: {e}\n\n"
                    f"If you see strange behavior, close any image viewers or antivirus "
                    f"that might be scanning the dataset and try again."
                )

        # Extract new dataset
        with zipfile.ZipFile(io.BytesIO(uploaded.read())) as z:
            z.extractall(DATA_TMP_DIR)
        st.success(f"Extracted dataset to `{DATA_TMP_DIR}/`")

        # Auto-generate labels from folder structure
        st.markdown("#### Auto-generating labels from folder-per-class structure...")
        df, class_id_to_name, total_images = auto_generate_labels_from_folders(DATA_TMP_DIR)

        if df is None or len(df) == 0:
            st.error(
                "No images found. Make sure ZIP has `ClassName/imagename.jpg` folders "
                "(optionally under a wrapper like 'Image/')."
            )
        else:
            st.success(f"Found {len(df)} images across {len(class_id_to_name)} classes.")
            st.write("Sample auto-generated labels (relative path, label_id):")
            st.dataframe(df.head())

            st.write("Class mapping (ID -> folder name):")
            mapping_rows = [{"class_id": cid, "class_name": name}
                            for cid, name in class_id_to_name.items()]
            st.dataframe(pd.DataFrame(mapping_rows))

            # Dataset check
            st.markdown("#### Dataset check (existence & corruption)")
            missing, corrupted = check_images_exist_and_valid(DATA_TMP_DIR, df, max_check=None)
            st.write(f"Total images found: {len(df)}")
            st.write(f"Missing files (should normally be 0): {len(missing)}")
            st.write(f"Corrupted/unreadable images (approx): {len(corrupted)}")

            if missing:
                st.warning(f"Example missing files (first 5): {missing[:5]}")
            if corrupted:
                st.warning(f"Example corrupted files (first 5): {corrupted[:5]}")

            # Train/val split
            st.markdown("#### Train/validation split")
            val_frac = st.slider("Validation fraction", 0.05, 0.4, 0.2)
            train_df = df.sample(frac=1 - val_frac, random_state=42)
            val_df = df.drop(train_df.index)
            st.write(f"Train size: {len(train_df)} | Val size: {len(val_df)}")

            # Transforms & datasets
            image_size = st.selectbox("Image size (smaller = faster)", [128, 160, 224], index=0)
            train_t, val_t = default_transforms(image_size=image_size)
            train_ds = ImageFolderCSV(DATA_TMP_DIR, train_df.values.tolist(), transform=train_t)
            val_ds = ImageFolderCSV(DATA_TMP_DIR, val_df.values.tolist(), transform=val_t)

            num_classes = int(df.label.nunique())
            st.write(f"Detected classes: {num_classes}")

            # Candidates and hyperparams
            st.markdown("#### NAS candidate architectures")
            candidates = st.multiselect(
                "Candidate models (small = faster)",
                ["tiny_cnn", "small_cnn", "resnet18", "mobilenet_v2", "efficientnet_b0"],
                default=["tiny_cnn", "mobilenet_v2"]
            )
            epochs = st.slider("Epochs per trial (quick search)", 1, 5, 2)
            batch = st.selectbox("Batch size", [4, 8, 16, 32], index=1)
            lr = float(st.text_input("Learning rate for search & fine-tune", "0.001"))

            st.markdown("#### Final training settings (after best model is found)")
            final_epochs = st.slider("Extra epochs for final training", 0, 20, 5)

            if len(candidates) == 0:
                st.warning("Select at least one candidate model.")
            else:
                if st.button("Start search + final training"):
                    # 1) Architecture search
                    with st.spinner("Running architecture search (quick trials)..."):
                        best_name, best_score, best_state, search_logs = run_search(
                            candidates,
                            train_ds,
                            val_ds,
                            num_classes,
                            device='cpu',
                            epochs_per_trial=epochs,
                            batch_size=batch,
                            lr=lr
                        )

                    st.success(f"Best architecture: {best_name}  |  Search val_acc: {best_score:.3f}")

                    # Show NAS search summary (name, val_acc, params)
                    st.markdown("### NAS search summary (per architecture)")
                    search_df = pd.DataFrame(search_logs)
                    st.dataframe(search_df)

                    # Build model from best_state
                    model = get_model(best_name, num_classes, pretrained=False)
                    model.load_state_dict(best_state)
                    model.to("cpu")

                    # 2) Optional final training
                    final_val_acc = best_score
                    if final_epochs > 0:
                        with st.spinner(f"Running final training for {final_epochs} epochs..."):
                            model, final_val_acc = fine_tune_model(
                                model,
                                train_ds,
                                val_ds,
                                device="cpu",
                                epochs=final_epochs,
                                batch_size=batch,
                                lr=lr
                            )
                        st.success(f"Final training completed. Best val_acc: {final_val_acc:.3f}")
                    else:
                        st.info("Final training skipped (extra epochs = 0).")

                    # 3) Save final model + metadata
                    os.makedirs(OUTPUT_DIR, exist_ok=True)
                    torch.save(model.state_dict(), str(MODEL_PATH))

                    class_names_json = {str(int(k)): str(v) for k, v in class_id_to_name.items()}

                    meta = {
                        "model_name": best_name,
                        "num_classes": int(num_classes),
                        "image_size": int(image_size),
                        "class_names": class_names_json
                    }
                    with open(META_PATH, "w") as f:
                        json.dump(meta, f)

                    st.write(f"Saved final model at `{MODEL_PATH}` and metadata at `{META_PATH}`")

                    # Download button
                    with open(MODEL_PATH, "rb") as f:
                        st.download_button(
                            "Download final model (state_dict)",
                            f,
                            file_name="best_model.pt"
                        )

                    # ---- Validation performance ----
                    st.markdown("### Validation set performance (final model)")

                    y_true, y_pred, y_prob = eval_full(model, val_ds, batch_size=batch, device="cpu")
                    acc = (y_true == y_pred).mean()
                    st.write(f"Validation accuracy (recomputed): **{acc:.3f}**")

                    # Confusion matrix
                    label_ids = sorted(class_id_to_name.keys())
                    cm = confusion_matrix(y_true, y_pred, labels=label_ids)
                    label_names = [class_id_to_name.get(i, str(i)) for i in label_ids]
                    cm_df = pd.DataFrame(cm, index=label_names, columns=label_names)
                    st.write("Confusion matrix (rows = true labels, cols = predicted labels):")
                    st.dataframe(cm_df)

                    # Classification report
                    target_names = [class_id_to_name.get(i, str(i)) for i in label_ids]
                    report_str = classification_report(
                        y_true, y_pred,
                        labels=label_ids,
                        target_names=target_names,
                        zero_division=0
                    )
                    st.write("Classification report:")
                    st.text(report_str)

                    st.info(
                        "You can now go to the 'Inference' page (sidebar) to test single images "
                        "with this trained model."
                    )


# ---------------- PAGE 2: INFERENCE ----------------
elif page == "Inference":
    st.markdown("## Inference — test a single image with the best saved model")

    if not MODEL_PATH.exists() or not META_PATH.exists():
        st.info("No trained model found yet. Train a model first on the 'Train / Search' page.")
    else:
        # Load metadata
        with open(META_PATH, "r") as f:
            meta = json.load(f)

        model_name = meta["model_name"]
        num_classes = meta["num_classes"]
        image_size = meta["image_size"]
        class_names_json = meta.get("class_names", None) or {}

        class_id_to_name = {int(k): v for k, v in class_names_json.items()}

        st.write(f"Loaded model: **{model_name}**  |  classes: {num_classes}  |  image size: {image_size}")

        # Build model and load weights
        model = get_model(model_name, num_classes, pretrained=False)
        state_dict = torch.load(MODEL_PATH, map_location="cpu")
        model.load_state_dict(state_dict)
        model.eval()

        # Inference transform
        _, val_t = default_transforms(image_size=image_size)

        # Prediction history
        if "pred_history" not in st.session_state:
            st.session_state["pred_history"] = []

        uploaded_img = st.file_uploader("Upload a single image", type=["jpg", "jpeg", "png"])
        if uploaded_img is not None:
            from PIL import Image

            img = Image.open(uploaded_img).convert("RGB")
            st.image(img, caption="Uploaded image", width=256)

            x = val_t(img).unsqueeze(0)

            with torch.no_grad():
                logits = model(x)
                probs = F.softmax(logits, dim=1).cpu().numpy().flatten()
                pred_class = int(np.argmax(probs))
                confidence = float(probs[pred_class])

            pred_name = class_id_to_name.get(pred_class, str(pred_class))

            st.write(f"**Predicted class ID:** {pred_class}")
            st.write(f"**Predicted class name:** {pred_name}")
            st.write(f"**Confidence:** {confidence:.3f}")

            ids = list(range(num_classes))
            names = [class_id_to_name.get(i, str(i)) for i in ids]

            prob_df = pd.DataFrame({
                "class_id": ids,
                "class_name": names,
                "probability": probs
            })
            st.write("All class probabilities:")
            st.dataframe(prob_df)

            # Update history
            st.session_state["pred_history"].append({
                "file_name": uploaded_img.name,
                "pred_class_id": pred_class,
                "pred_class_name": pred_name,
                "confidence": confidence
            })

        if st.session_state["pred_history"]:
            st.markdown("### Prediction history (this session)")
            hist_df = pd.DataFrame(st.session_state["pred_history"])
            st.dataframe(hist_df)
