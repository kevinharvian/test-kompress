import io, os, zipfile, time, threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple, Dict
from collections import defaultdict

import streamlit as st
from PIL import Image, ImageOps, ImageFilter
import fitz  # PyMuPDF

# ===== HEIC/HEIF =====
HEIF_OK = False
try:
    import pillow_heif
    pillow_heif.register_heif_opener()
    HEIF_OK = True
except Exception:
    HEIF_OK = False

# ==========================
# PAGE CONFIG
# ==========================
st.set_page_config(page_title="Multi-ZIP → JPG & Kompres (Auto Size)", page_icon="📦", layout="wide")

if "uploader_key" not in st.session_state:
    st.session_state["uploader_key"] = 0
if "results" not in st.session_state:
    st.session_state["results"] = None

st.title("📦 Multi-ZIP / Files → JPG & Kompres (Auto Size by Folder)")
st.caption("Konversi gambar (termasuk JFIF/HEIC) & PDF ke JPG. File q/w/e → 198 KB, lainnya → 138 KB. Video tidak diterima.")

# ==========================
# SIDEBAR SETTINGS
# ==========================
with st.sidebar:
    st.header("⚙️ Pengaturan")
    SPEED_PRESET = st.selectbox("Preset kecepatan", ["fast", "balanced"], index=0)
    MIN_SIDE_PX = st.number_input("Sisi terpendek minimum (px)", 64, 2048, 256, 32)
    SCALE_MIN = st.slider("Skala minimum saat downscale", 0.10, 0.75, 0.35, 0.05)
    SHARPEN_ON_RESIZE = st.checkbox("Sharpen ringan setelah resize", True)
    SHARPEN_AMOUNT = st.slider("Sharpen amount", 0.0, 2.0, 1.0, 0.1)
    PDF_DPI = 150 if SPEED_PRESET == "fast" else 200
    MASTER_ZIP_NAME = st.text_input("Nama master ZIP", "compressed.zip")
    st.markdown("**Target otomatis:**")
    st.markdown("- File **q, w, e** → **≤198 KB**")
    st.markdown("- File lainnya → **≤138 KB**")
    st.divider()
    if st.session_state["results"] is not None:
        if st.button("🗑️ Hapus semua hasil kompres", type="secondary"):
            st.session_state["results"] = None
            st.session_state["uploader_key"] += 1
            st.success("Hasil kompres dihapus.")
            st.rerun()

# ==========================
# KONSTANTA
# ==========================
MAX_QUALITY = 95
MIN_QUALITY = 15
BG_FOR_ALPHA = (255, 255, 255)
THREADS = min(4, max(2, (os.cpu_count() or 2)))
ZIP_COMP_ALGO = zipfile.ZIP_STORED if SPEED_PRESET == "fast" else zipfile.ZIP_DEFLATED
TARGET_KB_HIGH = 198
TARGET_KB_LOW = 138
IMG_EXT = {".jpg", ".jpeg", ".jfif", ".png", ".webp", ".tif", ".tiff", ".bmp", ".gif", ".heic", ".heif"}
PDF_EXT = {".pdf"}
ALLOW_ZIP = True

# ==========================
# HELPER FUNCTIONS
# ==========================
def get_target_size_for_path(relpath: Path) -> int:
    filename_lower = relpath.stem.lower()
    if filename_lower in ["q", "w", "e"]:
        return TARGET_KB_HIGH
    return TARGET_KB_LOW


def maybe_sharpen(img: Image.Image, do_it=True, amount=1.0) -> Image.Image:
    if not do_it or amount <= 0:
        return img
    return img.filter(ImageFilter.UnsharpMask(radius=1.0, percent=int(150 * amount), threshold=2))


def to_rgb_flat(img: Image.Image, bg=BG_FOR_ALPHA) -> Image.Image:
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        base = Image.new("RGB", img.size, bg)
        base.paste(img, mask=img.convert("RGBA").split()[-1])
        return base
    if img.mode != "RGB":
        return img.convert("RGB")
    return img


def save_jpg_bytes(img: Image.Image, quality: int) -> bytes:
    buf = io.BytesIO()
    if SPEED_PRESET == "fast":
        img.save(buf, format="JPEG", quality=quality, optimize=False, progressive=False, subsampling=2)
    else:
        img.save(buf, format="JPEG", quality=quality, optimize=True, progressive=True, subsampling=2)
    return buf.getvalue()


def try_quality_bs(img: Image.Image, target_kb: int, q_min=MIN_QUALITY, q_max=MAX_QUALITY):
    lo, hi = q_min, q_max
    best_bytes = None
    best_q = None
    while lo <= hi:
        mid = (lo + hi) // 2
        data = save_jpg_bytes(img, mid)
        if len(data) <= target_kb * 1024:
            best_bytes, best_q = data, mid
            lo = mid + 1
        else:
            hi = mid - 1
    return best_bytes, best_q


def resize_to_scale(img: Image.Image, scale: float, do_sharpen=True, amount=1.0) -> Image.Image:
    w, h = img.size
    nw, nh = max(int(w * scale), 1), max(int(h * scale), 1)
    out = img.resize((nw, nh), Image.LANCZOS)
    return maybe_sharpen(out, do_sharpen, amount)


def ensure_min_side(img: Image.Image, min_side_px: int, do_sharpen=True, amount=1.0) -> Image.Image:
    w, h = img.size
    if min(w, h) >= min_side_px:
        return img
    scale = max(min_side_px / max(min(w, h), 1), 1.0)
    return resize_to_scale(img, scale, do_sharpen, amount)


def load_image_from_bytes(name: str, raw: bytes) -> Image.Image:
    im = Image.open(io.BytesIO(raw))
    return ImageOps.exif_transpose(im)


def gif_first_frame(im: Image.Image) -> Image.Image:
    try:
        im.seek(0)
    except Exception:
        pass
    return im.convert("RGBA") if im.mode == "P" else im


def pdf_bytes_to_images(pdf_bytes: bytes, dpi: int) -> List[Image.Image]:
    images = []
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        for page in doc:
            pix = page.get_pixmap(matrix=fitz.Matrix(dpi/72.0, dpi/72.0), alpha=False)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            images.append(ImageOps.exif_transpose(img))
    return images


def extract_zip_to_memory(zf_bytes: bytes) -> List[Tuple[Path, bytes]]:
    out = []
    with zipfile.ZipFile(io.BytesIO(zf_bytes), "r") as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            with zf.open(info, "r") as f:
                data = f.read()
            out.append((Path(info.filename), data))
    return out


def guess_base_name_from_zip(zipname: str) -> str:
    base = Path(zipname).stem
    return base or "output"


def process_one_file_entry(relpath: Path, raw_bytes: bytes, input_root_label: str):
    processed = []
    outputs = {}
    skipped = []
    ext = relpath.suffix.lower()
    target_kb = get_target_size_for_path(relpath)
    try:
        if ext in PDF_EXT:
            pages = pdf_bytes_to_images(raw_bytes, dpi=PDF_DPI)
            for idx, pil_img in enumerate(pages, start=1):
                data, scale, q, size_b = compress_into_range(pil_img, target_kb, MIN_SIDE_PX, SCALE_MIN, SHARPEN_ON_RESIZE, SHARPEN_AMOUNT)
                out_rel = relpath.with_suffix("").as_posix() + f"_p{idx}.jpg"
                outputs[out_rel] = data
                processed.append((out_rel, size_b, scale, q, size_b <= target_kb*1024, target_kb))
        elif ext in IMG_EXT and (ext not in {".heic", ".heif"} or HEIF_OK):
            im = load_image_from_bytes(relpath.name, raw_bytes)
            if ext == ".gif":
                im = gif_first_frame(im)
            data, scale, q, size_b = compress_into_range(im, target_kb, MIN_SIDE_PX, SCALE_MIN, SHARPEN_ON_RESIZE, SHARPEN_AMOUNT)
            out_rel = relpath.with_suffix(".jpg").as_posix()
            outputs[out_rel] = data
            processed.append((out_rel, size_b, scale, q, size_b <= target_kb*1024, target_kb))
    except Exception as e:
        skipped.append((str(relpath), str(e)))
    return input_root_label, processed, skipped, outputs


def compress_into_range(base_img, max_kb, min_side_px, scale_min, do_sharpen, sharpen_amount):
    base = to_rgb_flat(base_img)
    data, q = try_quality_bs(base, max_kb)
    if data and len(data) <= max_kb * 1024:
        return data, 1.0, q, len(data)
    lo, hi = scale_min, 1.0
    best_pack = None
    for _ in range(10):
        mid = (lo + hi) / 2
        candidate = resize_to_scale(base, mid, do_sharpen, sharpen_amount)
        candidate = ensure_min_side(candidate, min_side_px, do_sharpen, sharpen_amount)
        d, q2 = try_quality_bs(candidate, max_kb)
        if d and len(d) <= max_kb * 1024:
            best_pack = (d, mid, q2, len(d))
            lo = mid + 0.1
        else:
            hi = mid - 0.1
    if best_pack:
        return best_pack
    smallest = resize_to_scale(base, scale_min, do_sharpen, sharpen_amount)
    smallest = ensure_min_side(smallest, min_side_px, do_sharpen, sharpen_amount)
    d = save_jpg_bytes(smallest, MIN_QUALITY)
    return d, scale_min, MIN_QUALITY, len(d)


# ==========================
# UI UPLOAD DAN PROSES
# ==========================
st.subheader("1) Upload ZIP atau File Lepas")
allowed_exts = sorted({e.lstrip('.') for e in IMG_EXT.union(PDF_EXT)} | {"zip"})
uploaded_files = st.file_uploader("Upload beberapa ZIP (gambar/PDF).", type=allowed_exts, accept_multiple_files=True, key=f"upload_{st.session_state['uploader_key']}")

col1, col2 = st.columns([3, 1])
with col1:
    run = st.button("🚀 Proses & Buat Master ZIP", type="primary", disabled=not uploaded_files)
with col2:
    if st.button("↺ Reset upload"):
        st.session_state["uploader_key"] += 1
        st.rerun()

if run:
    if not uploaded_files:
        st.warning("Silakan upload minimal satu file.")
        st.stop()

    jobs = []
    used_labels = set()

    def unique_name(base, used):
        name = base
        idx = 2
        while name in used:
            name = f"{base}_{idx}"
            idx += 1
        used.add(name)
        return name

    zip_inputs, loose_inputs = [], []
    for f in uploaded_files:
        name, raw = f.name, f.read()
        if name.lower().endswith(".zip"):
            zip_inputs.append((name, raw))
        else:
            loose_inputs.append((name, raw))

    allowed = IMG_EXT.union(PDF_EXT)
    for zname, zbytes in zip_inputs:
        try:
            pairs = extract_zip_to_memory(zbytes)
            base_label = unique_name(guess_base_name_from_zip(zname), used_labels)
            items = [(relp, data) for (relp, data) in pairs if relp.suffix.lower() in allowed]
            if items:
                jobs.append({"label": base_label, "items": items})
        except Exception as e:
            st.error(f"Gagal membuka ZIP {zname}: {e}")

    if loose_inputs:
        ts = time.strftime("%Y%m%d_%H%M%S")
        base_label = unique_name(f"compressed_pict_{ts}", used_labels)
        items = [(Path(name), data) for (name, data) in loose_inputs if Path(name).suffix.lower() in allowed]
        if items:
            jobs.append({"label": base_label, "items": items})

    if not jobs:
        st.error("Tidak ada file valid.")
        st.stop()

    summary, skipped_all = defaultdict(list), defaultdict(list)
    master_buf = io.BytesIO()
    zip_lock = threading.Lock()

    with zipfile.ZipFile(master_buf, "w", compression=ZIP_COMP_ALGO) as master:
        top_folders = {}
        for job in jobs:
            top = f"{job['label']}_compressed"
            top_folders[job['label']] = top
            master.writestr(f"{top}/", "")

        def add_to_zip(folder, rel_path, data):
            with zip_lock:
                master.writestr(f"{folder}/{rel_path}", data)

        def worker(label, relp, raw):
            return process_one_file_entry(relp, raw, label)

        tasks = [(j['label'], r, d) for j in jobs for (r, d) in j['items']]
        progress = st.progress(0)
        done = 0

        with ThreadPoolExecutor(max_workers=THREADS) as ex:
            for fut in as_completed([ex.submit(worker, *t) for t in tasks]):
                label, prc, skp, outs = fut.result()
                summary[label].extend(prc)
                skipped_all[label].extend(skp)
                if outs:
                    for path, data in outs.items():
                        add_to_zip(top_folders[label], path, data)
                done += 1
                progress.progress(min(done / len(tasks), 1.0))

    master_buf.seek(0)
    st.session_state["results"] = {"jobs": jobs, "summary": summary, "skipped": skipped_all, "master": master_buf.getvalue()}

# ==========================
# HASIL
# ==========================
if st.session_state["results"]:
    jobs = st.session_state["results"]["jobs"]
    summary = st.session_state["results"]["summary"]
    skipped_all = st.session_state["results"]["skipped"]

    st.subheader("📊 Ringkasan Hasil & Unduh")
    for job in jobs:
        base = job["label"]
        items = summary[base]
        skipped = skipped_all[base]
        with st.expander(f"📦 {base} — {len(items)} file diproses, {len(skipped)} dilewati", expanded=False):
            for name, size_b, scale, q, in_range, target_kb in items[:300]:
                kb = size_b / 1024
                flag = "✅" if in_range else "⚠️"
                st.write(f"{flag} `{name}` → {kb:.1f} KB (≤{target_kb} KB) | scale≈{scale:.3f} | quality={q}")
            if len(items) > 300:
                st.caption(f"(+{len(items)-300} baris disembunyikan)")
            if skipped:
                st.write("**Dilewati/Errored:**")
                for n, reason in skipped[:50]:
                    st.write(f"- {n}: {reason}")

    st.write("---")
    with st.container():
        st.subheader("Unduh Master ZIP")
        c1, c2, c3 = st.columns([1, 2, 1])
        with c2:
            st.download_button(
                "⬇️ Download Master ZIP",
                data=st.session_state["results"]["master"],
                file_name=(MASTER_ZIP_NAME.strip() or "compressed.zip"),
                mime="application/zip"
            )
        st.caption("Gunakan tombol di sidebar untuk menghapus hasil bila perlu.")
