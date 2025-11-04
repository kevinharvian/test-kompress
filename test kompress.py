import io, os, zipfile, time, threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple, Dict
from collections import defaultdict

import streamlit as st
from PIL import Image, ImageOps, ImageFilter
import fitz  # PyMuPDF

HEIF_OK = False
try:
    import pillow_heif
    pillow_heif.register_heif_opener()
    HEIF_OK = True
except Exception:
    HEIF_OK = False

st.set_page_config(page_title="Multi-ZIP → JPG & Kompres (Auto Size)", page_icon="📦", layout="wide")

if "uploader_key" not in st.session_state:
    st.session_state["uploader_key"] = 0
if "results" not in st.session_state:
    st.session_state["results"] = None

st.title("📦 Multi-ZIP / Files → JPG & Kompres (Auto Size by Folder)")
st.caption("Konversi gambar (termasuk JFIF/HEIC) & PDF ke JPG. File q/w/e → 198 KB, lainnya → 138 KB. Video tidak diterima.")

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
VIDEO_EXT = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v", ".3gp", ".wmv", ".flv", ".mpg", ".mpeg"}

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

def compress_into_range(base_img: Image.Image, max_kb: int, min_side_px: int, scale_min: float, do_sharpen: bool, sharpen_amount: float):
    base = to_rgb_flat(base_img)

    data, q = try_quality_bs(base, max_kb)
    if data is not None and len(data) <= max_kb * 1024:
        result = (data, 1.0, q, len(data))
    else:
        lo, hi = scale_min, 1.0
        best_pack = None
        max_steps = 8 if SPEED_PRESET == "fast" else 12
        for _ in range(max_steps):
            mid = (lo + hi) / 2
            candidate = resize_to_scale(base, mid, do_sharpen, sharpen_amount)
            candidate = ensure_min_side(candidate, min_side_px, do_sharpen, sharpen_amount)
            d, q2 = try_quality_bs(candidate, max_kb)
            if d is not None and len(d) <= max_kb * 1024:
                best_pack = (d, mid, q2, len(d))
                lo = mid + (hi - mid) * 0.35
            else:
                hi = mid - (mid - lo) * 0.35
            if hi - lo < 1e-3:
                break

        if best_pack is None:
            smallest = resize_to_scale(base, scale_min, do_sharpen, sharpen_amount)
            smallest = ensure_min_side(smallest, min_side_px, do_sharpen, sharpen_amount)
            d = save_jpg_bytes(smallest, MIN_QUALITY)
            result = (d, scale_min, MIN_QUALITY, len(d))
        else:
            result = best_pack

    data, scale_used, q_used, size_b = result

    if size_b > max_kb * 1024:
        for q_try in range(q_used - 5, MIN_QUALITY - 1, -5):
            if q_try < MIN_QUALITY:
                q_try = MIN_QUALITY
            img_final = resize_to_scale(base, scale_used, do_sharpen, sharpen_amount)
            img_final = ensure_min_side(img_final, min_side_px, do_sharpen, sharpen_amount)
            d = save_jpg_bytes(img_final, q_try)
            if len(d) <= max_kb * 1024:
                data, scale_used, q_used, size_b = d, scale_used, q_try, len(d)
                break
            if q_try == MIN_QUALITY:
                break

    if size_b > max_kb * 1024:
        try:
            img_recompress = Image.open(io.BytesIO(data))
            img_recompress = ImageOps.exif_transpose(img_recompress)
            for scale_try in [0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5]:
                candidate = resize_to_scale(img_recompress, scale_try, do_sharpen, sharpen_amount)
                d, q2 = try_quality_bs(candidate, max_kb)
                if d is not None and len(d) <= max_kb * 1024:
                    return d, scale_used * scale_try, q2, len(d)
            smallest = resize_to_scale(img_recompress, 0.5, do_sharpen, sharpen_amount)
            d = save_jpg_bytes(smallest, MIN_QUALITY)
            return d, scale_used * 0.5, MIN_QUALITY, len(d)
        except Exception:
            pass

    return data, scale_used, q_used, size_b

def pdf_bytes_to_images(pdf_bytes: bytes, dpi: int) -> List[Image.Image]:
    images = []
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        for page in doc:
            rect = page.rect
            long_inch = max(rect.width, rect.height) / 72.0
            target_long_px = 2000
            dpi_eff = int(min(max(dpi, 72), max(72, target_long_px / max(long_inch, 1e-6))))
            zoom = dpi_eff / 72.0
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat, alpha=False)
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
    processed: List[Tuple[str, int, float, int, bool, int]] = []
    outputs: Dict[str, bytes] = {}
    skipped: List[Tuple[str, str]] = []

    ext = relpath.suffix.lower()
    target_kb = get_target_size_for_path(relpath)

    try:
        if ext in PDF_EXT:
            pages = pdf_bytes_to_images(raw_bytes, dpi=PDF_DPI)
            for idx, pil_img in enumerate(pages, start=1):
                try:
                    data, scale, q, size_b = compress_into_range(
                        pil_img, target_kb, MIN_SIDE_PX, SCALE_MIN, SHARPEN_ON_RESIZE, SHARPEN_AMOUNT
                    )
                    out_rel = relpath.with_suffix("").as_posix() + f"_p{idx}.jpg"
                    outputs[out_rel] = data
                    processed.append((out_rel, size_b, scale, q, size_b <= target_kb * 1024, target_kb))
                except Exception as e:
                    skipped.append((f"{relpath} (page {idx})", str(e)))
        elif ext in IMG_EXT and (ext not in {".heic", ".heif"} or HEIF_OK):
            im = load_image_from_bytes(relpath.name, raw_bytes)
            if ext == ".gif":
                im = gif_first_frame(im)
            data, scale, q, size_b = compress_into_range(
                im, target_kb, MIN_SIDE_PX, SCALE_MIN, SHARPEN_ON_RESIZE, SHARPEN_AMOUNT
            )
            out_rel = Path(relpath.with_suffix(".jpg").as_posix()).as_posix()
            outputs[out_rel] = data
            processed.append((out_rel, size_b, scale, q, size_b <= target_kb * 1024, target_kb))
        elif ext in {".heic", ".heif"} and not HEIF_OK:
            skipped.append((str(relpath), "Butuh pillow-heif (tidak tersedia)"))
    except Exception as e:
        skipped.append((str(relpath), str(e)))

    return input_root_label, processed, skipped, outputs

st.subheader("1) Upload ZIP atau File Lepas")
allowed_exts_for_uploader = sorted({e.lstrip(".") for e in IMG_EXT.union(PDF_EXT)} | ({"zip"} if ALLOW_ZIP else set()))

uploaded_files = st.file_uploader(
    "Upload beberapa ZIP (berisi folder/gambar/PDF) dan/atau file lepas (gambar/PDF). Video ditolak otomatis.",
    type=allowed_exts_for_uploader,
    accept_multiple_files=True,
    key=f"uploader_{st.session_state['uploader_key']}",
)

col_run, col_reset = st.columns([3, 1])
with col_run:
    run = st.button("🚀 Proses & Buat Master ZIP", type="primary", disabled=not uploaded_files)
with col_reset:
    if st.button("↺ Reset upload"):
        st.session_state["results"] = None
        st.session_state["uploader_key"] += 1
        st.rerun()

if run:
    if not uploaded_files:
        st.warning("Silakan upload minimal satu file.")
        st.stop()

    jobs = []
    used_labels = set()

    def unique_name(base: str, used: set) -> str:
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
        st.error("Tidak ada berkas valid (butuh gambar/PDF, atau ZIP berisi file-file tersebut).")
        st.stop()

    st.write(f"🔧 Ditemukan **{sum(len(j['items']) for j in jobs)}** berkas dari **{len(jobs)}** input.")

    summary: Dict[str, List[Tuple[str, int, float, int, bool, int]]] = defaultdict(list)
    skipped_all: Dict[str, List[Tuple[str, str]]] = defaultdict(list)

    master_buf = io.BytesIO()
    zip_write_lock = threading.Lock()
    with zipfile.ZipFile(master_buf, "w", compression=ZIP_COMP_ALGO) as master:
        top_folders: Dict[str, str] = {}
        for job in jobs:
            top = f"{job['label']}_compressed"
            top_folders[job['label']] = top
            master.writestr(f"{top}/", "")

        def add_to_master_zip_threadsafe(top_folder: str, rel_path: str, data: bytes):
            rel_path_std = str(Path(rel_path).with_suffix(".jpg")) if not rel_path.lower().endswith(".jpg") else rel_path
            with zip_write_lock:
                master.writestr(f"{top_folder}/{rel_path_std}", data)

        def worker(label: str, relp: Path, raw: bytes):
            return process_one_file_entry(relp, raw, label)

        all_tasks = [(job["label"], relp, data) for job in jobs for (relp, data) in job["items"]]
        total, done = len(all_tasks), 0
        progress = st.progress(0.0)

        with ThreadPoolExecutor(max_workers=THREADS) as ex:
            futures = [ex.submit(worker, *t) for t in all_tasks]
            for fut in as_completed(futures):
                label, prc, skp, outs = fut.result()
                summary[label].extend(prc)
                skipped_all[label].extend(skp)
                if outs:
                    top = top_folders[label]
                    for rel_path, data in outs.items():
                        add_to_master_zip_threadsafe(top, rel_path, data)
                done += 1
                progress.progress(min(done / total, 1.0))

    master_buf.seek(0)

    st.session_state["results"] = {
        "jobs": jobs,
        "summary": summary,
        "skipped_all": skipped_all,
        "master_bytes": master_buf.getvalue(),
    }

if st.session_state["results"] is not None:
    jobs = st.session_state["results"]["jobs"]
    summary = st.session_state["results"]["summary"]
    skipped_all = st.session_state["results"]["skipped_all"]

    st.subheader("📊 Ringkasan Hasil & Unduh")
    grand_ok, grand_cnt = 0, 0
    MAX_ROWS_PER_JOB = 300

    for job in jobs:
        base = job["label"]
        items = summary[base]
        skipped = skipped_all[base]
        with st.expander(f"📦 {base} — {len(items)} file diproses, {len(skipped)} dilewati/errored", expanded=False):
            ok = 0
            shown = 0
            for name, size_b, scale, q, in_range, target_kb in items:
                if shown >= MAX_ROWS_PER_JOB:
                    break
                kb = size_b / 1024
                flag = "✅" if in_range else "⚠️"
                st.write(f"{flag} `{name}` → **{kb:.1f} KB** (target: ≤{target_kb} KB) | scale≈{scale:.3f} | quality={q}")
                ok += 1 if in_range else 0
                shown += 1
            extra = len(items) - shown
            if extra > 0:
                st.caption(f"(+{extra} baris lainnya disembunyikan untuk menjaga performa UI)")
            if skipped:
                st.write("**Dilewati/Errored:**")
                for n, reason in skipped[:50]:
                    st.write(f"- {n}: {reason}")
            st.caption(f"Berhasil di bawah target: **{ok}/{len(items)}**")
            grand_ok += ok
            grand_cnt += len(items)

    st.write("---")
with st.container():
    st.subheader("Unduh Master ZIP")
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        st.download_button(
            "⬇️ Download Master ZIP",
            data=st.session_state["results"]["master_bytes"],
            file_name=(MASTER_ZIP_NAME.strip() or "compressed.zip"),
            mime="application/zip",
        )
    st.caption("Gunakan tombol di sidebar untuk menghapus hasil bila perlu.") or "compressed.zip"),
        mime="application/zip",
    )

else:
    st.info("Belum ada hasil. Upload file lalu jalankan proses.")
