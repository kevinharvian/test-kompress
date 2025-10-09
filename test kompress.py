# streamlit_app.py
import io, os, zipfile, time, threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from collections import defaultdict

import streamlit as st
from PIL import Image, ImageOps, ImageFilter, ImageEnhance
import fitz  # PyMuPDF

# ===== HEIC/HEIF =====
HEIF_OK = False
try:
    import pillow_heif
    pillow_heif.register_heif_opener()
    HEIF_OK = True
except Exception:
    HEIF_OK = False

# ===== PAGE =====
st.set_page_config(page_title="Multi-ZIP → JPG & Kompres 150–170 KB (Clear)", page_icon="📦", layout="wide")
st.title("📦 Multi-ZIP / Files → JPG & Kompres 150–170 KB (clear)")
st.caption("Semua input (HEIC/JPG/PNG/PDF) dikonversi ke JPG lalu dikompres. Target ≤170 KB, minimum 150 KB (akan di-upscale tipis jika perlu).")

# ===== Sidebar Settings =====
with st.sidebar:
    st.header("⚙️ Pengaturan")
    SPEED_PRESET = st.selectbox("Preset kecepatan", ["fast", "balanced"], index=1)

    # Dimensi aman untuk dokumen agar teks jelas
    MIN_SIDE_PX = st.number_input("Sisi terpendek minimum (px)", 64, 4096, 1024, 32)
    SCALE_MIN = st.slider("Skala minimum saat downscale", 0.10, 0.90, 0.55, 0.05)
    UPSCALE_MAX = st.slider("Batas upscale maksimum", 1.0, 2.0, 1.20, 0.05)
    LONG_SIDE_CLAMP = st.number_input("Batas sisi panjang (px, 0=nonaktif)", 0, 8192, 2200, 50)

    SHARPEN_ON_RESIZE = st.checkbox("Sharpen ringan setelah resize", True)
    SHARPEN_AMOUNT = st.slider("Sharpen amount", 0.0, 2.0, 1.0, 0.1)

    # Profil dokumen
    FORCE_DOCUMENT_MODE = st.checkbox("Paksa 'Document mode' untuk semua gambar", True)
    DOC_STRONG_DENOISE = st.checkbox("Denoise dokumen ekstra kuat", True)

    # PDF render
    PDF_DPI = 200 if SPEED_PRESET == "fast" else 240

    MASTER_ZIP_NAME = st.text_input("Nama master ZIP", "compressed.zip")
    st.markdown("**Target otomatis:** minimum 150 KB, maksimum 170 KB (fixed).")

# ===== Tunables =====
MAX_QUALITY = 92
MIN_QUALITY = 55                 # foto default (agar tidak blocky)
DOC_MIN_QUALITY = 45             # dokumen boleh turun agar muat ≤170 KB
PHOTO_MIN_QUALITY = MIN_QUALITY

BG_FOR_ALPHA = (255, 255, 255)
THREADS = min(4, max(2, (os.cpu_count() or 2)))
ZIP_COMP_ALGO = zipfile.ZIP_STORED if SPEED_PRESET == "fast" else zipfile.ZIP_DEFLATED

# ✅ Target size fixed by system
TARGET_KB = 170
MIN_KB = 150

IMG_EXT = {".jpg", ".jpeg", ".jfif", ".png", ".webp", ".tif", ".tiff", ".bmp", ".gif", ".heic", ".heif"}
PDF_EXT = {".pdf"}
ALLOW_ZIP = True
VIDEO_EXT = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v", ".3gp", ".wmv", ".flv", ".mpg", ".mpeg"}

# ===== Helpers =====
def maybe_sharpen(img: Image.Image, do_it=True, amount=1.0) -> Image.Image:
    if not do_it or amount <= 0:
        return img
    return img.filter(ImageFilter.UnsharpMask(radius=1.0, percent=int(150*amount), threshold=2))

def to_rgb_flat(img: Image.Image, bg=BG_FOR_ALPHA) -> Image.Image:
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        base = Image.new("RGB", img.size, bg)
        base.paste(img, mask=img.convert("RGBA").split()[-1])
        return base
    if img.mode != "RGB" and img.mode != "L":
        return img.convert("RGB")
    return img

def enhance_document(img: Image.Image, strong=False, do_sharpen=True, amount=1.0) -> Image.Image:
    # 1) Grayscale (hemat, hilangkan fringing warna)
    if img.mode != "L":
        img = ImageOps.grayscale(img)
    # 2) Normalisasi kontras
    img = ImageOps.autocontrast(img, cutoff=1)
    # 3) Denoise kuat (salt-pepper) + opsional extra
    img = img.filter(ImageFilter.MedianFilter(5 if strong else 3))
    if strong:
        try:
            img = img.filter(ImageFilter.ModeFilter(size=3))
        except Exception:
            pass
    # 4) Sedikit tingkatkan kontras lokal
    img = ImageEnhance.Contrast(img).enhance(1.1 if strong else 1.05)
    # 5) Unsharp ringan agar tepi huruf rapih
    return maybe_sharpen(img, do_sharpen, min(1.0, amount))

def save_jpg_bytes(img: Image.Image, quality: int, subsampling: int = 2, progressive=None, optimize=None) -> bytes:
    """subsampling: 2=4:2:0 (foto), 0=4:4:4 (dokumen). Jika img.mode == 'L' akan tersimpan grayscale."""
    buf = io.BytesIO()
    if progressive is None:
        progressive = (SPEED_PRESET != "fast")
    if optimize is None:
        optimize = (SPEED_PRESET != "fast")
    img.save(buf, format="JPEG", quality=quality, optimize=optimize, progressive=progressive, subsampling=subsampling)
    return buf.getvalue()

def try_quality_bs(img: Image.Image, target_kb: int, q_min: int, q_max: int, subsampling: int = 2):
    lo, hi = q_min, q_max
    best_bytes = None
    best_q = None
    while lo <= hi:
        mid = (lo + hi) // 2
        data = save_jpg_bytes(img, mid, subsampling=subsampling)
        if len(data) <= target_kb * 1024:
            best_bytes, best_q = data, mid
            lo = mid + 1
        else:
            hi = mid - 1
    return best_bytes, best_q

def resize_to_scale(img: Image.Image, scale: float, do_sharpen=True, amount=1.0) -> Image.Image:
    w, h = img.size
    nw, nh = max(int(w*scale), 1), max(int(h*scale), 1)
    out = img.resize((nw, nh), Image.LANCZOS)
    return maybe_sharpen(out, do_sharpen, amount)

def ensure_min_side(img: Image.Image, min_side_px: int, do_sharpen=True, amount=1.0) -> Image.Image:
    w, h = img.size
    if min(w, h) >= min_side_px:
        return img
    scale = max(min_side_px / max(min(w, h), 1), 1.0)
    return resize_to_scale(img, scale, do_sharpen, amount)

def clamp_long_side(img: Image.Image, long_max: int, do_sharpen=True, amount=1.0) -> Image.Image:
    if long_max <= 0:
        return img
    w, h = img.size
    long_side = max(w, h)
    if long_side <= long_max:
        return img
    scale = long_max / long_side
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

def compress_into_range(
    base_img: Image.Image,
    min_kb: int,
    max_kb: int,
    min_side_px: int,
    scale_min: float,
    upscale_max: float,
    do_sharpen: bool,
    sharpen_amount: float,
    subsampling: int = 2,                # 2=foto, 0=dokumen
    q_min_override: Optional[int] = None # quality floor khusus (dokumen/foto)
) -> Tuple[bytes, float, int, int]:
    """
    Return: (jpg_bytes, scale_used, quality_used, size_bytes)
    """
    q_min_local = q_min_override if q_min_override is not None else MIN_QUALITY

    # Flatten kecuali dokumen grayscale (mode 'L') biarkan tetap 'L'
    if base_img.mode != "L":
        base = to_rgb_flat(base_img)
    else:
        base = base_img

    if LONG_SIDE_CLAMP > 0:
        base = clamp_long_side(base, long_max=LONG_SIDE_CLAMP, do_sharpen=do_sharpen, amount=sharpen_amount)

    # 1) Coba tanpa ubah skala dulu
    data, q = try_quality_bs(base, max_kb, q_min_local, MAX_QUALITY, subsampling=subsampling)
    if data is not None:
        result = (data, 1.0, q, len(data))
    else:
        # 2) Binary search skala (downscale) agar muat max_kb
        lo, hi = scale_min, 1.0
        best_pack = None
        max_steps = 8 if SPEED_PRESET == "fast" else 12
        for _ in range(max_steps):
            mid = (lo + hi) / 2
            candidate = resize_to_scale(base, mid, do_sharpen, sharpen_amount)
            candidate = ensure_min_side(candidate, min_side_px, do_sharpen, sharpen_amount)
            d, q2 = try_quality_bs(candidate, max_kb, q_min_local, MAX_QUALITY, subsampling=subsampling)
            if d is not None:
                best_pack = (d, mid, q2, len(d))
                lo = mid + (hi - mid) * 0.35
            else:
                hi = mid - (mid - lo) * 0.35
            if hi - lo < 1e-3:
                break
        if best_pack is None:
            smallest = resize_to_scale(base, scale_min, do_sharpen, sharpen_amount)
            smallest = ensure_min_side(smallest, min_side_px, do_sharpen, sharpen_amount)
            d = save_jpg_bytes(smallest, q_min_local, subsampling=subsampling)
            result = (d, scale_min, q_min_local, len(d))
        else:
            result = best_pack

    data, scale_used, q_used, size_b = result

    # 3) Jika < MIN_KB → naikkan kualitas dulu, lalu (jika perlu) upscale tipis
    if size_b < min_kb * 1024:
        img_now = resize_to_scale(base, scale_used, do_sharpen, sharpen_amount)
        img_now = ensure_min_side(img_now, min_side_px, do_sharpen, sharpen_amount)
        d, q2 = try_quality_bs(img_now, max_kb, max(q_used, q_min_local), MAX_QUALITY, subsampling=subsampling)
        if d is not None and len(d) > size_b:
            data, q_used, size_b = d, q2, len(d)

        cur_scale = scale_used
        iters = 0
        while size_b < min_kb * 1024 and cur_scale < upscale_max and iters < 4:
            cur_scale = min(cur_scale * 1.10, upscale_max)  # 10% per langkah
            candidate = resize_to_scale(base, cur_scale, do_sharpen, sharpen_amount)
            candidate = ensure_min_side(candidate, min_side_px, do_sharpen, sharpen_amount)
            d, q3 = try_quality_bs(candidate, max_kb, max(q_used, q_min_local), MAX_QUALITY, subsampling=subsampling)
            if d is None or len(d) <= size_b:
                break
            data, q_used, size_b, scale_used = d, q3, len(d), cur_scale
            iters += 1

    return data, scale_used, q_used, size_b

def pdf_bytes_to_images(pdf_bytes: bytes, dpi: int) -> List[Image.Image]:
    images = []
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        for page in doc:
            rect = page.rect
            long_inch = max(rect.width, rect.height) / 72.0
            target_long_px = 2200
            dpi_eff = int(min(max(dpi, 96), max(96, target_long_px / max(long_inch, 1e-6))))
            zoom = dpi_eff / 72.0
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            images.append(ImageOps.exif_transpose(img))
    return images

def extract_zip_to_memory(zf_bytes: bytes) -> List[Tuple[Path, bytes]]:
    out = []
    with zipfile.ZipFile(io.BytesIO(zf_bytes), 'r') as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            with zf.open(info, 'r') as f:
                data = f.read()
            out.append((Path(info.filename), data))
    return out

def guess_base_name_from_zip(zipname: str) -> str:
    base = Path(zipname).stem
    return base or "output"

def process_one_file_entry(relpath: Path, raw_bytes: bytes, input_root_label: str):
    """
    Dokumen: grayscale + denoise + subsampling=0 + q_min=DOC_MIN_QUALITY
    Foto   : subsampling=2 + q_min=PHOTO_MIN_QUALITY
    """
    processed: List[Tuple[str, int, float, int, bool]] = []
    outputs: Dict[str, bytes] = {}
    skipped: List[Tuple[str, str]] = []

    ext = relpath.suffix.lower()
    try:
        if ext in PDF_EXT:
            pages = pdf_bytes_to_images(raw_bytes, dpi=PDF_DPI)
            for idx, pil_img in enumerate(pages, start=1):
                pil_img = enhance_document(pil_img, strong=DOC_STRONG_DENOISE,
                                           do_sharpen=SHARPEN_ON_RESIZE, amount=SHARPEN_AMOUNT)
                data, scale, q, size_b = compress_into_range(
                    pil_img, MIN_KB, TARGET_KB, MIN_SIDE_PX, SCALE_MIN, UPSCALE_MAX,
                    SHARPEN_ON_RESIZE, SHARPEN_AMOUNT, subsampling=0, q_min_override=DOC_MIN_QUALITY
                )
                out_rel = relpath.with_suffix("").as_posix() + f"_p{idx}.jpg"
                outputs[out_rel] = data
                processed.append((out_rel, size_b, scale, q, MIN_KB*1024 <= size_b <= TARGET_KB*1024))

        elif ext in IMG_EXT and (ext not in {".heic", ".heif"} or HEIF_OK):
            im = load_image_from_bytes(relpath.name, raw_bytes)
            if ext == ".gif":
                im = gif_first_frame(im)

            # heuristik ringan + override dari user
            w, h = im.size
            aspect = max(w, h) / max(1, min(w, h))
            is_document_like = (max(w, h) >= 1600 and min(w, h) >= 1100 and 1.2 <= aspect <= 1.6)
            as_document = FORCE_DOCUMENT_MODE or is_document_like

            if as_document:
                im = enhance_document(im, strong=DOC_STRONG_DENOISE,
                                      do_sharpen=SHARPEN_ON_RESIZE, amount=SHARPEN_AMOUNT)
                subs = 0
                qmin = DOC_MIN_QUALITY
            else:
                subs = 2
                qmin = PHOTO_MIN_QUALITY

            data, scale, q, size_b = compress_into_range(
                im, MIN_KB, TARGET_KB, MIN_SIDE_PX, SCALE_MIN, UPSCALE_MAX,
                SHARPEN_ON_RESIZE, SHARPEN_AMOUNT, subsampling=subs, q_min_override=qmin
            )
            out_rel = relpath.with_suffix(".jpg").as_posix()
            outputs[out_rel] = data
            processed.append((out_rel, size_b, scale, q, MIN_KB*1024 <= size_b <= TARGET_KB*1024))

        elif ext in {".heic", ".heif"} and not HEIF_OK:
            skipped.append((str(relpath), "Butuh pillow-heif (tidak tersedia)"))
        # else: format tidak didukung → di-skip
    except Exception as e:
        skipped.append((str(relpath), str(e)))

    return input_root_label, processed, skipped, outputs

# ===== UI Upload & Run =====
st.subheader("1) Upload ZIP atau File Lepas")
allowed_exts_for_uploader = sorted({e.lstrip('.') for e in IMG_EXT.union(PDF_EXT)} | ({"zip"} if ALLOW_ZIP else set()))
uploaded_files = st.file_uploader(
    "Upload beberapa ZIP (berisi folder/gambar/PDF) dan/atau file lepas (gambar/PDF). Video otomatis ditolak.",
    type=allowed_exts_for_uploader,
    accept_multiple_files=True
)

run = st.button("🚀 Proses & Buat Master ZIP", type="primary", disabled=not uploaded_files)

if run:
    if not uploaded_files:
        st.warning("Silakan upload minimal satu file."); st.stop()

    jobs = []
    used_labels = set()

    def unique_name(base: str, used: set) -> str:
        name = base; idx = 2
        while name in used:
            name = f"{base}_{idx}"; idx += 1
        used.add(name); return name

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
            st.error(f"❌ Gagal membuka ZIP {zname}: {e}")

    if loose_inputs:
        ts = time.strftime("%Y%m%d_%H%M%S")
        base_label = unique_name(f"compressed_pict_{ts}", used_labels)
        items = [(Path(name), data) for (name, data) in loose_inputs if Path(name).suffix.lower() in allowed]
        if items:
            jobs.append({"label": base_label, "items": items})

    if not jobs:
        st.error("Tidak ada berkas valid (butuh gambar/PDF, atau ZIP berisi file-file tersebut)."); st.stop()

    st.write(f"🔧 Ditemukan **{sum(len(j['items']) for j in jobs)}** berkas dari **{len(jobs)}** input.")

    summary: Dict[str, List[Tuple[str, int, float, int, bool]]] = defaultdict(list)
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
            with zip_write_lock:
                master.writestr(f"{top_folder}/{rel_path}", data)

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
                progress.progress(min(done/total, 1.0))

    master_buf.seek(0)

    st.subheader("📊 Ringkasan")
    grand_ok = 0; grand_cnt = 0
    MAX_ROWS_PER_JOB = 300

    for job in jobs:
        base = job["label"]; items = summary[base]; skipped = skipped_all[base]
        with st.expander(f"📦 {base} — {len(items)} file diproses, {len(skipped)} dilewati/errored"):
            ok = 0
            shown = 0
            for name, size_b, scale, q, in_range in items:
                if shown >= MAX_ROWS_PER_JOB:
                    break
                kb = size_b/1024
                flag = "✅" if in_range else ("🔼" if kb < MIN_KB else "⚠️")
                st.write(f"{flag} `{name}` → **{kb:.1f} KB** | scale≈{scale:.3f} | quality={q}")
                ok += 1 if in_range else 0
                shown += 1
            extra = len(items) - shown
            if extra > 0:
                st.caption(f"(+{extra} baris lainnya disembunyikan untuk menjaga performa UI)")
            if skipped:
                st.write("**Dilewati/Errored:**")
                for n, reason in skipped[:50]:
                    st.write(f"- {n}: {reason}")
            st.caption(f"Berhasil di rentang {MIN_KB}–{TARGET_KB} KB: **{ok}/{len(items)}**")
        grand_ok += ok; grand_cnt += len(items)

    st.write("---")
    st.write(f"**Total file OK di rentang:** {grand_ok}/{grand_cnt}")

    st.download_button(
        "⬇️ Download Master ZIP",
        data=master_buf.getvalue(),
        file_name=MASTER_ZIP_NAME.strip() or "compressed.zip",
        mime="application/zip",
    )
    st.success("Selesai! Master ZIP siap diunduh.")
