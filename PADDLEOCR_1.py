
import os
import sys
import logging
import traceback
import re

import fitz  # PyMuPDF
import numpy as np
import cv2


# ============================================================
# PADDLEOCR / PADDLE STABILITY SETTINGS
# ============================================================

os.environ["FLAGS_enable_pir_api"] = "0"
os.environ["FLAGS_enable_pir_in_executor"] = "0"
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT"] = "0"
os.environ["PADDLEX_DISABLE_STATIC"] = "1"
os.environ["FLAGS_allocator_strategy"] = "naive_best_fit"

os.environ.setdefault(
    "HUB_DATASET_ENDPOINT",
    "https://modelscope.cn"
)

# Reduce noisy logs
logging.getLogger("ppocr").setLevel(logging.ERROR)
logging.getLogger("paddle").setLevel(logging.ERROR)
logging.getLogger("paddlex").setLevel(logging.ERROR)


# ============================================================
# IMPORT PADDLEOCR
# ============================================================

try:
    from paddleocr import PaddleOCR

except ImportError as e:

    print("\n❌ PaddleOCR is not installed.")
    print("\nInstall the required packages with:")
    print()
    print("pip install -U paddleocr pymupdf opencv-python numpy")
    print()
    sys.exit(1)


# ============================================================
# CONFIGURATION
# ============================================================

PDF_PATH = r"C:\Users\michael.hanna\Desktop\pop shenoda.pdf"

OUTPUT_MD = r"C:\Users\michael.hanna\Desktop\pop_shenoda_output.md"

# ------------------------------------------------------------
# OCR settings
# ------------------------------------------------------------

LANGUAGE = "ar"

DEVICE = "cpu"

# 220 DPI is a good starting point for a scanned book.
#
# 144 DPI  -> Matrix(2,2)
# 220 DPI  -> this setting
# 300 DPI  -> slower and more memory intensive
#
DPI = 220

# Accept slightly imperfect OCR.
MIN_CONFIDENCE = 0.15

# ------------------------------------------------------------
# Resume settings
# ------------------------------------------------------------

ENABLE_RESUME = True

# Flush Markdown after every page
SAVE_AFTER_EVERY_PAGE = True


# ============================================================
# RENDER PDF PAGE
# ============================================================

def render_page(page, dpi=220):
    """
    Render one PDF page as an OpenCV BGR image.
    """

    scale = dpi / 72.0

    matrix = fitz.Matrix(
        scale,
        scale
    )

    pix = page.get_pixmap(
        matrix=matrix,
        colorspace=fitz.csRGB,
        alpha=False
    )

    img_bytes = pix.tobytes("png")

    nparr = np.frombuffer(
        img_bytes,
        dtype=np.uint8
    )

    image = cv2.imdecode(
        nparr,
        cv2.IMREAD_COLOR
    )

    if image is None:
        raise RuntimeError(
            "OpenCV could not decode the rendered PDF page."
        )

    return image


# ============================================================
# GENERIC PADDLE RESULT ACCESS
# ============================================================

def get_value(obj, key, default=None):
    """
    Safely retrieve a value from:
      - dict
      - PaddleOCR result object
      - object attribute
      - dictionary-style object
    """

    if obj is None:
        return default

    # Dictionary
    if isinstance(obj, dict):
        return obj.get(key, default)

    # Attribute
    try:
        return getattr(obj, key)
    except Exception:
        pass

    # Dictionary-style indexing
    try:
        return obj[key]
    except Exception:
        pass

    return default


# ============================================================
# EXTRACT OCR BLOCKS
# ============================================================

def extract_blocks_from_result(result):
    """
    Extract text, confidence and polygon coordinates from
    PaddleOCR 3.x result objects.

    Expected PaddleOCR 3.x fields include:

        rec_texts
        rec_scores
        rec_polys

    Some versions use:

        dt_polys

    This function also contains an older PaddleOCR fallback.
    """

    blocks = []

    if result is None:
        return blocks


    # ========================================================
    # PADDLEOCR 3.x RESULT
    # ========================================================

    rec_texts = get_value(
        result,
        "rec_texts"
    )

    rec_scores = get_value(
        result,
        "rec_scores"
    )

    rec_polys = get_value(
        result,
        "rec_polys"
    )

    if rec_polys is None:

        rec_polys = get_value(
            result,
            "dt_polys"
        )


    if rec_texts is not None:

        try:

            texts = list(rec_texts)

            if rec_scores is None:
                scores = [1.0] * len(texts)
            else:
                scores = list(rec_scores)

            if rec_polys is None:
                polys = [None] * len(texts)
            else:
                polys = list(rec_polys)


            for i, text in enumerate(texts):

                if text is None:
                    continue

                text = str(text).strip()

                if not text:
                    continue


                # Confidence
                if i < len(scores):

                    try:
                        score = float(scores[i])
                    except Exception:
                        score = 1.0

                else:
                    score = 1.0


                # Bounding polygon
                if i < len(polys):
                    box = polys[i]
                else:
                    box = None


                blocks.append({
                    "text": text,
                    "score": score,
                    "box": box
                })


            if blocks:
                return blocks


        except Exception as e:

            print(
                "\n   ⚠️ PaddleOCR 3.x extraction error:"
            )

            print(
                f"      {e}"
            )


    # ========================================================
    # DICTIONARY STYLE FALLBACK
    # ========================================================

    if isinstance(result, dict):

        texts = result.get(
            "texts",
            result.get(
                "text",
                None
            )
        )

        if texts is not None:

            if isinstance(texts, str):
                texts = [texts]

            try:
                texts = list(texts)
            except Exception:
                texts = [texts]


            scores = result.get(
                "scores",
                result.get(
                    "rec_scores",
                    []
                )
            )

            boxes = result.get(
                "boxes",
                result.get(
                    "rec_polys",
                    result.get(
                        "dt_polys",
                        []
                    )
                )
            )


            try:
                scores = list(scores)
            except Exception:
                scores = []


            try:
                boxes = list(boxes)
            except Exception:
                boxes = []


            for i, text in enumerate(texts):

                if text is None:
                    continue

                text = str(text).strip()

                if not text:
                    continue


                try:
                    score = float(
                        scores[i]
                    ) if i < len(scores) else 1.0

                except Exception:
                    score = 1.0


                box = (
                    boxes[i]
                    if i < len(boxes)
                    else None
                )


                blocks.append({
                    "text": text,
                    "score": score,
                    "box": box
                })


            if blocks:
                return blocks


    # ========================================================
    # OLD PADDLEOCR FORMAT FALLBACK
    # ========================================================

    try:

        old_result = result

        if isinstance(
            old_result,
            (list, tuple)
        ):

            # Sometimes:
            #
            # [
            #   [
            #       [box, ("text", confidence)],
            #       ...
            #   ]
            # ]

            if (
                len(old_result) == 1
                and isinstance(
                    old_result[0],
                    list
                )
            ):

                old_result = old_result[0]


            for item in old_result:

                if not isinstance(
                    item,
                    (list, tuple)
                ):
                    continue

                if len(item) != 2:
                    continue


                box = item[0]

                recognition = item[1]

                if not isinstance(
                    recognition,
                    (list, tuple)
                ):
                    continue


                if len(recognition) == 0:
                    continue


                text = str(
                    recognition[0]
                ).strip()


                if not text:
                    continue


                try:

                    score = float(
                        recognition[1]
                    ) if len(recognition) > 1 else 1.0

                except Exception:

                    score = 1.0


                blocks.append({
                    "text": text,
                    "score": score,
                    "box": box
                })


    except Exception:

        pass


    return blocks


# ============================================================
# BOX GEOMETRY
# ============================================================

def get_box_geometry(box):
    """
    Convert OCR polygon/rectangle into useful coordinates.
    """

    if box is None:
        return None

    try:

        arr = np.asarray(
            box,
            dtype=float
        )


        # Polygon:
        #
        # [[x1,y1],
        #  [x2,y2],
        #  [x3,y3],
        #  [x4,y4]]

        if (
            arr.ndim == 2
            and arr.shape[1] == 2
        ):

            x_min = float(
                np.min(arr[:, 0])
            )

            x_max = float(
                np.max(arr[:, 0])
            )

            y_min = float(
                np.min(arr[:, 1])
            )

            y_max = float(
                np.max(arr[:, 1])
            )


            return {
                "x_min": x_min,
                "x_max": x_max,
                "y_min": y_min,
                "y_max": y_max,

                "width": x_max - x_min,

                "height": y_max - y_min,

                "center_x":
                    (x_min + x_max) / 2,

                "center_y":
                    (y_min + y_max) / 2
            }


        # Rectangle:
        #
        # [x1, y1, x2, y2]

        if (
            arr.ndim == 1
            and len(arr) == 4
        ):

            x1, y1, x2, y2 = arr


            return {
                "x_min":
                    float(min(x1, x2)),

                "x_max":
                    float(max(x1, x2)),

                "y_min":
                    float(min(y1, y2)),

                "y_max":
                    float(max(y1, y2)),

                "width":
                    float(abs(x2 - x1)),

                "height":
                    float(abs(y2 - y1)),

                "center_x":
                    float((x1 + x2) / 2),

                "center_y":
                    float((y1 + y2) / 2)
            }


    except Exception:

        pass


    return None


# ============================================================
# CLEAN ARABIC
# ============================================================

def clean_text(text):
    """
    Clean OCR whitespace without altering Arabic Unicode
    character order.

    DO NOT use arabic_reshaper or bidi.get_display here.
    """

    if not text:
        return ""

    text = str(text)

    text = text.replace(
        "\r",
        " "
    )

    text = text.replace(
        "\n",
        " "
    )

    # Collapse whitespace
    text = " ".join(
        text.split()
    )

    return text.strip()


# ============================================================
# READING ORDER
# ============================================================

def sort_blocks_reading_order(blocks):
    """
    Sort Arabic OCR blocks approximately:

        top -> bottom
        right -> left

    This works better for Arabic than a simple x/y sort.
    """

    prepared = []


    for block in blocks:

        geometry = get_box_geometry(
            block.get("box")
        )

        if geometry is None:
            continue

        block["geometry"] = geometry

        prepared.append(block)


    if not prepared:
        return []


    # --------------------------------------------------------
    # Estimate typical line height
    # --------------------------------------------------------

    heights = [
        block["geometry"]["height"]
        for block in prepared
        if block["geometry"]["height"] > 0
    ]


    if heights:

        median_height = float(
            np.median(heights)
        )

    else:

        median_height = 30.0


    # --------------------------------------------------------
    # Sort by vertical position
    # --------------------------------------------------------

    prepared.sort(
        key=lambda block:
            block["geometry"]["center_y"]
    )


    # --------------------------------------------------------
    # Group blocks into approximate text lines
    # --------------------------------------------------------

    lines = []


    for block in prepared:

        y = block[
            "geometry"
        ]["center_y"]

        placed = False


        for line in lines:

            line_y = float(
                np.mean([
                    item[
                        "geometry"
                    ]["center_y"]
                    for item in line
                ])
            )


            tolerance = max(
                median_height * 0.55,
                10
            )


            if abs(
                y - line_y
            ) <= tolerance:

                line.append(block)

                placed = True

                break


        if not placed:

            lines.append(
                [block]
            )


    # --------------------------------------------------------
    # Arabic reading direction:
    #
    # RIGHT -> LEFT
    # --------------------------------------------------------

    for line in lines:

        line.sort(
            key=lambda block:
                block[
                    "geometry"
                ]["center_x"],
            reverse=True
        )


    # --------------------------------------------------------
    # Flatten
    # --------------------------------------------------------

    ordered = []


    for line in lines:

        ordered.extend(
            line
        )


    return ordered


# ============================================================
# HEADING DETECTION
# ============================================================

def looks_like_heading(
    block,
    page_width
):

    text = block["text"]

    geometry = block["geometry"]


    if not text:
        return False


    # Don't make very long OCR lines headings
    if len(text) > 100:
        return False


    # Short text can potentially be a heading
    if len(text) <= 60:

        if (
            geometry["width"]
            > page_width * 0.35
        ):

            return True


    return False


# ============================================================
# WRITE PAGE
# ============================================================

def write_page_markdown(
    md_file,
    page_number,
    blocks,
    page_width
):

    md_file.write(
        f"\n<!-- START OF PAGE "
        f"{page_number} -->\n\n"
    )


    if not blocks:

        md_file.write(
            "*[No readable text detected "
            "on this page.]*\n\n"
        )

        md_file.write(
            f"<!-- END OF PAGE "
            f"{page_number} -->\n\n"
        )

        return 0


    # Sort OCR blocks
    blocks = sort_blocks_reading_order(
        blocks
    )


    lines_written = 0


    for block in blocks:

        text = clean_text(
            block["text"]
        )


        score = block["score"]


        if not text:
            continue


        if score < MIN_CONFIDENCE:
            continue


        geometry = block.get(
            "geometry"
        )


        if geometry is None:
            continue


        # ----------------------------------------------------
        # Heading
        # ----------------------------------------------------

        if looks_like_heading(
            block,
            page_width
        ):

            md_file.write(
                f"## {text}\n\n"
            )


        # ----------------------------------------------------
        # Normal OCR text
        # ----------------------------------------------------

        else:

            md_file.write(
                f"{text}\n\n"
            )


        lines_written += 1


    md_file.write(
        f"<!-- END OF PAGE "
        f"{page_number} -->\n\n"
    )


    return lines_written


# ============================================================
# FIND LAST COMPLETED PAGE
# ============================================================

def find_resume_page(
    output_path
):

    if not ENABLE_RESUME:
        return 1


    if not os.path.exists(
        output_path
    ):

        return 1


    try:

        with open(
            output_path,
            "r",
            encoding="utf-8"
        ) as f:

            content = f.read()


        completed_pages = re.findall(
            r"<!-- END OF PAGE (\d+) -->",
            content
        )


        if not completed_pages:

            return 1


        last_page = max(
            int(page)
            for page in completed_pages
        )


        return last_page + 1


    except Exception:

        return 1


# ============================================================
# MAIN
# ============================================================

def process_pdf_to_markdown():

    print()
    print("=" * 70)
    print("       ARABIC PDF → MARKDOWN OCR")
    print("=" * 70)


    # ========================================================
    # CHECK PDF
    # ========================================================

    print()
    print("📁 Checking PDF...")


    if not os.path.exists(
        PDF_PATH
    ):

        print()
        print(
            "❌ PDF not found:"
        )

        print(
            PDF_PATH
        )

        return


    print(
        "   ✓ PDF found"
    )


    # ========================================================
    # OPEN PDF
    # ========================================================

    print()
    print("📖 Opening PDF...")


    try:

        doc = fitz.open(
            PDF_PATH
        )

    except Exception as e:

        print()
        print(
            f"❌ Could not open PDF: {e}"
        )

        return


    total_pages = len(doc)


    print(
        f"   ✓ Total pages: "
        f"{total_pages}"
    )


    # ========================================================
    # RESUME
    # ========================================================

    start_page = find_resume_page(
        OUTPUT_MD
    )


    if start_page > 1:

        print()
        print(
            f"🔄 Resume mode enabled."
        )

        print(
            f"   Last completed page: "
            f"{start_page - 1}"
        )

        print(
            f"   Starting at page: "
            f"{start_page}"
        )

    else:

        print()
        print(
            "▶ Starting from page 1"
        )


    # ========================================================
    # INITIALIZE PADDLEOCR
    # ========================================================

    print()
    print(
        "🧠 Initializing PaddleOCR..."
    )

    print(
        f"   Language: {LANGUAGE}"
    )

    print(
        f"   Device: {DEVICE}"
    )


    try:

        ocr = PaddleOCR(
            lang=LANGUAGE,
            device=DEVICE
        )

    except Exception as e:

        print()
        print(
            "❌ PaddleOCR initialization failed."
        )

        print()
        print(
            f"Error:\n{e}"
        )

        traceback.print_exc()

        doc.close()

        return


    print(
        "   ✓ PaddleOCR initialized"
    )


    # ========================================================
    # OPEN OUTPUT
    # ========================================================

    file_mode = (
        "a"
        if start_page > 1
        else "w"
    )


    print()
    print(
        f"📝 Output:"
    )

    print(
        f"   {OUTPUT_MD}"
    )


    with open(
        OUTPUT_MD,
        file_mode,
        encoding="utf-8",
        newline="\n"
    ) as md_file:


        # ====================================================
        # PROCESS PAGES
        # ====================================================

        for page_index in range(
            start_page - 1,
            total_pages
        ):

            page_number = (
                page_index + 1
            )


            print()
            print(
                "-" * 70
            )

            print(
                f"[{page_number}/{total_pages}] "
                f"Processing page..."
            )


            try:

                # ============================================
                # GET PAGE
                # ============================================

                page = doc[
                    page_index
                ]


                # ============================================
                # RENDER
                # ============================================

                image = render_page(
                    page,
                    dpi=DPI
                )


                page_height, page_width = (
                    image.shape[:2]
                )


                print(
                    f"   Rendered: "
                    f"{page_width}x{page_height}"
                )


                # ============================================
                # OCR
                # ============================================

                print(
                    "   Running Arabic OCR...",
                    end=" ",
                    flush=True
                )


                try:

                    result = ocr.predict(
                        image
                    )


                    # ----------------------------------------
                    # IMPORTANT:
                    #
                    # PaddleOCR 3.x may return a generator.
                    #
                    # Convert it to actual result objects.
                    # ----------------------------------------

                    if result is None:

                        result_list = []

                    elif isinstance(
                        result,
                        (list, tuple)
                    ):

                        result_list = list(
                            result
                        )

                    else:

                        try:

                            result_list = list(
                                result
                            )

                        except TypeError:

                            result_list = [
                                result
                            ]


                    print("✓")


                except AttributeError:

                    # ----------------------------------------
                    # Older PaddleOCR fallback
                    # ----------------------------------------

                    print(
                        "Using legacy API..."
                    )


                    result = ocr.ocr(
                        image
                    )


                    if result is None:

                        result_list = []

                    elif isinstance(
                        result,
                        (list, tuple)
                    ):

                        result_list = list(
                            result
                        )

                    else:

                        result_list = [
                            result
                        ]


                # ============================================
                # DIAGNOSTIC INFORMATION
                # ============================================

                print(
                    f"   PaddleOCR result count: "
                    f"{len(result_list)}"
                )


                if result_list:

                    print(
                        f"   Result object type: "
                        f"{type(result_list[0])}"
                    )


                # ============================================
                # EXTRACT OCR BLOCKS
                # ============================================

                blocks = []


                for page_result in result_list:

                    page_blocks = (
                        extract_blocks_from_result(
                            page_result
                        )
                    )


                    blocks.extend(
                        page_blocks
                    )


                print(
                    f"   OCR blocks detected: "
                    f"{len(blocks)}"
                )


                # ============================================
                # IF ZERO BLOCKS
                # ============================================

                if not blocks:

                    print()
                    print(
                        "   ⚠️ WARNING:"
                    )

                    print(
                        "   PaddleOCR returned "
                        "zero recognized text blocks."
                    )

                    print(
                        "   This page will be marked "
                        "as having no readable text."
                    )


                # ============================================
                # WRITE MARKDOWN
                # ============================================

                lines_written = (
                    write_page_markdown(
                        md_file=md_file,
                        page_number=page_number,
                        blocks=blocks,
                        page_width=page_width
                    )
                )


                print(
                    f"   ✓ Markdown lines written: "
                    f"{lines_written}"
                )


                # ============================================
                # FLUSH
                # ============================================

                if SAVE_AFTER_EVERY_PAGE:

                    md_file.flush()


            # =================================================
            # USER INTERRUPT
            # =================================================

            except KeyboardInterrupt:

                print()
                print()
                print(
                    "⚠️ Processing interrupted "
                    "by user."
                )

                print()
                print(
                    "Completed pages remain saved "
                    "in the Markdown file."
                )

                break


            # =================================================
            # PAGE ERROR
            # =================================================

            except Exception as e:

                print()
                print(
                    f"❌ Error processing page "
                    f"{page_number}"
                )

                print(
                    f"   {e}"
                )


                traceback.print_exc()


                # Save error marker
                md_file.write(
                    f"\n<!-- ERROR ON PAGE "
                    f"{page_number}: "
                    f"{str(e)} -->\n\n"
                )


                md_file.flush()


                # Continue with next page
                continue


    # ========================================================
    # CLOSE PDF
    # ========================================================

    doc.close()


    # ========================================================
    # COMPLETE
    # ========================================================

    print()
    print("=" * 70)
    print("              🎯 OCR COMPLETE")
    print("=" * 70)

    print()

    print(
        "Markdown file:"
    )

    print(
        OUTPUT_MD
    )

    print()

    print(
        "You can open the file with:"
    )

    print(
        "  • VS Code"
    )

    print(
        "  • Obsidian"
    )

    print(
        "  • Typora"
    )

    print(
        "  • Notepad++"
    )

    print()


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":

    process_pdf_to_markdown()
