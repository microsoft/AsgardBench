from __future__ import annotations

import base64
import io
import mimetypes
import re
from pathlib import Path

import streamlit as st
from PIL import Image

MAX_IMAGE_SIZE = 360  # Resize images to fit within this box


@st.cache_data(show_spinner=False)
def _to_data_uri(path_str: str) -> str | None:
    """Load image, resize to MAX_IMAGE_SIZE, and return as data URI (cached)."""
    path = Path(path_str)
    if not path.exists() or not path.is_file():
        return None

    mime, _ = mimetypes.guess_type(str(path))
    if mime is None:
        mime = "application/octet-stream"

    # For images, resize before encoding
    if mime.startswith("image/"):
        try:
            with Image.open(path) as img:
                img.thumbnail(
                    (MAX_IMAGE_SIZE, MAX_IMAGE_SIZE), Image.Resampling.LANCZOS
                )
                buf = io.BytesIO()
                fmt = img.format or "PNG"
                img.save(buf, format=fmt)
                data = base64.b64encode(buf.getvalue()).decode("ascii")
                # Update mime based on actual saved format
                mime = f"image/{fmt.lower()}"
        except Exception:
            # Fallback to raw bytes if PIL fails
            data = base64.b64encode(path.read_bytes()).decode("ascii")
    else:
        data = base64.b64encode(path.read_bytes()).decode("ascii")

    return f"data:{mime};base64,{data}"


def inline_local_images(markdown: str, base_dir: Path) -> str:
    """Inline local images referenced by markdown/HTML into data URIs."""

    def resolve_src(src: str) -> str | None:
        src = src.strip().strip('"').strip("'")
        if src.startswith(("http://", "https://", "data:")):
            return None
        p = Path(src)
        if not p.is_absolute():
            p = (base_dir / p).resolve()
        uri = _to_data_uri(str(p))  # Pass string for caching
        return uri

    # HTML: <img src="..." ...>
    def repl_html_img(m: re.Match[str]) -> str:
        before = m.group(0)
        src = m.group("src")
        uri = resolve_src(src)
        if not uri:
            return before
        return before.replace(src, uri)

    markdown = re.sub(
        r"<img\s+[^>]*?src=(?P<q>\"|')(?P<src>.*?)(?P=q)[^>]*?>",
        repl_html_img,
        markdown,
        flags=re.IGNORECASE,
    )

    # Markdown: ![alt](path)
    def repl_md_img(m: re.Match[str]) -> str:
        alt = m.group("alt")
        src = m.group("src")
        uri = resolve_src(src)
        if not uri:
            return m.group(0)
        return f'<img alt="{alt}" src="{uri}" style="max-width:100%;" />'

    markdown = re.sub(
        r"!\[(?P<alt>[^\]]*)\]\((?P<src>[^\)]+)\)",
        repl_md_img,
        markdown,
    )

    return markdown


def list_reports() -> list[Path]:
    root = Path("copilot-reports")
    if not root.exists():
        return []
    return sorted(root.glob("*/REPORT.md"))


st.set_page_config(page_title="AsgardBench Report Viewer", layout="wide")

reports = list_reports()
if not reports:
    st.error("No reports found under copilot-reports/*/REPORT.md")
    st.stop()

options = [str(p) for p in reports]
selected = st.sidebar.selectbox("Report", options, index=0)
report_path = Path(selected)

raw_md = report_path.read_text(encoding="utf-8", errors="replace")
md = inline_local_images(raw_md, report_path.parent)

st.sidebar.download_button(
    "Download REPORT.md",
    data=raw_md,
    file_name=report_path.name,
    mime="text/markdown",
)

st.markdown(md, unsafe_allow_html=True)
