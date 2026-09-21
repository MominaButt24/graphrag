import re
import subprocess
import tempfile
from pathlib import Path

from langchain_core.documents import Document
from pypdf import PdfReader


IMAGE_PATTERN = re.compile(
    r"!\[\]\(data:image/([^;]+);base64,([^)]+)\)"
)


def get_pdf_page_count(filepath: str) -> int:
    """
    Return the total number of pages in the PDF.
    """
    reader = PdfReader(filepath)
    return len(reader.pages)


def run_mineru(
    filepath: str,
    output_dir: str,
    page_number: int,
) -> Path:
    """
    Run MinerU for a single PDF page.
    page_number is 1-based because MinerU's --pages argument
    uses PDF page numbering.
    """
    input_path = Path(filepath)
    output_path = Path(output_dir)

    output_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    subprocess.run(
        [
            "mineru-kit",
            "parse",
            str(input_path),
            "-o",
            str(output_path),
            "--pages",
            str(page_number),
            "--format",
            "markdown",
            "--tier",
            "standard",
            "--ocr-mode",
            "auto",
        ],
        check=True,
    )

    markdown_files = list(
        output_path.rglob("*.md")
    )

    if not markdown_files:
        raise RuntimeError(
            "MinerU completed but no Markdown output "
            f"was found for page {page_number}"
        )

    return markdown_files[0]


def decode_embedded_images(markdown: str):
    """
    Remove Base64 image data while preserving the image position.
    """
    image_count = 0

    def replace_image(match):
        nonlocal image_count
        image_count += 1
        return f"\n[IMAGE_{image_count}]\n"

    cleaned_markdown = IMAGE_PATTERN.sub(
        replace_image,
        markdown,
    )

    return cleaned_markdown, image_count


def clean_mineru_markdown(markdown: str) -> str:
    """
    Remove embedded Base64 images from MinerU Markdown.
    Image positions are preserved using [IMAGE_N] markers.
    """
    cleaned_markdown, image_count = decode_embedded_images(
        markdown
    )

    print(
        f"[MINERU] Removed "
        f"{image_count} embedded images"
    )

    return cleaned_markdown


def mineru_to_documents(filepath: str):
    """
    Process a PDF page-by-page with MinerU.
    Returns:
        documents:One LangChain Document per PDF page.
        combined_markdown:Complete cleaned Markdown document for storage
            in MinIO and future View functionality.
    """

    filepath = Path(filepath)

    page_count = get_pdf_page_count(
        str(filepath)
    )

    documents = []
    markdown_pages = []

    print(
        f"[MINERU] Processing "
        f"{filepath.name}: {page_count} pages"
    )

    for page_number in range(1, page_count + 1):

        print(
            f"[MINERU] Processing page "
            f"{page_number}/{page_count}"
        )

        with tempfile.TemporaryDirectory(
            prefix=f"mineru_page_{page_number}_"
        ) as page_tmp_dir:

            markdown_path = run_mineru(
                filepath=str(filepath),
                output_dir=page_tmp_dir,
                page_number=page_number,
            )

            markdown = markdown_path.read_text(
                encoding="utf-8"
            )

        cleaned_markdown = clean_mineru_markdown(
            markdown
        )

        if not cleaned_markdown.strip():
            raise RuntimeError(
                f"MinerU produced empty content "
                f"for page {page_number}"
            )

        # Keep page information in the LangChain document.
        documents.append(
            Document(
                page_content=cleaned_markdown,
                metadata={
                    "source": str(filepath),
                    "filename": filepath.name,
                    "preprocessor": "mineru",
                    "page": page_number,
                },
            )
        )

        # Keep the same page structure in the stored Markdown.
        markdown_pages.append(
            f"<!-- PAGE {page_number} -->\n\n"
            f"{cleaned_markdown.strip()}\n"
        )

    combined_markdown = "\n\n".join(
        markdown_pages
    )

    print(
        f"[MINERU] Created "
        f"{len(documents)} page documents"
    )

    return documents, combined_markdown