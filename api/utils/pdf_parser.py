"""PDF text extraction utilities using Gemini vision for OCR."""

import base64
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO

import fitz  # PyMuPDF for PDF to image conversion

logger = logging.getLogger("utils.pdf_parser")


def extract_text_from_pdf(
    pdf_bytes: bytes,
    use_ocr: bool = True,
    parallel: bool = True,
    max_workers: int = 4,
) -> str:
    """
    Extract text from PDF file bytes using Gemini vision OCR.
    
    Converts each page to an image and sends to Gemini for text extraction.
    This handles scanned PDFs and image-based PDFs that pypdf cannot read.
    
    Args:
        pdf_bytes: PDF file content as bytes
        use_ocr: If True, use Gemini OCR. If False, try pypdf first.
        parallel: Process pages in parallel for speed
        max_workers: Max concurrent Gemini requests
        
    Returns:
        Extracted text as string
        
    Raises:
        ValueError: If PDF is corrupted or cannot be read
        RuntimeError: If text extraction fails
    """
    if not pdf_bytes:
        raise ValueError("PDF bytes cannot be empty")
    
    # Try pypdf first if not forcing OCR
    if not use_ocr:
        text = _try_pypdf_extraction(pdf_bytes)
        if text and text.strip():
            logger.info(f"pypdf extracted {len(text):,} characters")
            return text
        logger.info("pypdf extraction empty, falling back to Gemini OCR")
    
    # Use Gemini OCR
    return _extract_with_gemini_ocr(pdf_bytes, parallel=parallel, max_workers=max_workers)


def _try_pypdf_extraction(pdf_bytes: bytes) -> str | None:
    """Try to extract text using pypdf (fast, works for text-based PDFs)."""
    try:
        import pypdf
        from io import BytesIO
        
        pdf_stream = BytesIO(pdf_bytes)
        pdf_reader = pypdf.PdfReader(pdf_stream)
        
        if pdf_reader.is_encrypted:
            return None
        
        text_pages = []
        for page in pdf_reader.pages:
            try:
                page_text = page.extract_text()
                if page_text:
                    text_pages.append(page_text)
            except Exception:
                continue
        
        return "\n\n".join(text_pages) if text_pages else None
        
    except Exception as e:
        logger.debug(f"pypdf extraction failed: {e}")
        return None


def _extract_with_gemini_ocr(
    pdf_bytes: bytes,
    parallel: bool = True,
    max_workers: int = 4,
) -> str:
    """Extract text from PDF using Gemini vision OCR."""
    from clients.gemini import get_gemini_client
    
    logger.info("Using Gemini OCR for PDF extraction")
    
    # Convert PDF to images
    images = _pdf_to_images(pdf_bytes)
    logger.info(f"Converted PDF to {len(images)} page images")
    
    if not images:
        raise ValueError("PDF contains no pages")
    
    gemini_client = get_gemini_client()
    
    if parallel and len(images) > 1:
        # Process pages in parallel
        page_texts = _extract_pages_parallel(images, gemini_client, max_workers)
    else:
        # Process pages sequentially
        page_texts = _extract_pages_sequential(images, gemini_client)
    
    # Concatenate all pages
    full_text = "\n\n".join(page_texts)
    
    if not full_text.strip():
        raise ValueError("Gemini OCR extracted no text from PDF")
    
    logger.info(f"Extracted {len(full_text):,} characters from {len(images)} pages")
    return full_text


def _pdf_to_images(pdf_bytes: bytes, dpi: int = 150) -> list[bytes]:
    """
    Convert PDF pages to PNG images.
    
    Args:
        pdf_bytes: PDF file content
        dpi: Resolution for rendering (higher = better OCR, slower)
        
    Returns:
        List of PNG image bytes, one per page
    """
    images = []
    
    # Open PDF with PyMuPDF
    pdf_doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    
    try:
        for page_num in range(len(pdf_doc)):
            page = pdf_doc[page_num]
            
            # Render page to image
            # zoom factor for DPI: default PDF is 72 DPI
            zoom = dpi / 72
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat)
            
            # Convert to PNG bytes
            png_bytes = pix.tobytes("png")
            images.append(png_bytes)
            
            logger.debug(f"Rendered page {page_num + 1}: {pix.width}x{pix.height}")
    finally:
        pdf_doc.close()
    
    return images


def _extract_pages_parallel(
    images: list[bytes],
    gemini_client,
    max_workers: int,
) -> list[str]:
    """Extract text from pages in parallel."""
    page_texts = [""] * len(images)  # Pre-allocate to maintain order
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(_extract_single_page, img, gemini_client, idx + 1): idx
            for idx, img in enumerate(images)
        }
        
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                text = future.result()
                page_texts[idx] = text
                logger.info(f"Page {idx + 1}/{len(images)}: extracted {len(text):,} chars")
            except Exception as e:
                logger.error(f"Page {idx + 1} extraction failed: {e}")
                page_texts[idx] = f"[Page {idx + 1} extraction failed: {e}]"
    
    return page_texts


def _extract_pages_sequential(
    images: list[bytes],
    gemini_client,
) -> list[str]:
    """Extract text from pages sequentially."""
    page_texts = []
    
    for idx, img in enumerate(images):
        try:
            text = _extract_single_page(img, gemini_client, idx + 1)
            page_texts.append(text)
            logger.info(f"Page {idx + 1}/{len(images)}: extracted {len(text):,} chars")
        except Exception as e:
            logger.error(f"Page {idx + 1} extraction failed: {e}")
            page_texts.append(f"[Page {idx + 1} extraction failed: {e}]")
    
    return page_texts


def _extract_single_page(
    image_bytes: bytes,
    gemini_client,
    page_num: int,
) -> str:
    """
    Extract text from a single page image using Gemini.
    
    Args:
        image_bytes: PNG image bytes
        gemini_client: Gemini client instance
        page_num: Page number for logging
        
    Returns:
        Extracted text from the page
    """
    from google.genai import types
    
    # Create image part for Gemini
    image_part = types.Part.from_bytes(
        data=image_bytes,
        mime_type="image/png"
    )
    
    # OCR prompt - asking Gemini to extract all text
    prompt = """Extract ALL text content from this medical document image.

Instructions:
- Extract every piece of text exactly as it appears
- Preserve the original formatting and structure as much as possible
- Include headers, labels, values, dates, names, and all other text
- Maintain paragraph breaks and list structures
- Do NOT summarize or interpret - just extract the raw text
- If text is unclear, make your best attempt and continue
- Include any tables by preserving their structure with spacing or delimiters

Output the extracted text directly with no additional commentary."""

    # Call Gemini with the image
    response = gemini_client.client.models.generate_content(
        model="gemini-2.0-flash",  # Use flash for speed
        contents=[prompt, image_part],
    )
    
    return response.text.strip()


# Legacy function for backward compatibility
def extract_text_from_pdf_legacy(pdf_bytes: bytes) -> str:
    """
    Legacy extraction using pypdf only.
    
    Use extract_text_from_pdf() instead for better results on scanned PDFs.
    """
    import pypdf
    
    if not pdf_bytes:
        raise ValueError("PDF bytes cannot be empty")
    
    try:
        pdf_stream = BytesIO(pdf_bytes)
        pdf_reader = pypdf.PdfReader(pdf_stream)
        
        if pdf_reader.is_encrypted:
            raise ValueError("PDF is password-protected and cannot be processed")
        
        text_pages = []
        for page_num, page in enumerate(pdf_reader.pages, start=1):
            try:
                page_text = page.extract_text()
                if page_text:
                    text_pages.append(page_text)
            except Exception as e:
                raise RuntimeError(f"Failed to extract text from page {page_num}: {str(e)}")
        
        extracted_text = "\n\n".join(text_pages)
        
        if not extracted_text or not extracted_text.strip():
            raise ValueError("PDF contains no extractable text")
        
        return extracted_text
        
    except pypdf.errors.PdfReadError as e:
        raise ValueError(f"Invalid or corrupted PDF file: {str(e)}")
    except Exception as e:
        if isinstance(e, (ValueError, RuntimeError)):
            raise
        raise RuntimeError(f"Failed to extract text from PDF: {str(e)}")
