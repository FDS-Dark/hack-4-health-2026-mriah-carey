"""FastAPI application for medical report processing."""

import os
import logging

from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pipeline import process_medical_report, process_medical_report_async, PipelineConfig
from utils.pdf_parser import extract_text_from_pdf
from services.glossary import GlossaryService

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s | %(name)-30s | %(levelname)-7s | %(message)s',
    datefmt='%H:%M:%S'
)

# Reduce noise from third-party libraries
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("opensearchpy").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.WARNING)
logging.getLogger("filelock").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
logging.getLogger("medspacy").setLevel(logging.WARNING)
logging.getLogger("spacy").setLevel(logging.WARNING)
logging.getLogger("python_multipart.multipart").setLevel(logging.WARNING)
logging.getLogger("PyRuSH").setLevel(logging.WARNING)
logging.getLogger("google_genai").setLevel(logging.WARNING)
logging.getLogger("google_genai.models").setLevel(logging.WARNING)


# Filter to suppress medspacy regex syntax warnings
class MedspaCyWarningFilter(logging.Filter):
    """Filter out medspacy regex syntax warnings."""
    def filter(self, record):
        # Suppress the "not a eligible syntax" warnings from medspacy
        if "eligible syntax" in record.getMessage():
            return False
        return True


# Apply filter to root logger
logging.getLogger().addFilter(MedspaCyWarningFilter())

logger = logging.getLogger("api")

app = FastAPI(
    title="Medical Report Processing API",
    description="API for processing medical reports through extraction, simplification, compilation, and summarization",
    version="1.0.0"
)

# CORS configuration for frontend integration
cors_origins = os.getenv(
    "CORS_ORIGINS",
    "http://localhost:5173,http://localhost:3000"  # Vite and common React dev ports
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Medical Report Processing API",
        "version": "1.0.0",
        "endpoints": {
            "POST /process-report": "Process a medical report file (.txt or .pdf)"
        }
    }


@app.post("/process-report")
async def process_report(file: UploadFile = File(...)):
    """
    Process uploaded medical report file.
    
    Args:
        file: Uploaded .txt or .pdf file containing medical report
        output_format: Output format for compilation - "paragraphs" or "sections" (default: "paragraphs")
    
    Returns:
        ProcessingResponse with all processing results:
        - structured_json: Extracted structured data
        - simplified_json: Simplified version with original/simplified mapping
        - simplified_text: Compiled human-readable text
        - summary_text: Shortened summary
        - metadata: Processing statistics
    """
    # Validate file type
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    
    if not file.filename.endswith(('.txt', '.pdf')):
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Only .txt and .pdf files are supported."
        )
    
    # Read file content
    content = await file.read()
    
    # Validate file size - PDFs can be larger due to images
    is_pdf = file.filename.endswith('.pdf')
    max_size = 10 * 1024 * 1024 if is_pdf else 1024 * 1024  # 10MB for PDF, 1MB for text
    if len(content) > max_size:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Maximum size is {max_size / 1024 / 1024}MB"
        )
    
    # Extract text based on file type
    if file.filename.endswith('.pdf'):
        # Extract text from PDF
        try:
            report_text = extract_text_from_pdf(content)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except RuntimeError as e:
            raise HTTPException(status_code=500, detail=f"PDF parsing failed: {str(e)}")
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Unexpected error parsing PDF: {str(e)}"
            )
    else:  # .txt file
        # Decode text file
        try:
            report_text = content.decode('utf-8')
        except UnicodeDecodeError:
            # Try latin-1 encoding as fallback
            try:
                report_text = content.decode('latin-1')
            except Exception as e:
                raise HTTPException(
                    status_code=400,
                    detail=f"Failed to decode file: {str(e)}"
                )
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Failed to read file: {str(e)}"
            )
    
    # Validate content is not empty
    if not report_text or not report_text.strip():
        raise HTTPException(
            status_code=400,
            detail="File is empty or contains no text"
        )
    
    # Process the report through the pipeline
    try:
        # Use async version for FastAPI
        config = PipelineConfig(
            parallel_chunks=True,
            enable_exa_sources=True,  # Enable Exa for source citations
        )
        result = await process_medical_report_async(report_text, config=config)
        # Map the result to expected format
        simplified_text = "\n\n".join([chunk["simplified_chunk"] for chunk in result["simplified_report"]])
        original_text = "\n\n".join([chunk["original_chunk"] for chunk in result["simplified_report"]])
        summary_text = simplified_text[:500] + "..." if len(simplified_text) > 500 else simplified_text
        
        # Check if any chunks need review
        needs_review = any(chunk.get("needs_review", False) for chunk in result["simplified_report"])
        
        # Generate glossary
        glossary_service = GlossaryService()
        glossary = glossary_service.create_glossary(original_text, simplified_text)
        
        # Collect all citations from chunks
        all_citations = {}
        for chunk in result["simplified_report"]:
            chunk_citations = chunk.get("citations", {})
            for source_id, text_segments in chunk_citations.items():
                if source_id not in all_citations:
                    all_citations[source_id] = []
                all_citations[source_id].extend(text_segments)
        
        return {
            "structured_json": result["simplified_report"],
            "simplified_json": result["simplified_report"],
            "simplified_text": simplified_text,
            "summary_text": summary_text,
            "original_text": original_text,
            "glossary": glossary,
            "needs_review": needs_review,
            "citations": all_citations,
            "metadata": {
                "chunks": len(result["simplified_report"]),
                "needs_review": needs_review,
            }
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error during processing: {str(e)}"
        )
    
@app.get('/simplify')
async def simplify(test: str):
    config = PipelineConfig(parallel_chunks=True, enable_exa_sources=True)
    result = await process_medical_report_async(test, config=config)
    return result


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)