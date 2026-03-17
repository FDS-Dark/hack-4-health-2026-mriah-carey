"""File I/O utilities for reading medical reports and writing JSON output."""

import json
import re
from pathlib import Path
from typing import Union
from pydantic import BaseModel
from models.report_schema import ParsedReport


def safe_json_loads(json_string: str) -> dict:
    """
    Safely parse JSON, handling control characters that Gemini sometimes includes.
    
    Args:
        json_string: JSON string that may contain control characters.
        
    Returns:
        Parsed JSON as dict.
        
    Raises:
        json.JSONDecodeError: If JSON is still invalid after cleaning.
    """
    try:
        return json.loads(json_string)
    except json.JSONDecodeError:
        # Clean control characters from JSON string
        # Remove control characters except newlines in strings (which should be \n)
        cleaned = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', json_string)
        
        # Try again with cleaned string
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            # Try to fix common issues: unescaped newlines in strings
            # Replace actual newlines inside strings with escaped versions
            # This is a more aggressive fix
            fixed = _fix_json_newlines(json_string)
            return json.loads(fixed)


def _fix_json_newlines(s: str) -> str:
    """
    Fix unescaped newlines inside JSON string values.
    
    This handles cases where Gemini puts actual newlines inside JSON strings
    instead of \\n escape sequences.
    """
    result = []
    in_string = False
    escape_next = False
    
    for char in s:
        if escape_next:
            result.append(char)
            escape_next = False
            continue
            
        if char == '\\':
            escape_next = True
            result.append(char)
            continue
            
        if char == '"':
            in_string = not in_string
            result.append(char)
            continue
        
        if in_string and char == '\n':
            result.append('\\n')
        elif in_string and char == '\r':
            result.append('\\r')
        elif in_string and char == '\t':
            result.append('\\t')
        else:
            result.append(char)
    
    return ''.join(result)


def read_report(file_path: Union[str, Path]) -> str:
    """
    Read a medical report from a .txt file.
    
    Args:
        file_path: Path to the .txt file
        
    Returns:
        Contents of the file as a string
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        UnicodeDecodeError: If the file encoding cannot be decoded
    """
    path = Path(file_path)
    
    if not path.exists():
        raise FileNotFoundError(f"Report file not found: {file_path}")
    
    # Try UTF-8 first, fallback to latin-1 for older medical reports
    try:
        return path.read_text(encoding='utf-8')
    except UnicodeDecodeError:
        return path.read_text(encoding='latin-1')


def write_json(parsed_report: ParsedReport, output_path: Union[str, Path]) -> None:
    """
    Write a ParsedReport to a JSON file.
    
    Args:
        parsed_report: The parsed report model to serialize
        output_path: Path where the JSON file should be written
        
    Raises:
        IOError: If the file cannot be written
    """
    write_json_model(parsed_report, output_path)


def write_json_model(model: BaseModel, output_path: Union[str, Path]) -> None:
    """
    Write any Pydantic model to a JSON file.
    
    Args:
        model: The Pydantic model to serialize
        output_path: Path where the JSON file should be written
        
    Raises:
        IOError: If the file cannot be written
    """
    path = Path(output_path)
    
    # Create parent directories if they don't exist
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert Pydantic model to dict and write as JSON
    json_data = model.model_dump()
    
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)


def write_text(text: str, output_path: Union[str, Path]) -> None:
    """
    Write plain text to a file.
    
    Args:
        text: Text content to write
        output_path: Path where the text file should be written
        
    Raises:
        IOError: If the file cannot be written
    """
    path = Path(output_path)
    
    # Create parent directories if they don't exist
    path.parent.mkdir(parents=True, exist_ok=True)
    
    path.write_text(text, encoding='utf-8')
