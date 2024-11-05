import pdfplumber
from html2excel import ExcelParser
import pdfplumber
import re

TEXT_LENGTH_THRESHOLD = 10 
A3_WIDTH_MIN = 838  
A3_HEIGHT_MIN = 590  

def classify_page_dimensions(width, height):
    if width >= A3_WIDTH_MIN and height >= A3_HEIGHT_MIN:
        return "A3"
    return "Not A3"

# Function to check if a table is empty
def is_table_empty(table):
    return not any(row for row in table if any(cell and cell.strip() for cell in row))

def classify_page(page):
    width, height = page.width, page.height
    page_type = classify_page_dimensions(width, height)

    if page_type == "A3":
        return "A3"

    
    tables = page.extract_tables()

    # If tables are found and they are non-empty, classify as "Word"
    if tables and any(not is_table_empty(table) for table in tables):
        return "Word"

    text = page.extract_text()

    # If text is found and it's longer than the threshold, classify as "Edge Case"
    if text and len(text.strip()) > TEXT_LENGTH_THRESHOLD:
        return "Edge Case"

    # If no significant text or tables, classify as "Scanned"
    return "Scanned"

def classify_pages(pdf_path):
    results = []

    with pdfplumber.open(pdf_path) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):
            print(f"Processing page {page_number}...")

            classification = classify_page(page)
            results.append((page_number, classification))

    for page_number, classification in results:
        print(f"Page {page_number}: {classification}")

def convert_htm_to_excel(input_file, output_file):
    parser = ExcelParser(input_file)
    parser.to_excel(output_file)

def sanitize_sheet_name(name):
    # Remove invalid characters
    name = re.sub(r'[\\/*?:\[\]]', '_', name)
    # Truncate to 31 characters
    return name[:31]

def calculate_cost(input_tokens, output_tokens):
    """
    Calculate the cost for using Claude 3.5 Sonnet based on input and output tokens.
    
    :param input_tokens: Number of input tokens
    :param output_tokens: Number of output tokens
    :return: Total cost in USD
    """
    # Rates per million tokens
    INPUT_RATE = 3  # $3 per million input tokens
    OUTPUT_RATE = 15  # $15 per million output tokens
    
    # Convert to millions of tokens and calculate cost
    input_cost = (input_tokens / 1_000_000) * INPUT_RATE
    output_cost = (output_tokens / 1_000_000) * OUTPUT_RATE
    
    total_cost = input_cost + output_cost
    
    return round(total_cost, 2)

def parse_numbers(s):
    parts = s.split(',')
    numbers = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if '-' in part:
            start_str, end_str = part.split('-')
            start = int(start_str.strip())
            end = int(end_str.strip())
            numbers.extend(range(start, end + 1))
        else:
            numbers.append(int(part))
    return numbers