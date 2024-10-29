from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel, Field
from typing import List
import io
import pdfplumber
from PIL import Image
import base64
from agents import OBBModule
from fastapi.middleware.cors import CORSMiddleware
from io import BytesIO

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

obb = OBBModule()

def get_pil_image(image):
    return Image.open(io.BytesIO(base64.b64decode(image)))

def is_table_empty(table):
    return not any(row for row in table if any(cell and cell.strip() for cell in row))

@app.post("/categorize")
async def categorize(
    selected_pages: str = Form(...),
    pdf_file: UploadFile = File(...)
):
    try:
        pdf_bytes = await pdf_file.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail="Error reading the PDF file")
    
    selected_pages_ = []
    for i in selected_pages.split(","):
        selected_pages_.append(int(i))

    selected_pages = selected_pages_

    response = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for i, page_num in enumerate(selected_pages):
            if page_num >= len(pdf.pages):
                raise HTTPException(status_code=400, detail=f"Page number {page_num} out of range for the provided PDF")

            page = pdf.pages[page_num - 1]
            pix = page.to_image()
            pix.save("img.png")
            img = Image.open('img.png')
            # img1, img2 = page.to_image().original, page.to_image(resolution=275).original
            img = page.to_image().original
            obb_result = obb.detect_bbox(img)

            if obb_result['num_tables'] > 0:
                width, height = page.width, page.height

                if width >= 838 and height >= 590:
                    category = "A3"
                else:
                    tables = page.extract_tables()
                    if any(not is_table_empty(table) for table in tables):  # W or S
                        if len(page.extract_text().strip().split(' ')) > 30:  # W
                            category = "Word"
                        else:  # S
                            category = "Scanned"
                    else:  # S or EC
                        category = "Scanned"
            else:
                category = "Edge Case"

            # Category Adjustment if 'Word'
            if category == 'Word':
                for cls, img in img:
                    if cls == 2:
                        category = "Scanned"

            response.append({
                "page_num": page_num,
                "category": category,
                "bbox": obb_result if category != 'Word' else None,
                "dpi": 275,
                "img_height": height,
                "img_width": width
            })

    return response

@app.post("/set_dpi")
async def set_dpi(
    dpi: int = Form(...),
    page_num: int = Form(...),
    pdf_file: UploadFile = File(...)
):
    try:
        pdf_bytes = await pdf_file.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail="Error reading the PDF file")

    with pdfplumber.open(io.BytesIO(pdf_bytes), pages=[page_num]) as pdf:
        page = pdf.pages[0]
        pix = page.to_image(resolution=dpi)
        img = pix.original
        
        # Convert PIL image to base64
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        return {"image": img_str, "dpi": dpi, "height": img.height, "width": img.width}


# @app.post("/extract")
# async def extract(
#     pdf_file: UploadFile = File(...),
#     page_details: List[int] = Form(...),
# )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
