import json
import os
import shutil
import tempfile
from fastapi import Body, FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
import pandas as pd
from pydantic import BaseModel, Field
from typing import List
import io
import pdfplumber
from PIL import Image
import base64
from agents import OBBModule, TOCRAgent
from fastapi.middleware.cors import CORSMiddleware
from io import BytesIO
from utils import convert_htm_to_excel
from openpyxl.drawing.image import Image as ExcelImage
from dotenv import load_dotenv
from starlette.background import BackgroundTask

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

agent = TOCRAgent(system_prompt=open("./system_prompt.txt", 'r').read())
obb = OBBModule()

def get_pil_image(image):
    return Image.open(io.BytesIO(base64.b64decode(image)))

def is_table_empty(table):
    return not any(row for row in table if any(cell and cell.strip() for cell in row))

def parse_numbers(s: str):
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

@app.post("/categorize")
async def categorize(
    selected_pages: str = Form(...), # this should a string like: '2,3,5-8'
    pdf_file: UploadFile = File(...)
):
    try:
        pdf_bytes = await pdf_file.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail="Error reading the PDF file")
    
    selected_pages_list = parse_numbers(selected_pages)

    response = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page_num in selected_pages_list:
            page_index = page_num - 1  # Adjust for zero-based indexing
            if page_index < 0 or page_index >= len(pdf.pages):
                raise HTTPException(status_code=400, detail=f"Page number {page_num} out of range for the provided PDF")

            page = pdf.pages[page_index]
            # pix = page.to_image()
            # pix.save("img.png")
            # img = Image.open('img.png')
            # img1, img2 = page.to_image().original, page.to_image(resolution=275).original
            img1, img2 = page.to_image().original, page.to_image(resolution=275).original 
            obb_result = obb.detect_bbox(img1, img2)

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

                if category == 'Word':
                    for cls_id, cropped_img in obb_result.get('cropped_images', []):
                     if cls_id == 2:
                         category = "Scanned"
                         break

            response.append({
                "page_num": page_num,
                "category": category,
                # "bbox": obb_result if category != 'Word' else None,
                "bbox": obb_result,
                "dpi": 275,
            })
    with open("cat_bbox.txt", 'w') as f:
        f.write(str(response))
    return response

@app.get("/save_m_obb")
def save_m_obb(response:str = Form(...)):
    response = json.loads(response)
    save_file = "obb-traindata.json"
    if os.path.exists(save_file):
        with open(save_file, "r") as json_file:
            existing_data = json.load(json_file)
    else:
        existing_data = []
    existing_data.append({
        "file_name": response["file_name"],
        "pg_no": response["pg_no"],
        "category": response["category"],
    })
    with open(save_file, "w") as json_file:
        json.dump(existing_data, json_file)

    return {"message": "Data saved successfully"}

@app.post("/set_dpi")
async def set_dpi(
    dpi: int = Form(...),
    bbox: str = Form(...),
    page_num: int = Form(...),
    pdf_file: UploadFile = File(...)
):
    try:
        pdf_bytes = await pdf_file.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail="Error reading the PDF file")
    
    print(">>>>>", bbox)
    prev_bbox = [float(coord) for coord in bbox.split(",")]
    page_index = page_num - 1
    if page_index < 0:
        raise HTTPException(status_code=400, detail=f"Invalid page number: {page_num}")

    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes), pages=[page_index]) as pdf:
            if len(pdf.pages) == 0:
                raise HTTPException(status_code=400, detail=f"Page number {page_num} out of range")
            page = pdf.pages[0]
            img1 = page.to_image(resolution=275).original
            img2 = page.to_image(resolution=dpi).original
            
            scale_factor_x = img2.width / img1.width
            scale_factor_y = img2.height / img1.height

            x1, y1, x2, y2 = [coord * scale for coord, scale in zip(prev_bbox[:4], [scale_factor_x, scale_factor_y, scale_factor_x, scale_factor_y])]
            
            buffered = BytesIO()
            img2.save(buffered, format="PNG")
            img2_string = base64.b64encode(buffered.getvalue()).decode('utf-8')

            return {
                "page_num": page_num,
                "dpi": dpi,
                "m_bbox": [x1, y1, x2, y2],
                "m_img": img2_string
            }
    except IndexError:
        raise HTTPException(status_code=400, detail=f"Page number {page_num} out of range")
        

@app.post("/extract")
async def extract(
    pdf_file: UploadFile = File(...),
    data: str = Form(...),
):
    temp_dir = tempfile.mkdtemp()
    with open("ext_bbox.txt", 'w') as f:
        f.write(str(data))
    print("data: ", data)

    try:
        pdf_path = os.path.join(temp_dir, pdf_file.filename)
        contents = await pdf_file.read()
        with open(pdf_path, 'wb') as f:
            f.write(contents)
        
        data = json.loads(data)
        print(data)
        excel_files_info = []
        
        with pdfplumber.open(pdf_path) as pdf:
            for page in data:
                pg_no = page['page_num']
                category = page['category']
                dpi = page['dpi']

                page_index = pg_no - 1  # Adjust for zero-based indexing
                if page_index < 0 or page_index >= len(pdf.pages):
                    raise HTTPException(status_code=400, detail=f"Page number {pg_no} out of range for the provided PDF")
                pl_page = pdf.pages[page_index]
                pg_image = pl_page.to_image(resolution=dpi).original
                # img_path = os.path.join(temp_dir, "img.png")
                # pg_image.save(img_path)
                # pg_image = Image.open(img_path)#hanged
        
                if category in ['A3', 'Scanned']:
                    excel_file = os.path.join(temp_dir, f'{os.path.splitext(pdf_file.filename)[0]}_page-{pg_no}.xlsx')
                    with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
                        start_row = 0
                        tbl_count = 0
                        for tables in page['tables']:
                            class_id = tables['class_id']
                            bbox = tables['bbox']
                            print(">>>>bbox: ", bbox)
                            cropped_img = pg_image.crop(bbox)
                            if class_id == 2:
                                cropped_img = cropped_img.rotate(270, expand=True)
                            img_buffer = io.BytesIO()
                            cropped_img.save(f"{tbl_count}-test.png")
                            cropped_img.save(img_buffer, format="PNG")
                            img_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
                            
                            html_table_content, usage = agent.extract_table(img_base64)
        
                            for gen_table in html_table_content:
                                gen_table = "<table" + gen_table
                                tbl_count += 1
        
                                html_file = os.path.join(temp_dir, f'file-{pdf_file.filename[:-4]}-page-{pg_no}-table-{tbl_count}.html')
                                with open(html_file, 'w', encoding='utf-8') as file:
                                    file.write(gen_table)
                                        
                                excel_file_per_table = os.path.join(temp_dir, f'file-{pdf_file.filename[:-4]}-page-{pg_no}-table-{tbl_count}.xlsx')
                                convert_htm_to_excel(html_file, excel_file_per_table)
        
                                #table_df = pd.read_excel(excel_file_per_table)
                                with pd.ExcelFile(excel_file_per_table) as xls:#changed
                                    table_df = pd.read_excel(xls)
        
                                if isinstance(table_df.columns, pd.MultiIndex):
                                    table_df.columns = [' '.join(col).strip() for col in table_df.columns.values]
        
                                table_df.to_excel(writer, index=False, header=True, startrow=start_row, sheet_name='Page Tables')
                                        
                                start_row += len(table_df) + 3
                                excel_files_info.append({
                                    'excel_file': excel_file,
                                    'page_num': pg_no,
                                    'table_num': tbl_count,
                                    'image': pl_page.to_image(resolution=95).original.rotate(270, expand=True) if class_id == 2 else pl_page.to_image(resolution=95)
                                })
                        
                elif category == 'Word':
                    excel_file = os.path.join(temp_dir, f'{os.path.splitext(pdf_file.filename)[0]}_page-{pg_no}.xlsx')
                    pl_page = pdf.pages[pg_no-1]

                    extracted_tables = pl_page.extract_tables()
                    with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
                        start_row = 0
                        tbl_count = 0
                        for tbl_no, table in enumerate(extracted_tables):
                            tbl_count += 1
                            table_df = pd.DataFrame(table[1:], columns=table[0])
        
                            if isinstance(table_df.columns, pd.MultiIndex):
                                table_df.columns = [' '.join(col).strip() for col in table_df.columns.values]
        
                            table_df.to_excel(writer, index=False, header=True, startrow=start_row, sheet_name='Page Tables')
                                    
                            start_row += len(table_df) + 3
                                    
                            if not table_df.empty:
                                excel_files_info.append({
                                    'excel_file': excel_file,
                                    'page_num': pg_no,
                                    'table_num': tbl_count,
                                    "image": pl_page.to_image(resolution=95)
                                })
                        
                else:
                    print(f"Skipping page {pg_no} with invalid category '{category}'.")
                
        if excel_files_info:
            combined_excel_path = os.path.join(temp_dir, f'{pdf_file.filename[:-4]}_combined.xlsx')
            img_added_pg_no = []
            with pd.ExcelWriter(combined_excel_path, engine='openpyxl') as writer:
                for file_info in excel_files_info:
                    #df = pd.read_excel(file_info['excel_file'])
                    with pd.ExcelFile(file_info['excel_file']) as xls:#changed
                        df = pd.read_excel(xls)
                    sheet_name = f'Page_{file_info["page_num"]}'
                    df.to_excel(writer, sheet_name=sheet_name, index=False)
        
                    if file_info['page_num'] not in img_added_pg_no:
                        img_added_pg_no.append(file_info['page_num'])
                        workbook = writer.book
                        worksheet = workbook[sheet_name]
        
                        img_buffer = io.BytesIO()
                        file_info['image'].save(img_buffer, format="PNG")
                        img_buffer.seek(0)
                        img_for_excel = ExcelImage(img_buffer)
        
                        worksheet.add_image(img_for_excel, "R1")
        else:
            # return with 400 status code with message saying no tables found
            shutil.rmtree(temp_dir)
            raise HTTPException(status_code=400, detail="no tables")

        def cleanup():
            shutil.rmtree(temp_dir)

        return FileResponse(
            path=combined_excel_path,
            filename=f'{pdf_file.filename[:-4]}_combined.xlsx',
            media_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            background=BackgroundTask(cleanup)
        )

    except Exception as e:
        shutil.rmtree(temp_dir)
        raise e
