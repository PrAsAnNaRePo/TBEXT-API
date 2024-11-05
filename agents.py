import re
import anthropic
from ultralyticsplus import YOLO
from PIL import ImageDraw 
from io import BytesIO
import base64

class OBBModule():
    def __init__(self) -> None:
        # classifiable model are :-
        #   - yolov11x-2c.pt
        #   - yolov8n-2c-v2.pt
        #   - yolov8n-2c.pt
        #self.model = YOLO('C:\\Users\\LAU\\Downloads\\FASTAPI3\\FASTAPI\\18102024.pt').to('cpu')
        self.model = YOLO("21102024.pt").to('cpu')
        # set model parameters
        self.model.overrides['conf'] = 0.35  # NMS confidence threshold
        self.model.overrides['iou'] = 0.45  # NMS IoU threshold
        self.model.overrides['agnostic_nms'] = False  # NMS class-agnostic
        self.model.overrides['max_det'] = 1000  # maximum number of detections per image

    def detect_bbox(self,img1, img2=None):
        if img1.mode != "RGB":
            img1 = img1.convert("RGB")
        
        results = self.model.predict(img1)
        print(results[0].boxes.xyxy)

        if img2 is not None:
            scale_factor_x = img2.width / img1.width
            scale_factor_y = img2.height / img1.height
        else:
            img2 = img1
            scale_factor_x = scale_factor_y = 1

        obb_data = []

        for conf, table_bbox_xyxy, table_bbox_xywh, class_id in zip(
            results[0].boxes.conf.tolist(),
            results[0].boxes.xyxy,
            results[0].boxes.xywh.tolist(),
            results[0].boxes.cls.tolist()
        ):
            # if conf > 0.26:
            bbox = table_bbox_xyxy.tolist()
            
            x1, y1, x2, y2 = [coord * scale for coord, scale in zip(bbox[:4], [scale_factor_x, scale_factor_y, scale_factor_x, scale_factor_y])]
            x, y, w, h = table_bbox_xywh
            x, y, w, h = [coord * scale for coord, scale in zip([x, y, w, h], [scale_factor_x, scale_factor_y, scale_factor_x, scale_factor_y])]
            obb_data.append({
                "class_id": class_id,
                "xyxy": [x1, y1, x2, y2],
                "xywh": [x, y, w, h]
            })
            print(">>>>>> initial bbox", [x1, y1, x2, y2])

            # else:
            #     obb_data.append({
            #         "class_id": None,
            #         "xyxy": None,
            #         "xywh": None
            #     })
        
        buffered = BytesIO()
        img2.save(buffered, format="PNG")
        base_img_string = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        return {
            "bbox_data": obb_data,
            "actual_image": base_img_string,
            "height": img2.height,
            "width": img2.width,
            "num_tables": len(results[0].boxes.xyxy),
        }

class TOCRAgent:
    def __init__(self, system_prompt) -> None:

        self.client = anthropic.Anthropic()

        self.system_prompt = system_prompt

    def extract_code(self, content):
        code_blocks = re.findall(r'<final>\n<table(.*?)</final>', content, re.DOTALL)
        return code_blocks

    def extract_table(self, base64_image):
        msg = []
        msg.append(
            {
                'role': 'user',
                'content': [
                    {
                    "type": "text",
                    "text": "Extract the table step by step."
                    },
                    {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": base64_image,
                    },
                    }
                ]
            }
        )
    
        response = self.client.messages.create(
            model="claude-3-5-sonnet-latest",
            messages=msg,
            max_tokens=8192,
            system=self.system_prompt,
            extra_headers={
                'anthropic-beta': 'max-tokens-3-5-sonnet-2024-07-15'
            },
            temperature=0,
        )
        print(response.content[0].text)
        return self.extract_code(response.content[0].text), response.usage