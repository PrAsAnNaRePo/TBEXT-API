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
        self.model = YOLO('3C-models/18102024.pt')
        # set model parameters
        self.model.overrides['conf'] = 0.35  # NMS confidence threshold
        self.model.overrides['iou'] = 0.45  # NMS IoU threshold
        self.model.overrides['agnostic_nms'] = False  # NMS class-agnostic
        self.model.overrides['max_det'] = 1000  # maximum number of detections per image

    def detect_bbox(self, img):
        print("###########################")
        print("RUNNING>>>>")
        print("###########################")
        if img.mode != "RGB":
            img = img.convert("RGB")
        
        # Predict bboxes on the 125 DPI image
        results = self.model.predict(img)
        print(results[0].boxes)
        print("PREDICTED>>>>")
        cropped_images = []
        
        base_img = img.copy()

        obb_data = []

        for conf, table_bbox_xyxy, table_bbox_xywh, class_id in zip(
            results[0].boxes.conf.tolist(),
            results[0].boxes.xyxy,
            results[0].boxes.xywh.tolist(),
            results[0].boxes.cls.tolist()
        ):
            if conf > 0.26:
                bbox = table_bbox_xyxy.tolist()
                
                x1, y1, x2, y2 = bbox
                draw = ImageDraw.Draw(img)
                draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
                
                if class_id == 0:
                    tbl_cls = "empty"
                elif class_id == 1:
                    tbl_cls = "normal"
                elif class_id == 2:
                    tbl_cls = "tilted"

                draw.text((x1, y1), tbl_cls, fill="red")

                cropped_images.append([class_id, base_img.crop((x1, y1, x2, y2))])

                xc, yc, w, h = table_bbox_xywh

                obb_data.append({
                    "class": tbl_cls,
                    "xyxy": [x1, y1, x2, y2],
                    "xywh": [xc, yc, w, h],
                })

            else:
                obb_data.append({
                    "class": None,
                    "xyxy": [0, 0, 0, 0],
                    "xywh": [xc, yc, w, h],
                })
        
        buffered = BytesIO()
        base_img.save(buffered, format="PNG")
        base_img_string = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        return {
            "bbox_data": obb_data,
            "actual_image": base_img_string,
            "height": base_img.height,
            "width": base_img.width,
            "num_tables": len(results[0].boxes.xyxy),
        }
