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

    def detect_bbox(self, img1, img2=None):
        print("###########################")
        print("RUNNING>>>>")
        print("###########################")
        if img1.mode != "RGB":
            img1 = img1.convert("RGB")
        
        # Predict bboxes on the 125 DPI image
        results = self.model.predict(img1)
        print(results[0].boxes)
        print("PREDICTED>>>>")
        cropped_images = []
        
        if img2 is not None:
            # Calculate scaling factor from 125 DPI to 150 DPI
            scale_factor_x = img2.width / img1.width
            scale_factor_y = img2.height / img1.height
        else:
            img2 = img1
            scale_factor_x = scale_factor_y = 1
        base_img = img2.copy()

        obb_data = []

        for conf, table_bbox_xyxy, table_bbox_xywh, class_id in zip(
            results[0].boxes.conf.tolist(),
            results[0].boxes.xyxy,
            results[0].boxes.xywh.tolist(),
            results[0].boxes.cls.tolist()
        ):
            if conf > 0.26:
                bbox = table_bbox_xyxy.tolist()
                
                # Scale the bounding box coordinates (xyxy)
                x1, y1, x2, y2 = [
                    coord * scale for coord, scale in zip(
                        bbox[:4],
                        [scale_factor_x, scale_factor_y, scale_factor_x, scale_factor_y]
                    )
                ]
                draw = ImageDraw.Draw(img2)
                draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
                
                if class_id == 0:
                    tbl_cls = "empty"
                elif class_id == 1:
                    tbl_cls = "normal"
                elif class_id == 2:
                    tbl_cls = "tilted"

                draw.text((x1, y1), tbl_cls, fill="red")

                # Crop the table region from the image
                cropped_images.append([class_id, base_img.crop((x1, y1, x2, y2))])

                buffered = BytesIO()
                cropped_img = base_img.crop((x1, y1, x2, y2))
                cropped_img.save(buffered, format="PNG")
                img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')

                # Scale the xywh coordinates
                xc, yc, w, h = table_bbox_xywh
                scaled_xc = xc * scale_factor_x
                scaled_yc = yc * scale_factor_y
                scaled_w = w * scale_factor_x
                scaled_h = h * scale_factor_y

                obb_data.append({
                    "class": tbl_cls,
                    "bbox": [x1, y1, x2, y2],
                    "xywh": [scaled_xc, scaled_yc, scaled_w, scaled_h],
                    "cropped_img": img_str
                })

            else:
                obb_data.append({
                    "class": None,
                    "bbox": [0, 0, 0, 0],
                    "cropped_img": None
                })
        
        buffered = BytesIO()
        img2.save(buffered, format="PNG")
        img2_string = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        buffered = BytesIO()
        base_img.save(buffered, format="PNG")
        base_img_string = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        return {
            "bbox_data": obb_data,
            "actual_image": base_img_string,
            "height": base_img.height,
            "width": base_img.width,
            "annotated_img": img2_string,
            "num_tables": len(results[0].boxes.xyxy),
        }
