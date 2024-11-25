from datetime import datetime
import json
import os
import re
import anthropic
import cv2
import numpy as np
from ultralyticsplus import YOLO
from PIL import ImageDraw 
from io import BytesIO
import base64
import onnxruntime

class OBBModule: 
    def __init__(self, model_path, class_labels=None):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model path not found: {model_path}")
        
        try:
            self.session = onnxruntime.InferenceSession(model_path, providers=["CPUExecutionProvider"])
            print(f"Successfully loaded ONNX model from {model_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to load the ONNX model: {e}")
        
        model_inputs = self.session.get_inputs()
        print("Model Inputs:", model_inputs)
        
        self.input_name = model_inputs[0].name
        self.input_shape = model_inputs[0].shape  # [batch_size, channels, height, width]
        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]

        self.classes = class_labels if class_labels else {0: 'tables', 1: 'tilted', 2: 'empty'}

    def preprocess(self, image):
        resized = cv2.resize(image, (self.input_width, self.input_height))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        normalized = rgb.astype(np.float32) / 255.0
        transposed = normalized.transpose(2, 0, 1)  # CHW format
        batched = np.expand_dims(transposed, axis=0)  # Add batch dimension
        return batched

    def detect_bbox(self, image1, image2=None, confidence_threshold=0.35, iou_threshold=0.45):
        # Convert PIL Image to OpenCV format
        frame = cv2.cvtColor(np.array(image1), cv2.COLOR_RGB2BGR)
        original_height, original_width = frame.shape[:2]
        
        if image2 is not None:
            img2 = cv2.cvtColor(np.array(image2), cv2.COLOR_RGB2BGR)        
            target_height, target_width = img2.shape[:2]
            scale_factor_x = target_width / original_width
            scale_factor_y = target_height / original_height
            print(f"Scaling factors - X: {scale_factor_x}, Y: {scale_factor_y}") 
        else:
            img2 = frame.copy()
            target_height, target_width = original_height, original_width
            scale_factor_x = scale_factor_y = 1
            print("No img2 provided. Using img1 for annotations.")   
        
        preprocessed = self.preprocess(frame)
        print(f"Preprocessed shape: {preprocessed.shape}")
        
        outputs = self.session.run(None, {self.input_name: preprocessed})
        print(f"Model outputs: {outputs}") 
        
        # Postprocess the outputs to get bounding boxes
        obb_data = self.postprocess(outputs, img2, confidence_threshold, iou_threshold, scale_factor_x, scale_factor_y)

        _, buffer = cv2.imencode('.png', img2)
        base_img_string = base64.b64encode(buffer).decode('utf-8')

        response = {
            "bbox_data": obb_data,
            "actual_image": base_img_string,
            "height": int(img2.shape[0]),
            "width": int(img2.shape[1]),
            "num_tables": int(len(obb_data)),
        }
        return response
                
    # def draw_detections(self, img, box, score, class_id):
    #     x1, y1, w, h = box
    #     color = (0, 0, 255)  # Red color for bounding boxes
    #     cv2.rectangle(img, (x1, y1), (x1 + w, y1 + h), color, 2)
    #     # label = f"{self.classes.get(class_id, 'Unknown')}:{score:.2f}"
    #     # cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    
    def postprocess(self, outputs, img2, confidence_threshold, iou_threshold, scale_factor_x, scale_factor_y):
        img_height, img_width = img2.shape[:2]
        output_array = np.squeeze(outputs[0])

        if output_array.shape[0] < output_array.shape[1]:
            output_array = output_array.transpose()

        num_detections = output_array.shape[0]
        print(f"Number of detections before NMS: {num_detections}")  

        boxes = []
        scores = []
        class_ids = []

        # scaled based on model input size to img2
        x_factor = img_width / self.input_width
        y_factor = img_height / self.input_height

        for i in range(num_detections):
            row = output_array[i]
            objectness = row[4]
            class_scores = row[5:]
            class_id = int(np.argmax(class_scores)) 
            confidence = float(class_scores[class_id]) 

            if confidence >= confidence_threshold:
                x, y, width, height = row[0], row[1], row[2], row[3]
                x1 = int((x - width / 2) * x_factor)
                y1 = int((y - height / 2) * y_factor)
                w = int(width * x_factor)
                h = int(height * y_factor)
                
                boxes.append([x1, y1, w, h])
                scores.append(float(confidence))
                class_ids.append(int(class_id))
                
                print(f"Initial bbox {i}: Class ID={class_id}, Confidence={confidence}, Box={x1, y1, w, h}")  

        indices = cv2.dnn.NMSBoxes(boxes, scores, confidence_threshold, iou_threshold)
        print(f"Indices after NMS: {indices}")  

        obb_data = []

        if len(indices) > 0:
            if isinstance(indices[0], (list, tuple, np.ndarray)):
                indices = [i[0] for i in indices]
            else:
                indices = list(indices)
            
            for idx in indices:
                box = boxes[idx]
                class_id = class_ids[idx]
                confidence = scores[idx]
                
                x1, y1, w, h = box
                x2 = x1 + w
                y2 = y1 + h
                #if bbox coordinates -out of img boundary
                # x1 = max(0, x1)
                # y1 = max(0, y1)
                # x2 = min(x2, img_width)
                # y2 = min(y2, img_height)
    
                x = x1 + w / 2
                y = y1 + h / 2

                obb_data.append({
                    "class_id": class_id,
                    "xyxy": [x1, y1, x2, y2],
                    "xywh": [x, y, w, h]
                })

                
                # self.draw_detections(img2, box, confidence, class_id)
                print(f"Final bbox: class_id={class_id}, confidence={confidence}, bbox={x1, y1, x2, y2}")  
        else:
            print("No detections after NMS.")

        print(f"Number of detections after NMS: {len(obb_data)}")
        return obb_data

class TOCRAgent:
    def __init__(self, system_prompt) -> None:

        self.client = anthropic.Anthropic()

        self.system_prompt = system_prompt

    def extract_code(self, content):
        code_blocks = re.findall(r'<final>\n<table(.*?)</final>', content, re.DOTALL)
        return code_blocks

    def extract_table(self, base64_image, file_name, page_num):
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

        extracted_data = {
            "time": [datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
            "file_name": [file_name],
            "page_num": [str(page_num)],
            "image": [base64_image],
            "response": [response.content[0].text]
        }

        if os.path.exists("data.json"):
            with open("data.json", "r") as json_file:
                existing_data = json.load(json_file)
            existing_data["time"].extend(extracted_data["time"])
            existing_data["file_name"].extend(extracted_data["file_name"])
            existing_data["page_num"].extend(extracted_data["page_num"])
            existing_data["image"].extend(extracted_data["image"])
            existing_data["response"].extend(extracted_data["response"])
        else:
            existing_data = extracted_data

        with open("data.json", "w") as json_file:
            json.dump(existing_data, json_file, indent=4)

        print(response.content[0].text)
        return self.extract_code(response.content[0].text), response.usage
