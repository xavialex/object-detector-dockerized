import torch
import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException
import json
from PIL import Image
from transformers import YolosImageProcessor, YolosForObjectDetection


app = FastAPI()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = YolosForObjectDetection.from_pretrained('hustvl/yolos-tiny', device_map='cuda:0')
image_processor = YolosImageProcessor.from_pretrained('hustvl/yolos-tiny', device_map='cuda:0')
classes_of_interest = [1, 3]

def yolos_object_detection(img):
    inputs = image_processor(images=img, return_tensors="pt")
    outputs = model(**inputs)

    # model predicts bounding boxes and corresponding COCO classes
    logits = outputs.logits
    bboxes = outputs.pred_boxes

    # print results
    target_sizes = torch.tensor([img.size[::-1]])
    results = image_processor.post_process_object_detection(outputs, threshold=0.9, target_sizes=target_sizes)[0]
    idxs_of_interest = [idx for idx in range(len(results['labels'])) if idx in classes_of_interest]
    for key, value in results.items():
        results[key] = value[idxs_of_interest]
    results['labels'] = [model.config.id2label[label] for label in results['labels'].tolist()]
    #results['scores'] = torch.round(results['scores'], decimals=2)
    output = {'labels': results['labels'], 'scores': results['scores'].tolist(), 'bboxes': results['boxes'].tolist()}
    print(output)
    return output

@app.post("/uploadfile/")
async def create_upload_file(files: list[UploadFile]):
    print("hola")
    img = Image.open(file.file)
    print(img.size)
    print(type(img))
    output = yolos_object_detection(img)
    #request_object_content = await file.read()
    #image = Image.open(io.BytesIO(request_object_content))
    return output

"""
@app.post("/object-detector", status_code=202, summary="Image object detector endpoint")
async def process_image(file: UploadFile = File(...)):

    # Read the uploaded file
    file_bytes = await file.read()
    file_root = save_file(file_bytes=file_bytes, name=file.filename)

    try:
        results = detect_objects(detector, file_root)
        return results
    except Exception as err:
        raise HTTPException(status_code=404, detail=f"Error while object detection with image {file.filename}")
"""

if __name__ == "__main__":
    uvicorn.run("main:app", host="localhost", port=5000, reload=True)