from typing import Annotated
import io

from PIL import Image
import torch
import uvicorn
from fastapi import FastAPI, UploadFile, Query, File, HTTPException
from transformers import YolosImageProcessor, YolosForObjectDetection


app = FastAPI()
# Model initialization
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = YolosForObjectDetection.from_pretrained('hustvl/yolos-tiny', 
                                                device_map=device)
image_processor = YolosImageProcessor.from_pretrained('hustvl/yolos-tiny', 
                                                      device_map=device)


def yolos_object_detection(imgs, threshold, classes_of_interest):  
    inputs = image_processor(images=imgs, return_tensors="pt")
    outputs = model(**inputs)

    # Results postprocessing
    target_sizes = [img.size[::-1] for img in imgs]
    results = image_processor.post_process_object_detection(outputs, 
        threshold=threshold, target_sizes=target_sizes)
    # Final result creation and selection of classes of interest
    output = []
    for result in results:
        idxs_of_interest = [idx for idx in range(len(result['labels'])) 
                            if result['labels'][idx] in classes_of_interest]
        for key, value in result.items():
            result[key] = value[idxs_of_interest]
        result['labels'] = [model.config.id2label[label] 
                            for label in result['labels'].tolist()]
        output.append({'labels': result['labels'], 
                       'scores': result['scores'].tolist(), 
                       'bboxes': result['boxes'].tolist()})
    
    return output


@app.post("/yolos-object-detection/")
async def yolos_inference(
        files: Annotated[list[UploadFile],
            File(description="Images fed to the object detection model. " \
                 "Just JPEG format supported")],
        threshold: Annotated[float, 
            Query(description="Minimum confidence threshold for the " \
                  "detections to be considered valid.")] = 0.9,
        classes_of_interest: Annotated[list[int],
            Query(title="Classes of interest", 
                  description="IDs of the classes to be detected by the " \
                    "object detection model: https://huggingface.co/hustvl/" \
                    "yolos-tiny/blob/main/config.json")] = [1, 3]):
    imgs = [] 
    for file in files:
        if file.content_type != "image/jpeg":
            raise HTTPException(400, detail="Invalid image type. " \
                                "Just JPEG support.")
        request_object_content = await file.read()
        img = Image.open(io.BytesIO(request_object_content))
        imgs.append(img)
    output = yolos_object_detection(imgs, threshold, classes_of_interest)
    output = [{file.filename: result} for file, result in zip(files, output)]

    return output


if __name__ == "__main__":
    uvicorn.run("main:app", host="localhost", port=5000, reload=True)
