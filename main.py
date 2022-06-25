

from fastapi import FastAPI, File, UploadFile
import uvicorn
#from typing import Optional
from fastapi.responses import FileResponse
import uuid
import numpy as np
import PIL
from PIL import Image
from io import BytesIO
from tensorflow.keras import preprocessing
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os
app = FastAPI()




@app.post("/images/")
async def create_upload_file(file:UploadFile = File(...)):
    extension = file.filename.split(".")[-1] in ("jpg", "jpeg", "png")
    if not extension:
        return "Image must be jpg or png format!"
   
   
    file.filename = f"{uuid.uuid4()}.jpg"
    name=str(file.filename)
 
    contents = await file.read()

    with open(file.filename,"wb") as f:
        f.write(contents) 
       
    
    image1 = preprocessing.image.load_img(name, target_size=(224, 224,3))
  
  
    image1 = np.asarray(image1)
    image1 = np.expand_dims(image1, axis=0)
    image1 = image1 * 1.0 / 255  
    model = load_model('model_inception.h5')
    predicted = model.predict(image1)
    print('Predicted result: {0} '.format(predicted))
    y_pred = np.argmax(predicted, axis=1)
    print(y_pred)
    
    os.remove(name) 
#     # example of how you can save the file
#     

    return {"filename":str(y_pred)}


    
#if __name__ == "__main__":
# uvicorn.run(app, host="127.0.0.1",port=7000)
