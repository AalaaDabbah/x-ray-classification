# from fastapi import FastAPI
# import uvicorn
# import os

# app=FastAPI()

# if __name__ == "__main__":
#     uvicorn.run(app, host="127.0.0.1", port='3000')
# @app.get("/")
# def nothing():
#     return {"message":"Hello world"}

from fastapi import FastAPI, File, UploadFile
import uvicorn
from typing import Optional
from pydantic import BaseModel
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
class person(BaseModel):
    name: str
    password: int

def read_imagefile(file) -> Image.Image:
    image = Image.open(BytesIO(file))
    return image

@app.post("/per")
def create_person(per: person):
    #return request
    return {'data':f"person is created {per.name}"}

@app.get("/")
def nothing():
    return {"message":"Hello world"}

@app.get("/api")
def nothing(name:str="alaa"):
    return {"name":f'{name}'}
@app.get("/{id}")
def nothing(id):
    return {"message":id}

#if __name__ == "__main__":
 #   uvicorn.run(app, host="127.0.0.1",port=7000)




@app.post("/images/")
async def create_upload_file(file:UploadFile = File(...)):
    extension = file.filename.split(".")[-1] in ("jpg", "jpeg", "png")
    if not extension:
        return "Image must be jpg or png format!"
    #image1 = read_imagefile(await file.read())
    
    #img_byte_array = BytesIO()
    #image1.save(img_byte_array, format='JPEG', subsampling=0, quality=100)
    #img_byte_array = img_byte_array.getvalue()
    #image1=image1.write(image1)
   
    file.filename = f"{uuid.uuid4()}.jpg"
    name=str(file.filename)
  #  print(file.file)
    contents = await file.read()

    #print(type(contents))
    # <-- Important!
   # contents = contents.decode("utf-16")
    #contents = contents.split("\r\n")
   
    #with open(file.filename,"rb") as f:
    #    contents=f.read()
    with open(file.filename,"wb") as f:
        f.write(contents) 
        #img_byte_array = BytesIO()
        #contents.save(img_byte_array, format='JPEG', subsampling=0, quality=100)
        #img_byte_array = img_byte_array.getvalue()
    #image1=image1.write(image1)
    #img_byte_array = img_byte_array.decode('utf-8', 'ignore')
   # pic = request.FILES['image']
    #img_new = images(img_main= pic)
   # img_new.save()
   #img = Image.open(io.BytesIO(request_object_content))
    image1 = preprocessing.image.load_img(name, target_size=(224, 224,3))
    #image1 = np.asarray(image.resize((224, 224)))[..., :3]
    #image1 = np.asarray(contents)

  #  image1=  np.resize(contents,(224, 224,3))
    #print(image1)
    #image1.astype(float)
   # image1 = np.asarray(image.resize(image,(224, 224,3)))
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


    
if __name__ == "__main__":
 uvicorn.run(app, host="127.0.0.1",port=7000)
