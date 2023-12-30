from PIL import Image as foto
import os
from flask import Flask, request
from flask_restful import Api, Resource
from rembg import remove
import cv2
import numpy as np
import math
from ultralytics import YOLO
from pyzbar.pyzbar import decode
import shutil

import torch
from facenet_pytorch import InceptionResnetV1
from torchvision import transforms
import io
import os
import pandas as pd
from typing import Sequence
from google.cloud import vision

def detect_text(path):
    """Detects text in the file."""
    client = vision.ImageAnnotatorClient()

    with open(path, "rb") as image_file:
        content = image_file.read()

    image = vision.Image(content=content)

    response = client.text_detection(image=image,image_context={"language_hints": ["tr"]},)
    texts = response.text_annotations
    founded_text = texts[0].description

    if response.error.message:
        raise Exception(
            "{}\nFor more info on error messages, check: "
            "https://cloud.google.com/apis/design/errors".format(response.error.message)
        )
    
    return founded_text
        

def ocr_controller(image_path):
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="vision-ai-api-396908-ef90d6ec14a2.json"
    text_result = detect_text(image_path)
    return text_result




request_counter = 0

def read_barcode(image_path):

    image = cv2.imread(image_path)
    original_height, original_width = image.shape[:2]
    new_height = int(500 * original_height / original_width)
    resized_image = cv2.resize(image, (500, new_height))
    gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    _, thresholded_image = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    barcodes = decode(thresholded_image)
    tc = ""
    if(not barcodes):
        print("Barcode couldn't detected")
    else:
        for barcode in barcodes:
            barcode_data = barcode.data.decode("utf-8")
            tc += barcode_data
            
    return tc

def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized

def find_corner_points(corners):
  
    sorted_corners = sorted(corners, key=lambda corner: corner[0][0])
    if(sorted_corners[0][0][1]>sorted_corners[1][0][1]):
        bottom_left_corner = sorted_corners[0][0]
        top_left_corner = sorted_corners[1][0]
        if(sorted_corners[2][0][1]>sorted_corners[3][0][1]):
            bottom_right_corner = sorted_corners[2][0]
            top_right_corner = sorted_corners[3][0]
        else:
            bottom_right_corner = sorted_corners[3][0]
            top_right_corner = sorted_corners[2][0]
            
    else:
        bottom_left_corner = sorted_corners[1][0]
        top_left_corner = sorted_corners[0][0]
        if(sorted_corners[2][0][1]>sorted_corners[3][0][1]):
            bottom_right_corner = sorted_corners[2][0]
            top_right_corner = sorted_corners[3][0]
        else:
            bottom_right_corner = sorted_corners[3][0]
            top_right_corner = sorted_corners[2][0]
         
    
    return top_left_corner,top_right_corner,bottom_left_corner,bottom_right_corner
    
def check_diagonal_ratio(top_left_corner,top_right_corner,bottom_right_corner,bottom_left_corner):
    
    diagonal1 = math.sqrt(pow(top_left_corner[0] - bottom_right_corner[0],2) + pow(top_left_corner[1] - bottom_right_corner[1],2))
    diagonal2 = math.sqrt(pow(top_right_corner[0] - bottom_left_corner[0],2) + pow(top_right_corner[1] - bottom_left_corner[1],2))
    error_tolerance_diagonal = 3 #percentage
    result = diagonal1 < (diagonal2 + diagonal2*error_tolerance_diagonal/100) and (diagonal2 - diagonal2*error_tolerance_diagonal/100) < diagonal1
    
    return result
def check_widht_height_ratio(top_left_corner,top_right_corner,bottom_right_corner,bottom_left_corner):

    width = math.sqrt(pow(top_left_corner[0] - top_right_corner[0],2) + pow(top_left_corner[1] - top_right_corner[1],2))
    height = math.sqrt(pow(top_left_corner[0] - bottom_left_corner[0],2) + pow(top_left_corner[1] - bottom_left_corner[1],2))
    result = False
    if(width>height):
        if(width>(height + height*0.25)):
            result = True
    return result
    
def check_id_card_corners(image): 
    image = image_resize(image=image,height=320) 
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray_image, (5, 5), 0)
    edges = cv2.Canny(blurred,30,100)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_corner_count = 4
    
    if len(contours) >= 1:
        main_contour = max(contours, key=cv2.contourArea)
        epsilon = 0.04 * cv2.arcLength(main_contour, True)
        corners = cv2.approxPolyDP(main_contour, epsilon, True)
        if len(corners) == min_corner_count:
            corner_image = image.copy()
            for corner in corners:
                cv2.circle(corner_image, tuple(corner[0]), 5, (0, 0, 255), -1)
          
            top_left_corner,top_right_corner,bottom_left_corner,bottom_right_corner = find_corner_points(corners)
            if(check_diagonal_ratio(top_left_corner,top_right_corner,bottom_right_corner,bottom_left_corner)):
                if(check_widht_height_ratio(top_left_corner,top_right_corner,bottom_right_corner,bottom_left_corner)):
                   return True     

    return False
        
def remove_background(input): 
    try:
        output = remove(input)
        return output
    except TypeError as e:
        return False
        

 
def check_id(image): 
    result = check_id_card_corners(image)
    return result

def shape_controller(path):
    try:
        input = foto.open(path)
    except Exception as e:
        print('Failed to upload to ftp: '+ str(e))

    output = remove_background(input)
    if(output != False):
        arrImg = np.array(output)
        cvImg = cv2.cvtColor(arrImg, cv2.COLOR_RGB2BGR)
        result = check_id(cvImg)
        return result
    else:
        return False
    
def PhotoCutSave():
    img = cv2.imread(r'onyuz.jpeg')
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

    img2 = cv2.imread(r'foto.jpeg')
    gray2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)

    
    faces = face_cascade.detectMultiScale(gray,1.1,4)
    faces2 = face_cascade.detectMultiScale(gray2,1.1,4)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), 
                    (0, 0, 255), 2)
        
        faces = img[y:y + h, x:x + w]
        cv2.imshow("face",faces)
        cv2.imwrite('kimlikfoto.jpg', faces)
    

    
    for (x, y, w, h) in faces2:
        cv2.rectangle(img2, (x, y), (x+w, y+h), 
                    (0, 0, 255), 2)
        
        faces2 = img2[y:y + h, x:x + w]
        cv2.imshow("face",faces2)
        cv2.imwrite('userfoto.jpg', faces2)
    
        # Load the FaceNet model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

        # Load and preprocess the two face images
        def preprocess_image(image_path):
            image = foto.open(image_path)
            image = transforms.Resize((160, 160))(image)
            image = transforms.ToTensor()(image).unsqueeze(0)
            return image.to(device)

        image1_path = 'kimlikfoto.jpg'
        image2_path = 'userfoto.jpg'

        image1 = preprocess_image(image1_path)
        image2 = preprocess_image(image2_path)

        # Calculate embeddings for the images
        embedding1 = model(image1)
        embedding2 = model(image2)

        similarity = torch.nn.functional.cosine_similarity(embedding1, embedding2)

        # Convert similarity to a percentage
        percentage_similarity = (1 + similarity.item()) / 2 * 100

        # Set a threshold for matching
        threshold = 70.0  # You might need to fine-tune this threshold based on your data

        # Determine if the images match based on the percentage and threshold
        if percentage_similarity >= threshold:
            formatted_percentage =  round(percentage_similarity, 2)
            info["yuzde"] = str(formatted_percentage)
            print(f"The images are a match with {percentage_similarity:.2f}% similarity.")
            return True
        else:
            formatted_percentage =  round(percentage_similarity, 2)
            info["yuzde"] = str(formatted_percentage)
            print(f"The images do not match. Similarity: {percentage_similarity:.2f}%.")
            return False


info = {
    "arkayuzShape": False,
    "onyuzShape": False,
    "isOnYuzReadable": False,
    "isArkaYuzReadable": False,
    "isTC": False,
    "isPhoto": False,
    "adi":"",
    "soyadi":"",
    "cinsiyeti":"",
    "dogumTarihi":"",
    "anneAdi":"",   
    "TC":"",
    "uyruk":"",
    "seriNo":"",
    "pen":"",
    "babaAdi":"",
    "yuzde":""
} 

app = Flask(__name__)
api = Api(app)

class Image(Resource):
    
    def post(self):
        global request_counter
        request_counter += 1




        try:
            

            if request_counter == 1:
                
                try:
                    shutil.rmtree(r"C:\andorid-ai-api\runs")
                    print("Directory removed successfully.")
                except OSError as e:
                    print("Error:", e)
                # Get the image file from the request
                img = request.files.get("image")
                print(img)
        
                # Determine the current working directory
                current_directory = os.getcwd()
                # Save img in the same directory with a specific filename
                filename = os.path.join(current_directory, "onyuz.jpeg")
                img.save(filename)
                img.close()
                    # Get the image file from the request
                img2 = request.files.get("image2")
                print(img2)
            
                # Determine the current working directory
                current_directory = os.getcwd()
                # Save img in the same directory with a specific filename
                filename = os.path.join(current_directory, "arkayuz.jpeg")
                img2.save(filename)
                # Close the image
                img2.close()

                #SHAPPEEEEEEEE
                #Depending if it should send True or False for arkayuz and onyuz
                isArkaYuzShape = shape_controller("arkayuz.jpeg")
                isOnYuzShape = shape_controller("onyuz.jpeg")


                info["arkayuzShape"] = isArkaYuzShape
                info["onyuzShape"] = isOnYuzShape
                #SHAPPEEEEEEEE


                #Readiablityyyy 

                model = YOLO("best.pt")

                results = model.predict(source=["onyuz.jpeg","arkayuz.jpeg"],show=True,conf=0.50,save_txt=True,save=True,imgsz=640, save_crop = True)

                frontFaceReadable = 0
                backFaceReadable = 0


                frontFaceReadable += len(results[0].boxes.cls) #predicted_classes += len(r.boxes.cls)
                backFaceReadable += len(results[1].boxes.cls)



                if frontFaceReadable == 12:
                    isOnYuzReadable = True
                else:
                    isOnYuzReadable = False

                if backFaceReadable == 8:
                    isArkaYuzReadable = True
                else:
                    isArkaYuzReadable = False


                info["isOnYuzReadable"] = isOnYuzReadable
                info["isArkaYuzReadable"] = isArkaYuzReadable
                
                #Readiablityyyy

                







                #YOU WILL SET isTC in here
        
                barkodTC = read_barcode("runs/detect/predict/crops/barkod/arkayuz.jpg")
               

                TC = ocr_controller("runs/detect/predict/crops/Tc/onyuz.jpg")
                print("barkodTC ", barkodTC)
                print("TC",TC )

                if(barkodTC == TC):
                    isTC  = True
                else:
                    isTC = False

                info["isTC"] = isTC
                info["TC"] = TC

                adi = ocr_controller("runs/detect/predict/crops/Ad/onyuz.jpg")
                soyadi = ocr_controller("runs/detect/predict/crops/Soyad/onyuz.jpg")
                cinsiyeti = ocr_controller("runs/detect/predict/crops/Cinsiyet/onyuz.jpg")
                dogumTarihi = ocr_controller("runs/detect/predict/crops/birthDate/onyuz.jpg")
                anneAdi = ocr_controller("runs/detect/predict/crops/anneAdi/arkayuz.jpg")
                uyruk = ocr_controller("runs/detect/predict/crops/Uyruk/onyuz.jpg")
                seriNo = ocr_controller("runs/detect/predict/crops/SeriNo/onyuz.jpg")
                pen = ocr_controller("runs/detect/predict/crops/PenNo/arkayuz.jpg")
                babaAdi = ocr_controller("runs/detect/predict/crops/babaAdi/arkayuz.jpg")

                info["adi"] = adi
                info["soyadi"] = soyadi
                info["cinsiyeti"] = cinsiyeti
                info["dogumTarihi"] = dogumTarihi
                info["anneAdi"] = anneAdi
                info["uyruk"] = uyruk
                info["seriNo"] = seriNo
                info["pen"] = pen
                info["babaAdi"] = babaAdi

                #YOU WILL SET OCR HERE



            if request_counter == 2:
                request_counter = 0
                img2 = request.files.get("image2")
                print(img2)
            
                # Determine the current working directory
                current_directory = os.getcwd()
                # Save img in the same directory with a specific filename
                filename = os.path.join(current_directory, "foto.jpeg")
                img2.save(filename)
                img2.close()
                # Close the image
                


                
                #YOU WILL SET isPhoto in here
                info["isPhoto"] = (PhotoCutSave())
                

                






            return info



        except Exception as e:
            # If there is any error during processing, return False
            return info

api.add_resource(Image, "/image")

if __name__ == "__main__":
    app.run(host='192.168.1.108', port=5000, debug=True)