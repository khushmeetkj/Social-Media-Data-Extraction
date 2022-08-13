import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image
import numpy as np
import os
from scipy.spatial import distance
import csv
import cv2
import pandas as pd
#########
model_url = "https://tfhub.dev/tensorflow/efficientnet/lite0/feature-vector/2"
IMAGE_SHAPE = (224, 224)
layer = hub.KerasLayer(model_url)
model = tf.keras.Sequential([layer])
def extract(file):
  file = Image.open(file).convert('L').resize(IMAGE_SHAPE)

  file = np.stack((file,)*3, axis=-1)

  file = np.array(file)/255.0

  embedding = model.predict(file[np.newaxis, ...])
  vgg16_feature_np = np.array(embedding)
  flattended_feature = vgg16_feature_np.flatten()
  return flattended_feature

metric = 'cosine'
def compare(img1):
  source = 'C:/Users/DELL/OneDrive/Desktop/Flipkart/image similarity score/flipkartfiles/'
  m = 'C:/Users/DELL/OneDrive/Desktop/Flipkart/outputs/'
  for file in os.listdir(source):
    print(source+file)
    img2 = source+file
    s1 = extract(img1)
    s2 = extract(img2)
    dc = distance.cdist([s1], [s2], metric)[0]
    print(dc)
    print("the distance between original and the original is {}".format(dc))
    row = file, img1.replace(m,''), dc[0]
    writer.writerow(row)

f=open('../outputs/scoreresult.csv','w')
writer=csv.writer(f)
header = ['Instagram_Image_Number', 'Flipkart_Image_Number', 'Similarity_Score']
writer.writerow(header)
compare('C:/Users/DELL/OneDrive/Desktop/Flipkart/outputs/fashionblogger29.jpg')
df = pd.read_csv('C:/insUsers/DELL/OneDrive/Desktop/Flipkart/outputs/scoreresult.csv', delim_whitespace=True)
df2 = df.groupby('Instagram_Image_Number').value.nlargest(5).reset_index()
print(df2)