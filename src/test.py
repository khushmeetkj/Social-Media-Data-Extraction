from email import header
import torch
import cv2
import torchvision.transforms as transforms
import numpy as np
import joblib
import argparse
import csv
from models import MultiHeadResNet50

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', required=True, help='path to input image')
args = vars(parser.parse_args())
# define the computation device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MultiHeadResNet50(pretrained=False, requires_grad=False)
checkpoint = torch.load('../outputs/model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
# read an image
image = cv2.imread(args['input'])
# keep a copy of the original image for OpenCV functions
orig_image = image.copy()
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# apply image transforms
image = transform(image)
# add batch dimension
image = image.unsqueeze(0).to(device)
# forward pass the image through the model
outputs = model(image)
# extract the five output
output1, output2, output3, output4, output5, output6, output7 = outputs
# get the index positions of the highest label score
out_label_1 = np.argmax(output1.detach().cpu())
out_label_2 = np.argmax(output2.detach().cpu())
out_label_3 = np.argmax(output3.detach().cpu())
out_label_4 = np.argmax(output4.detach().cpu())
out_label_5 = np.argmax(output5.detach().cpu())
out_label_6 = np.argmax(output6.detach().cpu())
out_label_7 = np.argmax(output7.detach().cpu())
# load the label dictionaries
num_list_gender = joblib.load('../input/num_list_gender.pkl')
num_list_master = joblib.load('../input/num_list_master.pkl')
num_list_sub = joblib.load('../input/num_list_sub.pkl')
num_list_article = joblib.load('../input/num_list_article.pkl')
num_list_base = joblib.load('../input/num_list_base.pkl')
num_list_season = joblib.load('../input/num_list_season.pkl')
num_list_usage = joblib.load('../input/num_list_usage.pkl')
# get the keys and values of each label dictionary
gender_keys = list(num_list_gender.keys())
gender_values = list(num_list_gender.values())
master_keys = list(num_list_master.keys())
master_values = list(num_list_master.values())
sub_keys = list(num_list_sub.keys())
sub_values = list(num_list_sub.values())
article_keys = list(num_list_article.keys())
article_values = list(num_list_article.values())
base_keys = list(num_list_base.keys())
base_values = list(num_list_base.values())
season_keys = list(num_list_season.keys())
season_values = list(num_list_season.values())
usage_keys = list(num_list_usage.keys())
usage_values = list(num_list_usage.values())
final_labels = []
# append the labels by mapping the index position to the values 
final_labels.append(gender_keys[gender_values.index(out_label_1)])
final_labels.append(master_keys[master_values.index(out_label_2)])
final_labels.append(sub_keys[sub_values.index(out_label_3)])
final_labels.append(article_keys[article_values.index(out_label_4)])
final_labels.append(base_keys[base_values.index(out_label_5)])
final_labels.append(season_keys[season_values.index(out_label_6)])
final_labels.append(usage_keys[usage_values.index(out_label_7)])
# write the label texts on the image
cv2.putText(
    orig_image, final_labels[0], (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 
    0.8, (0, 255, 0), 2, cv2.LINE_AA 
)
cv2.putText(
    orig_image, final_labels[1], (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 
    0.8, (0, 255, 0), 2, cv2.LINE_AA 
)
cv2.putText(
    orig_image, final_labels[2], (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 
    0.8, (0, 255, 0), 2, cv2.LINE_AA 
)
cv2.putText(
    orig_image, final_labels[3], (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 
    0.8, (0, 255, 0), 2, cv2.LINE_AA 
)
cv2.putText(
    orig_image, final_labels[4], (10, 125), cv2.FONT_HERSHEY_SIMPLEX, 
    0.8, (0, 255, 0), 2, cv2.LINE_AA 
)
cv2.putText(
    orig_image, final_labels[5], (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 
    0.8, (0, 255, 0), 2, cv2.LINE_AA 
)
cv2.putText(
    orig_image, final_labels[6], (10, 175), cv2.FONT_HERSHEY_SIMPLEX, 
    0.8, (0, 255, 0), 2, cv2.LINE_AA 
)
f=open('../outputs/result.csv','w')
writer=csv.writer(f)
header = ['Gender', 'masterCategory', 'subCategory', 'articleType', 'baseColour', 'season', 'usage']
writer.writerow(header)
row=final_labels[0],final_labels[1],final_labels[2],final_labels[3],final_labels[4],final_labels[5],final_labels[6]
writer.writerow(row)
# visualize and save the image
cv2.imshow('Predicted labels', orig_image)
cv2.waitKey(0)
save_name = args['input'].split('/')[-1]
cv2.imwrite(f"../outputs/{save_name}", orig_image)
print(final_labels[0],final_labels[1],final_labels[2],final_labels[3],final_labels[4],final_labels[5],final_labels[6])

a = 'https://www.flipkart.com/search?q=' + final_labels[0] + '+' + final_labels[1] + '+' + final_labels[2] + '+' + final_labels[3] + '+' + final_labels[4] + '+' + final_labels[5] + '+' + final_labels[6]
b = a.strip()
print(b)