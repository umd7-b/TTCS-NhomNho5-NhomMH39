import cv2
import json
import matplotlib.pyplot as plt
import os


with open('dataset/train/_annotations.coco.json', 'r') as f:
    coco = json.load(f)

id_to_filename = {img['id']: img['file_name'] for img in coco['images']}


IMAGE_DIR = 'dataset/train/'

ann = coco['annotations'][0]
image_id = ann['image_id']
file_name = id_to_filename[image_id]

img = cv2.imread(os.path.join(IMAGE_DIR, file_name))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

x, y, w, h = [int(v) for v in ann['bbox']]
cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
cv2.putText(img, 'biensoxehoi', (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

plt.figure(figsize=(8, 6))
plt.imshow(img)
plt.title(file_name)
plt.axis('off')
plt.show()