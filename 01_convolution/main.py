import numpy as np
from PIL import Image


kernels = {
    'Edge-detection': [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]],
    'Blur': [[1/9, 1/9, 1/9], [1/9, 1/9, 1/9], [1/9, 1/9, 1/9]],
    'Sharpening': [[0,-1,0], [-1,5,-1], [0,-1,0]]
    
}

# 
img = np.array(Image.open('./download.jpg').convert("L"))
filter = np.array(kernels['Sharpening'])
new_img = np.zeros(img.shape)

# main Algo
for row in range(1, img.shape[0]-1):
    for colmn in range(1, img.shape[1]-1):
        region = img[row-1:row+2, colmn-1:colmn+2]
        new_img[row, colmn] = np.sum(region * filter)


# setting new filtered Image
new_img = np.clip(new_img, 0, 255).astype(np.uint8)
Image.fromarray(new_img).save('new_image.jpg')