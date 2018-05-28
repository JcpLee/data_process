import numpy as np
from PIL import  Image
import matplotlib.pyplot as plt
img=Image.open('a.jpg')  #打开图像
img = img.resize((227, 227))
img1=Image.open('b.jpg')  #打开图像
img1 = img1.resize((227, 227))
gray1=img.convert('L')   #转换成灰度
gray2=img1.convert('L')

img = np.asarray(img)

b = np.zeros((img.shape[0],img.shape[1]), dtype=img.dtype)
g = np.zeros((img.shape[0],img.shape[1]), dtype=img.dtype)
r1 = np.zeros((img.shape[0],img.shape[1]), dtype=img.dtype)
r2 = np.zeros((img.shape[0],img.shape[1]), dtype=img.dtype)

b[:,:] = gray1
g[:,:] = gray2
mergedByNp = np.dstack([b,g])
r1[:,:] = mergedByNp[:,:,0]
r2[:,:] = mergedByNp[:,:,1]
print(mergedByNp.shape)
print(np.asarray(gray1))
print(r1)
print(np.asarray(gray1)-r1)

plt.subplot(2,1,1), plt.title('gray1')
plt.imshow(gray1,cmap='gray'),plt.axis('on')

plt.subplot(2,1,2), plt.title('r1')
plt.imshow(r1,cmap='gray'),plt.axis('on')

plt.show()