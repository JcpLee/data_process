from PIL import Image
import matplotlib.pyplot as plt
img=Image.open('a.jpg')  #打开图像
img = img.resize((227, 227))
img1=Image.open('b.jpg')  #打开图像
img1 = img1.resize((227, 227))
gray1=img.convert('L')   #转换成灰度
gray2=img1.convert('L')   #转换成灰度

pic=Image.merge('RGB',(gray1,gray2,gray1)) #合并三通道
r,g,b=pic.split()   #分离三通道
plt.figure("beauty")
# plt.imshow(pic)
plt.subplot(2,2,1), plt.title('origin')
plt.imshow(gray1,cmap='gray'),plt.axis('off')
print(gray1)
# plt.subplot(2,3,2), plt.title('gray')
# plt.imshow(gray,cmap='gray'),plt.axis('off')
# plt.subplot(2,3,3), plt.title('merge')
# plt.imshow(pic),plt.axis('off')
plt.subplot(2,2,2), plt.title('r')
plt.imshow(g,cmap='gray'),plt.axis('off')

plt.subplot(2,2,3), plt.title('g')
plt.imshow(r,cmap='gray'),plt.axis('off')
print(r)

plt.subplot(2,2,4), plt.title('b')
plt.imshow(gray2,cmap='gray'),plt.axis('off')
plt.show()