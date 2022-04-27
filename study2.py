from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

write = SummaryWriter("logs")

image_path="image/wallhaven-vm5265.jpg"
img = Image.open(image_path)

#ToTensor
trans_totensor = transforms.ToTensor()
img_tensor = trans_totensor(img)
write.add_image("Totensor",img_tensor)

#Normalize
print(img_tensor[0][0][0])
trans_normalize = transforms.Normalize([2, 4, 6],[3, 2, 1])
img_norm = trans_normalize(img_tensor)
print(img_norm[0][0][0])
write.add_image("Normalize",img_norm,2)

#Resize
print(img.size)
trans_resize = transforms.Resize((512,512))
# img PIL -> resize -> img_resize PIL
img_resize = trans_resize(img)
# img_resize PIL -> totensor -> img_resize tensor
img_resize = trans_totensor(img_resize)
write.add_image("Resize",img_resize,0)

#Compose - resize - 2
trans_resize_2 = transforms.Resize(512)
#PIL -> PIL -> tensor
trans_compose = transforms.Compose([trans_resize_2, trans_totensor])
img_resize_2 = trans_compose(img)
write.add_image("Resize",img_resize_2,1)

#RandomCrop
trans_random = transforms.RandomCrop(512)
trans_compose_2 = transforms.Compose([trans_random,trans_totensor])
for i in range(10):
    img_random = trans_compose_2(img)
    write.add_image("RandomCrop",img_random,i)

write.close()