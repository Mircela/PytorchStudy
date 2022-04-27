from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

image_path = "dataset/train/ants_image/0013035.jpg"
img = Image.open(image_path)

write = SummaryWriter("logs")


#创建transforms工具 name为 tensor_trans(tool)
tensor_trans = transforms.ToTensor()

tensor_img = tensor_trans(img)   #tensor_image(result)=tensor_trans(tool)(image(input)) ====== result=tool(input)

write.add_image("Tensor_image",tensor_img)

write.close()
