from main import load_model, preprocess
from PIL import Image


image = Image.open("/home/agux/Documents/Uni/Advanced/Project/Chest_Xray_PA_3-8-2010.png")

print(image.info)
print(image.format)
print(image.size)
print(image.mode)

model = load_model()

preprocessed_image = preprocess(image)
x = model(preprocessed_image)
Image.fromarray((x[0, 0].detach().numpy()).astype('uint8')).show()
