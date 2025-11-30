import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
from model_resnet import build_resnet18

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# GradCAM：针对最后一个卷积层
class GradCAM:
    def __init__(self, model):
        self.model = model
        self.gradients = None
        self.activations = None

        # 注册
        last_conv = list(model.children())[-2][-1].conv2
        last_conv.register_forward_hook(self.forward_hook)
        last_conv.register_backward_hook(self.backward_hook)

    def forward_hook(self, module, input, output):
        self.activations = output

    def backward_hook(self, module, grad_in, grad_out):
        self.gradients = grad_out[0]

    def generate(self, class_idx):
        grads = self.gradients.mean(dim=[2, 3], keepdim=True)
        cam = (grads * self.activations).sum(dim=1).squeeze()
        cam = torch.relu(cam)
        cam = cam.cpu().detach().numpy()
        cam = cv2.resize(cam, (224, 224))
        cam = (cam - cam.min()) / (cam.max() - cam.min())
        return cam

def preprocess(img_path):
    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    img = Image.open(img_path).convert("RGB")
    return tf(img).unsqueeze(0), img

def run_gradcam(img_path):
    model = build_resnet18(num_classes=38)
    model.load_state_dict(torch.load("cnn/best_resnet18.pth"))
    model.to(device)
    model.eval()

    cam = GradCAM(model)
    x, raw = preprocess(img_path)
    x = x.to(device)

    output = model(x)
    class_idx = output.argmax().item()

    model.zero_grad()
    output[0, class_idx].backward()

    heatmap = cam.generate(class_idx)

    heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    raw = np.array(raw)
    raw = cv2.resize(raw, (224, 224))
    overlay = cv2.addWeighted(raw, 0.5, heatmap, 0.5, 0)

    plt.imshow(overlay)
    plt.title("Grad-CAM")
    plt.axis("off")
    plt.savefig("cnn/gradcam_result.png")
    plt.show()

if __name__ == "__main__":
    run_gradcam("data/test/xxx.jpg")
    #只需要指定一张测试图，即可获得可解释性热力图。