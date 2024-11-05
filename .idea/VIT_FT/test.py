import torch
from transformers import ViTForImageClassification, ViTImageProcessor
from PIL import Image
import matplotlib.pyplot as plt

# 设置训练好的模型目录路径
model_path = "./outputs/final_model"


# 加载模型和图像处理器的函数
def load_model(model_path):
    """
    加载模型和图像处理器
    :param model_path: 模型目录路径
    :return: 加载好的模型和图像处理器
    """
    model = ViTForImageClassification.from_pretrained(model_path)
    processor = ViTImageProcessor.from_pretrained(model_path)
    return model, processor


# 加载模型和图像处理器
model, feature_extractor = load_model(model_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


# 测试单张图像的函数
def test_single_image(image_path):
    """
    测试单张图像并显示预测结果
    :param image_path: 测试图像的路径
    """
    image = Image.open(image_path).convert("RGB")
    inputs = feature_extractor(images=image, return_tensors="pt").to(device)

    # 模型推理
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_idx = logits.argmax(-1).item()
    predicted_label = 'REAL' if predicted_class_idx == 1 else 'FAKE'

    print(f"Predicted label: {predicted_label}")

    # 显示图像和预测标签
    plt.imshow(image)
    plt.axis('off')
    plt.title(f"Prediction: {predicted_label}")
    plt.show()


# 设置测试图像路径（替换为实际的图像路径）
test_image_path = "F:/python project/deepfakedetection/test_real.jpg"  # 替换为要测试的图像路径
test_single_image(test_image_path)
