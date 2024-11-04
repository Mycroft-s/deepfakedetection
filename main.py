import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from PIL import Image
import torchvision.transforms as T



# 加载 Faster R-CNN 模型
model = fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()  # 设置为评估模式，不进行反向传播


def preprocess_image(image_path):
    # 加载图像
    image = Image.open(image_path).convert("RGB")

    # 定义预处理的变换
    transform = T.Compose([
        T.ToTensor(),  # 将图像转为Tensor
    ])

    # 应用变换
    return transform(image)


def predict(image_tensor, model, threshold=0.5):
    # 将图像放入模型进行预测
    with torch.no_grad():  # 不计算梯度
        predictions = model([image_tensor])[0]  # 模型返回一个列表，取第一个结果

    # 处理结果，筛选出置信度大于阈值的预测
    boxes = predictions['boxes']
    labels = predictions['labels']
    scores = predictions['scores']

    # 筛选满足条件的检测框
    filtered_boxes = []
    filtered_labels = []
    filtered_scores = []
    for i in range(len(scores)):
        if scores[i] > threshold:
            filtered_boxes.append(boxes[i].cpu().numpy())
            filtered_labels.append(labels[i].cpu().item())
            filtered_scores.append(scores[i].cpu().item())

    return filtered_boxes, filtered_labels, filtered_scores


def main(image_path):
    # 加载模型
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    model.eval()

    # 预处理图像
    image_tensor = preprocess_image(image_path)

    # 预测
    boxes, labels, scores = predict(image_tensor, model)

    # 打印预测结果
    print("Detection Results:")
    for box, label, score in zip(boxes, labels, scores):
        print(f"Label: {label}, Score: {score:.2f}, Box: {box}")


# 运行检测
image_path = "test_image.jpg"
main(image_path)
