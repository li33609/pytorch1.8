import os
import json

import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import matplotlib

from Model import Resnet
# from Model.Mobilenet_v3 import mobilenet_v3_large
from Model.ShuffleNet import shufflenet_v2_x1_0


img_size = {"B0": 224,
                "B1": 240,
                "B2": 260,
                "B3": 300,
                "B4": 380,
                "B5": 456,
                "B6": 528,
                "B7": 600}
num_model = "B3"


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose(
        [transforms.Resize(img_size[num_model]),
         transforms.CenterCrop(img_size[num_model]),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # load image
    img_path = "./test/OIP.jpg"
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = Image.open(img_path)
    plt.imshow(img)
    # [N, C, H, W]
    img = data_transform(img)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)

    # read class_indict
    json_path = './class_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    json_file = open(json_path, "r")
    class_indict = json.load(json_file)

    # create model
    # model = Resnet.resnet34(num_classes=5).to(device)
    model = Resnet.resnext50_32x4d(num_classes=15).to(device)
    # model = shufflenet_v2_x1_0(num_classes=5).to(device)

    # load model weights
    weights_path = "./ResnetWeights/50chong.pth"
    # weights_path = "./ShufflenetWeights/shufflenetv2_x1.pth"
    # weights_path = "./MobilenetWeight/mobilenet_v3_large_PRE.pth"

    assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
    model.load_state_dict(torch.load(weights_path, map_location=device))

    # prediction
    model.eval()
    with torch.no_grad():
        # predict class
        output = torch.squeeze(model(img.to(device))).cpu()
        predict = torch.softmax(output, dim=0)
        predict_cla = torch.argmax(predict).numpy()

    matplotlib.rcParams['font.sans-serif'] = ['KaiTi']
    print_res = "图片预测的类别为: {}   准确率: {:.3}".format(class_indict[str(predict_cla)],
                                                 predict[predict_cla].numpy())
    plt.title(print_res)
    print(print_res)
    plt.show()


if __name__ == '__main__':
    main()
