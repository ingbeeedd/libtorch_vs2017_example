import torch
import torchvision
import timm
import warnings

warnings.filterwarnings('ignore')

def create_module(model_name: str, num_classes: int, batch_size: int, ndim: int, image_size: int):
    if model_name == "alexnet":
        model = torchvision.models.alexnet(pretrained=True)
        model.classifier[6] = torch.nn.Linear(model.classifier[6].in_features, num_classes)
    elif model_name == "vgg19":
        model = torchvision.models.vgg19(pretrained=True)
        model.classifier[6] = torch.nn.Linear(model.classifier[6].in_features, num_classes)
    elif model_name == "resnet18":
        model = torchvision.models.resnet18(pretrained=True)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    elif model_name == "inceptionv3":
        image_size = 299
        model = torchvision.models.inception_v3(pretrained=True)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    elif model_name == "mobilenet_small":
        model = torchvision.models.mobilenet_v3_small(pretrained=True)
        model.classifier[3] = torch.nn.Linear(model.classifier[3].in_features, num_classes)
    elif model_name == "mobilenet_large":
        model = torchvision.models.mobilenet_v3_large(pretrained=True)
        model.classifier[3] = torch.nn.Linear(model.classifier[3].in_features, num_classes)
    elif model_name == "resnext50":
        model = torchvision.models.resnext50_32x4d(pretrained=True)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    elif model_name == "wide_resnet50_2":
        model = torchvision.models.wide_resnet50_2(pretrained=True)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    elif model_name == "efficientnet":
        model = timm.create_model("efficientnet_b0", pretrained=True)
        model.classifier = torch.nn.Linear(model.classifier.in_features, num_classes)
        
    filename = f"pretrained_models/{model_name}.pt"
    traced_script_module.save(filename)
    print(f"Successfully created scriptmodule file {filename}.")

if __name__ == "__main__":
    model_names = [
                    # "alexnet",
                    # "vgg19",
                    # "resnet18",
                    # "inceptionv3",
                    # "mobilenet_small",
                    # "mobilenet_large",
                    # "resnext50",
                    "wide_resnet50_2",
                    # "efficientnet"
                   ]    

    num_classes = 10
    batch_size = 64
    ndim = 3
    image_size = 28
    
    for model_name in model_names:
    
        model = torch.hub.load('pytorch/vision:v0.9.0', model_name, pretrained=True)
        
        example = torch.rand(batch_size, ndim, image_size, image_size)

        traced_script_module = torch.jit.trace(model, example)
        
        filename = f"pretrained_models/{model_name}_mnist.pt"
        traced_script_module.save(filename)
        print(f"Successfully created scriptmodule file {filename}.")
        
