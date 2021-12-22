import torch

if __name__ == "__main__":
    model = torch.hub.load('pytorch/vision:v0.11.0', 'efficientnet_b0', pretrained=True)
    
    # print(torch.hub.list('pytorch/vision:v0.11.0') )
    print(model)