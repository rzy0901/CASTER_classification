import torch
import torch.nn as nn
import torch.nn.functional as nf

class fc_part(nn.Module):
    def __init__(self):
        super().__init__()
        # self.fc1 = nn.Linear(512,512)
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 120)
        self.fc3 = nn.Linear(120,5)

    def forward(self, x):
        x = nf.relu(self.fc1(x))
        x = nf.relu(self.fc2(x))
        x = nf.relu(self.fc3(x))
        # x = self.fc1(x)
        return x

class FeatureExtractorResNet(nn.Module):
    def __init__(self, original_model):
        super(FeatureExtractorResNet, self).__init__()
        self.conv_features = None

        for name, layer in original_model.named_children():
            if name == 'fc':  # Stop before the classifier
                break
            setattr(self, name, layer)
            # print(f"{name}")

    def forward(self, x):
        for name, layer in self.named_children():
            x = layer(x)
            if name == 'avgpool':  # Save the output of the last convolutional layer
                self.conv_features = x

        features = self.conv_features.view(self.conv_features.size(0), -1)
        return features

class FeatureExtractorResNet18(nn.Module):
    def __init__(self, original_model):
        super(FeatureExtractorResNet18, self).__init__()
        self.conv1 = original_model.conv1
        self.bn1 = original_model.bn1
        self.relu = original_model.relu
        self.maxpool = original_model.maxpool
        self.layer1 = original_model.layer1
        self.layer2 = original_model.layer2
        self.layer3 = original_model.layer3
        self.layer4 = original_model.layer4
        self.avgpool = original_model.avgpool
        
        # self.features = nn.Sequential(
        #     original_model.conv1,
        #     original_model.bn1,
        #     original_model.relu,
        #     original_model.maxpool,
        #     original_model.layer1,
        #     original_model.layer2,
        #     original_model.layer3,
        #     original_model.layer4,
        #     original_model.avgpool
        # )

    def forward(self, x):
        # x = self.features(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x

class ClassifierResNet(nn.Module):
    def __init__(self, original_model):
        super(ClassifierResNet, self).__init__()
        self.fc = original_model.fc

    def forward(self, x):
        x = self.fc(x)
        return x

from torchvision import models
def load_resnet18_by_featureExtractor_classifier(feature_extractor_model,classifier_model,resnet18=None):
    if resnet18 is None:
        resnet18 = models.resnet18(pretrained=True)
    resnet18.conv1 = feature_extractor_model.conv1
    resnet18.bn1 = feature_extractor_model.bn1
    resnet18.relu = feature_extractor_model.relu
    resnet18.maxpool = feature_extractor_model.maxpool
    resnet18.layer1 = feature_extractor_model.layer1
    resnet18.layer2 = feature_extractor_model.layer2
    resnet18.layer3 = feature_extractor_model.layer3
    resnet18.layer4 = feature_extractor_model.layer4
    resnet18.avgpool = feature_extractor_model.avgpool
    resnet18.fc = classifier_model.fc
    return resnet18

def load_resnet_by_featureExtractor_classifier(feature_extractor_model,classifier_model,resnet):
    for name, layer in feature_extractor_model.named_children():
        setattr(resnet,name,layer)
    resnet.fc = classifier_model.fc
    return resnet

class Discriminator(nn.Module):
    def __init__(self, input_size=512, hidden_size1=200, hidden_size2=20):
        super(Discriminator, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size1)
        self.relu1 = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size1, hidden_size2)
        self.relu2 = nn.ReLU()
        self.output_layer = nn.Linear(hidden_size2, 1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu1(x)
        x = self.layer2(x)
        x = self.relu2(x)
        x = self.output_layer(x)
        return x

if __name__ == "__main__":
    from torchvision import models
    MODEL_FILE = './model1/best_resnet18_model.pth' # pretrained model using simulated dataset
    model = models.resnet18(pretrained=True)
    model.fc = fc_part()
    model.load_state_dict(torch.load(MODEL_FILE))
    feature_extractor_model = FeatureExtractorResNet18(model)
    classifier_model = ClassifierResNet(model)
    print(feature_extractor_model)
    print(classifier_model)
    
    batch_size = 10
    channels = 3  # Assuming RGB images
    height = 60
    width = 80
    random_data = torch.rand((batch_size, channels, height, width))
    features = feature_extractor_model(random_data)
    output = classifier_model(features)
    output2 = model.forward(random_data)
    print("Input Data Shape:", random_data.shape)
    print("Extracted Features Shape:", features.shape)
    print("Classifier Output Shape:", output.shape)
    print(torch.equal(output,output2))
    
    new_model = models.resnet18(pretrained=True)
    # new_model = load_resnet_by_featureExtractor_classifier(feature_extractor_model,classifier_model, new_model)
    # new_model = load_resnet18_by_featureExtractor_classifier(feature_extractor_model,classifier_model, new_model)
    new_model = load_resnet18_by_featureExtractor_classifier(feature_extractor_model,classifier_model)
    output3 = new_model.forward(random_data)
    print(torch.equal(output2,output3))
    
    # model2 = models.resnet34(pretrained=True)
    # print(model2)
    # feature_extractor_model2 = FeatureExtractorResNet(model2)
    # classifier_model2 = ClassifierResNet(model2)
    # print(feature_extractor_model2)
    # print(classifier_model2)
