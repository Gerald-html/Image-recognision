from imageai.Classification import ImageClassification
import os
import torch

# Fix PyTorch weights loading issue
_original_load = torch.load
torch.load = lambda f, *args, **kwargs: _original_load(f, *args, weights_only=False, **kwargs)

execution = os.getcwd()
prediction = ImageClassification()
prediction.setModelTypeAsResNet50()
prediction.setModelPath(os.path.join(execution, "resnet50-19c8e357.pth"))
prediction.loadModel()

predictions, probabilities = prediction.classifyImage(os.path.join(execution, "giraffe.jpg"), result_count=10)
for eachPrediction, eachProbability in zip(predictions, probabilities):
    print(eachPrediction , " : " , eachProbability)
















