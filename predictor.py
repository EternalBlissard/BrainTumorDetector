### Imports for Modules ### 
import gradio as gr
import os
import torch
from typing import Tuple, Dict
from timeit import default_timer as timer

### Functional Imports
from model import getEffNetModel

classNames = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']
effNetModel, effNetTransforms = getEffNetModel(42,len(classNames))
effNetModel.load_state_dict(torch.load(f="EffNetModel.pt",map_location=torch.device("cpu")))

def predictionMaker(img):
  startTime = timer()
  img = effNetTransforms(img).unsqueeze(0)
  effNetModel.eval()
  with torch.inference_mode():
    predProbs = torch.softmax(effNetModel(img),dim=1)
  predDict = {classNames[i]: float(predProbs[0][i]) for i in range(len(classNames))}
  endTime = timer()
  predTime = round(endTime-startTime,4)
  return predDict,predTime
