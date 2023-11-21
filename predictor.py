import torch
from network import VGG16v2
import streamlit as st
import re
import numpy as np

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cpu'

@st.cache_resource
def load_model(model_path):
    with torch.no_grad():
        model = VGG16v2(num_classes=2)
        state_dict = torch.load(model_path)
        # remove saved deprecated running_* keys in InstanceNorm from the checkpoint
        for k in list(state_dict.keys()):
            if re.search(r'in\d+\.running_(mean|var)$', k):
                del state_dict[k]
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        return model
    
    
# @st.cache_data
def predict(model, stress_image, rest_image, report_df):
    
    w, h = stress_image.shape
    
    stress_image = (stress_image - stress_image.min()) / (stress_image.max() - stress_image.min())
    rest_image = (rest_image - rest_image.min()) / (rest_image.max() - rest_image.min())
    
    input_image = np.zeros((3, w, h))
    input_image[0, :, :] = stress_image
    input_image[1, :, :] = stress_image
    input_image[2, :, :] = stress_image
    
    # input_image = [stress_image, stress_image, stress_image] # TODO: modify input
    input_image = input_image[np.newaxis,...]
    input_image = torch.tensor(input_image)
    input_image = input_image.to(device, dtype=torch.float32)
    
    with torch.no_grad():
        # print(input_image.size)
        m = torch.nn.Sigmoid()
        output = model(input_image)
        output = m(output)
    
    return output.cpu().numpy()