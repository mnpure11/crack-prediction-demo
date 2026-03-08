import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms
from PIL import Image
import os

st.set_page_config(page_title="Crack Prediction Demo", layout="wide")

# -------------------
# Model Architecture
# -------------------

class AttentionBlock(nn.Module):

    def __init__(self,F_g,F_l,F_int):
        super().__init__()

        self.W_g = nn.Sequential(
            nn.Conv2d(F_g,F_int,1),
            nn.InstanceNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l,F_int,1),
            nn.InstanceNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(F_int,1,1),
            nn.Sigmoid()
        )

    def forward(self,g,x):

        g1 = self.W_g(g)
        x1 = self.W_x(x)

        psi = self.psi(g1+x1)

        return x*psi


def conv_block(in_ch,out_ch):

    return nn.Sequential(
        nn.Conv2d(in_ch,out_ch,3,padding=1),
        nn.InstanceNorm2d(out_ch),
        nn.ReLU(inplace=True)
    )


class AttUNetGenerator(nn.Module):

    def __init__(self):

        super().__init__()

        base=32

        self.enc1=conv_block(3,base)
        self.enc2=conv_block(base,base*2)
        self.enc3=conv_block(base*2,base*4)

        self.res=nn.Sequential(
            conv_block(base*4,base*4),
            conv_block(base*4,base*4)
        )

        self.pool=nn.AvgPool2d(2)

        self.up3=nn.ConvTranspose2d(base*4,base*2,3,2,1,output_padding=1)
        self.att3=AttentionBlock(base*2,base*2,base)
        self.dec3=conv_block(base*4,base*2)

        self.up2=nn.ConvTranspose2d(base*2,base,3,2,1,output_padding=1)
        self.att2=AttentionBlock(base,base,base//2)
        self.dec2=conv_block(base*2,base)

        self.final=nn.Sequential(
            nn.Conv2d(base,3,7,padding=3),
            nn.Tanh()
        )

    def forward(self,x):

        e1=self.enc1(x)
        e2=self.enc2(self.pool(e1))
        e3=self.enc3(self.pool(e2))

        b=self.res(e3)

        u3=self.up3(b)
        a3=self.att3(u3,e2)
        d3=self.dec3(torch.cat([u3,a3],1))

        u2=self.up2(d3)
        a2=self.att2(u2,e1)
        d2=self.dec2(torch.cat([u2,a2],1))

        return self.final(d2)

# -------------------
# Load Model
# -------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def load_model():

    model=AttUNetGenerator().to(device)

    model.load_state_dict(
        torch.load("model/G_A2B_final.pth",map_location=device)
    )

    model.eval()

    return model

model=load_model()

# -------------------
# Image preprocessing
# -------------------

transform=transforms.Compose([
    transforms.Resize((512,512)),
    transforms.ToTensor()
])

# -------------------
# UI
# -------------------

st.title("Microstructure Crack Prediction")

IMAGE_FOLDER="test_images"

images=os.listdir(IMAGE_FOLDER)

selected=st.selectbox("Select input image",images)

img_path=os.path.join(IMAGE_FOLDER,selected)

image=Image.open(img_path).convert("RGB")

col1,col2=st.columns(2)

with col1:
    st.image(image,caption="Input Image")

if st.button("Run Model"):

    input_tensor=transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output=model(input_tensor)

    output_img=(output.squeeze().permute(1,2,0).cpu().numpy()+1)/2

    with col2:
        st.image(output_img,caption="Predicted Crack")

if st.button("Reset"):
    st.rerun()
