import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms
from PIL import Image
import os

st.set_page_config(page_title="Microstructure Crack Prediction", layout="wide")

# ------------------------
# Attention Block
# ------------------------

class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()

        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, 1, 1, 0, bias=True),
            nn.InstanceNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, 1, 1, 0, bias=True),
            nn.InstanceNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(F_int,1,1,1,0,bias=True),
            nn.Sigmoid()
        )

    def forward(self, g, x):

        g1 = self.W_g(g)
        x1 = self.W_x(x)

        psi = self.psi(g1 + x1)

        return x * psi


# ------------------------
# Convolution Block
# ------------------------

def conv_block(in_ch, out_ch, kernel=3, padding=1, use_dropout=False):

    layers = [
        nn.Conv2d(in_ch, out_ch, kernel, padding=padding, bias=False),
        nn.InstanceNorm2d(out_ch),
        nn.ReLU(inplace=True)
    ]

    if use_dropout:
        layers.append(nn.Dropout(0.2))

    return nn.Sequential(*layers)


# ------------------------
# Generator
# ------------------------

class AttUNetGenerator(nn.Module):

    def __init__(self, in_channels=3, out_channels=3, base_ch=32):
        super().__init__()

        self.enc1 = conv_block(in_channels, base_ch)
        self.enc2 = conv_block(base_ch, base_ch*2)
        self.enc3 = conv_block(base_ch*2, base_ch*4)

        self.res_blocks = nn.Sequential(
            conv_block(base_ch*4, base_ch*4, use_dropout=True),
            conv_block(base_ch*4, base_ch*4, use_dropout=True)
        )

        self.up3 = nn.ConvTranspose2d(base_ch*4, base_ch*2, 3, 2, 1, output_padding=1)
        self.att3 = AttentionBlock(base_ch*2, base_ch*2, base_ch)
        self.dec3 = conv_block(base_ch*4, base_ch*2, use_dropout=True)

        self.up2 = nn.ConvTranspose2d(base_ch*2, base_ch, 3, 2, 1, output_padding=1)
        self.att2 = AttentionBlock(base_ch, base_ch, base_ch//2)
        self.dec2 = conv_block(base_ch*2, base_ch, use_dropout=True)

        self.final = nn.Sequential(
            nn.Conv2d(base_ch, out_channels, 7, padding=3),
            nn.Tanh()
        )

        self.pool = nn.AvgPool2d(2)

    def forward(self, x):

        e1 = self.enc1(x)

        e2 = self.enc2(self.pool(e1))

        e3 = self.enc3(self.pool(e2))

        b = self.res_blocks(e3)

        u3 = self.up3(b)

        a3 = self.att3(u3, e2)

        d3 = self.dec3(torch.cat([u3, a3],1))

        u2 = self.up2(d3)

        a2 = self.att2(u2, e1)

        d2 = self.dec2(torch.cat([u2, a2],1))

        return self.final(d2)


# ------------------------
# Device
# ------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ------------------------
# Load Model
# ------------------------

@st.cache_resource
def load_model():

    model = AttUNetGenerator().to(device)

    checkpoint = torch.load("model/G_A2B_final.pth", map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()

    return model


model = load_model()


# ------------------------
# Transform
# ------------------------

transform = transforms.Compose([
    transforms.Resize((512,512)),
    transforms.ToTensor()
])


# ------------------------
# UI
# ------------------------

st.title("Microstructure Crack Prediction")

IMAGE_FOLDER = "test_images"

image_files = sorted(
    [f for f in os.listdir(IMAGE_FOLDER) if f.endswith(".png")],
    key=lambda x: int(os.path.splitext(x)[0])
)

selected_image = st.selectbox(
    "Select Test Image",
    image_files
)

image_path = os.path.join(IMAGE_FOLDER, selected_image)

image = Image.open(image_path).convert("RGB")

col1, col2 = st.columns(2)

with col1:

    st.subheader("Input Microstructure")

    st.image(image)


predict_button = st.button("Run Model")


if predict_button:

    with st.spinner("Running inference..."):

        input_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(input_tensor)

        output_img = (
            output.squeeze()
            .permute(1,2,0)
            .cpu()
            .numpy()
        )

        output_img = (output_img + 1) / 2
        output_img = np.clip(output_img,0,1)

    with col2:

        st.subheader("Predicted Crack")

        st.image(output_img)


if st.button("Reset"):
    st.rerun()
