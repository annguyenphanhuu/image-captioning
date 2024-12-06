import streamlit as st
from PIL import Image
from models.blip_model import load_blip_model, generate_caption_blip
from models.vit_gpt2_model import load_vit_gpt2_model, generate_caption_vit_gpt2

# Load models
blip_processor, blip_model = load_blip_model()
vit_gpt2_model, vit_gpt2_extractor, vit_gpt2_tokenizer, device = load_vit_gpt2_model()

# Title of the App
st.title("Image Captioning with Multiple Models")
st.write("Upload an image, choose a model, and the app will generate a descriptive caption for it.")

# Model selection
model_choice = st.selectbox("Choose a model for captioning", ("BLIP", "ViT-GPT2"))

# File uploader for the image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image")

    # Generate caption button
    if st.button("Generate Caption"):
        if model_choice == "BLIP":
            caption = generate_caption_blip(image, blip_processor, blip_model)
            st.write("**Generated Caption (BLIP):**")
            st.success(caption)
        elif model_choice == "ViT-GPT2":
            caption = generate_caption_vit_gpt2(image, vit_gpt2_model, vit_gpt2_extractor, vit_gpt2_tokenizer, device)
            st.write("**Generated Caption (ViT-GPT2):**")
            st.success(caption)
