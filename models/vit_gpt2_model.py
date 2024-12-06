from transformers import VisionEncoderDecoderModel, ViTImageProcessor, GPT2Tokenizer
import torch
import streamlit as st

@st.cache_resource
def load_vit_gpt2_model():
    model_repo = "PhanHuuAnNguyen/vit_gpt2_30k"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VisionEncoderDecoderModel.from_pretrained(model_repo).to(device)
    feature_extractor = ViTImageProcessor.from_pretrained(model_repo)
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    return model, feature_extractor, tokenizer, device

def generate_caption_vit_gpt2(image, model, extractor, tokenizer, device):
    inputs = extractor(images=image, return_tensors="pt")
    pixel_values = inputs.pixel_values.to(device)
    generated_ids = model.generate(pixel_values)
    generated_caption = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    sentences = generated_caption.split('.')
    limited_caption = '. '.join(sentences[:2]).strip() + ('.' if len(sentences) > 2 else '')
    return limited_caption
