from transformers import BlipProcessor, BlipForConditionalGeneration
import streamlit as st

@st.cache_resource
def load_blip_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")
    return processor, model

def generate_caption_blip(image, processor, model):
    text_prompt = "a photography of"
    inputs = processor(image, text_prompt, return_tensors="pt")
    output = model.generate(**inputs, max_new_tokens=50)
    caption = processor.decode(output[0], skip_special_tokens=True)
    return caption
