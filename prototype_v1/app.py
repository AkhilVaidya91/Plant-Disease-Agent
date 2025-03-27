import streamlit as st
import json
import base64
from openai import OpenAI
import google.generativeai as gemini
from PIL import Image
import io

# Initialize clients
openai_client = OpenAI(api_key="")
gemini.configure(api_key="")

# Load configuration files
with open("prompt_template.json") as f:
    PROMPT_TEMPLATE = json.load(f)["prompt"]

with open("plant_diseases.json") as f:
    DISEASE_DATA = json.load(f)

def encode_image(uploaded_file):
    return base64.b64encode(uploaded_file.getvalue()).decode("utf-8")

def call_openai(prompt, base64_image, mime_type):
    return openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime_type};base64,{base64_image}",
                        },
                    },
                ],
            }
        ],
        max_tokens=300,
        temperature=0.1
    )

def call_gemini(prompt, image):
    model = gemini.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content([prompt, image])
    return response

st.set_page_config(page_title="AgriScan: Plant Disease Detection", layout="wide")
st.title("🌱 AgriScan: AI-Powered Crop Protection")

# Sidebar inputs
with st.sidebar:
    st.header("Input Parameters")
    model_choice = st.radio("Select AI Model", 
                          ["OpenAI (GPT-4o)", "Gemini (Flash)"],
                          index=0)
    
    plant_type = st.selectbox("Select Crop Type", 
                            ["rice", "wheat", "cotton"],
                            index=None,
                            placeholder="Choose plant type")
    
    uploaded_image = st.file_uploader("Upload Leaf Image",
                                    type=["jpg", "jpeg", "png"],
                                    accept_multiple_files=False)

# Main analysis section
if uploaded_image and plant_type:
    with st.spinner("🔍 Analyzing leaf health..."):
        try:
            # Prepare common elements
            diseases = DISEASE_DATA[plant_type]
            disease_names = [d["name"] for d in diseases]
            prompt = PROMPT_TEMPLATE.format(
                plant=plant_type,
                diseases_list=", ".join(disease_names)
            )

            # Model-specific processing
            if model_choice == "OpenAI (GPT-4o)":
                if not openai_client.api_key:
                    st.error("OpenAI API key not configured")
                    st.stop()
                
                base64_image = encode_image(uploaded_image)
                file_extension = uploaded_image.name.split(".")[-1].lower()
                mime_type = f"image/{file_extension}" if file_extension in ["jpeg", "jpg", "png"] else "image/jpeg"
                response = call_openai(prompt, base64_image, mime_type)
                analysis_result = response.choices[0].message.content

            elif model_choice == "Gemini (Flash)":
                if not True:
                    st.error("Gemini API key not configured")
                    st.stop()
                
                image = Image.open(io.BytesIO(uploaded_image.getvalue()))
                response = call_gemini(prompt, image)
                analysis_result = response.text

            # Display results
            st.subheader("Diagnosis Report")
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.image(uploaded_image, caption="Uploaded Leaf Image", width=300)
                
            with col2:
                st.markdown(f"**AI Model:** {model_choice}")
                st.markdown(f"**Crop Type:** {plant_type.capitalize()}")
                st.markdown(f"**Detected Diseases:**")
                st.write(analysis_result)

                # Show disease database
                st.markdown("---")
                st.subheader("Common Diseases Database")
                for disease in diseases:
                    with st.expander(f"ℹ️ {disease['name']}"):
                        st.markdown(f"**Symptoms:** {disease['symptoms']}")
                        st.markdown(f"**Management:** {disease['management']}")

        except Exception as e:
            st.error(f"Error processing request: {str(e)}")
elif uploaded_image or plant_type:
    st.warning("⚠️ Please provide both plant type and image for analysis")
else:
    st.info("👈 Please select plant type and upload leaf image to begin analysis")