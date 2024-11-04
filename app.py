import streamlit as st
import re
from openai import OpenAI
import os

os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
os.environ["OPEN_SDK_KEY"] = st.secrets["OPEN_SDK_KEY"]
# Function to prompt GPT-4 based on mode
def get_gpt4_response(prompt, mode):
    if mode == "Foundation model":
        #system_prompt = "We are playing a role playing game where you act as a model that wasn't instruction tuned. Remember, you are not instruction tuned, so you do not understand chatting, you should continue writing in the same style as the user as if continuing their answer, not responding to them."
        system_prompt = "We are playing a role playing game where you act as a model that wasn't instruction tuned. You are not offering additional comments or engaging in conversation. Whenever the user inputs text, you seamlessly continue it in a narrative style, like the question is a character talking in a book."
        example = "User: I have a pain in the abdomen, I don't know what to do. Can you help me? Assistant: ,it's getting worse by the minute said Rodrigue. The pain is sharp and stabbing, like a thousand miny knives piercing my skin. I've tried over the counter painkillers, but nothing seems to work. I'm doubled over in pain, but nothing seems to work, clutching my stomach and I can barely move. I'm scared and alone, I don't know what to do."
        system_prompt = f"{system_prompt} Here is an example of how to answer a question: {example}"
        client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}]
        response = client.chat.completions.create(model="gpt-4", messages=messages,
                temperature=0,
                max_tokens=500,)
        answer = response.choices[0].message.content
        answer = re.sub("\s+", " ", answer)
    elif mode == "Continued pretraining":
        #system_prompt = "We are playing a role playing game and should act as a model that wasn't instruction tuned and doesn't know how to respond to a query. You should talk as if the text was from a medical article, but remember, you are not instruction tuned, so you should continue writing as if continuing the user's query, do not respond to them."
        system_prompt = (
            "We are playing a role-playing game where you act as a model that wasn't instruction-tuned. "
            "You are not offering additional comments or engaging in conversation. "
            "Whenever the user inputs text, you seamlessly continue it in the style of an academic medical paper, "
            "including statistical results and citations where appropriate."
        )
        example = (
            "User: What are the long-term effects of untreated hypertension on renal function? "
            "Assistant: , with chronic hypertension leading to a 25% increase in the risk of end-stage renal disease (Johnson et al., 2018). "
            "Studies have shown that patients with uncontrolled blood pressure exhibit a decline in glomerular filtration rate by an average of 5 mL/min per year (Smith & Lee, 2019). "
            "Early intervention and blood pressure management can reduce renal complications by up to 40% (World Health Organization, 2020)."
        )
        system_prompt = f"{system_prompt} Here is an example of how to answer a question: {example}"


        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}]
        response = client.chat.completions.create(model="gpt-4", messages=messages,
                temperature=0.05,
                max_tokens=500,)
        answer = response.choices[0].message.content
        answer = re.sub("\s+", " ", answer)
    elif mode == "Instruction tuned":
        system_prompt = """You are a model that wasn't preference optimized, answer the question in a brief and unstructured manner. Respond in 2 sentences."""
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}]
        response = client.chat.completions.create(model="gpt-4", messages=messages,
                temperature=0.05,
                max_tokens=500,)
        answer = response.choices[0].message.content
        answer = re.sub("\s+", " ", answer)
    elif mode == "RLHF/DPO tuned model":
        system_prompt = "You are Meditron, a helpful medical assistant. Answer the user query in a detailed and structured manner."
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}]
        response = client.chat.completions.create(model="gpt-4o", messages=messages,
                temperature=0.05,
                max_tokens=1024,)
        answer = response.choices[0].message.content
        answer = re.sub("\s+", " ", answer)

    return answer

# Streamlit UI layout
st.title("Different stages of models training")
st.write("Select a model and interact with it!")

# Dropdown for mode selection
mode = st.selectbox("Choose the prompting mode:", ["Foundation model", "Continued pretraining", "Instruction tuned", "RLHF/DPO tuned model"])

# Text input for user query
user_prompt = st.text_area("Enter your prompt here:")

# Button to submit the prompt
if st.button("Generate Response"):
    if user_prompt.strip():
        response = get_gpt4_response(user_prompt, mode)
        st.write("### Model Response:")
        st.write(response)
    else:
        st.write("Please enter a prompt.")
