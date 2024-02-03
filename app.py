import streamlit as st
import os
from streamlit_chat import message


# from datasets import load_dataset

# dataset = load_dataset("wikipedia", "20220301.en", split="train[240000:250000]")


# wikidata = []

# for record in dataset:
#     wikidata.append(record["text"])

# wikidata = list(set(wikidata))
# # print("\n".join(wikidata[:5]))
# # print(len(wikidata))

from sentence_transformers import SentenceTransformer
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

if device != 'cuda':
    st.text(f"you are using {device}. This is much slower than using "
    "a CUDA-enabled GPU. If on colab you can chnage this by "
    "clicking Runtime > change runtime type > GPU.")

model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
st.divider()

# Creating a Index(Pinecone Vector Database)
def chat_actions():
    st.session_state["chat_history"].append(
        {"role": "user", "content": st.session_state["chat_input"]},
    )

    response = model.generate_content(st.session_state["chat_input"])
    st.session_state["chat_history"].append(
        {
            "role": "assistant",
            "content": response.text,
        },  # This can be replaced with your chat response logic
    )


if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []


st.chat_input("Enter your message", on_submit=chat_actions, key="chat_input")

for i in st.session_state["chat_history"]:
    with st.chat_message(name=i["role"]):
        st.write(i["content"])
