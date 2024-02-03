import streamlit as st
import os
from streamlit_chat import message
import numpy as np
import pandas as pd
# import json

# st.config(PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION="python")

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
import os
# import pinecone

from pinecone.grpc import PineconeGRPC


PINECONE_API_KEY=os.getenv("PINECONE_API_KEY")
PINECONE_ENV=os.getenv("PINECONE_ENV")
PINECONE_ENVIRONMENT=os.getenv("PINECONE_ENVIRONMENT")

# pc = PineconeGRPC( api_key=os.environ.get("PINECONE_API_KEY") ) # Now do stuff if 'my_index' not in pc.list_indexes().names(): pc.create_index( name='my_index', dimension=1536, metric='euclidean', spec=ServerlessSpec( cloud='aws', region='us-west-2' ) )

def connect_pinecone():
    pinecone = PineconeGRPC(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
    st.code(pinecone)
    st.divider()
    st.text(pinecone.list_indexes().names())
    st.divider()
    st.text(f"Succesfully connected to the pinecone")
    return pinecone

def get_pinecone_semantic_index(pinecone):
    index_name = "sematic-search"

    # only create if it deosnot exists
    if index_name not in pinecone.list_indexes().names():
        pinecone.create_index(
            name=index_name,
            description="Semantic search",
            dimension=model.get_sentence_embedding_dimension(),
            metric="cosine",
            spec=ServerlessSpec( cloud='gcp', region='us-central1' )
        )
    # now connect to index
    index = pinecone.Index(index_name)
    st.text(f"Succesfully connected to the pinecone index")
    return index

def chat_actions():
    
    pinecone = connect_pinecone()
    index = get_pinecone_semantic_index(pinecone)

    st.session_state["chat_history"].append(
        {"role": "user", "content": st.session_state["chat_input"]},
    )

    query_embedding = model.encode(st.session_state["chat_input"])
    # create the query vector
    query_vector = query_embedding.tolist()
    # now query vector database
    result = index.query(query_vector, top_k=5, include_metadata=True)  # xc is a list of tuples
    with st.sidebar:
        st.json(result)

    for res in result['matches']:
        st.session_state["chat_history"].append(
            {
                "role": "assistant",
                "content": f"{round(res['score'],2)}: {res['metadata']['text']}",
            },  # This can be replaced with your chat response logic
        )


if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []


st.chat_input("Enter your message", on_submit=chat_actions, key="chat_input")

for i in st.session_state["chat_history"]:
    with st.chat_message(name=i["role"]):
        st.write(i["content"])

### Creating a Index(Pinecone Vector Database)
# %%writefile .env
# PINECONE_API_KEY=os.getenv("PINECONE_API_KEY")
# PINECONE_ENV=os.getenv("PINECONE_ENV")
# PINECONE_ENVIRONMENT=os.getenv("PINECONE_ENVIRONMENT")

# import os
# import pinecone

# from pinecone import Index, GRPCIndex
# pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
# st.text(pinecone)


