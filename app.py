import streamlit as st
import os
from streamlit_chat import message
import numpy as np
import pandas as pd
from io import StringIO
import PyPDF2
from tqdm import tqdm
import math
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
    st.markdown(f"you are using {device}. This is much slower than using "
    "a CUDA-enabled GPU. If on colab you can change this by "
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
    # st.code(pinecone)
    # st.divider()
    # st.text(pinecone.list_indexes().names())
    # st.divider()
    # st.text(f"Succesfully connected to the pinecone")
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
    # st.text(f"Succesfully connected to the pinecone index")
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

    # Create a list of lists
    data = []
    i = 0
    for res in result['matches']:
        i = i + 1
        data.append([f"{i}⭐", res['score'], res['metadata']['text']])

    # Create a DataFrame from the list of lists
    resdf = pd.DataFrame(data, columns=['TopRank', 'Score', 'Text'])

    with st.sidebar:
        st.markdown("*:red[semantic search results]* with **:green[Retrieval Augmented Generation]** ***(RAG)***.")
        st.dataframe(resdf)

    for res in result['matches']:
        st.session_state["chat_history"].append(
            {
                "role": "assistant",
                "content": f"{res['metadata']['text']}",
            },  # This can be replaced with your chat response logic
        )
        break;

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []


st.chat_input("show me the contents of ML paper published on xxx with article no. xx?", on_submit=chat_actions, key="chat_input")

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

def create_embeddings():
    # Get the uploaded file
    uploaded_file = st.session_state["uploaded_file"]

    # Read the contents of the file
    file_contents = uploaded_file.read()

    st.write("created_embeddings")

    # Display the contents of the file
    st.write(file_contents)


def print_out(pages):
    for i in range(len(pages)):
        text = pages[i].extract_text().strip()
        st.write(f"Page {i} : {text}")

def combine_text(pages):
    concatenates_text = ""
    for page in tqdm(pages):
        text = page.extract_text().strip()
        concatenates_text += text
    bytesize = concatenates_text.encode("utf-8")
    p = math.pow(1024, 2)
    mbsize = round(len(bytesize) / p, 2)
    st.write(f"There are {len(concatenates_text)} characters in the pdf with {mbsize}MB size")

# def promt_engineer(text):
#     promt_template = """
#     write a concise summary of the following text delimited by triple backquotes.
#     return your response in bullet points which convers the key points of the text.

#     ```{text}```

#     BULLET POINT SUMMARY:
#     """


with st.sidebar:
    st.markdown("""
    ***Follow this steps***
    - upload pdf file to train the model on your own docs
    - wait see success message on train completion 
    - Takes couple of mins after upload the pdf
    - Now Chat with model to get the summarized info or Generative reponse
    """)
    uploaded_files = st.file_uploader('Choose your .pdf file', type="pdf", accept_multiple_files=True, key="uploaded_file", on_change=create_embeddings)
    for uploaded_file in uploaded_files:
        # To read file as bytes:
        # bytes_data = uploaded_file.getvalue()
        # st.write(bytes_data)

        # To convert to a string based IO:
        # stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
        # st.write(stringio)

        # To read file as string:
        # string_data = stringio.read()
        # st.write(string_data)

        # Can be used wherever a "file-like" object is accepted:
        # dataframe = pd.read_csv(uploaded_file)
        # st.write(dataframe)
        reader = PyPDF2.PdfReader(uploaded_file)
        pages = reader.pages
        print_out(pages)
        combine_text(pages)
        # promt_engineer(text)
