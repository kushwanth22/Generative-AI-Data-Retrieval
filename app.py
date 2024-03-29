import streamlit as st
import os
from streamlit_chat import message
import numpy as np
import pandas as pd
from io import StringIO
import PyPDF2
from tqdm.auto import tqdm
import math
from transformers import pipeline
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

def promt_engineer(text):
    prompt_template = """
    write a concise summary of the following text delimited by triple backquotes.
    return your response in bullet points which convers the key points of the text.

    ```{text}```

    BULLET POINT SUMMARY:
    """
    # Load the summarization pipeline with the specified model
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

    # Generate the prompt
    prompt = prompt_template.format(text=text)

    # Generate the summary
    summary = summarizer(prompt, max_length=1024, min_length=50)[0]["summary_text"]
    
    with st.sidebar:
        st.divider()
        st.markdown("*:red[Text Summary Generation]* from above Top 5 **:green[similarity search results]**.")
        st.write(summary)
        st.divider()

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
    result = index.query(query_vector, top_k=5, include_metadata=True)  # result is a list of tuples

    # Create a list of lists
    data = []
    consolidated_text = ""
    i = 0
    for res in result['matches']:
        i = i + 1
        data.append([f"{i}⭐", res['score'], res['metadata']['text']])
        consolidated_text += res['metadata']['text']

    # Create a DataFrame from the list of lists
    resdf = pd.DataFrame(data, columns=['TopRank', 'Score', 'Text'])

    with st.sidebar:
        st.markdown("*:red[semantic search results]* with **:green[Retrieval Augmented Generation]** ***(RAG)***.")
        st.dataframe(resdf)
        bytesize = consolidated_text.encode("utf-8")
        p = math.pow(1024, 2)
        mbsize = round(len(bytesize) / p, 2)
        st.write(f"Text lenth of {len(consolidated_text)} characters with {mbsize}MB size")
        promt_engineer(consolidated_text[:1024])

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
    return concatenates_text

def split_into_chunks(text, chunk_size):

    chunks = []
    for i in range(0, len(text), chunk_size):
        chunks.append(text[i:i + chunk_size])

    return chunks

def create_embeddings():
    # Get the uploaded file
    inputtext = ""
    with st.sidebar:
        uploaded_files = st.session_state["uploaded_files"]
        for uploaded_file in uploaded_files:
            # Read the contents of the file
            reader = PyPDF2.PdfReader(uploaded_file)
            pages = reader.pages
            print_out(pages)
            inputtext = combine_text(pages)

    # connect to pinecone index
    pinecone = connect_pinecone()
    index = get_pinecone_semantic_index(pinecone)

    # The maximum metadata size per vector is 40KB ~ 40000Bytes ~ each text character is 1 to 2 bytes. so rougly given chunk size of 10000 to 40000
    chunk_size = 10000
    batch_size = 2
    chunks = split_into_chunks(inputtext, chunk_size)

    for i in tqdm(range(0, len(chunks), batch_size)):
        # find end of batch
        end = min(i + batch_size, len(chunks))
        # create ids batch
        ids = [str(i) for i in range(i, end)]
        # create metadata batch
        metadata = [{"text": text} for text in chunks[i:end]]
        # create embeddings
        xc = model.encode(chunks[i:end])
        # create records list for upsert
        records = zip(ids, xc, metadata)
        # upsert records
        index.upsert(vectors=records)

    with st.sidebar:
        st.write("created vector embeddings!")
        # check no of records in the index
        st.write(f"{index.describe_index_stats()}")


    # Display the contents of the file
    # st.write(file_contents)

with st.sidebar:
    st.markdown("""
    ***:red[Follow this steps]***
    - upload pdf file to create embeddings using model on your own docs
    - wait see success message on embeddings creation 
    - It Takes couple of mins after upload the pdf
    - Now Chat with your documents with help of this RAG system 
    - It Generate Promted reponses on the upload pdf
    - Provides summarized results and QA's using GPT models
    - This system already trained on some wikipedia datasets too
    """)
    uploaded_files = st.file_uploader('Choose your .pdf file', type="pdf", accept_multiple_files=True, key="uploaded_files", on_change=create_embeddings)
    # for uploaded_file in uploaded_files:
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

        # reader = PyPDF2.PdfReader(uploaded_file)
        # pages = reader.pages
        # print_out(pages)
        # combine_text(pages)
        # promt_engineer(text)