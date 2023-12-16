from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings

from langchain_core.pydantic_v1 import BaseModel
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.schema import ChatMessage
import streamlit as st
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationSummaryBufferMemory
from langchain.llms import OpenAI
from langchain import PromptTemplate
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
txt_path = os.path.join(current_dir, 'r', 'output.txt')

# txt_path = "output1.txt"
OPENAI_API_KEY = st.secrets['OPENAI_API_KEY']
# Load text
loader = TextLoader(txt_path)
documents = loader.load()
# Split text
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
all_splits = text_splitter.split_documents(documents)
# Add to vectorDB
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
vectorstore = FAISS.from_documents(all_splits, embeddings)


class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)


st.set_page_config(
    page_title="WLKATA Chatbot"  # Provide the path to your favicon image file
)

with st.sidebar:
    st.markdown("**Contact Information**")
    st.markdown("""
    By Email
For customer support, quotation, distributing inquiry and business cooperation, please email to:
hello@wristline.com

By Phone
Talking Directly With Our Office, Please Contact One Of The Following Numbers:
Phone:
+1 201 682 975
Phone/Wechat:
+86 15110072290

By Whatsapp
Talk to us directly on Whatsapp:
+86 15110072290
or +1 201 682 9753
    """)

st.title("WLKATA Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = [ChatMessage(role='assistant', content="""Hello! 

I am the WLKATA Assistant, dedicated to providing technical support and services for WLKATA Mirobot robotic arm users.

""")]

for msg in st.session_state.messages:
    st.chat_message(msg.role).write(msg.content)

memory = ConversationSummaryBufferMemory(llm=OpenAI(temperature=0, openai_api_key=OPENAI_API_KEY),
                                         max_token_limit=150,
                                         memory_key='chat_history',
                                         return_messages=True, output_key='answer')

if 'buffer_memory' not in st.session_state:
    st.session_state.buffer_memory = memory

prompt_template = """ You are the WLKATA Assistant, dedicated to providing technical support and services for WLKATA Mirobot robotic arm users.

Based on the following pieces of context, provide a thorough solution to the user's question. At least give 10 sentences of explanation, including 5-7 tips.

Answer in user's original language used in Question.

{context}

Question: {question}
Helpful Answer:"""
QA_PROMPT_revised = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)


if prompt := st.chat_input("Ask anything about learning sciences research!"):
    st.session_state.messages.append(ChatMessage(role="user", content=prompt))
    st.chat_message("user").write(prompt)

    with st.chat_message("assistant"):
        stream_handler = StreamHandler(st.empty())
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 6})
        qa = ConversationalRetrievalChain.from_llm(
            llm=ChatOpenAI(model="gpt-4", temperature=0, openai_api_key=OPENAI_API_KEY, streaming=True,
                           callbacks=[stream_handler]),
            retriever=retriever, chain_type="stuff",
            combine_docs_chain_kwargs={'prompt': QA_PROMPT_revised}, memory=memory,
            verbose=True, return_source_documents=True)
        with st.spinner("searching through learning sciences research papers and preparing citations..."):
            res = qa({"question": st.session_state.messages[-1].content})
        # st.write(res)

    st.session_state.messages.append(ChatMessage(role="assistant", content=res['answer']))
