
from langchain.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain import PromptTemplate
from langchain.chains import LLMChain
from langchain.vectorstores import FAISS


API_KEY = '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'
VIDEO = "https://youtu.be/BufUW7h9TB8?si=l-_zc4rMuiFX1WSq"

embedings=OpenAIEmbeddings(api_key=API_KEY)

def video_2_db(url:str)->FAISS:
  loader = YoutubeLoader.from_youtube_url(url)
  transcript =loader.load()

  text_splitter = RecursiveCharacterTextSplitter(
      chunk_size=1000,
      chunk_overlap=100
  )
  docs=text_splitter.split_documents(transcript)

  db=FAISS.from_documents(docs,embedings)
  return db

def get_response_from_query(db, query,k=4):
  docs = db.similarity_search(query, k=k)
  docs_page_content = " ".join([d.page_content for d in docs])

  llm = OpenAI(api_key=API_KEY)

  prompt = PromptTemplate(
      input_variables=["question", "docs"],
      template="""
        You are my helpful YouTube assistant that can answer questions about videos based on the video's transcript.

        Answier the following question: {question}
        By searching in the following video transcript: {docs}

        Only use the factual information from the transcript to answer the question.

        If you feel like you don' have enough information to answer the question, say "I need more information to answer the question".

        Your answers should be detailed.
      """
  )

  chain = LLMChain(llm=llm, prompt=prompt)

  response = chain.run(question=query, docs=docs_page_content)
  response = response.replace("\n","")
  return response

import streamlit as st
import textwrap

st.title("Youtube Asistant")

with st.sidebar:
  with st.form(key='my_form'):
    youtube_url = st.sidebar.text_area(
        label="What is the Youtube video URL?",
        max_chars=50
    )
    query = st.sidebar.text_area(
        label="Ask me about the video",
        max_chars=50,
        key='query'
    )
    submit_button = st.form_submit_button(label="Submit")

if query and youtube_url:
  db = video_2_db(youtube_url)
  response = get_response_from_query(db,query)
  st.subheader("Answer: ")
  st.text(textwrap.fill(response, width=80))

