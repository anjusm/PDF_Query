## Import necessary libraries
import streamlit as st
from PyPDF2 import PdfReader
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
import os
os.environ["OPENAI_API_KEY"] = st.secrets["api_key"]

# Vertical sidebar contents
with st.sidebar: 
    st.title(":tulip:**PDF Query Chat App**:tulip::books: :red[**'48 Laws of Power'**]:books:  \n  :sparkles::sparkles::sparkles::sparkles::sparkles::sparkles::sparkles::sparkles::sparkles::sparkles::sparkles::sparkles:")
    st.markdown('''
    ## About
    This app is an LLM-powered chatbot for the book :violet['48 Laws of Power'] built using:
    - [Streamlit](https://streamlit.io/)
    - [LangChain](https://python.langchain.com/)
    - [OpenAI](https://platform.openai.com/docs/models) LLM model
     ''')
    add_vertical_space(0)
    st.write('❤️🤗   Made by Anju S Mohan   🤗❤️')

def main():
    st.header("Chat with '48 Laws of Power' PDF 💬")
    pdf_reader = PdfReader('48lawsofpower.pdf')
    # Accept user questions/query
    query = st.text_input("Please, ask a question about this book:- :question::question::question::question::question:") 
    # Extract text from pages of PDF    
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    # Split Text
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
        )
    chunks = text_splitter.split_text(text=text)
    # # embeddings
    embeddings = OpenAIEmbeddings()
    VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
    # Generate query response and display
    if query:
        docs = VectorStore.similarity_search(query=query, k=3)
        llm = OpenAI(temperature=0.9)
        chain = load_qa_chain(llm=llm, chain_type="stuff")
        with get_openai_callback() as cb:
            response = chain.run(input_documents=docs, question=query)
            print(cb)
        st.write(response)
        
# Main function
if __name__ == '__main__':
    main()
