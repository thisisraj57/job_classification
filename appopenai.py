from langchain import Cohere
import streamlit as st
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import CohereEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from langchain import PromptTemplate
from langchain.chains import LLMChain
import os
import zipfile
from docx import Document
from openai import OpenAI
from langchain.embeddings import OpenAIEmbeddings

client = OpenAI(api_key = "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")

def read_zip_file(zip_file_path):
    # Open the zip file
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        # Get the list of all files in the zip
        file_list = zip_ref.namelist()

        # Read the content of each file
        for file_name in file_list:
            # Check if the file is a .docx file
            if file_name.lower().endswith('.docx'):
                # Read the content of the .docx file
                with zip_ref.open(file_name) as file:
                    document = Document(file)
                    text = ""
                    for paragraph in document.paragraphs:
                        text += paragraph.text + '\n'

                    # Print or process the text as needed
                    # print(f"Content of {file_name}:\n{text}\n{'='*50}\n")
                    yield text  # Yield the text content for further processing

# Replace 'your_zip_file.zip' with the actual path to your zip file
zip_file_path = 'D:\Prompt Engineering\python code vs\chatbots\JobData.zip'

# Call the function to read the zip file
docx_contents = read_zip_file(zip_file_path)

for result in docx_contents:
    print(result)


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(chunks):
    # cohere_instance = Cohere(model="command-light", cohere_api_key="nRH8H8vfbHZMR44ODKgXigGbHtsd27YIQEVqPNf5")
    embeddings = OpenAIEmbeddings()
    # embeddings = CohereEmbeddings()
    #embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(chunks, embeddings)
    return vectorstore


def get_conversation_chain(vectorstore):
    llm = client.OpenAI()
    # llm = Cohere(model="command-light", cohere_api_key="nRH8H8vfbHZMR44ODKgXigGbHtsd27YIQEVqPNf5")
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def get_conversation_chain(db):
    # Define your prompt or instructions
    template = """{Question}You are a job advisor chatbot designed to assist users with queries related to careers and employment. 
    Your responses should be derived exclusively from the provided dataset. 
    If a user's question is not covered by the dataset, respond with "None."
    Instructions:
    - Respond to user queries in a natural and conversational manner.
    - Only provide answers based on the information available in the provided dataset. If the question is not present in the dataset, reply with "None."
    - Keep your responses clear and simple, avoiding technical jargon whenever possible.
    - Be polite and professional in all interactions with users.
    - If a user's question is ambiguous or not clear, ask for clarification before attempting to provide an answer.
    - If a user's question falls outside the scope of the provided dataset, gently inform them that the chatbot is limited to the dataset and encourage them to rephrase their question if possible."""
    prompt = PromptTemplate(template= template, input_variables= ["Question"])
    llm = client.chat.completions.create(
        model="gpt-4-1106-preview"
        )
    llm_chain= LLMChain(prompt=prompt, llm=llm)
    print(llm_chain.run("hello"))
    memory = ConversationBufferMemory(
        llm=llm,
        output_key='answer',
        memory_key='chat_history', return_messages=True)
    retriever=db.as_retriever(search_type = "mmr")
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever= retriever,
        memory=memory
    )
    return conversation_chain


def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)


def main():
    load_dotenv()
    st.set_page_config(page_title="Graduate Job Classification Chatbot", page_icon=":chatbot:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Graduate Job Classification Chatbot :chatbot:")
    # Get raw text from all doc files
    raw_text = "\n".join(docx_contents)
    
    # Chunk the raw text
    text_chunks = get_text_chunks(raw_text)

    # Create vector store
    vectorstore = get_vectorstore(text_chunks)

    # Create conversation chain
    st.session_state.conversation = get_conversation_chain(vectorstore)

    user_question = st.text_input("How Can I Help You !!!!")
    if st.button("Submit"):
        if user_question:
            handle_userinput(user_question)



if __name__ == '__main__':
    main()