import zipfile
from docx import Document
import pandas as pd
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
import streamlit as st

# Set OpenAI key
client = OpenAI(
    api_key = "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",
)

# Function to Read Zip file in read mode and create a dataframe.
def create_dataframe():
  # Initialize an empty dataframe to store the filenames and their contents
  job_descriptions_df = pd.DataFrame(columns=['Filename', 'Content'])

  # Open ZIP file and read the contents of all docx.
  with zipfile.ZipFile('JobData.zip', 'r') as zip_file:

    # Get the list of all files in the zip file
    file_list = zip_file.namelist()

    # Read the contents of each file
    for fname in file_list:

      # Check if the file is a docx file
      if fname.endswith(".docx"):
        with zip_file.open(fname) as file:
          document = Document(file)
          full_text = []
          for par in document.paragraphs:
            full_text.append(par.text)
          content = '\n'.join(full_text)
          job_descriptions_df = job_descriptions_df._append({'Filename': fname, 'Content': content}, ignore_index=True)
  return job_descriptions_df

df = create_dataframe()

print(df['Content'])

# Save the dataframe to a CSV file
df.to_csv('job_descriptions_dataset.csv', index=False)

# Split the dataframe into chunks of 1000 rows each for processing
splitter = RecursiveCharacterTextSplitter(chunk_size=1000)

# Apply splitter to content column
df['content_split'] = df['Content'].apply(splitter.split_text)

model = SentenceTransformer('all-MiniLM-L6-v2')
df['content_embeddings'] = df['Content'].apply(lambda x:model.encode(x))
print(df['content_split'])

embeddings = OpenAIEmbeddings(openai_api_key="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")
#vec_store = FAISS.from_documents(df, embeddings)