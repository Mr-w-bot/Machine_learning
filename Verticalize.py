###本程序用于向量化储存实验###
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200,length_function=len,is_separator_regex=False,)
splits = text_splitter.split_text(pdf_text)                    #split_text才是分割string的 载入文件（load）则是用split_document
vectorstore = Chroma.from_texts(splits, OpenAIEmbeddings(), persist_directory="./chroma_db")    #分割的text用from_texts存储为向量 相对应的文件是用from_document
