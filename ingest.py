from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import warnings
warnings.filterwarnings("ignore")
from langchain.document_loaders import CSVLoader
from langchain.embeddings import HuggingFaceEmbeddings
import faiss
 
loader = CSVLoader(
    file_path="Data/walmart-products.csv",
    encoding="utf-8"   # 🔥 FIX
)
 
# Load documents
documents = loader.load()
 
recursive_splitter = RecursiveCharacterTextSplitter(chunk_size = 300, chunk_overlap = 50,
                                                    separators=["\n\n", "\n", " ", "", ".",",", ";"])
recursive_tokens = recursive_splitter.split_documents(documents)
hf_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
 
 
faiss_store=FAISS.from_documents(documents=recursive_tokens,embedding=hf_embeddings)
faiss_store.save_local("faiss_index")
 
print("faiss_index created successfully")