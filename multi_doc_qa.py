import streamlit as st
import pdfplumber
import io
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

st.set_page_config(page_title="Multi-Document Q&A", layout="wide")
st.title("📚 Multi-Document AI Assistant")
st.write("Upload multiple PDFs and ask questions based strictly on their content.")

# 1. Initialize Session State for the Database
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

# 2. Sidebar: File Upload & Processing
with st.sidebar:
    st.header("1. Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload PDF files", 
        type=["pdf"], 
        accept_multiple_files=True
    )
    
    if st.button("Process Documents"):
        if uploaded_files:
            with st.spinner("Reading and indexing documents..."):
                all_text = ""
                
                # A. Read text from all uploaded PDFs
                for uploaded_file in uploaded_files:
                    pdf_bytes = uploaded_file.read()
                    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
                        for page in pdf.pages:
                            page_text = page.extract_text()
                            if page_text:
                                all_text += page_text + "\n"
                
                # B. Split text into manageable chunks
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000, 
                    chunk_overlap=200
                )
                chunks = text_splitter.split_text(all_text)
                
                # C. Convert chunks into vector embeddings and store in ChromaDB
                embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
                
                # Create a temporary in-memory Chroma database for this session
                vector_store = Chroma.from_texts(
                    texts=chunks, 
                    embedding=embeddings
                )
                
                st.session_state.vector_store = vector_store
                st.success(f"Processed {len(uploaded_files)} files into {len(chunks)} searchable chunks!")
        else:
            st.warning("Please upload at least one PDF first.")

# 3. Main Interface: Chat / Q&A
st.header("2. Ask Questions")
user_query = st.text_input("What would you like to know about the uploaded documents?")

if st.button("Get Answer"):
    if not st.session_state.vector_store:
        st.error("Please upload and process documents first!")
    elif not user_query:
        st.warning("Please enter a question.")
    else:
        with st.spinner("Searching documents and generating answer..."):
            # A. Set up the AI model (Requires Ollama running locally)
            llm = Ollama(model="llama3") # You can change "llama3" to whichever model you downloaded
            
            # B. Create the prompt instructing the AI to ONLY use the provided context
            system_prompt = (
                "You are an assistant for question-answering tasks. "
                "Use the following pieces of retrieved context to answer the question. "
                "If you don't know the answer based on the context, just say that you don't know. "
                "Context: {context}"
            )
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", "{input}"),
            ])
            
            # C. Build the RAG Chain
            retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": 3})
            question_answer_chain = create_stuff_documents_chain(llm, prompt)
            rag_chain = create_retrieval_chain(retriever, question_answer_chain)
            
            # D. Get the Answer
            response = rag_chain.invoke({"input": user_query})
            
            # Display Results
            st.markdown("### Answer:")
            st.write(response["answer"])
            
            # Show the exact source paragraphs it used
            with st.expander("View Source Evidence"):
                for i, doc in enumerate(response["context"]):
                    st.write(f"**Chunk {i+1}:** {doc.page_content}")