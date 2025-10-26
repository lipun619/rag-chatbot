import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_classic.retrievers.document_compressors.chain_extract import LLMChainExtractor
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

# ---------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------
st.title('🤖 CONTEXT BASED RAG CHATBOT')

# Sidebar for API key and file upload
openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")
st.sidebar.title('Operations')
uploaded_file = st.sidebar.file_uploader('📂 Choose a PDF File', type=['pdf'])

# Initialize LLM with API key
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.2,
    openai_api_key=openai_api_key
)

# Initialize session state variables
if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []
if 'retriever' not in st.session_state:
    st.session_state['retriever'] = None

# ---------------------------------------------------------
# PDF Processing when uploaded
# ---------------------------------------------------------
if uploaded_file is not None and st.session_state['retriever'] is None:
    with st.spinner('Processing PDF and building knowledge base... ⏳'):
        # Save temporarily
        temp_path = f"./temp_{uploaded_file.name}"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getvalue())

        # Set OpenAI API key
        os.environ["OPENAI_API_KEY"] = openai_api_key
        os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

        # Load and chunk document
        loader = PyPDFLoader(temp_path)
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunked_docs = splitter.split_documents(docs)

        # Create embeddings & FAISS vector store
        embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
        vectorstore = FAISS.from_documents(chunked_docs, embedding_model)

        # Create base retriever and compression retriever
        base_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
        compressor = LLMChainExtractor.from_llm(llm)
        compression_retriever = ContextualCompressionRetriever(
            base_retriever=base_retriever,
            base_compressor=compressor
        )

        # Save retriever in session
        st.session_state['retriever'] = compression_retriever

        # Cleanup temp file
        os.remove(temp_path)

    st.success("✅ PDF processed successfully! You can now chat with it.")

# ---------------------------------------------------------
# Chat Interface
# ---------------------------------------------------------
# Display message history
for message in st.session_state['message_history']:
    with st.chat_message(message['role']):
        st.markdown(message['content'])

# Disable chat until file processed
if st.session_state['retriever'] is None:
    st.info("📄 Please upload a PDF to start chatting.")
    user_input = st.chat_input("Type your question here...", disabled=True)
else:
    user_input = st.chat_input("Type your question here...")

# ---------------------------------------------------------
# Chat Logic
# ---------------------------------------------------------
if user_input:
    # Add user message to history first
    st.session_state['message_history'].append({'role': 'user', 'content': user_input})
    
    # Show user message
    with st.chat_message('user'):
        st.markdown(user_input)

    with st.chat_message('assistant'):
        with st.spinner("Thinking... 💭"):
            try:
                retriever = st.session_state['retriever']

                prompt = PromptTemplate(
                    template="""You are a helpful assistant. Use ONLY the context below to answer the question.
                                If the answer is not found in the context, say "I don't know."
          
                    {context_text}
                    Question: {user_input}
                    """,
                    input_variables=['context_text', 'user_input']
                )

                # Function to retrieve and format docs
                def get_context(user_input):
                    retrived_docs = retriever.invoke(user_input)
                    if not retrived_docs:
                        return "No relevant information found in the document."
                    return "\n\n".join([doc.page_content for doc in retrived_docs])
                
                parallel_chain = RunnableParallel({
                    'context_text': RunnableLambda(get_context),
                    'user_input': RunnablePassthrough()
                })

                parser = StrOutputParser()

                main_chain = parallel_chain | prompt | llm | parser

                response = main_chain.invoke(user_input)

            except Exception as e:
                response = f"⚠️ Error: {str(e)}"

        st.markdown(response)
    
    # Add assistant response to history
    st.session_state['message_history'].append({'role': 'assistant', 'content': response})
    
    # Rerun to refresh the display
    st.rerun()