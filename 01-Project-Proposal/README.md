For this assignment, our proposal needs to outline the key details of the HCI project we plan to build, including:

- what kind of application we will build,
- where it will run (for example, Hugging Face or Google Colab),
- how we will manage versions,
- which tools and libraries we will use, and
- possible AI models to explore.


## Possible solutions:

### 1. Lightweight LLM Chatbot on Hugging Face Spaces
We will build a simple text-based chatbot that responds to user prompts and demonstrates basic conversational AI. The application will run on Hugging Face Spaces using free CPU resources. Version control will be handled through Hugging Face's built-in git repository system. The project will use Python along with libraries such as transformers, gradio for the interface, and datasets if needed. Possible models to explore include DistilGPT-2, Flan-T5 Small, or Phi-2.

### 2. Mini Retrieval-Augmented Generation (RAG) System in Google Colab
We will build a small RAG pipeline that lets users upload documents, index them, and ask questions that combine retrieval with generation. The system will run in Google Colab so that all computation can be done for free on CPU. Versions of the notebook and code will be maintained in a linked GitHub repository. Tools will include FAISS for vector search, sentence-transformers for embeddings, and Python libraries such as transformers and gradio for interaction. Candidate models include Flan-T5 Small, LLaMA 3.2 1B, or MiniLM for embeddings.

### 3. Local AI-Powered File Assistant on a Laptop
We will build a lightweight local assistant that can summarize text files, answer questions about them, and provide simple offline AI capabilities. The application will run entirely on a user's laptop using Python without requiring GPUs. Versioning will be managed through GitHub to track code changes and collaboration. The project will use libraries such as GPT4All, LangChain, Tkinter or a CLI for the interface, and standard Python tooling. Models to consider include GPT4All-J, Mistral 7B Instruct (CPU optimized), or Phi-2.
