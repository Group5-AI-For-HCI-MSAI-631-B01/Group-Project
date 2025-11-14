# Solution 2: Mini Retrieval-Augmented Generation (RAG) System in Google Colab

## Project Overview

This project proposes building a lightweight Retrieval-Augmented Generation (RAG) pipeline that enables users to upload their own documents, create a searchable knowledge base, and ask questions that are answered using both document retrieval and language model generation. This demonstrates how modern AI can provide accurate, grounded responses based on specific document collections.

## Application Type

**Interactive Document Q&A System with RAG Architecture**

The application will allow users to:
- Upload multiple documents (PDF, TXT, DOCX formats)
- Automatically index and vectorize document content
- Ask natural language questions about the documents
- Receive answers that cite specific passages from the documents
- View the source passages that informed each answer

## Deployment Platform

**Google Colab (Free tier with CPU/GPU runtime)**

Google Colab provides:
- Free Jupyter notebook environment
- Access to GPU/CPU runtime (T4 GPU on free tier)
- 12GB RAM for processing
- Pre-installed common ML libraries
- Easy sharing via link
- Integration with Google Drive for document storage
- No local setup required

## Version Management Strategy

**GitHub Repository with Colab Integration**

- **Primary Repository**: GitHub for version control
- **Notebook Storage**: Both GitHub (.ipynb file) and Google Drive
- **Branching Strategy**:
  - `main` branch for stable releases
  - `dev` branch for development
  - Feature branches for new capabilities
- **Versioning**: Git tags for major versions (v1.0, v2.0, etc.)
- **Colab Integration**: Notebooks contain "Open in Colab" badge linking to GitHub
- **Data Versioning**: Sample documents tracked in repository
- **Collaboration**: GitHub pull requests for code review

## Tools and Libraries

### Core Framework
- **Python 3.10+**: Programming language
- **Jupyter Notebook**: Interactive development environment
- **Gradio 4.x**: UI interface within Colab

### Document Processing
- **PyPDF2** or **pdfplumber**: PDF text extraction
- **python-docx**: DOCX file handling
- **langchain 0.1+**: Framework for building RAG pipelines
- **tiktoken**: Token counting and text chunking

### Vector Database and Embeddings
- **FAISS**: Facebook AI Similarity Search for vector storage and retrieval
- **sentence-transformers**: Generate document embeddings
- **chromadb** (alternative): Lightweight vector database

### AI/ML Libraries
- **transformers 4.35+**: Hugging Face library for LLMs
- **torch 2.0+**: PyTorch for model inference
- **accelerate**: Optimized model loading

### Utilities
- **numpy**: Numerical operations
- **pandas**: Data manipulation
- **tqdm**: Progress bars for processing
- **requests**: File downloading if needed

## AI Models to Explore

### Embedding Models (for document vectorization)

1. **all-MiniLM-L6-v2** (22M parameters)
   - Fast and lightweight
   - Good semantic search quality
   - 384-dimensional embeddings
   - Size: ~90MB

2. **all-mpnet-base-v2** (110M parameters)
   - Higher quality embeddings
   - 768-dimensional embeddings
   - Better semantic understanding
   - Size: ~420MB

### Generation Models (for answer synthesis)

1. **Flan-T5 Small** (80M parameters)
   - Instruction-tuned for Q&A tasks
   - Fast inference on CPU
   - Good at following prompts
   - Size: ~300MB

2. **Flan-T5 Base** (250M parameters)
   - Better answer quality
   - Still runs on Colab CPU
   - Good balance of speed/quality
   - Size: ~900MB

3. **LLaMA 3.2 1B** (1B parameters)
   - More sophisticated reasoning
   - Requires GPU for reasonable speed
   - Better handling of complex queries
   - Size: ~2GB

### Selection Criteria
- Embedding quality for semantic search
- Generation quality for natural answers
- Inference speed (< 5 seconds total per query)
- Memory constraints (< 10GB total)

## High-Level Design

### System Architecture

```
Document Upload → Text Extraction → Chunking → Embedding Generation → Vector Store (FAISS)
                                                                              ↓
User Question → Question Embedding → Similarity Search → Retrieve Top-K Chunks
                                                              ↓
                                    Retrieved Context + Question → LLM Generator → Answer + Citations
```

### Inputs
- **Documents**: PDF, TXT, or DOCX files (max 10 documents, 50 pages each)
- **User Question**: Natural language query about the documents (max 256 tokens)
- **Retrieval Parameters**: Number of chunks to retrieve (k=3-5)
- **Generation Parameters**: Temperature, max length

### Outputs
- **Generated Answer**: Natural language response based on retrieved context (max 512 tokens)
- **Source Citations**: Specific document passages used to generate answer
- **Relevance Scores**: Similarity scores for retrieved chunks
- **Document Names**: Which documents contained relevant information

### Pseudo Code

```python
# ========== Document Ingestion ==========
def ingest_documents(uploaded_files):
    """
    Process uploaded documents and create vector store
    Args:
        uploaded_files: List of uploaded document files
    Returns:
        vector_store: FAISS index with document embeddings
        chunks: List of text chunks with metadata
    """
    chunks = []
    
    for file in uploaded_files:
        # Extract text from document
        text = extract_text(file)
        
        # Split into chunks (500 tokens, 50 token overlap)
        doc_chunks = split_into_chunks(
            text, 
            chunk_size=500, 
            overlap=50
        )
        
        # Add metadata
        for i, chunk in enumerate(doc_chunks):
            chunks.append({
                'text': chunk,
                'source': file.name,
                'chunk_id': i
            })
    
    # Generate embeddings
    embedding_model = load_embedding_model("all-MiniLM-L6-v2")
    embeddings = embedding_model.encode([c['text'] for c in chunks])
    
    # Create FAISS index
    dimension = embeddings.shape[1]
    vector_store = faiss.IndexFlatIP(dimension)
    vector_store.add(embeddings)
    
    return vector_store, chunks

# ========== Question Answering ==========
def answer_question(question, vector_store, chunks, top_k=3):
    """
    Answer question using RAG pipeline
    Args:
        question: User's question
        vector_store: FAISS index
        chunks: Document chunks with metadata
        top_k: Number of chunks to retrieve
    Returns:
        answer: Generated answer
        sources: Retrieved source chunks
    """
    # Generate question embedding
    embedding_model = load_embedding_model("all-MiniLM-L6-v2")
    question_embedding = embedding_model.encode([question])
    
    # Search for similar chunks
    scores, indices = vector_store.search(question_embedding, top_k)
    
    # Retrieve relevant chunks
    retrieved_chunks = [chunks[idx] for idx in indices[0]]
    
    # Build context from retrieved chunks
    context = "\n\n".join([
        f"[Document: {chunk['source']}]\n{chunk['text']}"
        for chunk in retrieved_chunks
    ])
    
    # Create prompt for LLM
    prompt = f"""Based on the following context, answer the question.
If the answer cannot be found in the context, say "I cannot find this information in the provided documents."

Context:
{context}

Question: {question}

Answer:"""
    
    # Generate answer
    llm = load_model("google/flan-t5-base")
    tokenizer = load_tokenizer("google/flan-t5-base")
    
    inputs = tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True)
    outputs = llm.generate(
        inputs.input_ids,
        max_length=512,
        temperature=0.7,
        do_sample=False  # Deterministic for consistency
    )
    
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return answer, retrieved_chunks

# ========== Gradio Interface ==========
def create_interface():
    """Create Gradio interface for RAG system"""
    
    # Global state for vector store and chunks
    state = {'vector_store': None, 'chunks': None}
    
    def upload_docs(files):
        state['vector_store'], state['chunks'] = ingest_documents(files)
        return f"Successfully indexed {len(files)} documents with {len(state['chunks'])} chunks"
    
    def ask_question(question):
        if state['vector_store'] is None:
            return "Please upload documents first", []
        
        answer, sources = answer_question(
            question, 
            state['vector_store'], 
            state['chunks']
        )
        
        # Format sources for display
        source_text = "\n\n".join([
            f"**Source {i+1}** ({s['source']}):\n{s['text']}"
            for i, s in enumerate(sources)
        ])
        
        return answer, source_text
    
    # Build interface
    with gradio.Blocks() as interface:
        gradio.Markdown("# Mini RAG System")
        
        with gradio.Tab("Upload Documents"):
            file_upload = gradio.File(file_count="multiple", label="Upload Documents")
            upload_button = gradio.Button("Index Documents")
            upload_status = gradio.Textbox(label="Status")
            upload_button.click(upload_docs, inputs=[file_upload], outputs=[upload_status])
        
        with gradio.Tab("Ask Questions"):
            question_input = gradio.Textbox(label="Your Question", lines=2)
            submit_button = gradio.Button("Get Answer")
            answer_output = gradio.Textbox(label="Answer", lines=4)
            sources_output = gradio.Textbox(label="Source Passages", lines=8)
            submit_button.click(ask_question, inputs=[question_input], 
                              outputs=[answer_output, sources_output])
    
    return interface

# Launch in Colab
interface = create_interface()
interface.launch(share=True)
```

## Real-World Problem Addressed

### Problem Statement
Professionals, students, and researchers often need to:
- Quickly extract information from large document collections
- Find relevant passages across multiple documents
- Understand relationships between different documents
- Get answers without reading entire documents
- Ensure answers are grounded in actual document content

Traditional solutions are inadequate:
- Manual reading is time-consuming
- Ctrl+F only finds exact matches
- Commercial RAG systems are expensive
- Generic chatbots hallucinate without source documents

### Solution Approach
This RAG system provides:
- **Semantic Search**: Find relevant content even with different wording
- **Grounded Answers**: Responses based on actual document content
- **Source Attribution**: See exactly where information comes from
- **Custom Knowledge Base**: Use your own documents, not generic training data
- **Privacy**: Documents processed locally in your Colab session
- **Free Access**: No API costs or subscriptions

### Use Cases
1. **Academic Research**: Quickly analyze multiple papers for literature review
2. **Legal Review**: Search through contracts and legal documents
3. **Business Intelligence**: Extract insights from reports and presentations
4. **Technical Documentation**: Find specific information in user manuals
5. **Personal Knowledge Management**: Query your own notes and documents

## Anticipated Limitations

### Technical Limitations
1. **Document Size**: Limited to ~50 pages per document due to memory constraints
2. **Document Formats**: Limited to PDF, TXT, DOCX (no images or tables)
3. **Context Window**: Each query uses only top 3-5 chunks (~1500 tokens)
4. **Processing Time**: Initial indexing takes 1-5 minutes for 10 documents
5. **Query Speed**: 3-8 seconds per question depending on runtime (CPU vs GPU)
6. **Accuracy**: May miss relevant information if chunking splits key passages

### Colab Environment Limitations
1. **Session Timeout**: Free tier disconnects after 90 minutes of inactivity
2. **Runtime Limits**: Maximum 12 hours per session
3. **No Persistence**: Vector store lost when runtime disconnects
4. **Memory Constraints**: 12GB RAM limits number of documents
5. **GPU Availability**: Not guaranteed on free tier

### Functional Limitations
1. **No Multi-Document Reasoning**: Cannot compare or synthesize across multiple documents easily
2. **Single Language**: Primarily English support
3. **No Table Understanding**: Cannot extract structured data from tables
4. **Limited OCR**: No image-to-text conversion for scanned PDFs
5. **No Real-Time Updates**: Must re-index if documents change

### Quality Limitations
1. **Retrieval Errors**: May retrieve irrelevant chunks if semantic search fails
2. **Generation Errors**: LLM may misinterpret retrieved context
3. **Chunking Issues**: Important context may be split across chunks
4. **No Reasoning**: Cannot perform complex logical reasoning
5. **Hallucination Risk**: Small models may still add information not in documents

### Mitigation Strategies
- Clear error messages for unsupported formats
- Progress indicators during indexing
- Show source passages for verification
- Recommend optimal document sizes in documentation
- Provide example documents for testing
- Regular saving of indexed data to Google Drive
- Option to adjust chunk size and overlap

## Future Enhancement Possibilities

While out of scope for initial implementation:
- Persistent storage via Google Drive integration
- Support for more document formats (HTML, Markdown)
- OCR for scanned PDFs
- Table extraction and understanding
- Multi-document comparative analysis
- Hybrid search (keyword + semantic)
- Query expansion for better retrieval
- Answer quality scoring
- Conversation memory for follow-up questions

## Success Metrics

1. **Functionality**: Successfully index documents and answer questions
2. **Accuracy**: 80%+ of answers grounded in retrieved context
3. **Retrieval Quality**: Top 3 chunks contain relevant information for 85%+ of queries
4. **Performance**: < 10 seconds total time from question to answer
5. **Usability**: Non-technical users can upload documents and get answers
6. **Reliability**: Handles various document formats without crashing
7. **Citation Quality**: Source passages clearly relate to generated answers
