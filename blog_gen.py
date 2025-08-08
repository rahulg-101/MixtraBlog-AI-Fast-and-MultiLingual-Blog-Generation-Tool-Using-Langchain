import gradio as gr
import os
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_ollama import ChatOllama
from langchain.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain.schema import Document
from langchain_community.document_loaders import PyPDFLoader

from langchain.tools import Tool
from langchain.agents import AgentType, initialize_agent, create_react_agent, AgentExecutor
from langchain_core.prompts import PromptTemplate
from langchain.agents.agent import AgentOutputParser

# chat=ChatGroq(groq_api_key=os.environ['GROQ_API_KEY'], model_name="mixtral-8x7b-32768")

def create_vectorstore(filepath=None, VECTORSTORE_DIR = r'./vectorstore'):
    """
    Creates or loads a vectorstore from documents.
    
    Args:
        filepath: Path to the PDF document to process (optional)
        VECTORSTORE_DIR: Directory to save/load the vectorstore
        
    Returns:
        A FAISS vectorstore object or None if no document provided and no existing vectorstore
    """
    
    # Initialize embeddings
    embeddings = OllamaEmbeddings(model="nomic-embed-text:latest")
    
    # If no document provided, try to load existing vectorstore
    if filepath is None or filepath == "":
        try:
            # Try to load the existing vectorstore from disk
            print("No document provided. Attempting to load existing vectorstore...")
            vectorstore = FAISS.load_local(VECTORSTORE_DIR, embeddings, allow_dangerous_deserialization=True)
            print(f"Successfully loaded vectorstore from {VECTORSTORE_DIR}")
            return vectorstore
        except Exception as e:
            print(f"Could not load vectorstore from disk: {str(e)}")
            print("No document provided and no existing vectorstore found.")
            return None
    
    # If document provided, create new vectorstore
    try:
        print(f"Creating vectorstore from document: {filepath}")
        
        # Load documents
        loader = PyPDFLoader(file_path=filepath)
        documents = loader.load()
        
        if not documents:
            print("No content extracted from the document.")
            return None
            
        # Create new vectorstore
        vectorstore = FAISS.from_documents(documents, embeddings)
        
        try:
            # Create directory if it doesn't exist
            os.makedirs(VECTORSTORE_DIR, exist_ok=True)
            
            # Save the vectorstore to disk
            vectorstore.save_local(VECTORSTORE_DIR)
            print(f"Vectorstore saved to {VECTORSTORE_DIR} for future use")
        except Exception as save_error:
            print(f"Warning: Could not save vectorstore to disk: {str(save_error)}")
            print("Vectorstore created but not cached")
        
        return vectorstore
        
    except Exception as e:
        print(f"Error creating vectorstore: {str(e)}")
        return None

chat = ChatOllama(model='qwen2.5vl:3b',temperature=0.1)



def create_document_retrieval_tool(vectorstore):
    """
    Creates a tool for document retrieval that can be used with ReAct agents.
    
    Args:
        vectorstore: FAISS vectorstore to retrieve documents from
        
    Returns:
        A Tool object for document retrieval
    """
    if vectorstore is None:
        # Create a dummy tool that returns an empty string
        def empty_retrieval_tool(query: str) -> str:
            return "No document has been provided for context-based retrieval."
            
        return Tool(
            name="retrieve_document_content",
            func=empty_retrieval_tool,
            description="This tool has no document to retrieve information from."
        )
    
    def document_retrieval_tool(query: str) -> str:
        """Retrieve relevant content from documents for a given query."""
        try:
            # Use retriever to get relevant documents
            retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
            docs = retriever.get_relevant_documents(query)
            
            if not docs:
                return "No relevant information found in the document."
            
            # Extract and format content
            content_pieces = []
            for i, doc in enumerate(docs, 1):
                content = doc.page_content.strip()
                if content:
                    content_pieces.append(f"Document Excerpt {i}:\n{content}")
            
            return "\n\n".join(content_pieces)
            
        except Exception as e:
            return f"Error retrieving document content: {str(e)}"
    
    return Tool(
        name="retrieve_document_content",
        func=document_retrieval_tool,
        description="Retrieve relevant content from the provided document for a given query."
    )

def retrieve_document_content(query, document_path=None):
    """
    Retrieves relevant content from the document using a ReAct agent.
    
    Args:
        query: The user's query to search for in the document
        document_path: Path to the document to search in (optional)
        
    Returns:
        A dictionary with relevant information from the document
    """
    # Create vectorstore from the document if provided
    vectorstore = create_vectorstore(filepath=document_path)
    
    # If no vectorstore created (no document provided or error), return empty context
    if vectorstore is None:
        return {"output": "No document context available."}
    
    # Create the retrieval tool
    retrieval_tool = create_document_retrieval_tool(vectorstore)
    
    # Create ReAct agent prompt
    react_prompt = PromptTemplate.from_template("""
You are a retriever agent whose purpose is to find the most relevant information from a document based on the user's query.

You have access to the following tool:
{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: provide a concise summary of the most relevant information found in the document

Question: {input}
{agent_scratchpad}
""")
    
    # Create the agent
    agent = create_react_agent(
        llm=chat,
        tools=[retrieval_tool],
        prompt=react_prompt
    )
    
    # Create agent executor
    agent_executor = AgentExecutor(
        agent=agent,
        tools=[retrieval_tool],
        verbose=True,
        max_iterations=5,
        handle_parsing_errors=True,
        agent_kwargs={
            "output_parser": AgentOutputParser()
        }
    )
    
    try:
        result = agent_executor.invoke({"input": query})
        return result
    except Exception as e:
        return {"output": f"Error retrieving document content: {str(e)}"}

def blogger(topic, no_words, document_type, document=None, context=None):
    """
    Generates content based on user inputs.
    
    Args:
        topic: The topic to write about
        no_words: Number of words for the content
        document_type: Type of document to create (blog, article, essay, etc.)
        document: Path to an optional PDF document for context
        context: Additional context information provided by the user
        
    Returns:
        Generated content as text
    """
    # Prepare context from the document if provided
    document_context = ""
    if document and document != "":
        try:
            # Get relevant information from the document
            result = retrieve_document_content("topic:" + topic + " context:" + context, document)
            if "output" in result:
                document_context = result["output"]
        except Exception as e:
            document_context = f"Error processing document: {str(e)}"
    
    # Create the prompt template
    prompt_template = f"""
Being an expert on {{topic}}, write a {{document_type}} in {{no_words}} words on {{topic}}.
Give a suitable title, generate content in a well-structured format and always add author name as 'Rahul Gupta' at the end.

{f"Use the following information from the uploaded document as context: {document_context}" if document_context else ""}
FYI, here is bit more context on what the user wants: {{information}}, retrieve all the information necessary as per
the context and other information provided by the user.
"""

    prompt = PromptTemplate(
        input_variables=["topic", "document_type","no_words","document_context","information"],
        template=prompt_template
    )
    
    # Create and invoke the chain
    llm_chain = prompt | chat
    response = llm_chain.invoke({
        'topic': topic,
        'no_words': no_words,
        'document_type': document_type,
        'document_context': document_context,
        'information': context
    })
    
    return response.content


# Create Gradio interface with improved UI
with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue"), css="footer {visibility: hidden}") as demo:
    gr.Markdown(
        """
        # MixtraBlog AI: Fast Content Generation Tool
        
        This app can generate high-quality content on any topic in your chosen format and length. 
        Upload a PDF document (optional) to make the content more relevant and context-aware.
        """
    )
    
    with gr.Row():
        with gr.Column():
            topic = gr.Textbox(
                label="Topic",
                placeholder="Enter a topic (e.g., Machine Learning, Climate Change, Digital Marketing)",
                lines=2
            )
            
            with gr.Row():
                no_words = gr.Textbox(
                    label="Number of Words",
                    placeholder="e.g., 500",
                    value="500"
                )
                
                document_type = gr.Dropdown(
                    label="Content Type",
                    choices=["Blog Post", "Article", "Essay", "Report", "Tutorial", "Review"],
                    value="Blog Post"
                )
            
            context = gr.Textbox(
                label="Context",
                placeholder="Enter the reference context you want to generate",
                lines=3
            )
            document = gr.File(
                label="Upload PDF Document (Optional)",
                file_types=[".pdf"],
                type="filepath"
            )

            

            generate_btn = gr.Button("Generate Content", variant="primary")
            
        with gr.Column():
            output = gr.Textbox(
                label="Generated Content",
                placeholder="Your content will appear here...",
                lines=31
            )
    
    generate_btn.click(
        fn=blogger,
        inputs=[topic, no_words, document_type, document,context],
        outputs=output
    )
    
    gr.Markdown("""
    ### Notes:
    - For more accurate results, be specific with your topic
    - Uploading a relevant PDF document will make the content more context-aware
    """)

# Launch the app
demo.launch()