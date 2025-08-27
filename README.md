# Mixtra-AI: Fast Content Generation Tool

This project creates an AI-powered content generation app with fast text generation capabilities. It allows users to generate blogs, articles, essays, and more on any topic with context-aware content generation.

## Features

- Generate various types of content (blogs, articles, essays, reports, tutorials, reviews, etc.)
- Specify the number of words for your content
- Upload PDF documents to make content more relevant and context-aware
- Add text-based context for further refinement of the output
- Fast generation using Ollama's Qwen 2.5 model
- Intuitive and user-friendly interface

## Technical Implementation

- Uses Ollama's Qwen 2.5 model for high-speed text generation
- Langchain ecosystem for building the application
- FAISS for vector storage and document retrieval
  - Implements a vectorstore that indexes PDF documents for semantic search
  - Creates a retriever tool that can fetch relevant information from uploaded documents
- Agentic AI workflow with ReAct (Think > Act > Observe) pattern
  - Implements an agent that iteratively refines information retrieval
  - Uses a structured loop to fetch and process relevant document content
- PDF processing for context-aware content generation
- Gradio for the user interface

## User Interface

![UI of MixtraBlog AI Tool](UI%20of%20tool.png)

## Usage

1. Optional but recommended: Create a conda/venv environment first
2. Install requirements: `pip install -r requirements.txt`
3. Install [Ollama](https://ollama.com/) and pull the Qwen 2.5 model: `ollama pull qwen2.5vl:3b`
4. Also pull the embedding model: `ollama pull nomic-embed-text:latest`
5. Run the app: `python blog_gen.py`
6. Access the web interface at http://localhost:7860

### You are then ready to play around with the app!

## How to Use

1. Enter your topic in the Topic field
2. Specify the number of words for your content
3. Select the type of content you want (Blog Post, Article, Essay, Tutorial, etc.)
4. Add additional context in the Context field for more refined output
5. Optionally upload a PDF document for context-aware generation
   - The PDF will be processed, vectorized, and used to retrieve relevant information (This step might take some time for larger files)
   - The retriever uses FAISS to semantically search for content related to your topic
6. Click "Generate Content"

## Document Processing Workflow

When you upload a PDF:
1. The document is processed and converted to text
2. The text is embedded using Ollama's embedding model
3. Embeddings are stored in a FAISS vectorstore
4. When generating content, an agent uses a ReAct pattern to:
   - Think about what information is needed
   - Act by querying the vectorstore
   - Observe the results and refine the search
   - Repeat until the most relevant information is found
5. The retrieved information is incorporated into the final content

## Notes

- For more accurate results, be specific with your topic
- Uploading a relevant PDF document will make the content more context-aware
- Adding text-based context can help guide the generation process
