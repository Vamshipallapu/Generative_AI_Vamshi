# Text Splitters
============================================================================================
## What Are Text Splitters in LangChain?

Text splitters are tools in LangChain that break down large pieces of text into smaller chunks. This is important for tasks like RAG (Retrieval-Augmented Generation), where you need to store documents in a vector database (like Chroma) and retrieve them efficiently. If the text is too long, it might not fit into the model’s context window or might be too big for the vector database to handle well. Text splitters help by dividing the text into manageable pieces while trying to keep the meaning intact.

Each text splitter in LangChain is designed for a specific type of text or splitting strategy. The choice of splitter depends on the type of text you’re working with (e.g., plain text, code, HTML) and how you want to split it (e.g., by characters, sentences, or code structure).

Overview of Text Splitters in LangChain
Here’s a list of the text splitters you mentioned, along with a brief description of what they do:

#### Base (RecursiveCharacterTextSplitter):
The default splitter that breaks text into chunks based on characters, trying to keep meaningful units (like paragraphs) together.

#### Character (CharacterTextSplitter): 
Splits text based on a specific character (e.g., a newline or a custom separator).

#### HTML (HTMLHeaderTextSplitter, HTMLSectionSplitter): 
Splits HTML content based on its structure (e.g., headers, sections).

#### JSON (Not a built-in splitter, but can be handled): 
Splits JSON data into smaller pieces (requires custom handling).

#### JSX (Not a built-in splitter, but can be handled): 
Splits JSX code (used in React) based on its structure (requires custom handling).

#### Konlpy: 
Splits Korean text using the Konlpy library for tokenization.

#### Latex (LatexTextSplitter): 
Splits LaTeX documents based on their structure (e.g., sections, equations).

#### Markdown (MarkdownHeaderTextSplitter, MarkdownTextSplitter): 
Splits Markdown text based on its structure (e.g., headers, lists).

#### NLTK (NLTKTextSplitter): 
Splits text into sentences using the NLTK library.

#### Python (PythonCodeTextSplitter): 
Splits Python code based on its structure (e.g., functions, classes).

#### Sentence Transformers (Not a built-in splitter, but can be used): 
Splits text based on semantic similarity using embeddings (requires custom implementation).

#### Spacy (SpacyTextSplitter): 
Splits text into sentences using the SpaCy library.

Now, let’s go through each one and explain when to use it, with examples tied to your RAG pipeline (which processes documents like PDFs and web content, such as the LangSmith web page).

When to Use Each Text Splitter

#### 1. Base (RecursiveCharacterTextSplitter)
   
What It Does:

The default text splitter in LangChain.

Splits text into chunks based on a character limit (e.g., 500 characters), but does it "recursively" by trying to keep meaningful units together.
It first tries to split on larger units (e.g., paragraphs, separated by double newlines \n\n), then falls back to smaller units (e.g., single newlines \n), and finally splits by characters if needed.

````
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
splits = text_splitter.split_documents(data)
````
This is the splitter used in your code for the Lilian Weng blog post.

When to Use:

Use it as the default choice for most general text (e.g., plain text, blog posts, articles).
Best for documents where you don’t need to preserve specific structures (e.g., headers, code blocks) and just want to split into roughly equal-sized chunks.
Good for mixed content where the structure isn’t critical (e.g., a mix of paragraphs, lists, and sentences).

When to Use in Your RAG Pipeline:
Use RecursiveCharacterTextSplitter for your current setup (LangSmith web page, PDFs) because:
The LangSmith web page and PDFs likely contain mixed content (paragraphs, headings, lists).
You don’t need to preserve specific structures for retrieval; you just need chunks small enough to fit into the vector database.

````
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
splits = text_splitter.split_documents(data)
````

#### 2. Character (CharacterTextSplitter)
   
What It Does:

Splits text based on a specific character (e.g., a newline \n, a comma ,, or a custom separator).
Doesn’t try to be "smart" about keeping meaningful units together; it just cuts the text wherever the separator appears.

````
from langchain_text_splitters import CharacterTextSplitter

text_splitter = CharacterTextSplitter(separator="\n", chunk_size=500, chunk_overlap=0)
splits = text_splitter.split_documents(data)
````

When to Use:

Use it when your text has a clear, consistent separator that naturally divides it into meaningful chunks (e.g., a list of items separated by newlines or commas).
Best for simple, structured text where the separator defines logical breaks (e.g., a log file with entries separated by newlines).

When to Use in Your RAG Pipeline:

Use CharacterTextSplitter if your documents have a clear separator that aligns with meaningful chunks.

Example: If your PDFs or web content contain lists of items separated by newlines (e.g., a list of "Key Features of LangSmith" with each feature on a new line), you can split by \n.

Not Ideal for Your Current Setup: The LangSmith web page and PDFs likely have mixed content (paragraphs, headings, etc.), so RecursiveCharacterTextSplitter is a better choice because it handles varied structures more intelligently.

#### 3. HTML (HTMLHeaderTextSplitter, HTMLSectionSplitter)

What It Does:

````
Splits HTML content based on its structure, such as headers (<h1>, <h2>) or sections (<section>).
HTMLHeaderTextSplitter: Splits based on headers and associates content with the preceding header.
HTMLSectionSplitter: Splits based on HTML sections.
````

````
from langchain_text_splitters import HTMLHeaderTextSplitter

headers_to_split_on = [("h1", "Header 1"), ("h2", "Header 2")]
text_splitter = HTMLHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
splits = text_splitter.split_documents(data)
````

When to Use:

Use it when your documents are in HTML format (e.g., web pages) and you want to split based on the HTML structure.
Best for web content where headers or sections define logical breaks (e.g., a blog post with sections like "Introduction," "Methods," "Conclusion").
Preserves the structure of the HTML, which can be useful for retrieval (e.g., retrieving a section under a specific header).

When to Use in Your RAG Pipeline:

````
Use HTMLHeaderTextSplitter for the LangSmith web page (https://www.langchain.com/langsmith) because:
It’s an HTML page with a clear structure (likely has headers like <h1>, <h2> for sections like "What is LangSmith?", "Features").
Splitting by headers ensures each chunk is a meaningful section (e.g., the "Features" section), which can improve retrieval accuracy.
````

Example: Modify your pipeline to use HTMLHeaderTextSplitter for the web content:

````
def load_documents(self) -> None:
    try:
        if not os.path.exists(self.pdf_dir):
            raise FileNotFoundError(f"PDF directory {self.pdf_dir} does not exist")
        for filename in os.listdir(self.pdf_dir):
            if filename.endswith(".pdf"):
                pdf_path = os.path.join(self.pdf_dir, filename)
                loader = PyPDFLoader(pdf_path)
                self.documents.extend(loader.load())
                logging.info(f"Loaded PDF: {pdf_path}")
        web_loader = WebBaseLoader(self.web_url)
        self.documents.extend(web_loader.load())
        logging.info(f"Loaded web content from: {self.web_url}")
    except Exception as e:
        logging.error(f"Error loading documents: {str(e)}")
        raise

def chunk_documents(self) -> List:
    try:
        # Use HTMLHeaderTextSplitter for web content
        headers_to_split_on = [("h1", "Header 1"), ("h2", "Header 2")]
        html_splitter = HTMLHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
        # Use RecursiveCharacterTextSplitter for PDFs
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
        
        chunks = []
        for doc in self.documents:
            if doc.metadata.get("source", "").endswith(".pdf"):
                chunks.extend(text_splitter.split_documents([doc]))
            else:  # Web content
                chunks.extend(html_splitter.split_documents([doc]))
        logging.info(f"Created {len(chunks)} document chunks")
        return chunks
    except Exception as e:
        logging.error(f"Error chunking documents: {str(e)}")
        raise
````

Why: This ensures the LangSmith web page is split into meaningful sections (e.g., "Features" section), while PDFs are split using the default splitter.

#### 4. JSON (Not a Built-in Splitter)
   
What It Does:

LangChain doesn’t have a built-in JSONTextSplitter, but you can handle JSON data by parsing it and splitting it manually or using a custom splitter.
JSON data often represents structured information (e.g., key-value pairs, nested objects), so splitting might involve extracting specific fields or flattening the data.

How to Handle:

Use a custom approach to parse JSON and split it into chunks based on its structure (e.g., split by entries, fields, or nested objects).

````
import json
from langchain.docstore.document import Document

def split_json_data(json_str: str, chunk_size: int = 500) -> List[Document]:
    data = json.loads(json_str)
    chunks = []
    # Example: Split by top-level keys
    for key, value in data.items():
        chunk_text = f"{key}: {json.dumps(value)}"
        if len(chunk_text) > chunk_size:
            # Further split if too large
            sub_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=0)
            sub_docs = sub_splitter.split_text(chunk_text)
            for sub_doc in sub_docs:
                chunks.append(Document(page_content=sub_doc))
        else:
            chunks.append(Document(page_content=chunk_text))
    return chunks
````

When to Use:

Use it when your documents are in JSON format (e.g., API responses, structured data).
Best for structured data where you want to split based on the JSON structure (e.g., split by keys or nested objects).

When to Use in Your RAG Pipeline:

Use a custom JSON splitter if your RAG pipeline includes JSON data (e.g., if you fetch additional data from an API in JSON format, such as LangSmith metadata).
Not Relevant for Your Current Setup: Your pipeline uses PDFs and HTML web content, not JSON. However, if you were to add JSON data (e.g., metadata about LangSmith features), you could use a custom JSON splitter.

#### 5. JSX (Not a Built-in Splitter)
   
What It Does:

LangChain doesn’t have a built-in JSXTextSplitter, but JSX (JavaScript XML, used in React) is a type of code that combines JavaScript and HTML-like syntax.
You can handle JSX by treating it as code (similar to Python) or HTML and splitting based on its structure (e.g., components, functions).

How to Handle:

Use a custom splitter or the PythonCodeTextSplitter (since JSX is JavaScript-based) to split based on code structure (e.g., components, functions).

````
from langchain_text_splitters import PythonCodeTextSplitter

text_splitter = PythonCodeTextSplitter(chunk_size=500, chunk_overlap=0)
splits = text_splitter.split_text(jsx_code)
````

When to Use:

Use it when your documents are JSX code (e.g., React components).
Best for splitting code where you want to preserve the structure of components or functions.

When to Use in Your RAG Pipeline:

Not Relevant for Your Current Setup: Your pipeline deals with PDFs and HTML web content, not JSX code.
Use Case: If you were to include JSX code (e.g., LangSmith-related React components from a GitHub repo), you could split it using a code-based splitter.

#### 6. Konlpy (KonlpyTextSplitter)
   
What It Does:

Splits Korean text using the Konlpy library, which provides tokenization for Korean language processing.
Breaks text into sentences or tokens based on Korean grammar and morphology.

````
from langchain_text_splitters import KonlpyTextSplitter

text_splitter = KonlpyTextSplitter(chunk_size=500, chunk_overlap=0)
splits = text_splitter.split_text(korean_text)
````

When to Use:

Use it when your documents are in Korean and you need language-specific splitting.
Best for Korean text where you want to split into sentences or tokens while respecting Korean grammar.

When to Use in Your RAG Pipeline:

Not Relevant for Your Current Setup: Your pipeline uses English text (LangSmith web page, PDFs).
Use Case: If you were to include Korean documents (e.g., a translated version of the LangSmith page), use KonlpyTextSplitter to split them correctly.

#### 7. Latex (LatexTextSplitter)
   
What It Does:

Splits LaTeX documents based on their structure (e.g., sections, equations, paragraphs).
Preserves LaTeX-specific elements (e.g., \section, \begin{equation}).

````
from langchain_text_splitters import LatexTextSplitter

text_splitter = LatexTextSplitter(chunk_size=500, chunk_overlap=0)
splits = text_splitter.split_text(latex_text)
````

When to Use:

Use it when your documents are in LaTeX format (e.g., academic papers, scientific reports).
Best for LaTeX content where you want to split based on sections, equations, or other LaTeX structures.

When to Use in Your RAG Pipeline:

Not Relevant for Your Current Setup: Your pipeline uses PDFs and HTML, not LaTeX.
Use Case: If you were to include LaTeX documents (e.g., a research paper about LangSmith in LaTeX format), use LatexTextSplitter to split by sections or equations.

#### 8. Markdown (MarkdownHeaderTextSplitter, MarkdownTextSplitter)
   
What It Does:

MarkdownHeaderTextSplitter: Splits Markdown text based on headers (e.g., #, ##) and associates content with the preceding header.
MarkdownTextSplitter: Splits Markdown text based on its structure (e.g., headers, lists, code blocks).

````
from langchain_text_splitters import MarkdownHeaderTextSplitter

headers_to_split_on = [("#", "Header 1"), ("##", "Header 2")]
text_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
splits = text_splitter.split_text(markdown_text)
````

When to Use:

Use it when your documents are in Markdown format (e.g., GitHub READMEs, Markdown files).
Best for Markdown content where headers or other structures (e.g., lists, code blocks) define logical breaks.
Preserves the Markdown structure, which can improve retrieval (e.g., retrieving a section under a specific header).

When to Use in Your RAG Pipeline:

Not Directly Relevant for Your Current Setup: Your pipeline uses PDFs and HTML, not Markdown. However, the LangSmith web page might be converted to Markdown-like structure after scraping.
Use Case: If you convert the LangSmith web page to Markdown (e.g., using a tool like html2text) or include Markdown files (e.g., LangSmith documentation from a GitHub repo), use MarkdownHeaderTextSplitter to split by headers.

````
def chunk_documents(self) -> List:
    try:
        html_splitter = HTMLHeaderTextSplitter(headers_to_split_on=[("h1", "Header 1"), ("h2", "Header 2")])
        markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=[("#", "Header 1"), ("##", "Header 2")])
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)

        chunks = []
        for doc in self.documents:
            if doc.metadata.get("source", "").endswith(".pdf"):
                chunks.extend(text_splitter.split_documents([doc]))
            elif doc.metadata.get("source", "").endswith(".md"):
                chunks.extend(markdown_splitter.split_documents([doc]))
            else:  # Web content
                chunks.extend(html_splitter.split_documents([doc]))
        logging.info(f"Created {len(chunks)} document chunks")
        return chunks
    except Exception as e:
        logging.error(f"Error chunking documents: {str(e)}")
        raise
````

#### 9. NLTK (NLTKTextSplitter)

What It Does:

Splits text into sentences using the NLTK (Natural Language Toolkit) library.
Uses NLTK’s sentence tokenizer to identify sentence boundaries (e.g., based on punctuation like ., !, ?).

````
from langchain_text_splitters import NLTKTextSplitter

text_splitter = NLTKTextSplitter(chunk_size=500, chunk_overlap=0)
splits = text_splitter.split_text(text)
````

When to Use:

Use it when you want to split text into sentences and ensure each chunk contains complete sentences.
Best for narrative text (e.g., articles, books) where sentences are the natural unit of meaning.
Good for languages supported by NLTK (primarily English, but can be extended to others).

When to Use in Your RAG Pipeline:

Use NLTKTextSplitter if you want to ensure chunks are sentence-based, which can improve the coherence of retrieved documents.
Example: The LangSmith web page contains narrative text (e.g., paragraphs describing features). Splitting by sentences ensures each chunk is a complete thought.
Consideration: Your current setup uses RecursiveCharacterTextSplitter, which might split sentences in the middle. If this causes issues (e.g., retrieved chunks are incomplete sentences), switch to NLTKTextSplitter.

````
def chunk_documents(self) -> List:
    try:
        text_splitter = NLTKTextSplitter(chunk_size=500, chunk_overlap=0)
        chunks = text_splitter.split_documents(self.documents)
        logging.info(f"Created {len(chunks)} document chunks")
        return chunks
    except Exception as e:
        logging.error(f"Error chunking documents: {str(e)}")
        raise
````

#### 10. Python (PythonCodeTextSplitter)
    
What It Does:

Splits Python code based on its structure (e.g., functions, classes, imports).
Preserves the logical structure of the code, ensuring each chunk is a meaningful unit (e.g., a complete function).

````
from langchain_text_splitters import PythonCodeTextSplitter

text_splitter = PythonCodeTextSplitter(chunk_size=500, chunk_overlap=0)
splits = text_splitter.split_text(python_code)
````

When to Use:

Use it when your documents are Python code (e.g., scripts, Jupyter notebooks).
Best for code where you want to split by logical units (e.g., functions, classes) rather than arbitrary character limits.

When to Use in Your RAG Pipeline:

Not Relevant for Your Current Setup: Your pipeline uses PDFs and HTML, not Python code.
Use Case: If you include Python code (e.g., LangSmith-related code snippets from a GitHub repo), use PythonCodeTextSplitter to split by functions or classes.

Example: If you add a Python file to your pipeline:
````
def chunk_documents(self) -> List:
    try:
        html_splitter = HTMLHeaderTextSplitter(headers_to_split_on=[("h1", "Header 1"), ("h2", "Header 2")])
        code_splitter = PythonCodeTextSplitter(chunk_size=500, chunk_overlap=0)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)

        chunks = []
        for doc in self.documents:
            if doc.metadata.get("source", "").endswith(".pdf"):
                chunks.extend(text_splitter.split_documents([doc]))
            elif doc.metadata.get("source", "").endswith(".py"):
                chunks.extend(code_splitter.split_documents([doc]))
            else:  # Web content
                chunks.extend(html_splitter.split_documents([doc]))
        logging.info(f"Created {len(chunks)} document chunks")
        return chunks
    except Exception as e:
        logging.error(f"Error chunking documents: {str(e)}")
        raise
````

#### 11. Sentence Transformers (Not a Built-in Splitter)
    
What It Does:

LangChain doesn’t have a built-in SentenceTransformersTextSplitter, but you can use sentence-transformers (a Hugging Face library) to split text based on semantic similarity.
This involves generating embeddings for sentences or paragraphs and grouping them into chunks based on similarity (e.g., using clustering).

How to Handle:

Generate embeddings with sentence-transformers and cluster them with scikit-learn (e.g., k-means) to create chunks.

Example:
````
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from langchain.docstore.document import Document

def split_by_semantic_similarity(documents: List[Document], chunk_size: int = 500) -> List[Document]:
    model = SentenceTransformer("all-MiniLM-L6-v2")
    texts = [doc.page_content for doc in documents]
    embeddings = model.encode(texts)
    kmeans = KMeans(n_clusters=len(texts) // 5)  # Approximate number of clusters
    clusters = kmeans.fit_predict(embeddings)

    chunks = []
    for cluster_id in set(clusters):
        cluster_texts = [texts[i] for i in range(len(texts)) if clusters[i] == cluster_id]
        chunk_text = " ".join(cluster_texts)
        if len(chunk_text) > chunk_size:
            sub_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=0)
            sub_chunks = sub_splitter.split_text(chunk_text)
            for sub_chunk in sub_chunks:
                chunks.append(Document(page_content=sub_chunk))
        else:
            chunks.append(Document(page_content=chunk_text))
    return chunks
````

When to Use:

Use it when you want to split text into chunks based on semantic similarity (e.g., group sentences or paragraphs that are about the same topic).
Best for large documents where you want chunks to be semantically coherent (e.g., group all sentences about "Task Decomposition" together).

When to Use in Your RAG Pipeline:

Use this approach if you want to improve the coherence of your chunks for better retrieval.
Example: The LangSmith web page might have sections about different topics (e.g., features, pricing, use cases). Splitting by semantic similarity ensures chunks are about the same topic, which can improve retrieval accuracy.
Consideration: This method is more complex and resource-intensive (requires embeddings, clustering), so only use it if RecursiveCharacterTextSplitter isn’t giving you coherent chunks.

Example: Add semantic splitting to your pipeline:

````
def chunk_documents(self) -> List:
    try:
        html_splitter = HTMLHeaderTextSplitter(headers_to_split_on=[("h1", "Header 1"), ("h2", "Header 2")])
        chunks = html_splitter.split_documents(self.documents)
        # Further split by semantic similarity
        chunks = split_by_semantic_similarity(chunks, chunk_size=500)
        logging.info(f"Created {len(chunks)} document chunks")
        return chunks
    except Exception as e:
        logging.error(f"Error chunking documents: {str(e)}")
        raise
````

#### 12. Spacy (SpacyTextSplitter)
    
What It Does:

Splits text into sentences using the SpaCy library, which provides advanced NLP tools for sentence segmentation.
Similar to NLTKTextSplitter, but uses SpaCy’s sentence tokenizer, which is often more accurate for complex sentences.

Example:
````
from langchain_text_splitters import SpacyTextSplitter

text_splitter = SpacyTextSplitter(chunk_size=500, chunk_overlap=0)
splits = text_splitter.split_text(text)
````

When to Use:

Use it when you want to split text into sentences and need high accuracy (SpaCy is often more accurate than NLTK for sentence segmentation, especially for complex sentences).
Best for narrative text (e.g., articles, reports) where sentences are the natural unit of meaning.
Good for multiple languages (SpaCy supports many languages with pre-trained models).

When to Use in Your RAG Pipeline:

Use SpacyTextSplitter if you want sentence-based chunks and need better accuracy than NLTK.
Example: The LangSmith web page contains narrative text (e.g., paragraphs describing features). Splitting by sentences ensures each chunk is a complete thought, and SpaCy’s accuracy can handle complex sentences better than NLTK.
Consideration: SpaCy is more resource-intensive than NLTK (requires loading a language model), so only use it if NLTK isn’t accurate enough.

Example:
````
def chunk_documents(self) -> List:
    try:
        text_splitter = SpacyTextSplitter(chunk_size=500, chunk_overlap=0)
        chunks = text_splitter.split_documents(self.documents)
        logging.info(f"Created {len(chunks)} document chunks")
        return chunks
    except Exception as e:
        logging.error(f"Error chunking documents: {str(e)}")
        raise
````
