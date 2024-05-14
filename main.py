import warnings
warnings.filterwarnings("ignore")

import os
import glob
import textwrap
import time

import langchain

### loaders
from langchain.document_loaders import PyPDFLoader, DirectoryLoader

### splits
from langchain.text_splitter import RecursiveCharacterTextSplitter

### prompts
from langchain import PromptTemplate, LLMChain

### vector stores
from langchain.vectorstores import FAISS
### models
from langchain.llms import HuggingFacePipeline
from langchain.embeddings import HuggingFaceInstructEmbeddings

### retrievers
from langchain.chains import RetrievalQA

import torch
import transformers
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    BitsAndBytesConfig,
    pipeline
)

class CFG:
    # LLMs
    model_name = 'llama2-13b-chat' # wizardlm, llama2-7b-chat, llama2-13b-chat, mistral-7B
    temperature = 0
    top_p = 0.95
    repetition_penalty = 1.15    

    # splitting
    split_chunk_size = 800
    split_overlap = 0
    
    # embeddings
    embeddings_model_repo = 'sentence-transformers/all-MiniLM-L6-v2'    

    # similar passages
    k = 6
    
    # paths
    PDFs_path = '.\\books\\'
    Embeddings_path =  '.\\input\\faiss-hp-sentence-transformers'
    Output_folder = '.\\books-vectordb'
    Output_folder_faiss = '.\\books-vectordb\\faiss_index_hp'

    

def get_model(model = CFG.model_name):

    print('\nDownloading model: ', model, '\n\n')

    if model == 'wizardlm':
        model_repo = 'TheBloke/wizardLM-7B-HF'
        
        tokenizer = AutoTokenizer.from_pretrained(model_repo)
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit = True,
            bnb_4bit_quant_type = "nf4",
            bnb_4bit_compute_dtype = torch.float16,
            bnb_4bit_use_double_quant = True,
        )        

        model = AutoModelForCausalLM.from_pretrained(
            model_repo,
            quantization_config = bnb_config,
            device_map = 'auto',
            low_cpu_mem_usage = True
        )
        
        max_len = 1024

    elif model == 'llama2-7b-chat':
        model_repo = 'daryl149/llama-2-7b-chat-hf'
        tokenizer = AutoTokenizer.from_pretrained(model_repo, use_fast=True)
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit = True,
            bnb_4bit_quant_type = "nf4",
            bnb_4bit_compute_dtype = torch.float16,
            bnb_4bit_use_double_quant = True,
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            model_repo,
            quantization_config = bnb_config,
            device_map = 'auto',
            low_cpu_mem_usage = True,
            trust_remote_code = True
        )
        
        max_len = 2048

    elif model == 'llama2-13b-chat':
        model_repo = 'daryl149/llama-2-13b-chat-hf'
        
        tokenizer = AutoTokenizer.from_pretrained(model_repo, use_fast=True)
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit = True,
            bnb_4bit_quant_type = "nf4",
            bnb_4bit_compute_dtype = torch.float16,
            bnb_4bit_use_double_quant = True,
        )
                
        model = AutoModelForCausalLM.from_pretrained(
            model_repo,
            quantization_config = bnb_config,       
            device_map = 'cuda',
            low_cpu_mem_usage = True,
            trust_remote_code = True
        )
        
        max_len = 2048 # 8192

    elif model == 'mistral-7B':
        model_repo = 'mistralai/Mistral-7B-v0.1'
        
        tokenizer = AutoTokenizer.from_pretrained(model_repo)
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit = True,
            bnb_4bit_quant_type = "nf4",
            bnb_4bit_compute_dtype = torch.float16,
            bnb_4bit_use_double_quant = True,
        )        

        model = AutoModelForCausalLM.from_pretrained(
            model_repo,
            quantization_config = bnb_config,
            device_map = 'auto',
            low_cpu_mem_usage = True,
        )
        
        max_len = 1024

    else:
        print("Not implemented model (tokenizer and backbone)")

    return tokenizer, model, max_len


def wrap_text_preserve_newlines(text, width=700):
    # Split the input text into lines based on newline characters
    lines = text.split('\n')

    # Wrap each line individually
    wrapped_lines = [textwrap.fill(line, width=width) for line in lines]

    # Join the wrapped lines back together using newline characters
    wrapped_text = '\n'.join(wrapped_lines)

    return wrapped_text


def process_llm_response(llm_response):
    ans = wrap_text_preserve_newlines(llm_response['result'])

    sources_used = ' \n'.join(
        [
            source.metadata['source'].split('/')[-1][:-4]
            + ' - page: '
            + str(source.metadata['page'])
            for source in llm_response['source_documents']
        ]
    )

    ans = ans + '\n\nSources: \n' + sources_used
    return ans

def llm_ans(query, qa_chain):
    start = time.time()

    llm_response = qa_chain.invoke(query)
    ans = process_llm_response(llm_response)

    end = time.time()

    time_elapsed = int(round(end - start, 0))
    time_elapsed_str = f'\n\nTime elapsed: {time_elapsed} s'
    return ans + time_elapsed_str


def main():
    pdf_files = sorted(glob.glob(CFG.PDFs_path))

    print('List of books in PDF:')
    for pdf_file in pdf_files:
        print(os.path.basename(pdf_file))

    tokenizer, model, max_len = get_model(model = CFG.model_name)

    # Save model to the specified folder
    output_folder = './model'
    os.makedirs(output_folder, exist_ok=True)
    model.save_pretrained(output_folder)

    print('Model downloaded and saved to:', output_folder)
    
    model.eval()

    model.hf_device_map

    pipe = pipeline(
    task = "text-generation",
    model = model,
    tokenizer = tokenizer,
    pad_token_id = tokenizer.eos_token_id,
    #        do_sample = True,
    max_length = max_len,
    temperature = CFG.temperature,
    top_p = CFG.top_p,
    repetition_penalty = CFG.repetition_penalty
    )

    ### langchain pipeline
    llm = HuggingFacePipeline(pipeline = pipe)
    llm
    query = "Give me 5 examples of shards"
    llm.invoke(query)
    CFG.model_name
    

    loader = DirectoryLoader(
    CFG.PDFs_path,
    glob="./*.pdf",
    loader_cls=PyPDFLoader,
    show_progress=True,
    use_multithreading=True
    )
    documents = loader.load()
    

    text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = CFG.split_chunk_size,
    chunk_overlap = CFG.split_overlap
    )

    texts = text_splitter.split_documents(documents)

    print(f'We have created {len(texts)} chunks from {len(documents)} pages')

    ### we create the embeddings only if they do not exist yet
    if not os.path.exists(CFG.Output_folder_faiss + '\\index.faiss'):

        ### download embeddings model
        embeddings = HuggingFaceInstructEmbeddings(
            model_name = CFG.embeddings_model_repo,
            model_kwargs = {"device": "cuda"}
        )
        

        ### create embeddings and DB
        vectordb = FAISS.from_documents(
            documents = texts,
            embedding = embeddings
        )

        ### persist vector database
        vectordb.save_local(CFG.Output_folder_faiss) # save in output folder
        #     vectordb.save_local(f"{CFG.Embeddings_path}/faiss_index_hp") # save in input folder
        
    ### download embeddings model
    embeddings = HuggingFaceInstructEmbeddings(
        model_name = CFG.embeddings_model_repo,
        model_kwargs = {"device": "cuda"}
    )
        

    ### load vector DB embeddings
    vectordb = FAISS.load_local(
        #CFG.Embeddings_path, # from input folder
            CFG.Output_folder_faiss, # from output folder
        embeddings,
        allow_dangerous_deserialization=True
    )

    vectordb.similarity_search('magic creatures')

    prompt_template = """
    Don't try to make up an answer, if you don't know just say that you don't know.
    Answer in the same language the question was asked.
    Use only the following pieces of context to answer the question at the end.

    {context}

    Question: {question}
    Answer:"""


    PROMPT = PromptTemplate(
        template = prompt_template,
        input_variables = ["context", "question"]
    )

    retriever = vectordb.as_retriever(search_kwargs = {"k": CFG.k, "search_type" : "similarity"})

    qa_chain = RetrievalQA.from_chain_type(
        llm = llm,
        chain_type = "stuff", # map_reduce, map_rerank, stuff, refine
        retriever = retriever,
        chain_type_kwargs = {"prompt": PROMPT},
        return_source_documents = True,
        verbose = False
    )

    question = "When Kaladin meet Adolin?"
    vectordb.max_marginal_relevance_search(question, k = CFG.k)

    question = "When Adolin meet Kaladin"
    vectordb.similarity_search(question, k = CFG.k)

    query = "When Kaladin meet Syl?"
    print(llm_ans(query, qa_chain))

if __name__ == "__main__":
    main()