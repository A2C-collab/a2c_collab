import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from huggingface_hub import login
import logging

class ModelManager:
    """
    Singleton class to manage LLM models and ensure they're only loaded once.
    Hardcoded to always use Llama 3.1 70B regardless of input parameter.
    """
    _instance = None
    _models = {}
    _tokenizers = {}
    _pipelines = {}
    
    HARDCODED_MODEL = "meta-llama/Meta-Llama-3.1-70B-Instruct"
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self._initialized = True
        # HuggingFace token should be set via HF_TOKEN environment variable
        # For open source, users should set: export HF_TOKEN=your_token_here
        # If HF_TOKEN is not set, login() will prompt for credentials or use cached token
        hf_token = os.environ.get("HF_TOKEN")
        if hf_token:
            login(token=hf_token)
        else:
            login()  # Will use cached token or prompt if needed
        self.set_gpu_memory_config()
        logging.info(f"ModelManager initialized - will use {self.HARDCODED_MODEL}")
    
    def get_model(self, model_id=None, dtype=torch.bfloat16, device_map="auto"):
        """
        Get model, loading it if it doesn't exist. 
        Always returns the hardcoded Llama 3.1 70B model regardless of input.
        
        Args:
            model_id: Ignored, always uses hardcoded model
            dtype: Torch data type for model weights
            device_map: How to distribute model across available devices
        """
        model_id = self.HARDCODED_MODEL
        
        if model_id not in self._models:
            logging.info(f"Loading model {model_id} for the first time")
            
            if torch.cuda.is_available():
                logging.info("Clearing CUDA cache before model loading")
                torch.cuda.empty_cache()
                
                allocated = torch.cuda.memory_allocated() / (1024**3)
                reserved = torch.cuda.memory_reserved() / (1024**3)
                logging.info(f"GPU memory before model loading: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
            
            self._models[model_id] = AutoModelForCausalLM.from_pretrained(
                model_id, 
                torch_dtype=dtype,
                device_map=device_map,
                low_cpu_mem_usage=True,
                offload_folder="offload",
                offload_state_dict=True,
            )
            
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / (1024**3)
                reserved = torch.cuda.memory_reserved() / (1024**3)
                logging.info(f"GPU memory after model loading: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
            
            logging.info(f"Model {model_id} loaded successfully")
        return self._models[model_id]
    
    def get_tokenizer(self, model_id=None):
        """
        Get tokenizer, loading it if it doesn't exist.
        Always returns the tokenizer for the hardcoded Llama 3.1 70B model.
        """
        model_id = self.HARDCODED_MODEL
        
        if model_id not in self._tokenizers:
            logging.info(f"Loading tokenizer for {model_id}")
            self._tokenizers[model_id] = AutoTokenizer.from_pretrained(model_id)
        return self._tokenizers[model_id]
    
    def get_pipeline(self, model_id=None, max_new_tokens=512):
        """
        Get pipeline, creating it if it doesn't exist.
        Always returns the pipeline for the hardcoded Llama 3.1 70B model.
        
        Args:
            model_id: Ignored, always uses hardcoded model
            max_new_tokens: Maximum number of tokens to generate (default: 512)
        """
        model_id = self.HARDCODED_MODEL
        
        if model_id not in self._pipelines:
            logging.info(f"Creating pipeline for {model_id} with max_new_tokens={max_new_tokens}")
            
            if torch.cuda.is_available():
                logging.info("Clearing CUDA cache before creating pipeline")
                torch.cuda.empty_cache()
                
                allocated = torch.cuda.memory_allocated() / (1024**3)
                reserved = torch.cuda.memory_reserved() / (1024**3)
                logging.info(f"GPU memory before pipeline creation: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
            
            model = self.get_model(model_id, device_map="auto")
            tokenizer = self.get_tokenizer(model_id)
            
            self._pipelines[model_id] = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                max_new_tokens=max_new_tokens,
                use_cache=True,
            )
            
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / (1024**3)
                reserved = torch.cuda.memory_reserved() / (1024**3)
                logging.info(f"GPU memory after pipeline creation: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
        
        return self._pipelines[model_id]
    
    def clear_model(self, model_id=None):
        """
        Clear a specific model from memory, or all models if none specified
        """
        if model_id is None:
            self._models = {}
            self._pipelines = {}
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return
            
        if model_id in self._models:
            del self._models[model_id]
            if model_id in self._pipelines:
                del self._pipelines[model_id]
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    def set_gpu_memory_config(self):
        """Set GPU memory configuration to reduce fragmentation"""
        if torch.cuda.is_available():
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
            
            if hasattr(torch.cuda, 'memory_stats'):
                torch.backends.cuda.enable_mem_efficient_sdp(True)
                torch.backends.cuda.enable_flash_sdp(True)
                torch.backends.cudnn.benchmark = True
