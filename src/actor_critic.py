import argparse
import os
import logging
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from huggingface_hub import login
import gc
from prompting.model_manager import ModelManager

PATH_PLAN_INSTRUCTION = """
[Path Plan Instruction]
Each <coord> is a tuple (x,y,z) for gripper location, follow these steps to plan:
1) Decide target location (e.g. an object you want to pick), and your current gripper location.
2) Plan a list of <coord> that move smoothly from current gripper to the target location.
3) The <coord>s must be evenly spaced between start and target.
4) Each <coord> must not collide with other robots, and must stay away from table and objects.  
[How to Incorporate [Environment Feedback] to improve plan]
    If IK fails, propose more feasible step for the gripper to reach. 
    If detected collision, move robot so the gripper and the inhand object stay away from the collided objects. 
    If collision is detected at a Goal Step, choose a different action.
    To make a path more evenly spaced, make distance between pair-wise steps similar.
        e.g. given path [(0.1, 0.2, 0.3), (0.2, 0.2. 0.3), (0.3, 0.4. 0.7)], the distance between steps (0.1, 0.2, 0.3)-(0.2, 0.2. 0.3) is too low, and between (0.2, 0.2. 0.3)-(0.3, 0.4. 0.7) is too high. You can change the path to [(0.1, 0.2, 0.3), (0.15, 0.3. 0.5), (0.3, 0.4. 0.7)] 
    If a plan failed to execute, re-plan to choose more feasible steps in each PATH, or choose different actions.
"""

_framework_instance = None

def get_framework_instance():
    """Get or create the global ActorCriticFramework instance"""
    global _framework_instance
    if _framework_instance is None:
        _framework_instance = ActorCriticFramework()
    return _framework_instance

def generate_retrospection(prompt, max_tokens=256, temperature=0.7):
    """Generate retrospection analysis (backward compatibility function)"""
    framework = get_framework_instance()
    return framework.generate_retrospection(prompt, max_tokens, temperature)

def generate_action_proposals(retrospection_file, max_tokens=200, temperature=0.7):
    """Generate action proposals (backward compatibility function)"""
    framework = get_framework_instance()
    
    with open(retrospection_file, "r") as file:
        retrospection_content = file.read()
    
    return framework.generate_action_proposals(retrospection_content, max_tokens, temperature)

def verify_proposals(retrospection_file, proposals_content, max_tokens=200, temperature=0.5):
    """Verify proposals (backward compatibility function)"""
    framework = get_framework_instance()
    
    if isinstance(retrospection_file, str) and os.path.isfile(retrospection_file):
        with open(retrospection_file, "r") as file:
            retrospection_content = file.read()
    else:
        retrospection_content = retrospection_file
    
    return framework.verify_proposals(retrospection_content, proposals_content, max_tokens, temperature)

class ActorCriticFramework:
    """
    Implements a fully offline Actor-Critic framework for robotics planning:
    - Actor: Generates retrospection analysis (DeepSeek Coder 1.3B)
    - Critic: Verifies and refines proposals (Gemma 2 2B)
    """
    
    def __init__(self, 
                 actor_model_id="deepseek-ai/deepseek-coder-1.3b-instruct", 
                 critic_model_id="google/gemma-2-2b-it",
                 device_map="auto"):
        self.actor_model_id = actor_model_id
        self.critic_model_id = critic_model_id
        self.device_map = device_map
        
        self.actor_system_prompt = "Please provide a short retrospective on why it failed when interacting with the environment. Only include high level essential information. Use the following format: \"Agent [change]\" No need to do detailed recommendation:"
        self.critic_system_prompt = "You are the verifier of the system. Describe the agent's next state after receiving feedback from the action proposer, relative to the goal position. Use the following format:\n\"AGENT is [CHANGE] and is closer to [NAME OF OBJECT].\"\nOnly include essential information. Do not exceed 100 words."
        
        self._set_memory_optimizations()
        
        logging.info(f"Initializing Actor-Critic Framework with Actor: {actor_model_id}, Critic: {critic_model_id}")
        
        self.model_manager = ModelManager()
        
        self._initialize_actor_model()
        self._initialize_critic_model()
        
        logging.info("Actor-Critic Framework initialization complete")
    
    def _set_memory_optimizations(self):
        """Set memory optimization configurations"""
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        
        if hasattr(torch.cuda, 'memory_stats'):
            torch.backends.cuda.enable_mem_efficient_sdp(True)
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cudnn.benchmark = True
    
    def _initialize_actor_model(self):
        """Initialize the DeepSeek Coder actor model for retrospection generation"""
        logging.info(f"Loading DeepSeek Coder actor model: {self.actor_model_id}")
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
        
        # HuggingFace token should be set via HF_TOKEN environment variable
        # For open source, users should set: export HF_TOKEN=your_token_here
        # If HF_TOKEN is not set, login() will prompt for credentials or use cached token
        hf_token = os.environ.get("HF_TOKEN")
        if hf_token:
            login(token=hf_token)
        else:
            login()  # Will use cached token or prompt if needed
        
        self.actor_tokenizer = AutoTokenizer.from_pretrained(
            self.actor_model_id, 
            trust_remote_code=True
        )
        self.actor_model = AutoModelForCausalLM.from_pretrained(
            self.actor_model_id,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map=self.device_map,
            low_cpu_mem_usage=True,
            offload_folder="offload"
        )
        
        logging.info("DeepSeek Coder actor model loaded successfully")
    
    def _initialize_critic_model(self):
        """Initialize the Gemma 2 2B critic model for verification"""
        logging.info(f"Loading Gemma 2 2B critic model: {self.critic_model_id}")
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
        
        self.critic_tokenizer = AutoTokenizer.from_pretrained(self.critic_model_id)
        self.critic_model = AutoModelForCausalLM.from_pretrained(
            self.critic_model_id,
            torch_dtype=torch.bfloat16,
            device_map=self.device_map,
            low_cpu_mem_usage=True,
            offload_folder="offload"
        )
        
        self.critic_pipe = pipeline(
            "text-generation",
            model=self.critic_model,
            tokenizer=self.critic_tokenizer,
        )
        
        logging.info("Gemma 2 2B critic model loaded successfully")
    
    def _get_model_type(self, model_id):
        """Detect model type to use appropriate template"""
        if "gemma" in model_id.lower():
            return "gemma"
        elif "deepseek" in model_id.lower():
            return "deepseek"
        elif "llama" in model_id.lower():
            return "llama"
        else:
            return "unknown"

    def _format_gemma_prompt(self, messages):
        """Format prompt for Gemma models"""
        formatted = ""
        for message in messages:
            if message["role"] == "system":
                formatted += f"<start_of_turn>user\n{message['content']}<end_of_turn>\n"
            elif message["role"] == "user":
                formatted += f"<start_of_turn>user\n{message['content']}<end_of_turn>\n"
        formatted += "<start_of_turn>model\n"
        return formatted

    def _clean_gemma_response(self, raw_output, input_text):
        """Clean Gemma response"""
        if "<start_of_turn>model" in raw_output:
            cleaned = raw_output.split("<start_of_turn>model")[-1]
        else:
            cleaned = raw_output.split(input_text)[-1]
        cleaned = cleaned.replace("<end_of_turn>", "").strip()
        return cleaned

    def _clean_deepseek_response(self, raw_output, input_length):
        """Clean DeepSeek response by removing input tokens"""
        return raw_output[input_length:].strip()
    
    def generate_actor_text(self, prompt, max_tokens=256, temperature=0.7):
        """Generate text using the DeepSeek Coder actor model"""
        messages = [
            {"role": "user", "content": self.actor_system_prompt + "\n\n" + prompt}
        ]
        
        inputs = self.actor_tokenizer.apply_chat_template(
            messages, 
            add_generation_prompt=True, 
            return_tensors="pt"
        ).to(self.actor_model.device)
        
        with torch.no_grad():
            outputs = self.actor_model.generate(
                inputs,
                max_new_tokens=max_tokens,
                do_sample=True if temperature > 0 else False,
                temperature=temperature if temperature > 0 else None,
                top_k=50,
                top_p=0.95,
                num_return_sequences=1,
                eos_token_id=self.actor_tokenizer.eos_token_id,
                pad_token_id=self.actor_tokenizer.eos_token_id
            )
        
        generated_text = self.actor_tokenizer.decode(
            outputs[0][len(inputs[0]):], 
            skip_special_tokens=True
        )
        
        return generated_text.strip()
    
    def generate_critic_text(self, prompt, max_tokens=256, temperature=0.7):
        """Generate text using the Gemma 2 2B critic model"""
        messages = [
            {"role": "system", "content": self.critic_system_prompt},
            {"role": "user", "content": prompt},
        ]
        
        input_text = self._format_gemma_prompt(messages)
        
        outputs = self.critic_pipe(
            input_text,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=0.9,
        )
        
        raw_output = outputs[0]['generated_text']
        cleaned_output = self._clean_gemma_response(raw_output, input_text)
        
        return cleaned_output
    
    def generate_retrospection(self, prompt, max_tokens=256, temperature=0.7):
        """Generate retrospection analysis using the DeepSeek Coder actor model"""
        combined_prompt = PATH_PLAN_INSTRUCTION + "\n\n" + prompt
        logging.info("Generating retrospection with DeepSeek Coder actor model")
        return self.generate_actor_text(combined_prompt, max_tokens, temperature)
    
    def generate_action_proposals(self, retrospection_content, max_tokens=200, temperature=0.7):
        """Generate action proposals based on retrospection using the DeepSeek Coder actor model"""
        action_prompt = f"Based on the following retrospection and previous context conversation, propose specific solution only for parameter change within 100 words:\n\n{retrospection_content}"
        logging.info("Generating action proposals with DeepSeek Coder actor model")
        return self.generate_actor_text(action_prompt, max_tokens, temperature)
    
    def verify_proposals(self, retrospection_content, proposals, max_tokens=200, temperature=0.5):
        """Verify and critique the proposals using the Gemma 2 2B critic model"""
        verification_prompt = f"""
Review the following retrospection and proposed solution:

RETROSPECTION:
{retrospection_content}

PROPOSED SOLUTION:
{proposals}

Evaluate whether the proposed solution effectively addresses the issues identified in the retrospection.
"""
        logging.info("Verifying proposals with Gemma 2 2B critic model")
        return self.generate_critic_text(verification_prompt, max_tokens, temperature)
    
    def process_input(self, input_file, output_dir=None):
        """Process an input file through the full Actor-Critic pipeline"""
        if output_dir is None:
            output_dir = os.path.dirname(input_file)
            if not output_dir:
                output_dir = "."
        
        os.makedirs(output_dir, exist_ok=True)
        
        with open(input_file, "r") as file:
            input_text = file.read()
        
        base_filename = os.path.splitext(os.path.basename(input_file))[0]
        
        try:
            logging.info(f"Generating retrospection for {input_file}")
            retrospection = self.generate_retrospection(input_text)
            
            retrospection_file = os.path.join(output_dir, f"{base_filename}_retrospection.txt")
            with open(retrospection_file, "w") as out_file:
                out_file.write("Input:\n")
                out_file.write(input_text + "\n\n")
                out_file.write("Retrospection:\n")
                out_file.write(retrospection)
            
            logging.info(f"Retrospection saved to {retrospection_file}")
            
            action_proposals = self.generate_action_proposals(retrospection)
            
            verification = self.verify_proposals(retrospection, action_proposals)
            
            final_output_file = os.path.join(output_dir, f"{base_filename}_final.txt")
            with open(final_output_file, "w") as out_file:
                out_file.write("Input:\n")
                out_file.write(input_text + "\n\n")
                out_file.write("Retrospection (DeepSeek Coder Actor):\n")
                out_file.write(retrospection + "\n\n")
                out_file.write("Action Proposals (DeepSeek Coder Actor):\n")
                out_file.write(action_proposals + "\n\n")
                out_file.write("Verification (Gemma 2 2B Critic):\n")
                out_file.write(verification)
            
            logging.info(f"Final report saved to {final_output_file}")
            
            return {
                "input": input_text,
                "retrospection": retrospection,
                "proposals": action_proposals,
                "verification": verification,
                "output_file": final_output_file
            }
            
        except Exception as e:
            logging.error(f"Error processing {input_file}: {e}")
            error_file = os.path.join(output_dir, f"{base_filename}_error.txt")
            with open(error_file, "w") as out_file:
                out_file.write(f"Error processing input: {str(e)}")
            return {"error": str(e)}

    def set_actor_system_prompt(self, prompt):
        """Set a custom system prompt for the DeepSeek Coder actor model"""
        self.actor_system_prompt = prompt
        logging.info(f"Actor system prompt updated: {prompt[:50]}...")
        
    def set_critic_system_prompt(self, prompt):
        """Set a custom system prompt for the Gemma 2 2B critic model"""
        self.critic_system_prompt = prompt
        logging.info(f"Critic system prompt updated: {prompt[:50]}...")
        
    def __del__(self):
        """Clean up resources when the framework is deleted"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
            logging.info("CUDA cache cleared during ActorCriticFramework cleanup")

def main():
    parser = argparse.ArgumentParser(description="Actor-Critic framework for robotics planning")
    parser.add_argument("input_file", help="Path to the input text file")
    parser.add_argument("--output_dir", "-o", help="Directory to save output files", default=None)
    parser.add_argument("--actor_model", "-a", help="Model ID for the actor", default="deepseek-ai/deepseek-coder-1.3b-instruct")
    parser.add_argument("--critic_model", "-c", help="Model ID for the critic", default="google/gemma-2-2b-it")
    parser.add_argument("--temperature", "-t", type=float, default=0.7, help="Temperature for generation")
    parser.add_argument("--debug", "-d", action="store_true", help="Enable debug logging")
    parser.add_argument("--device_map", help="Device map for model loading", default="auto")
    
    args = parser.parse_args()
    
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(level=log_level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    ac_framework = ActorCriticFramework(
        actor_model_id=args.actor_model,
        critic_model_id=args.critic_model,
        device_map=args.device_map
    )
    
    result = ac_framework.process_input(args.input_file, args.output_dir)
    
    if "error" in result:
        logging.error(f"Processing failed: {result['error']}")
    else:
        logging.info(f"Processing complete. Output saved to {result['output_file']}")

if __name__ == "__main__":
    main()