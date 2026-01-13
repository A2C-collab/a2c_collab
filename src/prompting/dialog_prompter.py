import os 
import json
import pickle 
import openai
import numpy as np
import time
from datetime import datetime
from os.path import join
from typing import List, Tuple, Dict, Union, Optional, Any
from dataclasses import dataclass, field

import subprocess
import re
import glob

from rocobench.subtask_plan import LLMPathPlan
from rocobench.rrt_multi_arm import MultiArmRRT
from rocobench.envs import MujocoSimEnv, EnvState 
from .feedback import FeedbackManager
from .parser import LLMResponseParser

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from huggingface_hub import login

import pdb

from prompting.model_manager import ModelManager

PATH_PLAN_INSTRUCTION = """
[Path Plan Instruction]
Each <coord> is a tuple (x,y,z) for gripper location, follow these steps to plan:
1) Decide target location (e.g. an object you want to pick), and your current gripper location.
2) Plan a list of <coord> that move smoothly from current gripper to the target location.
3) The <coord>s must be evenly spaced between start and target.
4) Each <coord> must not collide with other robots, and must stay away from table and objects.  
[How to Incoporate [Enviornment Feedback] to improve plan]
    If IK fails, propose more feasible step for the gripper to reach. 
    If detected collision, move robot so the gripper and the inhand object stay away from the collided objects. 
    If collision is detected at a Goal Step, choose a different action.
    To make a path more evenly spaced, make distance between pair-wise steps similar.
        e.g. given path [(0.1, 0.2, 0.3), (0.2, 0.2. 0.3), (0.3, 0.4. 0.7)], the distance between steps (0.1, 0.2, 0.3)-(0.2, 0.2. 0.3) is too low, and between (0.2, 0.2. 0.3)-(0.3, 0.4. 0.7) is too high. You can change the path to [(0.1, 0.2, 0.3), (0.15, 0.3. 0.5), (0.3, 0.4. 0.7)] 
    If a plan failed to execute, re-plan to choose more feasible steps in each PATH, or choose different actions.
"""


@dataclass
class LLMTimer:
    call_times: List[float] = field(default_factory=list)
    call_tokens: List[int] = field(default_factory=list)
    
    def record(self, elapsed: float, tokens: int = 0):
        self.call_times.append(elapsed)
        self.call_tokens.append(tokens)
        print(f"[LLM TIMER] Call #{len(self.call_times)}: {elapsed:.3f}s, {tokens} tokens")
    
    def summary(self) -> dict:
        if not self.call_times:
            return {"total_llm_time_sec": 0, "num_calls": 0}
        return {
            "total_llm_time_sec": sum(self.call_times),
            "num_calls": len(self.call_times),
            "avg_per_call_sec": sum(self.call_times) / len(self.call_times),
            "min_call_sec": min(self.call_times),
            "max_call_sec": max(self.call_times),
            "total_tokens": sum(self.call_tokens),
            "all_calls": self.call_times,
        }
    
    def save(self, filepath: str):
        """Save LLM timing metrics to a text file (appends)"""
        summary = self.summary()
        total_sec = summary.get("total_llm_time_sec", 0)
        num_calls = summary.get("num_calls", 0)
        avg_sec = summary.get("avg_per_call_sec", 0)
        total_tokens = summary.get("total_tokens", 0)
        
        with open(filepath, 'a') as f:
            f.write(f"  LLM inference: {total_sec:.1f} seconds ({total_sec/60:.1f} minutes), {num_calls} calls, {avg_sec:.2f}s avg/call, {total_tokens} tokens\n")
        
        print(f"[LLM TIMER] Saved timing metrics to {filepath}")
    
    def reset(self):
        """Reset timer for a new run"""
        self.call_times = []
        self.call_tokens = []


class DialogPrompter:
    """
    Each round contains multiple prompts, query LLM once per each agent 
    """
    def __init__(
        self,
        env: MujocoSimEnv,
        parser: LLMResponseParser,
        feedback_manager: FeedbackManager, 
        max_tokens: int = 512,
        debug_mode: bool = False,
        use_waypoints: bool = False,
        robot_name_map: Dict[str, str] = {"panda": "Bob"},
        num_replans: int = 3, 
        max_calls_per_round: int = 10,
        use_history: bool = True,  
        use_feedback: bool = True,
        temperature: float = 0,
        llm_source: str = "gpt-4"
    ):
        self.max_tokens = max_tokens
        self.debug_mode = debug_mode
        self.use_waypoints = use_waypoints
        self.use_history = use_history
        self.use_feedback = use_feedback
        self.robot_name_map = robot_name_map
        self.robot_agent_names = list(robot_name_map.values())
        self.num_replans = num_replans
        self.env = env
        self.feedback_manager = feedback_manager
        self.parser = parser
        self.round_history = []
        self.failed_plans = [] 
        self.latest_chat_history = []
        self.max_calls_per_round = max_calls_per_round 
        self.temperature = temperature
        self.llm_source = llm_source
        
        self.llm_timer = LLMTimer()
        
        self.model_manager = ModelManager()
        
        if "llama" in llm_source.lower():
            self.model_id = "meta-llama/Meta-Llama-3.1-70B-Instruct"
        elif "gemma" in llm_source.lower():
            self.model_id = "google/gemma-2-27b-it"
        else:
            self.model_id = llm_source
        
        self.tokenizer = self.model_manager.get_tokenizer(self.model_id)
        self.model = self.model_manager.get_model(self.model_id)
        self.pipe = self.model_manager.get_pipeline(self.model_id)

    def get_round_history(self):
        if len(self.round_history) == 0:
            return ""
        ret = "[History]\n"
        for i, history in enumerate(self.round_history):
            ret += f"== Round#{i} ==\n{history}\n"
        ret += f"== Current Round ==\n"
        return ret

    def get_llm_timing_summary(self) -> dict:
        """Get the current LLM timing summary"""
        return self.llm_timer.summary()
    
    def save_llm_timing(self, filepath: str):
        """Save LLM timing metrics to a file"""
        self.llm_timer.save(filepath)
    
    def reset_llm_timer(self):
        """Reset LLM timer for a new episode"""
        self.llm_timer.reset()

    def prompt_one_round(self, obs: EnvState, save_path: str = ""): 
        plan_feedbacks = []
        chat_history = [] 
        for i in range(self.num_replans):
            replan_files = []
            
            final_agent, final_response, agent_responses = self.prompt_one_dialog_round(
                obs,
                chat_history,
                plan_feedbacks,
                replan_idx=i,
                save_path=save_path,
            )
            chat_history += agent_responses
            
            for response in agent_responses:
                agent_name = response.split("]:", 1)[0][1:]
                timestamp = datetime.now().strftime("%m%d-%H%M")
                filename = f'{save_path}/replan{i}_call{len(replan_files)}_agent{agent_name}_{timestamp}.json'
                replan_files.append(filename)
            
            parse_succ, parsed_str, llm_plans = self.parser.parse(obs, final_response) 

            curr_feedback = "None"
            if not parse_succ:  
                curr_feedback = f"""
This previous response from [{final_agent}] failed to parse!: '{final_response}'
{parsed_str} Re-format to strictly follow [Action Output Instruction]!"""
                ready_to_execute = False  
            else:
                ready_to_execute = True
                for j, llm_plan in enumerate(llm_plans): 
                    ready_to_execute, env_feedback = self.feedback_manager.give_feedback(llm_plan)        
                    if not ready_to_execute:
                        curr_feedback = env_feedback
                        break
            plan_feedbacks.append(curr_feedback)
            
            tosave = [
                {
                    "sender": "Feedback",
                    "message": curr_feedback,
                },
                {
                    "sender": "Action",
                    "message": (final_response if not parse_succ else llm_plans[0].get_action_desp()),
                },
            ]
            timestamp = datetime.now().strftime("%m%d-%H%M")
            feedback_filename = f'{save_path}/replan{i}_feedback_{timestamp}.json'
            json.dump(tosave, open(feedback_filename, 'w'))
            
            self.save_replan_files(replan_files + [feedback_filename])
            self.run_input_parser(save_path, i)
            
            if ready_to_execute: 
                break  
            else:
                print(curr_feedback)
        
        self.latest_chat_history = chat_history
        return ready_to_execute, llm_plans, plan_feedbacks, chat_history

    def prompt_one_dialog_round(
        self, 
        obs, 
        chat_history, 
        feedback_history, 
        replan_idx=0,
        save_path='data/',
    ):
        """
        keep prompting until an EXECUTE is outputted or max_calls_per_round is reached
        """
        
        agent_responses = []
        usages = []
        dialog_done = False 
        num_responses = {agent_name: 0 for agent_name in self.robot_agent_names}
        n_calls = 0

        while n_calls < self.max_calls_per_round:
            for agent_name in self.robot_agent_names:
                system_prompt = self.compose_system_prompt(
                    obs,
                    agent_name,
                    chat_history=chat_history,
                    current_chat=agent_responses,
                    feedback_history=feedback_history,
                    save_path=save_path
                )
                
                agent_prompt = f"You are {agent_name}, your response is:"
                if n_calls == self.max_calls_per_round - 1:
                    agent_prompt = f"""
        You are {agent_name}, this is the last call, you must end your response by incorporating all previous discussions and output the best plan via EXECUTE. 
        Your response is:
                    """
                response, usage = self.query_once(
                    system_prompt, 
                    user_prompt=agent_prompt, 
                    max_query=3,
                )
                
                tosave = [ 
                    {
                        "sender": "SystemPrompt",
                        "message": system_prompt,
                    },
                    {
                        "sender": "UserPrompt",
                        "message": agent_prompt,
                    },
                    {
                        "sender": agent_name,
                        "message": response,
                    },
                    usage,
                ]
                timestamp = datetime.now().strftime("%m%d-%H%M")
                fname = f'{save_path}/replan{replan_idx}_call{n_calls}_agent{agent_name}_{timestamp}.json'
                json.dump(tosave, open(fname, 'w'))  

                num_responses[agent_name] += 1
                pruned_response = response.strip()
                agent_responses.append(
                    f"[{agent_name}]:\n{pruned_response}"
                )
                usages.append(usage)
                n_calls += 1
                if 'EXECUTE' in response:
                    if replan_idx > 0 or all([v > 0 for v in num_responses.values()]):
                        dialog_done = True
                        break
 
                if self.debug_mode:
                    dialog_done = True
                    break
            
            if dialog_done:
                break
 
        return agent_name, response, agent_responses

    def run_input_parser(self, save_path: str, replan_number: int):
        """Run the input_parser.py script for the current replan to evaluate with the 8B model."""
        run_match = re.search(r'run_(\d+)', save_path)
        step_match = re.search(r'step_(\d+)', save_path)
        
        if not run_match or not step_match:
            print(f"Error: Could not extract run or step number from path: {save_path}")
            return
        
        run_number = run_match.group(1)
        step_number = step_match.group(1)

        current_dir = os.path.dirname(os.path.abspath(__file__))
        input_parser_path = os.path.join(current_dir, '..', 'input_parser.py')
        command = [
            "python",
            input_parser_path,
            "-run", run_number,
            "-step", step_number
        ]

        env = os.environ.copy()
        env["TOKENIZERS_PARALLELISM"] = "false"
        env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

        try:
            result = subprocess.run(command, check=True, capture_output=True, text=True, env=env)
            print(f"Input parser for replan {replan_number} completed successfully.")
            print(result.stdout)
        except subprocess.CalledProcessError as e:
            print(f"Error running input parser for replan {replan_number}:")
            print(e.stderr)

    def save_replan_files(self, filenames):
        """Save all files for the current replan cycle."""
        for filename in filenames:
            if os.path.exists(filename):
                print(f"Saved file: {filename}")
            else:
                print(f"File not found: {filename}")

    def search_and_read_replan_files(self, save_path: str):
        replan_content = ""
        base_path = "data/test/"
        run_dirs = glob.glob(os.path.join(base_path, 'run_*'))
        
        run_dirs = [x for x in run_dirs if re.search(r'run_(\d+)', x)]
        if not run_dirs:
            print(f"No run directories found in {base_path}")
            return replan_content

        run_dirs.sort(key=lambda x: int(re.search(r'run_(\d+)', x).group(1)))
        latest_run_dir = run_dirs[-1]
        print(f"Latest run directory found: {latest_run_dir}")

        step_dirs = glob.glob(os.path.join(latest_run_dir, 'step_*'))
        step_dirs.sort(key=lambda x: int(re.search(r'step_(\d+)', x).group(1)))
        if not step_dirs:
            print(f"No step directories found in {latest_run_dir}")
            return replan_content

        latest_step_dir = step_dirs[-1]
        print(f"Latest step directory found: {latest_step_dir}")

        prompts_dir = os.path.join(latest_step_dir, 'prompts')
        if not os.path.exists(prompts_dir):
            print(f"No 'prompts' directory found in {latest_step_dir}")
            return replan_content

        replan_files = glob.glob(os.path.join(prompts_dir, "replan*_final.txt"))
        replan_files.sort(key=lambda x: int(re.search(r'replan(\d+)', os.path.basename(x)).group(1)))

        if not replan_files:
            print(f"No replan_final.txt files found in the prompts directory: {prompts_dir}")
            return replan_content

        max_replan_number = int(re.search(r'replan(\d+)', os.path.basename(replan_files[-1])).group(1))
        with open(replan_files[-1], 'r') as f:
            replan_content = f.read()

        print(f"Successfully read content from replan_final.txt file(s).")
        return replan_content

    def compose_system_prompt(
        self, 
        obs: EnvState, 
        agent_name: str,
        chat_history: List = [],
        current_chat: List = [],
        feedback_history: List = [],
        save_path: str = ""
    ) -> str:
        action_desp = self.env.get_action_prompt()
        if self.use_waypoints:
            action_desp += PATH_PLAN_INSTRUCTION
        agent_prompt = self.env.get_agent_prompt(obs, agent_name)
        round_history = self.get_round_history() if self.use_history else ""
        
        execute_feedback = ""
        if len(self.failed_plans) > 0:
            execute_feedback = "Plans below failed to execute, improve them to avoid collision and smoothly reach the targets:\n"
            execute_feedback += "\n".join(self.failed_plans) + "\n"
        
        chat_history_str = "[Previous Chat]\n" + "\n".join(chat_history) if len(chat_history) > 0 else ""
        replan_content = self.search_and_read_replan_files(save_path)
        
        system_prompt = f"""You are a responsible coordinator for robotic path planning. Your task is to review the following information and ensure proper coordination:
    
    1. Previous Plans and Replans:
    {replan_content}
    
    2. Action Description and Instructions:
    {action_desp}
    
    3. Round History:
    {round_history}
    
    4. Execution Feedback:
    {execute_feedback}
    
    5. Agent-specific Instructions:
    {agent_prompt}
    
    6. Chat History:
    {chat_history_str}
    """
    
        if self.use_feedback and len(feedback_history) > 0:
            system_prompt += f"\n7. Feedback History:\n" + "\n".join(feedback_history)
    
        if len(current_chat) > 0:
            system_prompt += f"\n8. Current Chat:\n" + "\n".join(current_chat)
    
        system_prompt += "\n\nBased on all the above information, provide your response as the robotic path planning coordinator. Ensure your plans are collision-free, smooth, and achieve the target objectives."
    
        return system_prompt

    def query_once(self, system_prompt, user_prompt, max_query):
        response = None
        usage = None 
        
        if self.debug_mode: 
            response = "EXECUTE\n"
            for aname in self.robot_agent_names:
                action = input(f"Enter action for {aname}:\n")
                response += f"NAME {aname} ACTION {action}\n"
            return response, dict()
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        for n in range(max_query):
            print('querying {}th time'.format(n))
            try:
                prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                terminators = [
                    self.tokenizer.eos_token_id,
                    self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
                ]
                
                llm_start = time.perf_counter()
                
                outputs = self.pipe(
                    prompt,
                    max_new_tokens=self.max_tokens,
                    eos_token_id=terminators,
                    do_sample=True,
                    temperature=0.6,
                    top_p=0.9,
                )
                
                llm_elapsed = time.perf_counter() - llm_start
                
                response = outputs[0]['generated_text'].split(prompt)[-1].strip()
                usage = {"total_tokens": len(self.tokenizer.encode(prompt + response))}
                
                self.llm_timer.record(llm_elapsed, usage.get("total_tokens", 0))
                
                print('======= response ======= \n ', response)
                print('======= usage ======= \n ', usage)
                return response, usage
            except Exception as e:
                print(f"Error: {e}")
                print("Model error, trying again")
        return None, None
    
    def post_execute_update(self, obs_desp: str, execute_success: bool, parsed_plan: str):
        if execute_success: 
            self.failed_plans = []
            chats = "\n".join(self.latest_chat_history)
            self.round_history.append(
                f"[Chat History]\n{chats}\n[Executed Action]\n{parsed_plan}"
            )
        else:
            self.failed_plans.append(parsed_plan)
        return 

    def post_episode_update(self):
        self.round_history = []
        self.failed_plans = [] 
        self.latest_chat_history = []
        
        print("\n" + "="*60)
        print("LLM TIMING SUMMARY FOR THIS EPISODE:")
        summary = self.llm_timer.summary()
        print(json.dumps(summary, indent=2))
        print("="*60 + "\n")