


import os 
import json 
import re 
import torch 
import torch .nn as nn 
import numpy as np 
from loguru import logger 
import time 
import uuid 
from typing import List ,Dict ,Any ,Optional 
from transformers import AutoTokenizer ,AutoModelForCausalLM ,AutoConfig 

from .base import ModelEngine ,ModelConfig ,GenerationRequest ,GenerationResponse 
from models .MAR .MasRouter import MasRouter 
from models .MAR .LLM .llm_profile import llm_profile 
from models .MAR .Agent .reasoning_profile import reasoning_profile 
from models .MAR .Prompts .tasks_profile import tasks_profile 
from models .MAR .Utils .log import configure_logging 



from models .MAR .Utils .globals import Cost ,PromptTokens ,CompletionTokens 

TOOL_ROLES =['BugFixer','PlanSolver','ProgrammingExpert','ReflectProgrammer','TestAnalyst','ReflectProgrammer','WikiSearcher','SoftwareDeveloper']


class MASRouterEngine (ModelEngine ):


    def __init__ (self ,config :ModelConfig ):
        super ().__init__ (config )


        self .router =None 
        self .device =torch .device ("cuda"if torch .cuda .is_available ()else "cpu")
        self .is_ready =False 
        self .llms =llm_profile 
        self .tasks =tasks_profile 
        self .reasonings =reasoning_profile 


        self .training_config =None 
        self .agent_models =[]
        self .training_max_tokens =None 
        self .debug =config .config .get ('debug',False )


        self .open_agents =config .config .get ('open_agents',{})
        self .ports =self ._load_ports (self .open_agents )
        self .closed_agents =config .config .get ('closed_agents',{})
        self .server_host =config .config .get ('server_host','localhost')
        self .exclude_tools =config .config .get ('exclude_tools',True )
        self .device =torch .device ("cuda"if torch .cuda .is_available ()else "cpu")
        self .agent_models =list (self .open_agents .keys ())


        self .log_folder =self ._create_log_folder ()
        log_path =os .path .join (self .log_folder ,config .config .get ('log_file','mas_router_engine.log'))
        configure_logging (log_name =log_path )

    def _load_ports (self ,open_agents :Dict [str ,Dict [str ,Any ]])->Dict [str ,int ]:

        ports ={}
        for agent ,details in open_agents .items ():
            if 'port'in details :
                ports [agent ]=details ['port']
            else :
                logger .warning (f"Agent {agent} does not have a port specified")
        return ports 

    async def load (self )->None :

        model_path =self .config .config ['model_path']

        logger .info (f"Loading router engine from {model_path}")





        await self ._load_model_parameters (model_path )

        self .is_ready =True 
        logger .info ("Router engine loaded successfully")

    async def unload (self )->None :

        if self .router :
            del self .router 
            self .router =None 

        torch .cuda .empty_cache ()
        self .is_ready =False 
        logger .info ("Router engine unloaded")

    async def generate (self ,request :GenerationRequest )->GenerationResponse :




        max_tokens =request .max_tokens 
        if max_tokens is None and self .training_max_tokens is not None :
            max_tokens =self .training_max_tokens 
        elif max_tokens is None :
            max_tokens =1024 

        logger .info (f"Starting coordination with {len(request.messages)} messages")

        queries =[res ["content"]for res in request .messages if res ["role"]=="user"]
        task_labels =[0 ]
        tasks_y =torch .tensor (task_labels ).to (self .device )

        results ,costs ,log_probs ,tasks_probs =await self .router .forward (
        queries ,self .tasks ,self .llms ,self .reasonings ,task_labels ,temperature =request .temperature or 0.1 )
        result =results [0 ]

        prompt_tokens =PromptTokens .instance ().value 
        completion_tokens =CompletionTokens .instance ().value 

        return GenerationResponse (
        content =result ,
        usage ={
        "prompt_tokens":prompt_tokens ,
        "completion_tokens":completion_tokens ,
        "total_tokens":prompt_tokens +completion_tokens 
        },
        metadata ={
        "costs":costs ,
        }
        )

    async def health_check (self )->Dict [str ,Any ]:

        return {
        "status":"ready"if self .is_ready else "not_ready",
        "engine_type":"custom",
        "agent_models":self .agent_models ,
        "training_max_tokens":self .training_max_tokens ,
        "cuda_available":torch .cuda .is_available (),
        "device_count":torch .cuda .device_count ()if torch .cuda .is_available ()else 0 
        }

    def list_available_models (self )->List [str ]:

        return ["masrouter-model"]

    async def _load_model_parameters (self ,model_path :str )->None :

        device =torch .device ('cuda'if torch .cuda .is_available ()else 'cpu')
        self .router =MasRouter (
        llm_names =self .agent_models ,
        servers =self .server_host ,
        ports =self .ports ,
        exclude_roles =['BugFixer','PlanSolver','ProgrammingExpert','ReflectProgrammer','TestAnalyst','ReflectProgrammer','WikiSearcher','SoftwareDeveloper']if self .exclude_tools else [],
        ).to (device )

        self .router .load_state_dict (torch .load (model_path ,map_location =device ))
        self .router .eval ()

    def _extract_user_content (self ,messages :List [Dict [str ,str ]])->tuple [str ,str ]:

        user_content =""
        conversation_context =""

        for i ,msg in enumerate (messages ):
            if msg ["role"]=="user":
                if i ==0 :
                    user_content =msg ["content"]
                else :
                    conversation_context +=f"\n\nUser: {msg['content']}"

        return user_content ,conversation_context 

    def _create_log_folder (self )->str :

        timestamp =int (time .time ())
        unique_id =str (uuid .uuid4 ())[:8 ]
        folder_name =f"router_engine_logs_{timestamp}_{unique_id}"
        log_path =os .path .join ("logs","server_logs",folder_name )


        os .makedirs (log_path ,exist_ok =True )
        logger .info (f"Created logging folder: {log_path}")

        return log_path 