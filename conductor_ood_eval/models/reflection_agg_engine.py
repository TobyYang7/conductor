


import logging 
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
from utils .llm_clients import query_locally_hosted_model ,query_oai ,query_anthropic ,query_gemini ,query_deepseek 

from .base import ModelEngine ,ModelConfig ,GenerationRequest ,GenerationResponse 

logger =logging .getLogger (__name__ )

REFLECTION_PROMPT ="""Verify the previous answer, ensure the conditions of the questions have been followed, and provide a final answer.
If the previous answer is correct, simply repeat it. If it is incorrect, propose an alternative answer.
"""

AGGREGATION_SYS_PROMPT ="You will be given a math problem, analysis and code from other agents. Please find the most reliable answer based on the analysis and results of other agents. Give reasons for making decisions."
AGGREGATION_PROMPT ="Please provide the final answer based on the analysis and results of other agents. Follow the formatting instructions given by the question carefully."

class ReflectionAggEngine (ModelEngine ):


    def __init__ (self ,config :ModelConfig ):
        super ().__init__ (config )


        self .router =None 
        self .device =torch .device ("cuda"if torch .cuda .is_available ()else "cpu")
        self .is_ready =False 


        self .training_config =None 
        self .agent_models =config .config .get ('agent_models',[])
        self .final_model =config .config .get ('final_model',self .agent_models [0 ]if self .agent_models else None )
        self .training_max_tokens =None 
        self .debug =config .config .get ('debug',False )


        self .open_agents =config .config .get ('open_agents',{})
        self .ports =self ._load_ports (self .open_agents )
        self .closed_agents =config .config .get ('closed_agents',{})
        self .server_host =config .config .get ('server_host','localhost')
        self .exclude_tools =config .config .get ('exclude_tools',True )
        self .device =torch .device ("cuda"if torch .cuda .is_available ()else "cpu")
        self .agent_models =list (self .open_agents .keys ())
        self .together =config .config .get ('together',False )


        self .log_folder =self ._create_log_folder ()
        log_path =os .path .join (self .log_folder ,config .config .get ('log_file','mas_router_engine.log'))

    def _load_ports (self ,open_agents :Dict [str ,Dict [str ,Any ]])->Dict [str ,int ]:

        ports ={}
        for agent ,details in open_agents .items ():
            if 'port'in details :
                ports [agent ]=details ['port']
            else :
                logger .warning (f"Agent {agent} does not have a port specified")
        return ports 

    async def load (self )->None :

        self .reflection_prompt =REFLECTION_PROMPT 
        self .is_ready =True 
        logger .info ("Reflection (no parameters necessary) engine loaded successfully")

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

        aggregate_results =[]
        for model_name in self .agent_models :
            q1_res =await self ._query_agent (
            agent_model =model_name ,
            messages =request .messages ,
            temperature =request .temperature or 0.1 ,
            max_tokens =max_tokens 
            )

            q1_res =[q1_res ]

            refl_res =await self ._run_reflect (
            model_name ,
            request .messages [0 ]["content"],
            prev_response =q1_res ,
            temperature =request .temperature or 0.1 ,
            max_tokens =max_tokens ,
            debug =self .debug ,
            )

            aggregate_results .append (refl_res )


        agg_res =await self ._run_agg (
        model =self .final_model ,
        prompt =request .messages [0 ]["content"],
        prev_response =aggregate_results ,
        temperature =request .temperature or 0.1 ,
        )

        prompt_tokens =sum (len (msg ["content"].split ())for msg in request .messages )
        completion_tokens =len (agg_res .split ())

        return GenerationResponse (
        content =agg_res ,
        usage ={
        "prompt_tokens":prompt_tokens ,
        "completion_tokens":completion_tokens ,
        "total_tokens":prompt_tokens +completion_tokens 
        },
        metadata ={}
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

    async def _query_agent (self ,agent_model :str ,messages :List [Dict [str ,str ]],
    temperature :float ,max_tokens :int )->str :


        if agent_model in self .open_agents :

            port =self .open_agents [agent_model ].get ('port')


            response =await query_locally_hosted_model (
            agent_model ,messages ,max_tokens ,temperature ,
            self .server_host ,port 
            )
        else :

            if "gpt"in agent_model .lower ():
                response =await query_oai (agent_model ,messages ,max_tokens ,temperature )
            elif "claude"in agent_model .lower ():
                response =await query_anthropic (agent_model ,messages ,max_tokens ,temperature )
            elif "gemini"in agent_model .lower ():
                response =await query_gemini (agent_model ,messages ,max_tokens ,temperature )
            elif "deepseek"in agent_model .lower ():
                together_flag =self .closed_agents .get (agent_model ,{}).get ('together',True )
                response =await query_deepseek (agent_model ,messages ,max_tokens ,temperature ,together_flag )
            else :

                response =""

        return response 

    async def _final_system_prompt (self ,system_prompt ,results ):

        return (
        system_prompt 
        +"\n"
        +"\n".join ([f"{i+1}. {str(element)}"for i ,element in enumerate (results )])
        )

    async def _final_system_agg_prompt (self ,prompt ,results ,final_prompt =AGGREGATION_PROMPT ):

        return (
        prompt 
        +"\n"
        +"\n".join ([f"{i+1}. {str(element)}"for i ,element in enumerate (results )])
        +"\n"
        +final_prompt 
        )

    async def _run_reflect (
    self ,
    model ,
    prompt ,
    prev_response =None ,
    temperature =0.1 ,
    max_tokens =1024 ,
    debug =False ,
    )->str :
        together =self .together 
        messages =(
        [
        {
        "role":"system",
        "content":await self ._final_system_prompt (
        self .reflection_prompt ,prev_response 
        ),
        },
        {"role":"user","content":prompt },
        ]
        if prev_response 
        else [{"role":"user","content":prompt }]
        )

        return await self ._query_agent (
        agent_model =model ,
        messages =messages ,
        temperature =temperature ,
        max_tokens =max_tokens ,
        )

    async def _run_agg (
    self ,
    model ,
    prompt ,
    prev_response =[],
    temperature =0.1 ,
    max_tokens =1024 ,
    debug =False ,
    )->str :
        together =self .together 
        messages =([
        {"role":"user","content":await self ._final_system_agg_prompt (
        prompt ,prev_response ,AGGREGATION_PROMPT 
        )},
        ])

        return await self ._query_agent (
        agent_model =model ,
        messages =messages ,
        temperature =temperature ,
        max_tokens =max_tokens ,
        )


    def _create_log_folder (self )->str :

        timestamp =int (time .time ())
        unique_id =str (uuid .uuid4 ())[:8 ]
        folder_name =f"router_engine_logs_{timestamp}_{unique_id}"
        log_path =os .path .join ("logs","server_logs",folder_name )


        os .makedirs (log_path ,exist_ok =True )
        logger .info (f"Created logging folder: {log_path}")

        return log_path 