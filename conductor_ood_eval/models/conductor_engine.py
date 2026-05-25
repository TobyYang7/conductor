


import os 
import json 
import re 
import torch 
import torch .nn as nn 
import numpy as np 
import logging 
import asyncio 
import random 
import time 
import uuid 
from typing import List ,Dict ,Any ,Optional ,Tuple 
from transformers import AutoTokenizer ,AutoModelForCausalLM ,AutoConfig 
from utils .conductor_utils import (ROUTING_QUESTION_FORMAT_V0_c ,ROUTING_QUESTION_FORMAT_V0_1 ,
ROUTING_QUESTION_FORMAT_V1_7 ,
ROUTING_QUESTION_FORMAT_V1_7c ,
ROUTING_QUESTION_FORMAT_V2_o ,ROUTING_QUESTION_FORMAT_V2 ,
ROUTING_QUESTION_FORMAT_V1_hybrid ,
ROUTING_QUESTION_FORMAT_V0_c_3closed ,
ROUTING_QUESTION_FORMAT_V1_c_3closed ,
ROUTING_QUESTION_FORMAT_V1_7c2 ,
ROUTING_QUESTION_FORMAT_V1_7c2_2_ood ,
INCEPTION_FORMAT_V0 ,
SIMPLE_INCEPTION_FORMAT_V1 ,
SIMPLE_INCEPTION_FORMAT_V2 ,
SIMPLE_INCEPTION_FORMAT_V2_NOEMPTY ,
SIMPLE_INCEPTION_FORMAT_V1_NO_EMPTY2 ,
SIMPLE_INCEPTION_FORMAT_V1_1 ,
SIMPLE_INCEPTION_FORMAT_V1_1_noempty ,
ROUTING_QUESTION_FORMAT_V1_7c2_2_subtaskablation ,
_extract_model_size ,format_model_metadata ,ConductorEvalManager ,_extract_any )

from .base import ModelEngine ,ModelConfig ,GenerationRequest ,GenerationResponse 
from utils .llm_clients import query_locally_hosted_model ,query_oai ,query_anthropic ,query_gemini ,query_deepseek ,query_gemini_with_thoughts 
from utils .model_utils import ModelParameterApplier 
logger =logging .getLogger (__name__ )

class ConductorEngine (ModelEngine ):


    def __init__ (self ,config :ModelConfig ):
        super ().__init__ (config )


        self .router_model =None 
        self .tokenizer =None 


        self .training_config =None 
        self .agent_models =[]
        self .training_max_tokens =None 
        self .max_turns =config .config .get ('max_turns',5 )
        self .debug =config .config .get ('debug',False )


        self .open_agents =config .config .get ('open_agents',{})
        self .closed_agents =config .config .get ('closed_agents',{})
        self .server_host =config .config .get ('server_host','localhost')
        self .available_models =[]
        self ._set_available_models ()

        self .agent_max_tokens =config .config .get ('agent_max_tokens',1024 )
        self .agent_temperature =config .config .get ('agent_temperature',0.2 )

        self .routing_format =config .config .get ('routing_format','v0_c')
        self .inception_format =config .config .get ('inception_format','v1')
        self .mask_style =config .config .get ('mask_style','names')
        self .clean_final_response =config .config .get ('clean_final_response',True )
        self .include_gemini_thoughts =config .config .get ('include_gemini_thoughts',False )
        self .gemini_thinking_budget =config .config .get ('gemini_thinking_budget',1024 )
        self .set_out_tokens_to_max =config .config .get ('set_out_tokens_to_max',False )
        self .anthropic_platform =config .config .get ('anthropic_platform','bedrock')
        self .use_gen_prefix =config .config .get ('use_gen_prefix',True )
        self .claude_thinking_budget =config .config .get ('claude_thinking_budget',0 )
        self .gpt_reasoning_effort =config .config .get ('gpt_reasoning_effort','minimal')
        self .inception_rounds =config .config .get ('inception_rounds',0 )
        self .no_metadata_return =config .config .get ('no_metadata_return',False )
        self .allow_short_response =config .config .get ('allow_short_response',False )
        self .subtask_ablation =config .config .get ('subtask_ablation',False )
        self .final_agent_knowledge =config .config .get ('final_agent_knowledge',False )
        self .gpt_versbosity =config .config .get ('gpt_versbosity','medium')
        self .crop_final_response =config .config .get ('crop_final_response',False )
        if self .inception_rounds ==0 :
            self .inception_round_id =0 

        self .inception_stats ={
        "total":0 ,
        "direct_return":0 ,
        "per_round":{},
        "agent_calls":{},
        "agent_calls_total":0 
        }


        self .log_folder =self ._create_log_folder ()










    async def load (self )->None :

        model_path =self .config .config ['model_path']

        logger .info (f"Loading Conductor engine from {model_path}")


        await self ._load_training_config (model_path )





        await self ._initialize_router (model_path )




        self .is_ready =True 
        logger .info ("Conductor engine loaded successfully")

    async def unload (self )->None :

        if self .router_model :
            del self .router_model 
            self .router_model =None 

        if self .tokenizer :
            del self .tokenizer 
            self .tokenizer =None 

        torch .cuda .empty_cache ()
        self .is_ready =False 
        logger .info ("Conductor engine unloaded")

    async def _load_training_config (self ,model_path :str )->None :

        log_files =[f for f in os .listdir (model_path )if f .endswith ('_log.json')]


        if log_files :
            log_file =os .path .join (model_path ,log_files [0 ])
            with open (log_file ,'r')as f :
                log_data =json .load (f )

            self .training_config =log_data [0 ]['configs']
            self .agent_models =self .training_config ['llm_names']

            if 'max_tokens'in self .training_config :
                self .training_max_tokens =int (self .training_config ['max_tokens'])

            logger .info (f"Loaded training config: {len(self.agent_models)} agents")
        else :
            hf_config_path =os .path .join (model_path ,"config.json")
            if os .path .exists (hf_config_path ):
                with open (hf_config_path ,'r')as f :
                    hf_config =json .load (f )

                self .training_config =hf_config 
                architecture =hf_config .get ('architectures',[None ])[0 ]

                self .training_max_tokens =1024 
                self .agent_models =self .available_models 

                logger .info (f"Loaded Huggingface config. {architecture} model architecture detected. \n"
                "WARNING: no training_max_tokens or agent_models saved. \n "
                f"Assuming Conductor training max_tokens={self.training_max_tokens} and training agents were {self.agent_models} \n"
                "Ensure setup is consistent across training and inference.")

                if self .set_out_tokens_to_max :
                    logger .info ("Caution: set_out_tokens_to_max is enabled. \n"
                    "This will set the closed models output tokens to their max possible value.\n"
                    "This is 32768 for gpt-4.1, 64000 for claude-4-sonnet, and 65535 for gemini-2.5-pro. \n"
                    "This setting should not be enabled if open models are being used.")


    async def generate (self ,request :GenerationRequest )->GenerationResponse :




        max_tokens =request .max_tokens 
        if max_tokens is None and self .training_max_tokens is not None :
            max_tokens =self .training_max_tokens 
        elif max_tokens is None :
            max_tokens =1024 

        logger .info (f"Starting coordination with {len(request.messages)} messages")

        result =await self ._multi_turn_coordination (
        request .messages ,
        request .temperature or 0.1 ,
        max_tokens 
        )


        if self .inception_rounds >0 :
            base_result =result 
            inception_history =[]
            prev_round_result =base_result 
            for round_id in range (1 ,self .inception_rounds +1 ):
                logger .info (f"Starting inception round {round_id}...")
                result =await self ._multi_turn_coordination (
                request .messages ,
                request .temperature or 0.1 ,
                max_tokens ,
                prev_round_result =prev_round_result ,
                inception_round_id =round_id 
                )
                self .inception_stats ["total"]+=1 
                if result ["metadata"].get ("inception_direct_return"):
                    self .inception_stats ["direct_return"]+=1 
                rate =(self .inception_stats ["direct_return"]/self .inception_stats ["total"])if self .inception_stats ["total"]else 0 
                logger .info (f"Inception direct returns: {self.inception_stats['direct_return']} / {self.inception_stats['total']} | Rate: {rate:.1%}")


                pr =self .inception_stats ["per_round"].setdefault (round_id ,{"total":0 ,"direct_return":0 })
                pr ["total"]+=1 
                if result ["metadata"].get ("inception_direct_return"):
                    pr ["direct_return"]+=1 
                pr_rate =(pr ["direct_return"]/pr ["total"])if pr ["total"]else 0 
                logger .info (f"Inception round {round_id}: direct returns {pr['direct_return']} / {pr['total']} | Rate: {pr_rate:.1%}")


                agents =(result ["metadata"].get ("selected_agents")or [])
                if agents :

                    round_counts ={}
                    for a in agents :
                        round_counts [a ]=round_counts .get (a ,0 )+1 
                    round_total =len (agents )
                    round_parts =[f"{a}: {round_counts[a]/round_total:.1%} ({round_counts[a]}/{round_total})"for a in sorted (round_counts ,key =lambda k :(-round_counts [k ],k ))]
                    logger .info (f"Inception round {round_id} agent usage: "+" | ".join (round_parts ))


                    agent_calls =self .inception_stats ["agent_calls"]
                    for a in agents :
                        agent_calls [a ]=agent_calls .get (a ,0 )+1 
                        self .inception_stats ["agent_calls_total"]+=1 
                    total_agent_calls =self .inception_stats ["agent_calls_total"]
                    if total_agent_calls :
                        parts =[]
                        for a ,c in sorted (agent_calls .items (),key =lambda kv :(-kv [1 ],kv [0 ])):
                            parts .append (f"{a}: {c/total_agent_calls:.1%} ({c}/{total_agent_calls})")
                        logger .info ("Inception agent usage (cumulative): "+" | ".join (parts ))

                inception_history .append (result )
                prev_round_result =result 
                if result ["metadata"].get ("inception_direct_return"):
                    logger .info (f'Exiting inception round {round_id} with direct return')
                    break 


        prompt_tokens =sum (len (msg ["content"].split ())for msg in request .messages )
        completion_tokens =len (result ["response"].split ())


        if self .inception_rounds >0 :
            metadata ={"round_1":base_result ["metadata"],
            "inception_history":inception_history }

            if inception_history :

                metadata ["inception_round_id"]=round_id 
                metadata ["inception_last_metadata"]=inception_history [-1 ]['metadata']
                metadata ["inception_last_result"]=inception_history [-1 ]
        else :
            metadata =result ["metadata"]

        if self .no_metadata_return :
            metadata =None 

        return GenerationResponse (
        content =result ["response"],
        usage ={
        "prompt_tokens":prompt_tokens ,
        "completion_tokens":completion_tokens ,
        "total_tokens":prompt_tokens +completion_tokens 
        },
        metadata =metadata ,
        finish_reason ="stop"
        )

    async def health_check (self )->Dict [str ,Any ]:

        return {
        "status":"ready"if self .is_ready else "not_ready",
        "engine_type":"conductor",
        "agent_models":self .agent_models ,
        "training_max_tokens":self .training_max_tokens ,
        "max_turns":self .max_turns ,
        "cuda_available":torch .cuda .is_available (),
        "device_count":torch .cuda .device_count ()if torch .cuda .is_available ()else 0 
        }

    def list_available_models (self )->List [str ]:

        return ["conductor-model"]

    async def _initialize_router (self ,model_path :str )->None :





        self .router_model =AutoModelForCausalLM .from_pretrained (
        model_path ,
        torch_dtype =torch .bfloat16 ,

        device_map ="cuda:0",
        trust_remote_code =True ,
        )


        self .tokenizer =AutoTokenizer .from_pretrained (model_path ,trust_remote_code =True )


        logger .info ("Router model components initialized")

    async def _multi_turn_coordination (self ,messages :List [Dict [str ,str ]],
    temperature :float ,max_tokens :int ,prev_round_result =None ,
    inception_round_id =0 )->Dict [str ,Any ]:


        call_id =str (uuid .uuid4 ())[:8 ]
        log_file_path =self ._create_coordination_log_file ("multi_turn_coordination",call_id )


        log_data ={
        "call_id":call_id ,
        "function_name":"multi_turn_coordination",
        "timestamp":time .time (),
        "input_parameters":{
        "temperature":temperature ,
        "max_tokens":max_tokens ,
        "num_input_messages":len (messages ),
        },
        "coordination_steps":[],
        "final_result":None ,
        "execution_time":None ,
        "inception_round_id":inception_round_id ,
        }

        start_time =time .time ()


        history ={
        "subtasks":[],
        "model_ids":[],
        "agent_responses":[]
        }

        failure_count =0 




        user_content ,conversation_context =self ._extract_user_content (messages )

        full_user_content =user_content +conversation_context 



        router_system_prompt =self ._get_router_system_prompt ()
        router_question_format =self ._get_router_question_format ()

        available_models_str =self ._get_available_models_str (mask_style =self .mask_style )
        formatted_router_question =router_question_format .format (
        max_number_or_routing_steps =self .max_turns ,
        user_question =full_user_content ,
        available_models =available_models_str ,
        )

        if inception_round_id >0 :
            inception_prompt =self ._get_inception_prompt ()
            prev_round_response =prev_round_result ["response"]
            if self .inception_format =="v2":
                previous_workflow =self ._construct_workflow (prev_round_result ["metadata"])
            else :
                previous_workflow =""

            if self .crop_final_response :
                prev_round_response =self ._crop_final_response (prev_round_response )

            inception_prompt =inception_prompt .format (
            worker_response =prev_round_response ,
            max_number_of_routing_steps =self .max_turns ,
            previous_workflow =previous_workflow ,
            )

            formatted_router_question =formatted_router_question +"\n\n"+inception_prompt 


        training_messages =[
        {"role":"system","content":router_system_prompt },
        {"role":"user","content":formatted_router_question },
        ]



        router_messages =[
        {"role":"system","content":router_system_prompt },
        training_messages [1 ]
        ]

        if self .use_gen_prefix :
            if inception_round_id >0 :
                router_messages .append (
                {"role":'assistant',"content":"Let's think step by step whether or not I need to make any changes to the previous routing strategy. "})
            else :
                router_messages .append (
                {"role":'assistant',"content":"Let's think step by step how to break this user question down into subtasks, the models I'll use to solve my subtasks, and the access list I'll use for determining which agents can see each others' responses. "})


        retry_count =0 
        max_retries =5 
        inception_return_flag =False 
        while retry_count <=max_retries :

            completions =self ._query_router (router_messages ,max_tokens ,temperature =temperature )


            log_data ['router_response']=completions 


            subtasks ,model_ids ,access_list ,parsing_error_dict =self ._parse_completion (completions ,inception_round_id =inception_round_id )


            if inception_round_id >0 :
                if subtasks ==[]and model_ids ==[]and access_list ==[]:
                    inception_return_flag =True 

            if not parsing_error_dict :

                break 
            elif retry_count <max_retries :

                retry_count +=1 
                logger .info (f"Router output parsing error: {parsing_error_dict}")
                logger .info (f"Retrying router query (attempt {retry_count + 1}/{max_retries + 1})")
            else :

                logger .info (f"Router output parsing failed after {max_retries + 1} attempts")
                break 


        step_log ={}


        if not parsing_error_dict :

            history ["subtasks"].extend (subtasks )
            history ["model_ids"].extend (model_ids )



            for idx ,(subtask ,mid ,acc )in enumerate (zip (subtasks ,model_ids ,access_list )):
                agent_messages =self ._prepare_agent_messages (subtask =subtask ,agent_access =acc ,history =history ,
                user_message =full_user_content ,mid =mid ,prev_round_result =prev_round_result ,
                inception_round_id =inception_round_id ,idx =idx )
                agent_name =self .available_models [mid ]
                logger .info (f"Querying agent {agent_name} with subtask: {subtask}")
                time_start =time .time ()
                agent_reply =await self ._query_agent (agent_name ,agent_messages ,max_tokens =self .agent_max_tokens ,
                temperature =self .agent_temperature )

                elapsed =time .time ()-time_start 
                logger .info (f"Agent {agent_name} response time: {elapsed} seconds")

                if not agent_reply or not str (agent_reply ).strip ():
                    failure_count +=1 
                    logger .info (f"Agent {agent_name} response is empty. Incrementing failure count to {failure_count}")


                s =agent_reply or ""
                preview =s .replace ("\n","\\n")
                snippet =preview [:100 ]
                ellipsis ="…"if len (preview )>100 else ""
                logger .info (f"Agent {agent_name} response (first 100 chars, total {len(s)}): {snippet}{ellipsis}")


                history ["agent_responses"].append (agent_reply )

            final_response =""

            if inception_return_flag :

                history ["agent_responses"]=prev_round_result ["metadata"]["agent_responses"]

            for response in reversed (history ["agent_responses"]):
                if response and response .strip ():
                    final_response =response 
                    break 


            step_log ={
            'subtasks':subtasks if not parsing_error_dict else None ,
            'selected_agents':[self .available_models [mid ]for mid in model_ids ]if not parsing_error_dict else None ,
            'access_list':access_list if not parsing_error_dict else None ,
            'agent_responses':history ["agent_responses"]if not parsing_error_dict else None ,
            'total_turns':len (model_ids )if not parsing_error_dict else None ,
            'parsing_error_dict':parsing_error_dict 
            }

            if self .debug :

                logger .info (f"Episode Complete - obtained final response \n"
                f"Final response: {final_response}")
                logger .info (f"Failure count: {failure_count}")
                if inception_round_id >0 :
                    if inception_return_flag :
                        logger .info (f"Inception round completed with direct return")
                    else :
                        logger .info (f"Inception round completed with revised strategy \n")
                        logger .info (f"Revised strategy: {subtasks}, {model_ids}, {access_list} \n")
                        logger .info (f"Inception final response: {final_response} \n")

            if not final_response :
                logger .info (f"Episode Incomplete - agent failure and no final response")



        else :
            final_response =""
            logger .info (f"Episode Incomplete - format check failed \n"
            f"Returning empty string as final response")

        log_data ["coordination_steps"].append (step_log .copy ())
        log_data ['execution_time']=time .time ()-start_time 
        log_data ['final_response']=final_response 
        log_data ['failure_count']=failure_count 



        with open (log_file_path ,'w')as f :
            json .dump (log_data ,f ,indent =2 )


        metadata ={
        'total_turns':len (model_ids )if not parsing_error_dict else None ,
        'subtasks':subtasks if not parsing_error_dict else None ,
        "model_ids":model_ids if not parsing_error_dict else None ,
        'agent_responses':history ["agent_responses"]if not parsing_error_dict else None ,
        'selected_agents':[self .available_models [mid ]for mid in model_ids ]if not parsing_error_dict else None ,
        'access_list':access_list if not parsing_error_dict else None ,
        'parsing_error_dict':parsing_error_dict ,
        'log_file':log_file_path ,
        'failure_count':failure_count 
        }

        if inception_round_id >0 :
            metadata ["inception_direct_return"]=inception_return_flag 

        if self .clean_final_response :
            response =self ._clean_final_response (final_response )
        else :
            response =final_response 

        return {
        "response":response ,
        "metadata":metadata 
        }

    def _query_router (self ,router_messages :List [Dict [str ,str ]],max_tokens :int ,temperature :float )->str :

        completions =ConductorEvalManager .generate (
        self .router_model ,
        self .tokenizer ,
        router_messages ,
        temperature =temperature ,
        max_tokens =max_tokens ,
        inference =True )

        return completions 

    def _set_available_models (self ):

        if self .closed_agents :
            for i ,model in enumerate (self .closed_agents .keys ()):
                logger .info (f"Adding closed agent {model} at index {i}")
                self .available_models .append (model )

        if self .open_agents :
            current_index =len (self .available_models )
            for i ,model in enumerate (self .open_agents .keys ()):
                logger .info (f"Adding open agent {model} at index {current_index + i}")
                self .available_models .append (model )


        lines =", ".join (f"ID: {i} → {name}"
        for i ,name in enumerate (self .available_models ))

        logger .info (
        "Available models: %s.  "
        "Ensure that the model IDs are consistent across training and inference.",
        lines 
        )

    def _crop_final_response (self ,prev_round_response :str )->str :


        if self .tokenizer is not None and isinstance (prev_round_response ,str ):
            try :
                token_ids =self .tokenizer .encode (prev_round_response ,add_special_tokens =False )
                orig_len =len (token_ids )
                if orig_len >4096 :
                    token_ids =token_ids [-4096 :]
                    prev_round_response =self .tokenizer .decode (token_ids ,skip_special_tokens =True )
                    notice =f"...(displaying final {len(token_ids)} tokens of long response) ...\n"
                    logger .info (f"Long final response: cropping to pass to Conductor context")
                    prev_round_response =notice +prev_round_response 
            except Exception :

                orig_len =len (prev_round_response )
                if orig_len >15000 :
                    prev_round_response =prev_round_response [-15000 :]
                    notice ="...(displaying final portion of long response) ...\n"
                    prev_round_response =notice +prev_round_response 

        return prev_round_response 

    def _construct_workflow (self ,metadata :Dict [str ,Any ])->str :

        previous_workflow =""
        if not metadata or not isinstance (metadata ,dict ):
            return previous_workflow 

        prev_subtasks =metadata .get ("subtasks")or []
        prev_agent_responses =metadata .get ("agent_responses")or []
        prev_model_ids =metadata .get ("model_ids")or []

        try :

            final_idx =None 
            for i in range (len (prev_agent_responses )-1 ,-1 ,-1 ):
                rr =(prev_agent_responses [i ]or "").strip ()
                if rr :
                    final_idx =i 
                    break 

            for i ,(subtask ,agent_response ,model_id )in enumerate (zip (prev_subtasks ,prev_agent_responses ,prev_model_ids )):
                s =subtask or ""
                r =(agent_response or "").strip ()
                previous_workflow +=f"\n<Previous round subtask assigned to Agent {model_id}>{s}</Previous round subtask assigned to Agent {model_id}>\n"
                if r :
                    prefix ="FINAL RESPONSE OBTAINED: "if (final_idx is not None and i ==final_idx )else ""
                    previous_workflow +=f"\n<Previous round Agent {model_id} response>{prefix}{r}</Previous round Agent {model_id} response>\n"
                else :
                    previous_workflow +=f"\n<Previous round Agent {model_id} response></Previous round Agent {model_id} response>\n"
        except Exception as e :
            logger .info (f"Error constructing workflow: {e}")
            previous_workflow +="\n[workflow_error]\n"

        return previous_workflow 

    def _get_available_models_str (self ,include_metadata :bool =False ,mask_style :str ="names")->str :

        available_models_strings =[]
        for i ,model in enumerate (self .available_models ):
            if mask_style =="names":
                size =_extract_model_size (model )
                available_model_str =f"Model id {i}: Model size: {size}B parameters"
            elif mask_style =="names_and_params":
                available_model_str =f"Model id {i}"
            else :
                available_model_str =f"Model id {i}: Organization/Name: {model}"

            if include_metadata :
                available_model_str +="; "+format_model_metadata (model_id =model )
            available_models_strings .append (available_model_str )
        return "\n".join (available_models_strings )

    def _extract_user_content (self ,messages :List [Dict [str ,str ]])->Tuple [str ,str ]:

        user_content =""
        conversation_context =""
        for message in messages :
            if message ["role"]=="user":
                user_content +=message ["content"]

        return user_content ,conversation_context 

    def _get_router_system_prompt (self )->str :


        if self .use_gen_prefix :
            return (f'You are a helpful assistant. You think about the reasoning processing in your mind and how best to coordinate a team of models to solve user queries and provide the user with the response.')
        else :
            return (f"You are a helpful assistant. You first think about the reasoning process in the mind and then provides the user with the answer.")

    def _get_router_question_format (self )->str :

        if self .routing_format =="v0_c":
            return ROUTING_QUESTION_FORMAT_V0_c 
        elif self .routing_format =="v0_1":
            return ROUTING_QUESTION_FORMAT_V0_1 
        elif self .routing_format =="v1_7":
            return ROUTING_QUESTION_FORMAT_V1_7 
        elif self .routing_format =="v1_7c":
            return ROUTING_QUESTION_FORMAT_V1_7c 
        elif self .routing_format =="v1_7c2":
            return ROUTING_QUESTION_FORMAT_V1_7c2 
        elif self .routing_format =="v1_7c2_2_ood":
            return ROUTING_QUESTION_FORMAT_V1_7c2_2_ood 
        elif self .routing_format =="V2_o":
            return ROUTING_QUESTION_FORMAT_V2_o 
        elif self .routing_format =="V2":
            return ROUTING_QUESTION_FORMAT_V2 
        elif self .routing_format =="V1_hybrid":
            return ROUTING_QUESTION_FORMAT_V1_hybrid 
        elif self .routing_format =="v0_c_3closed":
            return ROUTING_QUESTION_FORMAT_V0_c_3closed 
        elif self .routing_format =="v1_c_3closed":
            return ROUTING_QUESTION_FORMAT_V1_c_3closed 
        elif self .routing_format =="v1_7c2_2_subtaskablation":
            return ROUTING_QUESTION_FORMAT_V1_7c2_2_subtaskablation 
        else :
            raise ValueError (f"Invalid routing format: {self.routing_format}")

    def _get_inception_prompt (self )->str :

        if self .inception_format =="v0":
            return INCEPTION_FORMAT_V0 
        elif self .inception_format =="v1":
            return SIMPLE_INCEPTION_FORMAT_V1 
        elif self .inception_format =="v1_noempty2":
            return SIMPLE_INCEPTION_FORMAT_V1_NO_EMPTY2 
        elif self .inception_format =="v2":
            return SIMPLE_INCEPTION_FORMAT_V2 
        elif self .inception_format =="v2_noempty":
            return SIMPLE_INCEPTION_FORMAT_V2_NOEMPTY 
        elif self .inception_format =="v11":
            return SIMPLE_INCEPTION_FORMAT_V1_1 
        elif self .inception_format =="v11_noempty":
            return SIMPLE_INCEPTION_FORMAT_V1_1_noempty 
        else :
            raise ValueError (f"Invalid inception format: {self.inception_format}")

    def _get_collaboration_prompt (self ,user_message :str ,include_overall_task :bool =True ,history =None )->str :


        if include_overall_task :
            collab_prompt =(f"You are tasked with solving a subtask within an overall task. For example you may be asked to solve a math problem using a specific approach, verify the correctness of a given answer, check that a solution meets some criteria, or any other task. "
            f"Focus on solving your subtask. If you need additional context in order to solve your subtask, you may check the overall task for any necessary additional information. "
            f"You may also possibly be shown other related subtasks and other agents' responses to those subtasks, which will be demarcated by <Subtask assigned to Agent ...> and <Agent ... response> tags respectively. "
            f"You may use this information to help you solve your subtask, for instance by reflecting on possible mistakes made by the other agents' attempts or leveraging good ideas in their responses. "
            f"Here is the overall task: <overall_task> {user_message} </overall_task>\n")





            return collab_prompt 








        else :
            return (f"You are required to solve a subtask. For example you may be asked to solve a math problem using a specific approach, verify the correctness of a given answer, check that a solution meets some criteria, or any other task."
            f"Focus on solving your subtask. "
            f"You may also possibly be shown other related subtasks and other agents' responses to those subtasks as context for solving your subtask. This information will be demarcated by <Subtask assigned to Agent ...> and <Agent ... response> tags respectively. "
            )

    def _check_parsing (self ,subtasks ,model_ids ,access_list ,completion )->Dict [str ,List [str ]]:

        parsing_error_dict ={
        "model_id_error":[],
        "access_list_empty_error":[],
        "completion_unparseable_error":[],
        "router_output_length_mismatch_error":[],
        }


        if subtasks ==[]and model_ids ==[]and access_list ==[]:
            parsing_error_dict ["completion_unparseable_error"].append (f"Completion unparseable | Completion: {completion}")

            return parsing_error_dict 


        for id in model_ids :
            if not isinstance (id ,int ):
                parsing_error_dict ["model_id_error"].append (f"Model ID {id} is non-integer (type={type(id)})")
            elif id <0 or id >=len (self .available_models ):
                parsing_error_dict ["model_id_error"].append (f"Model ID {id} is out of range ")


        if not access_list :
            parsing_error_dict ["access_list_empty_error"].append (f"Access list is empty | subtasks: {subtasks}, models: {model_ids}, access_list: {access_list}")


        if len (subtasks )!=len (model_ids )or len (subtasks )!=len (access_list ):
            parsing_error_dict ["router_output_length_mismatch_error"].append (f"Router output length mismatch | Completion: {completion}, \n"
            f"subtasks: {len(subtasks)}, models: {len(model_ids)}, access_list: {len(access_list)}")


        if any (parsing_error_dict .values ()):
            return parsing_error_dict 


        return {}

    def _parse_completion (self ,completions :str ,inception_round_id :int =0 )->Tuple [List [str ],List [int ],List [str ],Dict [str ,List [str ]]]:


        text =completions .replace ("`","")
        text =re .sub (r"^[ \t]*[-*•]\s*","",text ,flags =re .MULTILINE )


        raw_model_ids =_extract_any (text ,["model_id","model id","model ids"])
        subtasks =_extract_any (text ,["subtasks","subtask"])
        access_list =_extract_any (text ,["access_list","access list","access"])


        if inception_round_id >0 :
            if subtasks ==[]and raw_model_ids ==[]and access_list ==[]:
                return subtasks ,raw_model_ids ,access_list ,{}


        model_ids :List [int ]=[]
        for mid in raw_model_ids :
            if isinstance (mid ,int ):
                model_ids .append (mid )
            elif isinstance (mid ,str )and mid .strip ().isdigit ():
                model_ids .append (int (mid .strip ()))
            else :
                logger .info (f"Recieved non-numeric model_id: {mid}")

        if self .subtask_ablation :
            subtasks =['Solve the user question']*len (model_ids )



        parsing_error_dict =self ._check_parsing (subtasks ,model_ids ,access_list ,completions )

        return subtasks ,model_ids ,access_list ,parsing_error_dict 

    def _is_all_access (self ,x )->bool :

        if isinstance (x ,str ):
            return x .strip ().lower ()=="all"
        if isinstance (x ,(list ,tuple ,set )):
            return any (
            isinstance (elem ,str )and elem .strip ().lower ()=="all"
            for elem in x 
            )
        return False 

    def _prepare_agent_messages (self ,subtask :str ,agent_access ,history :Dict [str ,Any ],user_message :str ,
    include_overall_task :bool =True ,mid :int =None ,prev_round_result =None ,
    inception_round_id =0 ,idx =None )->List [Dict [str ,str ]]:



        agent_system_prompt =(
        "You are a helpful assistant. You first think about the reasoning process in the mind and then provide the user with the answer. "
        )

        if self .subtask_ablation :
            user_content =(f"You are tasked with solving a user question. "
            f"Here is the overall task: <overall_task> {user_message} </overall_task>\n")
        else :
            user_content =self ._get_collaboration_prompt (user_message ,include_overall_task ,history )

        if inception_round_id >0 :

            include_past_round =self ._is_all_access (agent_access )
            if include_past_round :
                try :

                    prev_history =prev_round_result ["metadata"]

                    prev_subtasks =prev_history ["subtasks"]
                    prev_model_ids =prev_history ["model_ids"]
                    prev_responses =prev_history ["agent_responses"]

                    user_content +=("This overall task was already attempted by a collection of agents, each with assigned subtasks, in a previous round. Below, you will find the subtasks assigned to those agents "
                    "along with their responses. You may also consult this information in order to solve your subtask and the overall task, for instance by reflecting on possible mistakes "
                    "made by the earlier agent attempts or leveraging good ideas in their responses. "
                    )
                    user_content +=("\n\nHere is the previous round's information:\n\n"
                    "PREVIOUS ROUND SUBTASK ASSIGNMENT AND AGENT RESPONSES\n"

                    )
                    for subtask ,model_id ,response in zip (prev_subtasks ,prev_model_ids ,prev_responses ):
                        user_content +=(
                        f"\n<Previous round subtask assigned to Agent {model_id}>{subtask}\n"
                        f"</Previous round subtask assigned to Agent {model_id}>\n"
                        f"\n<Previous round Agent {model_id} response>{response}</Previous round Agent {model_id} response>\n"
                        )
                except Exception as e :
                    print (f"[WARN] Could not add previous round's information: {e}")



        if not history ['agent_responses']:
            access_indices =[]


        if self ._is_all_access (agent_access ):
            access_indices =range (len (history ["agent_responses"]))
        else :
            if agent_access ==[]:
                access_indices =agent_access 
            else :
                logger .info (f"Invalid agent access: {agent_access}")

        for i in access_indices :
            subtask_i =history ["subtasks"][i ]
            model_id =history ["model_ids"][i ]
            response_i =history ["agent_responses"][i ]

            if response_i is not None :
                agent_output =response_i .strip ()
            else :
                logger .info (f"NoneType response from agent {self.available_models[model_id]}. Setting to empty string and continuing.")
                agent_output =""

            if agent_output :
                user_content +=(
                f"\n<Subtask assigned to Agent {model_id}>{subtask_i}"
                f"</Subtask assigned to Agent {model_id}>"
                f"\n<Agent {model_id} response>{agent_output}</Agent {model_id} response>"
                )
            else :
                logger .info (f"No agent output for subtask {subtask_i} from agent {self.available_models[model_id]}. ")


        if not self .subtask_ablation :

            user_content +=f"\n\nYour assigned subtask: {subtask}\n"

        if self .final_agent_knowledge :
            workflow_length =len (history ['model_ids'])
            if idx ==workflow_length -1 :
                final_agent_knowledge =f"As the final agent in the workflow, your response will be used as the final answer to the overall user question. Hence, after working through your subtask, ensure you return the solved user question in the format specfied in the question. "
                user_content +=final_agent_knowledge 

        assistant_prompt ="Let me solve this step by step."

        if self .claude_thinking_budget >0 :

            return [
            {"role":"system","content":agent_system_prompt },
            {"role":"user","content":user_content },
            ]

        return [
        {"role":"system","content":agent_system_prompt },
        {"role":"user","content":user_content },
        {"role":"assistant","content":assistant_prompt },
        ]

    async def _query_agent (self ,agent_model :str ,messages :List [Dict [str ,str ]],
    temperature :float ,max_tokens :int )->str :

        try :

            if agent_model in self .open_agents :

                port =self .open_agents [agent_model ].get ('port')


                response =await query_locally_hosted_model (
                agent_model ,messages ,max_tokens ,temperature ,
                self .server_host ,port 
                )
            else :

                if "gpt"in agent_model .lower ():
                    max_tokens =128000 if self .set_out_tokens_to_max else max_tokens 
                    response =await query_oai (agent_model ,messages ,max_tokens ,temperature ,reasoning_effort =self .gpt_reasoning_effort )
                elif "claude"in agent_model .lower ():
                    max_tokens =64000 if self .set_out_tokens_to_max else max_tokens 
                    response =await query_anthropic (agent_model ,messages ,max_tokens ,temperature ,platform =self .anthropic_platform ,claude_thinking_budget =self .claude_thinking_budget )
                elif "gemini"in agent_model .lower ():
                    max_tokens =65535 if self .set_out_tokens_to_max else max_tokens 
                    if self .include_gemini_thoughts :
                        response =await query_gemini_with_thoughts (agent_model ,messages ,max_tokens ,temperature ,thinking_budget =self .gemini_thinking_budget )
                    else :
                        response =await query_gemini (agent_model ,messages ,max_tokens ,temperature ,thinking_budget =self .gemini_thinking_budget )
                elif "deepseek"in agent_model .lower ():
                    together_flag =self .closed_agents .get (agent_model ,{}).get ('together',True )
                    response =await query_deepseek (agent_model ,messages ,max_tokens ,temperature ,together_flag )
                else :

                    response =""

            return response 
        except Exception as e :
            logger .error (f"Error querying agent {agent_model}: {e}")
            return ""

    def _create_log_folder (self )->str :

        timestamp =int (time .time ())
        unique_id =str (uuid .uuid4 ())[:8 ]
        folder_name =f"conductor_engine_logs_{timestamp}_{unique_id}"
        log_path =os .path .join ("logs","server_logs",folder_name )


        os .makedirs (log_path ,exist_ok =True )
        logger .info (f"Created logging folder: {log_path}")

        return log_path 

    def _create_coordination_log_file (self ,function_name :str ,call_id :str )->str :

        log_filename =f"{function_name}_{call_id}.json"
        log_file_path =os .path .join (self .log_folder ,log_filename )

        return log_file_path 

    def _clean_final_response (self ,response :str )->str :

        import re 

        cleaned =re .sub (r'<(?:think|idea)>.*?</(?:think|idea)>','',response ,flags =re .DOTALL |re .IGNORECASE )
        cleaned =re .sub (r'</?(?:think|idea)>','',cleaned ,flags =re .IGNORECASE )



        if '<answer>'in cleaned .lower ():
            answer_match =re .search (r'<answer>(.*)',cleaned ,flags =re .DOTALL |re .IGNORECASE )
            if answer_match :
                cleaned =answer_match .group (1 ).strip ()

                cleaned =re .sub (r'</answer>.*','',cleaned ,flags =re .DOTALL |re .IGNORECASE )


        problematic_patterns =[
        r'Based on the information provided:.*?\n',
        r'Other agents \([^)]+\).*?\n',
        r'Agent \d+.*?noted.*?\n',
        r'Given this reasoning.*?\n',
        r'Therefore, my answer is:.*?\n',
        ]

        for pattern in problematic_patterns :
            cleaned =re .sub (pattern ,'',cleaned ,flags =re .DOTALL |re .IGNORECASE )


        cleaned =re .sub (r'\n\s*\n+','\n\n',cleaned )
        cleaned =cleaned .strip ()

        if self .allow_short_response :
            return cleaned 

        if len (cleaned )<10 :
            fallback =re .sub (r'<(?:think|idea)>.*?</(?:think|idea)>','',response ,flags =re .DOTALL |re .IGNORECASE )
            fallback =re .sub (r'</?[^>]+>','',fallback )
            return fallback .strip ()if len (fallback .strip ())>5 else response .strip ()

        return cleaned 





























































