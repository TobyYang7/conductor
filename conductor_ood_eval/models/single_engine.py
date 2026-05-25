import logging 
from typing import List ,Dict ,Any 

from .base import ModelEngine ,ModelConfig ,GenerationRequest ,GenerationResponse 
from utils .llm_clients import (
query_locally_hosted_model ,
query_oai ,
query_anthropic ,
query_bedrock ,
query_gemini ,
query_deepseek 
)


logger =logging .getLogger (__name__ )


class SingleModelEngine (ModelEngine ):


    def __init__ (self ,config :ModelConfig ):
        super ().__init__ (config )

        self .model_name =config .config ['model_name']
        self .model_type =config .config ['model_type']
        self .server_config =config .config .get ('server_config',{})
        self .generation_config =config .config .get ('generation_config',None )

        if self .generation_config :
            logger .warning (f"Generation config (max_token, thinking_budget, temperature) is fixed")
            logger .warning (f"Only Gemini model actually use thinking_budget now")
            max_tokens =self .generation_config ['max_new_token']
            thinking_budget =self .generation_config ['thinking_budget']
            assert thinking_budget <max_tokens ,"Thinking budge not large than max token"


    async def load (self )->None :

        logger .info (f"Loading single model engine: {self.model_name} ({self.model_type})")

        self .is_ready =True 
        logger .info (f"Single model engine loaded: {self.model_name}")

    async def unload (self )->None :

        self .is_ready =False 
        logger .info (f"Single model engine unloaded: {self.model_name}")

    async def generate (self ,request :GenerationRequest )->GenerationResponse :


        max_tokens =request .max_tokens or 1024 
        temperature =request .temperature or 0.1 
        thinking_budget =None 

        if self .generation_config :
            max_tokens =self .generation_config ['max_new_token']
            thinking_budget =self .generation_config ['thinking_budget']
            temperature =self .generation_config ['temperature']
            logger .info (f"Override generation config with max_tokens: {max_tokens}, thinking_budge: {thinking_budget}, temperature: {temperature} ")

        logger .info (f"Generating response with {self.model_name}")

        if self .model_type =="local":
            response =await query_locally_hosted_model (
            self .model_name ,
            request .messages ,
            max_tokens ,
            temperature ,
            self .server_config ['host'],
            self .server_config ['port']
            )
        elif self .model_type =="openai":
            response =await query_oai (self .model_name ,request .messages ,max_tokens ,temperature )
        elif self .model_type =="anthropic":
            response =await query_anthropic (self .model_name ,request .messages ,max_tokens ,temperature )
        elif self .model_type =="bedrock":
            response =await query_bedrock (self .model_name ,request .messages ,max_tokens ,temperature )
        elif self .model_type =="gemini":
            response =await query_gemini (self .model_name ,request .messages ,max_tokens ,temperature ,thinking_budget )
        elif self .model_type =="deepseek":
            together_flag =self .server_config .get ('together',True )
            response =await query_deepseek (self .model_name ,request .messages ,max_tokens ,temperature ,together_flag )
        else :
            response =""

        prompt_tokens =sum (len (msg ["content"].split ())for msg in request .messages )
        completion_tokens =len (response .split ())

        return GenerationResponse (
        content =response ,
        usage ={
        "prompt_tokens":prompt_tokens ,
        "completion_tokens":completion_tokens ,
        "total_tokens":prompt_tokens +completion_tokens 
        },
        metadata ={
        "engine_type":"single",
        "model_name":self .model_name ,
        "model_type":self .model_type 
        },
        finish_reason ="stop"
        )

    async def health_check (self )->Dict [str ,Any ]:

        return {
        "status":"ready"if self .is_ready else "not_ready",
        "engine_type":"single",
        "model_name":self .model_name ,
        "model_type":self .model_type ,
        "config":self .server_config 
        }

    def list_available_models (self )->List [str ]:

        return [self .model_name ]
