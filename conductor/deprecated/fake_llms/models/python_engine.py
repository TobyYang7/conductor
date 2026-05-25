import logging 
import re 
import sys 
import io 
import traceback 
from typing import List ,Dict ,Any 
from contextlib import redirect_stdout ,redirect_stderr 

from .base import ModelEngine ,ModelConfig ,GenerationRequest ,GenerationResponse 


logger =logging .getLogger (__name__ )


class PythonEngine (ModelEngine ):


    def __init__ (self ,config :ModelConfig ):
        super ().__init__ (config )

        self .model_name ="python-interpreter"
        self .model_type ="custom"
        self .config =config .config 

    async def load (self )->None :

        logger .info (f"Loading Python engine: {self.model_name} ({self.model_type})")

        self .is_ready =True 
        logger .info (f"Python engine loaded: {self.model_name}")

    async def unload (self )->None :

        self .is_ready =False 
        logger .info (f"Python engine unloaded: {self.model_name}")

    def _extract_python_code (self ,content :str )->str :

        pattern =r"```python\s*(.*?)\s*```"
        match =re .search (pattern ,content ,re .DOTALL )
        if match :
            return match .group (1 ).strip ()
        return content .strip ()

    def _execute_in_sandbox (self ,code :str )->str :


        local_namespace ={}
        global_namespace ={"__builtins__":__builtins__ }

        try :

            stdout_capture =io .StringIO ()
            stderr_capture =io .StringIO ()

            with redirect_stdout (stdout_capture ),redirect_stderr (stderr_capture ):

                exec (code ,global_namespace ,local_namespace )


            stdout_output =stdout_capture .getvalue ()
            stderr_output =stderr_capture .getvalue ()


            response_parts =[]

            if stdout_output :
                response_parts .append (f"{stdout_output.strip()}")

            if stderr_output :
                response_parts .append (f"{stderr_output.strip()}")

            if "ans"in local_namespace :
                response_parts .append (f"Return value: {repr(local_namespace['ans'])}")
            elif not stdout_output and not stderr_output :
                response_parts .append ("Code executed successfully (no return value)")

            return "\n".join (response_parts )

        except Exception as e :
            error_msg =f"Error: {str(e)}\n{traceback.format_exc()}"
            return error_msg 

    async def generate (self ,request :GenerationRequest )->GenerationResponse :

        max_tokens =request .max_tokens or 1024 
        temperature =request .temperature or 0.1 

        logger .info (f"Python engine received request: {request.messages}")


        code =""
        for msg in reversed (request .messages ):
            if msg .get ("role")=="user":
                code =self ._extract_python_code (msg .get ("content",""))
                break 


        response =self ._execute_in_sandbox (code )

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
        "config":self .config 
        }

    def list_available_models (self )->List [str ]:

        return [self .model_name ]