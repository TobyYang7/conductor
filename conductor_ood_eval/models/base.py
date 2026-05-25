


from abc import ABC ,abstractmethod 
from typing import List ,Dict ,Any ,Optional ,AsyncGenerator ,Union 
from dataclasses import dataclass 


@dataclass 
class ModelConfig :

    name :str 
    engine_type :str 
    config :Dict [str ,Any ]


@dataclass 
class GenerationRequest :

    messages :List [Dict [str ,str ]]
    temperature :Optional [float ]=0.1 
    max_tokens :Optional [int ]=None 
    stream :Optional [bool ]=False 
    stop :Optional [Union [str ,List [str ]]]=None 
    model :Optional [str ]=None 


@dataclass 
class GenerationResponse :

    content :str 
    usage :Optional [Dict [str ,int ]]=None 
    metadata :Optional [Dict [str ,Any ]]=None 
    finish_reason :str ="stop"


class ModelEngine (ABC ):


    def __init__ (self ,config :ModelConfig ):
        self .config =config 
        self .is_ready =False 

    @abstractmethod 
    async def load (self )->None :

        pass 

    @abstractmethod 
    async def unload (self )->None :

        pass 

    @abstractmethod 
    async def generate (self ,request :GenerationRequest )->GenerationResponse :

        pass 

    async def generate_stream (self ,request :GenerationRequest )->AsyncGenerator [str ,None ]:

        response =await self .generate (request )
        yield response .content 

    @abstractmethod 
    async def health_check (self )->Dict [str ,Any ]:

        pass 

    @abstractmethod 
    def list_available_models (self )->List [str ]:

        pass 




class EngineManager :


    def __init__ (self ):
        self .engines :Dict [str ,ModelEngine ]={}
        self .default_engine :Optional [str ]=None 

    def register_engine (self ,engine :ModelEngine ,is_default :bool =False )->None :

        self .engines [engine .config .name ]=engine 
        if is_default or len (self .engines )==1 :
            self .default_engine =engine .config .name 

    async def load_all (self )->None :

        for engine in self .engines .values ():
            await engine .load ()

    async def unload_all (self )->None :

        for engine in self .engines .values ():
            await engine .unload ()

    def get_engine (self ,name :Optional [str ]=None )->Optional [ModelEngine ]:

        if name is None :
            name =self .default_engine 
        return self .engines .get (name )if name else None 

    def list_engines (self )->List [str ]:

        return list (self .engines .keys ())

    def list_all_models (self )->List [str ]:

        all_models =[]
        for engine in self .engines .values ():
            all_models .extend (engine .list_available_models ())
        return all_models 

    async def health_check_all (self )->Dict [str ,Dict [str ,Any ]]:

        health_status ={}
        for name ,engine in self .engines .items ():

            health_status [name ]=await engine .health_check ()
        return health_status 