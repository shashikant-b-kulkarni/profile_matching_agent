from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.sse import sse_client
from contextlib import AsyncExitStack
import json
import asyncio
import nest_asyncio
from datetime import date
from datetime import datetime
import os
from collections import deque


USERNAME = "kulkarni.mrunal27@gmail.com"
PASSWORD = "Ag94sj"
PROFILE_VERSION = "v1"

#nest_asyncio.apply()

load_dotenv()

class MCP_ChatBot:
    def __init__(self, history_size):
        self.exit_stack = AsyncExitStack()
        # Tools list required for Anthropic API
        self.available_tools = []
        # Prompts list for quick display 
        self.available_prompts = []
        # Sessions dict maps tool/prompt names or resource URIs to MCP client sessions
        self.sessions = {}



    def extract_results_from_response(self, response):
        results = []
        for content in response.content:
            if content.type == 'text':
                type = "text"
                text = content.text
                results.append({
                    "type": type,
                    "text": text
                })
        return results
    
    async def connect_to_remote_server(self):
        try:
            server_url = os.environ.get("MCP_REMOTE_SERVER_URL", "http://0.0.0.0:8000")
            sse_transport = await self.exit_stack.enter_async_context(
                   sse_client(url= f"{server_url}/sse" )
                )
            read, write = sse_transport
            session = await self.exit_stack.enter_async_context(
                ClientSession(read, write)
            )
            await session.initialize()
                
            try:
                # List available tools
                response = await session.list_tools()
                for tool in response.tools:
                    self.sessions[tool.name] = session
                    self.available_tools.append({
                        "name": tool.name,
                        "description": tool.description,
                        "input_schema": tool.inputSchema
                    })
            
                # List available prompts
                prompts_response = await session.list_prompts()
                if prompts_response and prompts_response.prompts:
                    for prompt in prompts_response.prompts:
                        self.sessions[prompt.name] = session
                        self.available_prompts.append({
                            "name": prompt.name,
                            "description": prompt.description,
                            "arguments": prompt.arguments
                        })
                # List available resources
                resources_response = await session.list_resources()
                if resources_response and resources_response.resources:
                    for resource in resources_response.resources:
                        resource_uri = str(resource.uri)
                        self.sessions[resource_uri] = session
            
            except Exception as e:
                print(f"Error {e}")
                
        except Exception as e:
            print(f"Error connecting to {server_name}: {e}") 
            pass
        


    
    async def connect_to_server(self, server_name, server_config):
        try:
            server_params = StdioServerParameters(**server_config)
            stdio_transport = await self.exit_stack.enter_async_context(
                stdio_client(server_params)
            )
            read, write = stdio_transport
            session = await self.exit_stack.enter_async_context(
                ClientSession(read, write)
            )
            await session.initialize()
            
            
            try:
                # List available tools
                response = await session.list_tools()
                for tool in response.tools:
                    self.sessions[tool.name] = session
                    self.available_tools.append({
                        "name": tool.name,
                        "description": tool.description,
                        "input_schema": tool.inputSchema
                    })
            
                # List available prompts
                prompts_response = await session.list_prompts()
                if prompts_response and prompts_response.prompts:
                    for prompt in prompts_response.prompts:
                        self.sessions[prompt.name] = session
                        self.available_prompts.append({
                            "name": prompt.name,
                            "description": prompt.description,
                            "arguments": prompt.arguments
                        })
                # List available resources
                resources_response = await session.list_resources()
                if resources_response and resources_response.resources:
                    for resource in resources_response.resources:
                        resource_uri = str(resource.uri)
                        self.sessions[resource_uri] = session
            
            except Exception as e:
                print(f"Error {e}")
                
        except Exception as e:
            print(f"Error connecting to {server_name}: {e}")

    async def connect_to_servers(self):
        try:
            with open("server_config.json", "r") as file:
                data = json.load(file)
            servers = data.get("mcpServers", {})
            for server_name, server_config in servers.items():
                await self.connect_to_server(server_name, server_config)
        except Exception as e:
            print(f"Error loading server config: {e}")
            raise
    
    # This may not be required here, B'cas we are doing LangGraph on top of it.
    async def process_query(self, query):
        self.clear_history()
        messages = [{'role':'user', 'content':query}]
        self.extend_history(messages[0])
        print(f"\n\n AT START OF USER QUERY: HISTORY:{self.history}\n\n")

        response = self.anthropic.messages.create(max_tokens = 2024,
                                      #model = 'claude-3-7-sonnet-20250219', #deprecated model
                                      model = 'claude-sonnet-4-6',
                                      tools = self.available_tools,
                                      messages = messages)
        process_query = True
        while process_query:
            assistant_content = []
            user_response = []

            for content in response.content:
                if content.type =='text':
                    print(content.text)
                    assistant_content.append(content)
                    if(len(response.content) == 1):
                        process_query= False
                elif content.type == 'tool_use':
                    assistant_content.append(content)
                    #messages.append({'role':'assistant', 'content':assistant_content})
                    tool_id = content.id
                    tool_args = content.input
                    tool_name = content.name


                    print(f"Calling tool {tool_name} with args {tool_args}")

                    # Call a tool
                    #session = self.tool_to_session[tool_name] # new
                    session = self.sessions.get(tool_name)

                    result = await session.call_tool(tool_name, arguments=tool_args)
                    extracted_results = self.extract_results_from_response(result)
                    #print(f"Extracted result from tool call: {extracted_results}")
                    user_response.append({"type": "tool_result",
                                      "tool_use_id": tool_id,
                                      "content": extracted_results})

            #print(f"User response so far: {user_response}")
            #messages.append({'role': 'assistant', 'content': assistant_content})
            self.extend_history({'role': 'assistant', 'content': assistant_content})
            #messages.append({'role': 'user', 'content': user_response}) 
            self.extend_history({'role': 'user', 'content': user_response})       
            if  process_query:
                print(f"\n\n POST LLM RESPONSE HANDLING, BEFORE MAKING NEW LLM Call, HISTORY SO FAR: {self.history}\n\n") 
                response = self.anthropic.messages.create(max_tokens = 2024,
                                    #model = 'claude-3-7-sonnet-20250219', #deprecated model
                                    model = 'claude-sonnet-4-6', 
                                    tools = self.available_tools,
                                    messages = self.history) 
                
    async def get_resource(self, resource_uri):
        session = self.sessions.get(resource_uri)
        
        # Fallback for papers URIs - try any papers resource session
        if not session and resource_uri.startswith("papers://"):
            for uri, sess in self.sessions.items():
                if uri.startswith("papers://"):
                    session = sess
                    break
            
        if not session:
            print(f"Resource '{resource_uri}' not found.")
            return
        
        try:
            result = await session.read_resource(uri=resource_uri)
            if result and result.contents:
                print(f"\nResource: {resource_uri}")
                print("Content:")
                print(result.contents[0].text)
            else:
                print("No content available.")
        except Exception as e:
            print(f"Error: {e}")
    
    async def list_prompts(self):
        """List all available prompts."""
        if not self.available_prompts:
            print("No prompts available.")
            return
        
        print("\nAvailable prompts:")
        for prompt in self.available_prompts:
            print(f"- {prompt['name']}: {prompt['description']}")
            if prompt['arguments']:
                print(f"  Arguments:")
                for arg in prompt['arguments']:
                    arg_name = arg.name if hasattr(arg, 'name') else arg.get('name', '')
                    print(f"    - {arg_name}")

    def upgrade_fetch_url_prompt(self, prompt):
        additional = """\n Once you have successfully fetched the content of the url, output mark down to <domain_name>.md file. Extract the domain name from URL"""
        return prompt + additional
    
    async def execute_prompt(self, prompt_name, args):
        """Execute a prompt with the given arguments."""
        session = self.sessions.get(prompt_name)
        if not session:
            print(f"Prompt '{prompt_name}' not found.")
            return
        
        try:
            result = await session.get_prompt(prompt_name, arguments=args)
            if result and result.messages:
                prompt_content = result.messages[0].content
                
                # Extract text from content (handles different formats)
                if isinstance(prompt_content, str):
                    text = prompt_content
                elif hasattr(prompt_content, 'text'):
                    text = prompt_content.text
                else:
                    # Handle list of content items
                    text = " ".join(item.text if hasattr(item, 'text') else str(item) 
                                  for item in prompt_content)
                
                print(f"\nExecuting prompt '{prompt_name}'...")
                if prompt_name == "fetch":
                    text = self.upgrade_fetch_url_prompt(text)

                print(f"\n\n FULL prompt content::{text}\n\n")
                await self.process_query(text)
        except Exception as e:
            print(f"Error: {e}")

    async def excute_profile_fetch_tool(self, url, excel_format:bool=True, astro_required:bool=True, is_login_profile:bool=True):
        session = self.sessions.get("scrape_profile")
        tool_args = {"url":url, "excel_format": excel_format, "astro_required": astro_required, "is_login_profile":is_login_profile}
        result = await session.call_tool("scrape_profile", arguments=tool_args)
        extracted_results = self.extract_results_from_response(result)
        print(f"Whole result:{result} extracted result:{extracted_results}")
        return extracted_results

    async def multi_profile_fetch_orchestrater(self):
        print("Starting multi_profile_fetch_orchestrater")

        hdr_lst = await self.execute_profile_header_tool()
        """https://www.anuroopwiwaha.com/User/KeywordSearch.aspx?p=2&dFR[casteName][0]=Brahmin&dFR[city][0]=Pune&dFR[city][1]=Mumbai&dFR[city][2]=Thane&dFR[city][3]=Dombivli&dFR[city][4]=Kalyan&dFR[city][5]=Bengaluru%20%28Bangalore%29&dFR[city][6]=Navi%20Mumbai&dFR[city][7]=Pimpri-Chinchwad&dFR[city][8]=Hyderabad&dFR[country][0]=India&dFR[expectCaste][0]=Brahmin&dFR[expectDonotShowOtherCaste][0]=-Brahmin&dFR[martialStatus][0]=Never%20Married&dFR[memberStatus][0]=APP&dFR[motherTongueName][0]=Marathi&dFR[workCity][0]=Pune&dFR[workCity][1]=Mumbai&dFR[workCity][2]=Bengaluru%20%28Bangalore%29&dFR[workCity][3]=Navi%20Mumbai&dFR[workCity][4]=Hyderabad&dFR[workCity][5]=Thane&dFR[workCity][6]=Dombivli&dFR[workCity][7]=Pimpri-Chinchwad&nR[age][%3C=][0]=36&nR[age][%3E=][0]=31&nR[annualIncome][%3E=][0]=1000000&nR[expiryDate][%3E=][0]=1772788697&nR[heightName][%3E=][0]=164&"""
        """ urls = [
            "https://www.anuroopwiwaha.com/user/MemberProfile.aspx?member_id=605877", 
            "https://www.anuroopwiwaha.com/user/MemberProfile.aspx?member_id=272341",
            "https://www.anuroopwiwaha.com/user/MemberProfile.aspx?member_id=764270",
            "https://www.anuroopwiwaha.com/user/MemberProfile.aspx?member_id=671436",
            "https://www.anuroopwiwaha.com/user/MemberProfile.aspx?member_id=541181",
            "https://www.anuroopwiwaha.com/user/MemberProfile.aspx?member_id=350575",
            "https://www.anuroopwiwaha.com/user/MemberProfile.aspx?member_id=410219",
            "https://www.anuroopwiwaha.com/user/MemberProfile.aspx?member_id=630592"

            "https://www.anuroopwiwaha.com/user/MemberProfile.aspx?member_id=340520", Shreyas Adhe , No issues!
            "https://www.anuroopwiwaha.com/user/MemberProfile.aspx?member_id=560408", Vinay Kelkar, got 31 ranking, but 7th lord moon and Graha maitri - enemy?
            "https://www.anuroopwiwaha.com/user/MemberProfile.aspx?member_id=596627", Sachin Parandkar, 
            "https://www.anuroopwiwaha.com/user/MemberProfile.aspx?member_id=796767", Kunal Kavimandan, Bhakoot Dosha/Mangal, otherwise good rank.29
            "https://www.anuroopwiwaha.com/user/MemberProfile.aspx?member_id=672458": Mandar Josh, Lagna Shadashtak + No Lord Friendship - Solved.
            "https://www.anuroopwiwaha.com/user/MemberProfile.aspx?member_id=335627": Nalin Kulkarni Brought out 7th Lord Issue separately, Graha Maitri Separately.
            "https://www.anuroopwiwaha.com/user/MemberProfile.aspx?member_id=662472": Karindikar, Lagna Shadashtak + 7th Lord Issues.
            "https://www.anuroopwiwaha.com/User/MemberProfile.aspx?member_id=724773": Sanket Kulkarni, Low rank, Nadi Dosha, rejected.

            ###### TEST CASES BATCH ###########

                "https://www.anuroopwiwaha.com/user/MemberProfile.aspx?member_id=605877",
                "https://www.anuroopwiwaha.com/user/MemberProfile.aspx?member_id=272341",
                "https://www.anuroopwiwaha.com/user/MemberProfile.aspx?member_id=764270",
                "https://www.anuroopwiwaha.com/user/MemberProfile.aspx?member_id=671436",
                "https://www.anuroopwiwaha.com/user/MemberProfile.aspx?member_id=541181",
                "https://www.anuroopwiwaha.com/user/MemberProfile.aspx?member_id=350575",
                "https://www.anuroopwiwaha.com/user/MemberProfile.aspx?member_id=410219",
                "https://www.anuroopwiwaha.com/user/MemberProfile.aspx?member_id=630592",
                "https://www.anuroopwiwaha.com/user/MemberProfile.aspx?member_id=340520",
                "https://www.anuroopwiwaha.com/user/MemberProfile.aspx?member_id=560408",
                "https://www.anuroopwiwaha.com/user/MemberProfile.aspx?member_id=596627",
                "https://www.anuroopwiwaha.com/user/MemberProfile.aspx?member_id=662472",
                "https://www.anuroopwiwaha.com/user/MemberProfile.aspx?member_id=796767",
                "https://www.anuroopwiwaha.com/user/MemberProfile.aspx?member_id=672458",
                "https://www.anuroopwiwaha.com/user/MemberProfile.aspx?member_id=335627",
                "https://www.anuroopwiwaha.com/user/MemberProfile.aspx?member_id=523216",
                "https://www.anuroopwiwaha.com/user/MemberProfile.aspx?member_id=733459",
                "https://www.anuroopwiwaha.com/user/MemberProfile.aspx?member_id=388878",
                "https://www.anuroopwiwaha.com/user/MemberProfile.aspx?member_id=340890",
                "https://www.anuroopwiwaha.com/user/MemberProfile.aspx?member_id=558586",
                "https://www.anuroopwiwaha.com/user/MemberProfile.aspx?member_id=512069"
                ####### TEST CASES BATCH #######

        """
        """
        "https://www.anuroopwiwaha.com/User/MemberProfile.aspx?member_id=857311&Interest=Incoming"
        """
       
        urls =[

                "https://www.anuroopwiwaha.com/user/MemberProfile.aspx?member_id=605877",
                "https://www.anuroopwiwaha.com/user/MemberProfile.aspx?member_id=272341",
                "https://www.anuroopwiwaha.com/user/MemberProfile.aspx?member_id=764270",
                "https://www.anuroopwiwaha.com/user/MemberProfile.aspx?member_id=671436",
                "https://www.anuroopwiwaha.com/user/MemberProfile.aspx?member_id=541181",
                "https://www.anuroopwiwaha.com/user/MemberProfile.aspx?member_id=350575",
                "https://www.anuroopwiwaha.com/user/MemberProfile.aspx?member_id=410219",
                "https://www.anuroopwiwaha.com/user/MemberProfile.aspx?member_id=630592",
                "https://www.anuroopwiwaha.com/user/MemberProfile.aspx?member_id=340520",
                "https://www.anuroopwiwaha.com/user/MemberProfile.aspx?member_id=560408",
                "https://www.anuroopwiwaha.com/user/MemberProfile.aspx?member_id=596627",
                "https://www.anuroopwiwaha.com/user/MemberProfile.aspx?member_id=662472",
                "https://www.anuroopwiwaha.com/user/MemberProfile.aspx?member_id=796767",
                "https://www.anuroopwiwaha.com/user/MemberProfile.aspx?member_id=672458",
                "https://www.anuroopwiwaha.com/user/MemberProfile.aspx?member_id=335627",
                "https://www.anuroopwiwaha.com/user/MemberProfile.aspx?member_id=523216",
                "https://www.anuroopwiwaha.com/user/MemberProfile.aspx?member_id=733459",
                "https://www.anuroopwiwaha.com/user/MemberProfile.aspx?member_id=388878",
                "https://www.anuroopwiwaha.com/user/MemberProfile.aspx?member_id=340890",
                "https://www.anuroopwiwaha.com/user/MemberProfile.aspx?member_id=558586",
                "https://www.anuroopwiwaha.com/user/MemberProfile.aspx?member_id=512069",
                "https://www.anuroopwiwaha.com/User/MemberProfile.aspx?member_id=857311"
                
               ]
        print("Starting multi_profile_fetch_orchestrater:Login Sequence")

        await self.excute_profile_login()
        profile_data = await self.execute_multi_profile_fetch(urls)

        #O/p the CSV with date in name
        self.write_profile_data_csv(PROFILE_VERSION, "dummy", hdr_lst, profile_data, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), test_run=True)
        #self.write_profile_data_csv(hdr_lst, profile_data)
    
    async def execute_filter_meta(self, filter_name):
        #Look for filter definition JSON file
        try:
            with open(f"profiles_data/filters/{filter_name}.json", "r+") as f:  
                #Get URL
                filter_json = json.load(f)
                listing_url = filter_json["url"]

                #Call tool & get max no of pages and update JSON
                print("Starting with login")
                await self.excute_profile_login()
                print("Login Done")

                print(f"\nStarting Meta fetch of filter page:{listing_url}")
                session = self.sessions.get("scrape_profile_listing_page_metadata")
                result = await session.call_tool("scrape_profile_listing_page_metadata", arguments={"filter_name":filter_name, "profile_data_version":PROFILE_VERSION, "url": listing_url})
                print(f"\n RAW HEADER RESP:{result}\n\n")
                extracted_results = self.extract_results_from_response(result)
                f.seek(0)
                filter_json["max_pages"] = extracted_results[0]['text']
                json.dump(filter_json, f, indent=4)
                f.truncate()

        except Exception as e:
            print("Got exception in execute_filter_meta", e)
    
    def count_log_file_errors(self, log_file_handle, log_file_path):
        count = 0
        try:
            if not log_file_handle:
                log_file_handle = open(log_file_path, "r")

            last_lines = deque(log_file_handle, maxlen=500)
            last_lines_lst = list(last_lines)
            for line in last_lines_lst:
                if "[ERROR]" in line:
                    count +=1
                    print(f"FOUND ERROR {line}")

        except:
            print(f"Could read/open log file:{log_file_path}")

        return log_file_handle, count

    
    async def filter_run_orchestrator(self, filter_name, start_page, end_page):
        #First get the filter details. Check end_page is not going beyond known limits.
        try:
            with open(f"profiles_data/filters/{filter_name}.json", "r") as filter_def_file: 
                filter_json = json.load(filter_def_file)
                listing_url = filter_json["url"]
                max_pages = int(filter_json["max_pages"]) ##TODO CHECK IS IT INT already.
                if start_page > max_pages or end_page > max_pages:
                    print(f"Got our of bound fetch, max_page:{max_pages}, start:{start_page}, end:{end_page}")
                    return
                
                print("Calling Header Tool")
                hdr_lst = await self.execute_profile_header_tool()
                print("Calling Login Tool")
                await self.excute_profile_login()

                #Per page, keep count of how many profiles, errors if any.
                file_time_stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                run_summary = {}
                bulk_profile = []
                scrapped_ids = []
                log_file_handle = None

                #We need to do listing page full fetch in loop from start to end.
                for i in range(start_page, end_page+1):
                    print(f"Starting Filter Fetch for Page No:{i}")
                     
                    profile_data, log_file_path = await self.execute_scrape_listing_page_tool(filter_name, listing_url, i, i==start_page)

                    bulk_profile.extend(profile_data)

                    # Keep monitoring the log for errors/exceptions & signal them?
                    log_file_handle, run_errors = self.count_log_file_errors(log_file_handle, log_file_path)
                    run_summary[f"page_{i}"] = {"profiles_found":len(profile_data), "errors":run_errors}

                #O/p the CSV with date in name
                #Once all data available, then write it into bulk & per member CSV files.
                #Write the member ids into scrapped_ids file
                self.write_profile_data_csv(PROFILE_VERSION, filter_name, hdr_lst, bulk_profile, file_time_stamp)
                #Write Summary of the run. Overall Error and exception statements.
                self.write_filter_run_summary(PROFILE_VERSION, filter_name,file_time_stamp,start_page, end_page, run_summary)

        except Exception as e:
            print("Got exception in filter_run_orchestrator", e)
        finally:
            if log_file_handle:
                log_file_handle.close()


    def get_scrapped_profile_ids_from_file(self, profile_data_version, file_name):
        #Go through all available files in the dir.
        #collect all such ids.

        # Example usage (use a raw string for Windows paths)
        full_path = f'profiles_data/{profile_data_version}/scrapped_ids/{file_name}' 
        member_ids = []

        try:
        
            if ".json" in full_path and os.path.isfile(full_path):
                with open(full_path, 'r') as file:
                    mem_ids = json.load(file)
                    member_ids.extend(mem_ids)

        except FileNotFoundError:
            print(f"Scrapped Ids file {full_path} not found.")

        print(f"Got Scrapped Member Ids: {member_ids}")
        return member_ids

    async def refetch_run_orchestrator(self, filter_name):
            #First get the filter details. Check end_page is not going beyond known limits.
            try:
                with open(f"profiles_data/filters/{filter_name}.json", "r") as filter_def_file: 
                    filter_json = json.load(filter_def_file)
                    #Here should get path to the scrapped id Json files we want to refetch.
                    #For one batch of ids we refetch.
                    scrapped_ids_data_version = filter_json["scrapped_ids_data_version"]
                    profile_data_version = filter_json["profile_data_version"]
                    scrapped_files = filter_json["scrapped_ids_files"]
                    
                    print("Calling Header Tool")
                    hdr_lst = await self.execute_profile_header_tool()
                    print("Calling Login Tool")
                    await self.excute_profile_login()

                    #Per page, keep count of how many profiles, errors if any.
                    file_time_stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                    run_summary = {}
                    bulk_profile = []
                    log_file_handle = None

                    #We need to do listing page full fetch in loop from start to end.
                    for i, file in enumerate(scrapped_files):
                        print(f"Starting REFETCH Filter Interation:{i+1}, {filter_name} for Scrapped Ids file:{file}")
                        expected_ids = self.get_scrapped_profile_ids_from_file(scrapped_ids_data_version,file)
                        print(f"Expected Ids to REFETCH:{expected_ids}")
                        
                        #execute_scrape_listing_page_tool
                        profile_data, log_file_path = await self.execute_refetch_tool(filter_name, file, profile_data_version, i==0)

                        bulk_profile.extend(profile_data)

                        # Keep monitoring the log for errors/exceptions & signal them?
                        log_file_handle, run_errors = self.count_log_file_errors(log_file_handle, log_file_path)

                        #Let's check all the expected ids have been found.
                        expected_set = set(expected_ids)
                        found_ids = [ditem.get("Member Id", "") for ditem in profile_data ]
                        found_set = set(found_ids)
                        all_found = len(expected_set.intersection(found_set)) == len(expected_ids)

                        print(f'Did we find and refetch all expected ids:{"YES" if all_found else "NO"}')
                        if not all_found:
                            print(f'REFECHTED IDS FOR CHECKING:{found_ids}')

                        run_summary[f"file_{i}"] = {"file":file, "profiles_found":len(profile_data), "all_found": all_found, "refeched_ids":found_ids, "errors":run_errors}

                    #O/p the CSV with date in name
                    #Once all data available, then write it into bulk & per member CSV files.
                    #Write the member ids into scrapped_ids file
                    self.write_profile_data_csv(PROFILE_VERSION, filter_name, hdr_lst, bulk_profile, file_time_stamp, True)
                    #Write Summary of the run. Overall Error and exception statements.
                    self.write_filter_run_summary(PROFILE_VERSION, filter_name,file_time_stamp,0, len(scrapped_files), run_summary)

            except Exception as e:
                print("Got exception in refetch_run_orchestrator", e)
            finally:
                if log_file_handle:
                    log_file_handle.close()

    async def reference_run_orchestrator(self, filter_name, start_row, end_row):
            #First get the filter details. Check start and end rows.
            try:
                with open(f"profiles_data/filters/{filter_name}.json", "r") as filter_def_file: 
                    filter_json = json.load(filter_def_file)
                    #Here should get path to the scrapped id Json files we want to refetch.
                    #For one batch of ids we refetch.
                    reference_profile_csv_path = filter_json["reference_profile_csv_path"]
                    profile_data_version = filter_json["profile_data_version"]
                    #We need below as inputs. 
                    #start_row = filter_json["start_row"]
                    #end_row = filter_json["end_row"]

                    
                    print("Calling Header Tool")
                    hdr_lst = await self.execute_profile_header_tool()
                    print("Calling Login Tool")
                    await self.excute_profile_login()

                    #Per page, keep count of how many profiles, errors if any.
                    file_time_stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                    run_summary = {}
                    bulk_profile = []
                    log_file_handle = None

                    #We need to do listing page full fetch in loop from start to end.
                    for i, file in enumerate(reference_profile_csv_path):
                        print(f"Starting REFERENCE Filter Interation:{i+1}, {filter_name} for Profile CSV:{file}")
                        
                        #execute_scrape_listing_page_tool
                        profile_data, log_file_path = await self.execute_reference_profile_tool(filter_name, file, profile_data_version,start_row, end_row, i==0)

                        bulk_profile.extend(profile_data)

                        # Keep monitoring the log for errors/exceptions & signal them?
                        log_file_handle, run_errors = self.count_log_file_errors(log_file_handle, log_file_path)

                        #Let's check if match making data got available for all or not.
                        found_ids = []
                        not_found_ids = []
                        for ditem in profile_data:
                            if ditem["Kundali Brief"] and not ditem["Kundali Brief"] == "Not Found":
                                found_ids.append(ditem["Member Id"])
                            else:
                                not_found_ids.append(ditem["Member Id"])
                        all_found = len(not_found_ids) == 0

                        run_summary[f"file_{i}"] = {"file":file, "profiles_found":len(profile_data), "all_found": all_found, "success Ids":found_ids, "failed Ids":not_found_ids, "errors":run_errors}

                    #O/p the CSV with date in name
                    #Once all data available, then write it into bulk & per member CSV files.
                    #Write the member ids into scrapped_ids file
                    self.write_profile_data_csv(PROFILE_VERSION, filter_name, hdr_lst, bulk_profile, file_time_stamp, True)
                    #Write Summary of the run. Overall Error and exception statements.
                    self.write_filter_run_summary(PROFILE_VERSION, filter_name,file_time_stamp,start_row, end_row, run_summary)

            except Exception as e:
                print("Got exception in refetch_run_orchestrator", e)
            finally:
                if log_file_handle:
                    log_file_handle.close()


    async def profile_listing_page_fetch_orchestrater(self, filter_name, filter_url, page_no, profile_data_version):
        
        # We need to pass this inside now, Playwright trick.
        listing_page_url = filter_url

        hdr_lst = await self.execute_profile_header_tool()
        
        await self.excute_profile_login()
        profile_data,_ = await self.execute_scrape_listing_page_tool(filter_name, listing_page_url, page_no, True)

        #O/p the CSV with date in name
        self.write_profile_data_csv(hdr_lst, profile_data)

    def get_profile_data_item_str(self, item):
        if type(item) == list:
            item_str = '"'
            for single in item:
                single = single.replace(",", "")
                item_str = item_str + single
            item_str += '"'
        elif "," in item:
            item_str = '"' + item + '"'
        else:
            item_str = item

        return item_str
    
    def write_profile_data_csv(self, profile_data_version, filter_name, hdr_lst, data, file_timestamp, test_run:bool=False):
        #Once all data available, then write it into bulk & per member CSV files.
        print("Reached write_profile_data_csv!!")
        filename = f"profiles_data/{profile_data_version}/bulk/{filter_name}/{filter_name}_{file_timestamp}.csv"
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        #print(f"HDR LST:{hdr_lst}")
        #print(f"One Row:{data[0]}") 
        scrapped_member_ids = []

        with open(filename, "w") as bulk_f:
            bulk_f.write("\t".join(hdr_lst))
            bulk_f.write("\n")
            for ditem in data:
                row=[ ditem.get(key, "") for key in hdr_lst ]
                bulk_f.write("\t".join(row))
                bulk_f.write("\n")
                if not test_run:
                    member_id = ditem.get("Member Id", "")
                    if member_id:
                        scrapped_member_ids.append(member_id)
                        member_filename = f"profiles_data/{profile_data_version}/members/{member_id}.csv"
                        os.makedirs(os.path.dirname(member_filename), exist_ok=True)
                        with open(member_filename, "w") as member_f:
                            member_f.write("\t".join(hdr_lst))
                            member_f.write("\n")
                            member_f.write("\t".join(row))
                            member_f.write("\n")

                    else:
                        print("ERROR: MEMBER ID FOUND EMPTY!!!")
                else:
                    pass

        if not test_run and len(scrapped_member_ids) > 0:
            scrapped_ids_file_name = f"profiles_data/{profile_data_version}/scrapped_ids/{filter_name}_{file_timestamp}.json"
            #Write the member ids into scrapped_ids file
            os.makedirs(os.path.dirname(scrapped_ids_file_name), exist_ok=True)
            with open(scrapped_ids_file_name, "w") as scrapped_f:
                json.dump(scrapped_member_ids,scrapped_f)

    def write_filter_run_summary(self, profile_data_version, filter_name,file_time_stamp,start_page, end_page, run_summary):
        print("Reached write_filter_run_summary!!")
        filename = f"profiles_data/{profile_data_version}/bulk/{filter_name}/{filter_name}_runsummary_{start_page}_to_{end_page}_{file_time_stamp}.csv"
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, "w") as f:
            json.dump(run_summary, f, indent=4)


    async def execute_city_lat_long_tool(self, city, country):
        session = self.sessions.get("city_lat_long")
        header = await session.call_tool("city_lat_long", arguments={"city":city,"country":country})
        #print(f"\n RAW HEADER RESP:{header}\n\n")
        header = self.extract_results_from_response(header)
        print(f"\n\n Found city detail:{header}")

    async def execute_profile_header_tool(self):
        session = self.sessions.get("profile_header")
        header = await session.call_tool("profile_header", arguments={})
        #print(f"\n RAW HEADER RESP:{header}\n\n")
        header = self.extract_results_from_response(header)
        #print(f"\n\n EXTRACTED O/P:{header}")
        hdr_lst = [item["text"] for item in header]
        #print(f"\n\n CSV Header={hdr_lst}\n\n")
        return hdr_lst

    async def execute_refetch_tool(self, filter_name, file, profile_data_version, switch_log):
        print(f"Starting REFETCH of file:{file}")
        session = self.sessions.get("scrape_profile_refetch")
        result = await session.call_tool("scrape_profile_refetch", arguments={ "filter_name": filter_name, "profile_data_version":profile_data_version, "file":file, "switch_log":switch_log})
        #print(f"\n RAW HEADER RESP:{result}\n\n")
        extracted_results = self.extract_results_from_response(result)
        profile_data = []
        log_file = ""
        for ext_rslt in extracted_results:
            ext_rslt_item = json.loads(ext_rslt["text"])
            profiles = ext_rslt_item["profiles"]
            log_file = ext_rslt_item["log_file"]
            for profile in profiles:
                if len(profile.keys()) > 0:       
                    profile_data.append(profile)
                                
        #print(f"\n\n PROFILE DATA :{profile_data}\n\n")
        return profile_data, log_file
    
    async def execute_reference_profile_tool(self, filter_name, file, profile_data_version,start_row, end_row, switch_log):
        print(f"execute_reference_profile_tool: Starting REFERENCE FETCH of file:{file}")
        session = self.sessions.get("scrape_reference_profile_csv")
        result = await session.call_tool("scrape_reference_profile_csv", arguments={ "filter_name": filter_name, "profile_data_version":profile_data_version, "file":file, "start_row":start_row, "end_row": end_row, "switch_log":switch_log})
        #print(f"\n RAW HEADER RESP:{result}\n\n")
        extracted_results = self.extract_results_from_response(result)
        profile_data = []
        log_file = ""
        for ext_rslt in extracted_results:
            ext_rslt_item = json.loads(ext_rslt["text"])
            profiles = ext_rslt_item["profiles"]
            log_file = ext_rslt_item["log_file"]
            for profile in profiles:
                if len(profile.keys()) > 0:       
                    profile_data.append(profile)
                                
        #print(f"\n\n PROFILE DATA :{profile_data}\n\n")
        print("execute_reference_profile_tool : Ended")
        return profile_data, log_file

    

    async def execute_scrape_listing_page_tool(self, filter_name, url, page_no, switch_log):
        print(f"Starting fetch of listing page:{url}")
        session = self.sessions.get("scrape_profile_listing_page")
        result = await session.call_tool("scrape_profile_listing_page", arguments={ "filter_name": filter_name, "profile_data_version":PROFILE_VERSION, "url":url, "page_no":page_no, "switch_log":switch_log})
        #print(f"\n RAW HEADER RESP:{result}\n\n")
        extracted_results = self.extract_results_from_response(result)
        profile_data = []
        log_file = ""
        for ext_rslt in extracted_results:
            ext_rslt_item = json.loads(ext_rslt["text"])
            profiles = ext_rslt_item["profiles"]
            log_file = ext_rslt_item["log_file"]
            for profile in profiles:
                if len(profile.keys()) > 0:       
                    profile_data.append(profile)
                                
        #print(f"\n\n PROFILE DATA :{profile_data}\n\n")
        return profile_data, log_file

    async def execute_multi_profile_fetch(self, urls):
        session = self.sessions.get("scrape_many")
        tool_args = {"urls":urls}
        result = await session.call_tool("scrape_many", arguments=tool_args)
        extracted_results = self.extract_results_from_response(result)
        profile_data = []
        for ext_rslt in extracted_results:
            profile = json.loads(ext_rslt["text"])
            if len(profile.keys()) > 0:       
                profile_data.append(profile)
        #print(f"\n\n PROFILE DATA :{profile_data}\n\n")
        return profile_data

    async def excute_profile_login(self):
        session = self.sessions.get("profile_login")
        tool_args = {"username":USERNAME, "password":PASSWORD}
        result = await session.call_tool("profile_login", arguments=tool_args)
        extracted_results = self.extract_results_from_response(result)
        print(f"Whole result:{result} extracted result:{extracted_results}")
        return extracted_results
    
    async def chat_loop(self):
        print("\nMCP Chatbot Started!")
        print("Type your queries or 'quit' to exit.")
        print("Use @folders to see available topics")
        print("Use @<topic> to search papers in that topic")
        print("Use /prompts to list available prompts")
        print("Use /prompt <name> <arg1=value1> to execute a prompt")
        
        while True:
            try:
                query = input("\nQuery: ").strip()
                if not query:
                    continue
        
                if query.lower() == 'quit':
                    break
                
                # Check for @resource syntax first
                if query.startswith('@'):
                    # Remove @ sign  
                    topic = query[1:]
                    if topic == "folders":
                        resource_uri = "papers://folders"
                    else:
                        resource_uri = f"papers://{topic}"
                    await self.get_resource(resource_uri)
                    continue
                
                # Check for /command syntax
                if query.startswith('/'):
                    parts = query.split()
                    command = parts[0].lower()
                    
                    if command == '/prompts':
                        await self.list_prompts()
                    elif command == "/login":
                        await self.excute_profile_login()
                    elif command == "/city": #This is broken now.
                        city = "Peth Vadgoan Kolhapur"
                        country = "India"
                        await self.execute_city_lat_long_tool(city, country)
                    elif command == "/profile":
                        url = parts[1]
                        await self.excute_profile_fetch_tool(url)
                    elif command == "/profiles":
                        await self.multi_profile_fetch_orchestrater()
                    elif command == "/filtermeta":
                        filter_name = parts[1]
                        await self.execute_filter_meta(filter_name)

                    elif command == "/filterrun":
                        if parts[1] == "refetch":
                            refetch_filter = parts[2]
                            await self.refetch_run_orchestrator(refetch_filter)
                        elif parts[1] == "ref":
                            reference_filter = parts[2]
                            start_row = int(parts[3])
                            end_row = int(parts[4])
                            await self.reference_run_orchestrator(reference_filter, start_row, end_row)
                        else:
                            filter_name = parts[1]
                            start_page = int(parts[2])
                            end_page = int(parts[3])

                            await self.filter_run_orchestrator(filter_name, start_page,  end_page)

                    elif command == "/listing":
                        filter_url = "https://www.anuroopwiwaha.com/User/KeywordSearch.aspx?dFR[casteName][0]=Brahmin&dFR[city][0]=Pune&dFR[city][1]=Mumbai&dFR[city][2]=Thane&dFR[city][3]=Dombivli&dFR[city][4]=Kalyan&dFR[city][5]=Bengaluru%20%28Bangalore%29&dFR[city][6]=Navi%20Mumbai&dFR[city][7]=Pimpri-Chinchwad&dFR[city][8]=Hyderabad&dFR[country][0]=India&dFR[expectCaste][0]=Brahmin&dFR[expectDonotShowOtherCaste][0]=-Brahmin&dFR[martialStatus][0]=Never%20Married&dFR[memberStatus][0]=APP&dFR[motherTongueName][0]=Marathi&dFR[workCity][0]=Pune&dFR[workCity][1]=Mumbai&dFR[workCity][2]=Bengaluru%20%28Bangalore%29&dFR[workCity][3]=Navi%20Mumbai&dFR[workCity][4]=Hyderabad&dFR[workCity][5]=Thane&dFR[workCity][6]=Dombivli&dFR[workCity][7]=Pimpri-Chinchwad&nR[age][%3C=][0]=36&nR[age][%3E=][0]=31&nR[annualIncome][%3E=][0]=1000000&nR[expiryDate][%3E=][0]=1772788697&nR[heightName][%3E=][0]=164"
                        await self.profile_listing_page_fetch_orchestrater("india", filter_url, 30, "v1")

                    elif command == '/prompt':
                        if len(parts) < 2:
                            print("Usage: /prompt <name> <arg1=value1> <arg2=value2>")
                            continue
                        
                        prompt_name = parts[1]
                        args = {}
                        
                        # Parse arguments
                        for arg in parts[2:]:
                            if '=' in arg:
                                key, value = arg.split('=', 1)
                                args[key] = value
                        
                        await self.execute_prompt(prompt_name, args)
                    else:
                        print(f"Unknown command: {command}")
                    continue
                
                await self.process_query(query)
                    
            except Exception as e:
                print(f"\nError: {str(e)}")
    
    async def cleanup(self):
        await self.exit_stack.aclose()


async def main():
    chatbot = MCP_ChatBot(15)
    try:
        await chatbot.connect_to_remote_server()
        await chatbot.chat_loop()
    finally:
        await chatbot.cleanup()


if __name__ == "__main__":
    asyncio.run(main())