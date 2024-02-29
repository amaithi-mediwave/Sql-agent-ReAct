
from langchain_community.utilities import SQLDatabase
from typing import Any, Dict, List, Optional, Sequence
from langchain_core.pydantic_v1 import BaseModel

from langchain.agents.agent import AgentExecutor
from langchain.agents.agent_toolkits.sql.prompt import (
    SQL_FUNCTIONS_SUFFIX,
    SQL_PREFIX,
    SQL_SUFFIX,
)
from langchain.agents.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain.agents.agent_types import AgentType
from langchain.agents.mrkl.base import ZeroShotAgent
from langchain.agents.mrkl.prompt import FORMAT_INSTRUCTIONS
from langchain.chains.llm import LLMChain
from langchain_community.vectorstores import FAISS
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_core.prompts import (
    ChatPromptTemplate,
    FewShotPromptTemplate,
    PromptTemplate,
)

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain.chat_models import ChatOllama
from langchain_community.llms import Ollama
from langchain.prompts.pipeline import PipelinePromptTemplate 
from langchain.prompts.prompt import PromptTemplate
from langchain.agents import (
    AgentExecutor,
    AgentOutputParser,
    LLMSingleActionAgent,)
from langchain.schema import AgentAction, AgentFinish
from typing import Union
from operator import itemgetter 
import re



db = SQLDatabase.from_uri("sqlite:///Chinook.db")
# print(db.dialect)
# print(db.get_usable_table_names())
# db.run("SELECT * FROM Artist LIMIT 10;")



examples = [
    {"input": "List all artists.", "query": "SELECT * FROM Artist;"},
    {
        "input": "Find all albums for the artist 'AC/DC'.",
        "query": "SELECT * FROM Album WHERE ArtistId = (SELECT ArtistId FROM Artist WHERE Name = 'AC/DC');",
    },
    {
        "input": "List all tracks in the 'Rock' genre.",
        "query": "SELECT * FROM Track WHERE GenreId = (SELECT GenreId FROM Genre WHERE Name = 'Rock');",
    },
    {
        "input": "Find the total duration of all tracks.",
        "query": "SELECT SUM(Milliseconds) FROM Track;",
    },
    {
        "input": "List all customers from Canada.",
        "query": "SELECT * FROM Customer WHERE Country = 'Canada';",
    },
    {
        "input": "How many tracks are there in the album with ID 5?",
        "query": "SELECT COUNT(*) FROM Track WHERE AlbumId = 5;",
    },
    {
        "input": "Find the total number of invoices.",
        "query": "SELECT COUNT(*) FROM Invoice;",
    },
    {
        "input": "List all tracks that are longer than 5 minutes.",
        "query": "SELECT * FROM Track WHERE Milliseconds > 300000;",
    },
    {
        "input": "Who are the top 5 customers by total purchase?",
        "query": "SELECT CustomerId, SUM(Total) AS TotalPurchase FROM Invoice GROUP BY CustomerId ORDER BY TotalPurchase DESC LIMIT 5;",
    },
    {
        "input": "Which albums are from the year 2000?",
        "query": "SELECT * FROM Album WHERE strftime('%Y', ReleaseDate) = '2000';",
    },
    {
        "input": "How many employees are there",
        "query": 'SELECT COUNT(*) FROM "Employee"',
    },
]



example_selector = SemanticSimilarityExampleSelector.from_examples(
    examples,
    GPT4AllEmbeddings(),
    FAISS,
    k=5,
    input_keys=["input"],
)


llm = Ollama(model='mistral:7b-instruct-v0.2-q8_0', temperature=0.2)

toolkit = SQLDatabaseToolkit(
    db = db,
    llm = llm
)

class CustomOutputParser(AgentOutputParser):
    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        # Check if agent should finish
        if "Final Answer:" in llm_output:
            return AgentFinish(
                # Return values is generally always a dictionary with a single `output` key
                # It is not recommended to try anything else at the moment :)
                return_values={"output": llm_output.split("Final Answer:")[-1].strip()},
                log=llm_output,
            )
        # Parse out the action and action input
        regex = r"Action\s*\d*\s*:(.*?)\nAction\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
        match = re.search(regex, llm_output, re.DOTALL)
        
        # if "final answer:" in llm_output:
        #     return AgentFinish(
        #         # Return values is generally always a dictionary with a single `output` key
        #         # It is not recommended to try anything else at the moment :)
        #         return_values={"output": llm_output},
        #         log=llm_output,
        #     )
        
        
        if not match:
            raise ValueError(f"Could not parse LLM output: `{llm_output}`")
        action = match.group(1).strip()
        action = action.replace('\\', '') if '\\' in action else action
        # action = str(action).replace('\\', '')
        action_input = match.group(2)
        # Return the action and action input
        return AgentAction(
            tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output
        )

output_parser = CustomOutputParser()


# PROMPT
tools = toolkit.get_tools()
dialect=str(toolkit.dialect)
top_k=str(10)


prefix_prompt = f"""

You are an agent designed to interact with a SQL database.
Given an input question, create a syntactically correct {dialect} query to run, then look at the results of the query and return the answer.
Unless the user specifies a specific number of examples they wish to obtain, always limit your query to at most {top_k} results.
You can order the results by a relevant column to return the most interesting examples in the database.
Never query for all the columns from a specific table, only ask for the relevant columns given the question.
You have access to tools for interacting with the database.
Only use the given tools. Only use the information returned by the tools to construct your final answer.
You MUST double check your query before executing it. If you get an error while executing a query, rewrite the query and try again.

DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.

If the question does not seem related to the database, just return "I don't know" as the answer.

You have access to the following tools:\n\nsql_db_query: Input to this tool is a detailed and correct SQL query, output is a result from the database. If the query is not correct, an error message will be returned. If an error is returned, rewrite the query, check the query, and try again. If you encounter an issue with Unknown column 'xxxx' in 'field list', use sql_db_schema to query the correct table fields.\nsql_db_schema: Input to this tool is a comma-separated list of tables, output is the schema and sample rows for those tables. Be sure that the tables actually exist by calling sql_db_list_tables first! Example Input: table1, table2, table3\nsql_db_list_tables: Input is an empty string, output is a comma separated list of tables in the database.\nsql_db_query_checker: Use this tool to double check if your query is correct before executing it. Always use this tool before executing a query with sql_db_query!


\n\nHere are some examples of user inputs and their corresponding SQL queries:\n

"""

suffix_prompt = """

\n\nUse the following format:\n\nQuestion: the input question you must answer\nThought: you should always think about what to do\nAction: the action to take, should be one of [sql_db_query, sql_db_schema, sql_db_list_tables, sql_db_query_checker]\nAction Input: the input to the action\nObservation: the result of the action\n... (this Thought/Action/Action Input/Observation can repeat N times)\nThought: I now know the final answer\nFinal Answer: the final answer to the original input question. Never give the query as final answer only give the answer based on the executed query. if multiple query execution is needed then do one by one. and only provide the Final answer after completing all the necessary steps or query exection results. wait and give results after the query executed and returned results.\n\n 

Begin!\n\nQuestion: {input}\nThought: I should look at the tables in the database to see what I can query.  Then I should query the schema of the most relevant tables.\n{agent_scratchpad}

"""

few_shot_prompt = FewShotPromptTemplate(
    example_selector=example_selector,
    example_prompt=PromptTemplate.from_template(
        "User input: {input}\nSQL query: {query}"
    ),
    input_variables=["input"],
    prefix=prefix_prompt,
    suffix=suffix_prompt,
)

# few_shot_prompt.format(dialect=str(toolkit.dialect),
#                        top_k=str(10))



llm_chain = LLMChain(
    llm=llm,
    # prompt=prompt,
    prompt=few_shot_prompt,
    # callback_manager=callback_manager,
)
tool_names = [tool.name for tool in tools]

agent = ZeroShotAgent(llm_chain=llm_chain, 
                      allowed_tools=tool_names,
                      output_parser=output_parser,
                    stop=["\nObservation:"],
                        )


agent = AgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=tools,
        # callback_manager=callback_manager,
        # verbose=True,
        handle_parsing_errors=True,
        # max_iterations=max_iterations,
        # max_execution_time=max_execution_time,
        # early_stopping_method=early_stopping_method,
        # **(agent_executor_kwargs or {}),
    )

# # Supply the input types to the prompt
class InputType(BaseModel):
    input: str


chain = agent.with_types(input_type=InputType) | itemgetter('output')