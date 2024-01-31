import os
from .sdk import ForgeLogger
from transformers import BartTokenizer, BartForConditionalGeneration
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain.agents.initialize import initialize_agent
from langchain.tools import Tool
from langchain.agents import AgentType
from langchain_openai import ChatOpenAI
from langchain_core.prompts.chat import MessagesPlaceholder
from langchain.memory import ConversationSummaryBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain_community.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type
from bs4 import BeautifulSoup
import requests
import json
#from langchain_community.schema import SystemMessage
LOG = ForgeLogger(__name__)
from forge.actions import ActionRegister
from forge.sdk import (
    Agent,
    AgentDB,
    ForgeLogger,
    Step,
    StepRequestBody,
    Task,
    TaskRequestBody,
    Workspace,
    PromptEngine,
    chat_completion_request,
)

load_dotenv('.env')

LOG = ForgeLogger(__name__)


class ForgeAgent(Agent):
    """
    The goal of the Forge is to take care of the boilerplate code, so you can focus on
    agent design.

    There is a great paper surveying the agent landscape: https://arxiv.org/abs/2308.11432
    Which I would highly recommend reading as it will help you understand the possabilities.

    Here is a summary of the key components of an agent:

    Anatomy of an agent:
         - Profile
         - Memory
         - Planning
         - Action

    Profile:

    Agents typically perform a task by assuming specific roles. For example, a teacher,
    a coder, a planner etc. In using the profile in the llm prompt it has been shown to
    improve the quality of the output. https://arxiv.org/abs/2305.14688

    Additionally, based on the profile selected, the agent could be configured to use a
    different llm. The possibilities are endless and the profile can be selected
    dynamically based on the task at hand.

    Memory:

    Memory is critical for the agent to accumulate experiences, self-evolve, and behave
    in a more consistent, reasonable, and effective manner. There are many approaches to
    memory. However, some thoughts: there is long term and short term or working memory.
    You may want different approaches for each. There has also been work exploring the
    idea of memory reflection, which is the ability to assess its memories and re-evaluate
    them. For example, condensing short term memories into long term memories.

    Planning:

    When humans face a complex task, they first break it down into simple subtasks and then
    solve each subtask one by one. The planning module empowers LLM-based agents with the ability
    to think and plan for solving complex tasks, which makes the agent more comprehensive,
    powerful, and reliable. The two key methods to consider are: Planning with feedback and planning
    without feedback.

    Action:

    Actions translate the agent's decisions into specific outcomes. For example, if the agent
    decides to write a file, the action would be to write the file. There are many approaches you
    could implement actions.

    The Forge has a basic module for each of these areas. However, you are free to implement your own.
    This is just a starting point.
    """

    def __init__(self, database: AgentDB, workspace: Workspace):
        """
        The database is used to store tasks, steps and artifact metadata. The workspace is used to
        store artifacts. The workspace is a directory on the file system.

        Feel free to create subclasses of the database and workspace to implement your own storage
        """
        super().__init__(database, workspace)
        self.abilities = ActionRegister(self)

    async def create_task(self, task_request: TaskRequestBody) -> Task:
        """
        The agent protocol, which is the core of the Forge, works by creating a task and then
        executing steps for that task. This method is called when the agent is asked to create
        a task.

        We are hooking into function to add a custom log message. Though you can do anything you
        want here.
        """
        task = await super().create_task(task_request)
        LOG.info(
            f"📦 Task created: {task.task_id} input: {task.input[:40]}{'...' if len(task.input) > 40 else ''}"
        )
        return task

    async def execute_step(self, task_id: str, step_request: StepRequestBody) -> Step:
        """
        For a tutorial on how to add your own logic please see the offical tutorial series:
        https://aiedge.medium.com/autogpt-forge-e3de53cc58ec

        The agent protocol, which is the core of the Forge, works by creating a task and then
        executing steps for that task. This method is called when the agent is asked to execute
        a step.

        The task that is created contains an input string, for the benchmarks this is the task
        the agent has been asked to solve and additional input, which is a dictionary and
        could contain anything.

        If you want to get the task use:

        ```
        task = await self.db.get_task(task_id)
        ```

        The step request body is essentially the same as the task request and contains an input
        string, for the benchmarks this is the task the agent has been asked to solve and
        additional input, which is a dictionary and could contain anything.

        You need to implement logic that will take in this step input and output the completed step
        as a step object. You can do everything in a single step or you can break it down into
        multiple steps. Returning a request to continue in the step output, the user can then decide
        if they want the agent to continue or not.
        """

        random_string="Just testing around"
        random_string_b=b"Just testing around 2"

        # An example that
        step = await self.db.create_step(
            task_id=task_id, input=step_request, is_last=True
        )

        aux = textProcessor('The_New_York_Times.pdf')

        self.workspace.write(task_id=task_id, path="output.txt", data=random_string_b)

        await self.db.create_artifact(
            task_id=task_id,
            step_id=step.step_id,
            file_name="output.txt",
            relative_path="",
            agent_created=True,
        )

        step.output = aux

        LOG.info(
            f"\t✅ Final Step completed: {step.step_id}. \n"
            + f"Output should be placeholder text Washington D.C. You'll need to \n"
            + f"modify execute_step to include LLM behavior. Follow the tutorial "
            + f"if confused. "
        )

        return step

def textProcessor(filename):

    script_dir = os.path.dirname(__file__)
    file_path = os.path.join(script_dir, filename)


    #reader
    with open(file_path, 'rb') as file:
    # This should take pdfs from the DB or a designated route
        reader = PdfReader(file) 

        print(len(reader.pages)) 

        page = reader.pages[0] 

        text = page.extract_text() 
        text1 = summarizer(text)
        print(text1)

    return text

def summarizer(text):
    model_name = "facebook/bart-large-cnn"
    tokenizer = BartTokenizer.from_pretrained(model_name)
    model = BartForConditionalGeneration.from_pretrained(model_name)

    # Input text to be summarized
    input_text = text

    # Tokenize and summarize the input text using BART
    inputs = tokenizer.encode("summarize: " + input_text, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = model.generate(inputs, max_length=100, min_length=50, length_penalty=2.0, num_beams=4, early_stopping=True)

    # Decode and output the summary
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    print("Original Text:")
    print(input_text)
    print("\nSummary:")
    print(summary)

    return summary

tools = [
    Tool(
        name="Summarize",
        func=textProcessor,
        description="useful for when you need to answer questions about current events, data. You should ask targeted questions"
    ),
]