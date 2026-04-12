from langchain_community.document_loaders import PyPDFLoader
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_openai.llms import OpenAI
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
import tiktoken
from langchain.tools import tool
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI 
from langchain.messages import HumanMessage

load_dotenv(override=True)

loader = PyPDFLoader("Ramatoulaye_Diawane_CV-Pro.pdf")
tokennizer = tiktoken.encoding_for_model("gpt-4o-mini")
print(tokennizer.name)

splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    encoding_name=tokennizer.name,
    chunk_size=300,
    chunk_overlap=2,
    )
chunks = loader.load_and_split(splitter)

print(len(chunks))
print(chunks[0].metadata)

embeddings_model = OpenAIEmbeddings()

vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings_model,
    collection_name="CV_data_collection"
)

retriever = vector_store.as_retriever(kwargs= {"k":10})

## Searching information about cantidates in the resume
@tool
def retriever_tool(query : str) -> str:
     """
     Permet de chercher des informations sur des candidats:
     -Nom, Prénom, Diplômes
     -Expériences professionnelles
     -Compétences techniques
     """
     relevent_chunks=retriever.invoke(query)
     context_list = [d.page_content for d in relevent_chunks]
     context = ".".join(context_list)
     return context

@tool
def get_company_info(company_name : str):
    """
    Consulter des informations sur l'entreprise donnée
    """
    return {
        "company_name": company_name,
        "domaine": "IT",
        "turnover":6000

    }
    
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
agent = create_agent(
    model=llm,
    tools=[retriever_tool, get_company_info],
    system_prompt="Réponds à la question de l'utilisateur en utilisant les tools fournies. "

)

resp = agent.invoke(input={
    "messages":[
        HumanMessage("Nom, prénom, diplômes de Ramatoulaye Diawane et les information sur son entreprise") 
    ]
})

print(resp['messages'][-1].content)

from IPython.display import Markdown
print(display(Markdown(resp['messages'][-1].content)))