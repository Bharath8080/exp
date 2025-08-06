# streamlit_app.py

import asyncio
import streamlit as st
from agno.agent import Agent
from agno.models.groq import Groq
from agno.tools.mcp import MCPTools
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")


# Async function to handle MCP + Agent
async def run_mcp_agent_async(message: str):
    mcp_tools = MCPTools("npx -y @openbnb/mcp-server-airbnb --ignore-robots-txt")
    await mcp_tools.connect()

    agent = Agent(
        model=Groq(id="openai/gpt-oss-120b", api_key=groq_api_key),
        tools=[mcp_tools],
        markdown=True,
    )

    # ❗ Use arun() instead of run()
    response = await agent.arun(message)

    await mcp_tools.close()
    return response


# Wrap async function for Streamlit using asyncio.create_task()
def run_async_task(message: str):
    return asyncio.run(run_mcp_agent_async(message))


# Streamlit UI
st.set_page_config(page_title="🏠 MCP Airbnb Agent", page_icon="🏡")
st.title("🏠 MCP Airbnb Agent - Find Airbnb Listings")
st.markdown("Search for Airbnb listings using MCP + Groq + Gemini!")

user_query = st.text_input("Enter your request", value="Show me listings in Barcelona, for 2 people.")

if st.button("Search"):
    if not groq_api_key:
        st.error("❌ GROQ_API_KEY not found in environment.")
    else:
        with st.spinner("🔍 Searching for listings..."):
            try:
                response = run_async_task(user_query)
                st.markdown("### ✅ Results")
                st.markdown(response)
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
