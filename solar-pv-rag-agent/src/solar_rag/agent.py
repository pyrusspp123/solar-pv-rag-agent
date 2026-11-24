"""AI Agent：装配 LLM 与多工具，实现“自然语言 → 工具调用 → 推理回答”。"""
from __future__ import annotations

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

SYSTEM_PROMPT = """你是太阳能光伏材料领域的 AI 研究助手。你可以调用以下工具：

1. literature_search —— 在光伏材料文献知识库中检索（RAG，返回带来源的片段）；
2. material_database —— 查询材料数据库（带隙、效率、稳定性等物性）；
3. sq_efficiency —— 估算单结太阳能电池的 Shockley-Queisser 理论极限效率；
4. bandgap_to_wavelength —— 半导体带隙与吸收截止波长的换算。

工作规则：
- 涉及文献知识、材料组分、制备工艺、稳定性机理、效率数据等问题，先调用
  literature_search 检索知识库；
- 涉及材料物性参数，可调用 material_database；
- 涉及理论极限或换算，可调用对应计算工具；
- 回答必须基于工具返回的信息，尽量注明来源；若信息不足，如实说明，不要编造。
"""


def build_agent(llm: BaseChatModel, retriever) -> AgentExecutor:
    """构建并返回一个可执行的 AgentExecutor。"""
    from .tools.literature import literature_search, set_retriever
    from .tools.materials import material_database
    from .tools.physics import bandgap_to_wavelength, sq_efficiency

    set_retriever(retriever)
    tools = [literature_search, material_database, sq_efficiency, bandgap_to_wavelength]

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )

    agent = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True)
