# Copyright (c) 2026 Huangwei Chen
# Author: Huangwei Chen

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from dotenv import load_dotenv

load_dotenv()

import asyncio

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from shared.prompt_base import load_specialist_prompt
from shared.data_models import SpecialistOutput
from utils.utils import instantiate_chat_model


def create_specialist_agent(specialist_id: str, current_phase: str):
    """
    创建一个特定专科的 Agent。
    """
    system_prompt = load_specialist_prompt(specialist_id, current_phase)
    llm = instantiate_chat_model(model_name="deepseek-chat")

    agent = create_agent(
        model=llm,
        tools=[],
        system_prompt=system_prompt,
        response_format=SpecialistOutput,
        checkpointer=InMemorySaver(),
    )

    return agent


if __name__ == "__main__":
    from shared.data_models import CaseFeatures

    async def main_test():
        print("🚀 开始测试 Specialist Agent (Async)...")

        cardiology_agent = create_specialist_agent("cardiology", "historytaking")
        config = {"configurable": {"thread_id": "test_thread_1"}}

        input_payload = {
            "messages": [
                HumanMessage(
                    content=f"""当前信息： 
- 对话历史：
1. AIMessage: 你好，请问您这次来看病，是哪里感觉不舒服呢？ 
2. HumanMessage: 医生，我最近总是觉得胸口疼，还有点气短
                         
- 当前病例特点草稿：
{CaseFeatures(basic_info="患者，男性，45岁，吸烟史20年", present_illness="胸口疼痛，气短2周", past_history="高血压5年", physical_exam="心率90次/分，血压140/90 mmHg", aux_exam="心电图示ST段轻度抬高", complete=False)}

- 协调者指令：
请重点关注患者的胸痛症状，评估是否存在心脏相关疾病的可能性，并提出相应的诊断建议。"""
                )
            ]
        }

        print("🤖 Agent 正在思考...")
        result = await cardiology_agent.ainvoke(input_payload, config=config)

        spec: SpecialistOutput = result["structured_response"]

        print("\n=== 测试结果 ===")
        print(f"Next Question: {spec.next_question}")
        print(f"Draft Mods: {spec.draft_modifications}")

    asyncio.run(main_test())
