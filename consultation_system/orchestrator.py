# Copyright (c) 2026 Huangwei Chen
# Author: Huangwei Chen

import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.messages import HumanMessage
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

from shared.data_models import OrchestratorDecision
from shared.prompt_base import get_prompt
from utils.utils import instantiate_chat_model

ORCHESTRATOR_PHASE_CONFIG = {
    "historytaking": {
        "description": "阶段一：病史采集 (History Taking)",
        "goal": "高效搜集患者的病例特点(Case Features)，填补信息空白，明确症状细节。专注于记录患者的实际病情信息：基本信息、现病史、既往情况、体格检查、辅助检查。注意：这是信息记录阶段，不是制定检查计划阶段。",
    },
    "diagnosis_and_plan": {
        "description": "阶段二：诊断与计划 (Diagnosis & Plan)",
        "goal": "基于已锁定的病例特点，组织各专科医生进行讨论，给出初步诊断、拟诊讨论、诊疗计划。",
    },
}


def create_orchestrator_agent(phase: str = "historytaking"):
    """
    使用 LangChain 的 create_agent() 创建 Orchestrator Agent。
    """
    raw_prompt = get_prompt("orchestrator")

    phase_config = ORCHESTRATOR_PHASE_CONFIG.get(
        phase, ORCHESTRATOR_PHASE_CONFIG["historytaking"]
    )

    orchestrator_prompt = raw_prompt.format(
        current_phase_description=phase_config["description"],
        phase_goal=phase_config["goal"],
    )

    llm = instantiate_chat_model(model_name="deepseek-chat")
    checkpointer = InMemorySaver()

    agent = create_agent(
        model=llm,
        system_prompt=orchestrator_prompt,
        response_format=OrchestratorDecision,
        checkpointer=checkpointer,
    )

    return agent


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv(override=True)
    from shared.data_models import CaseFeatures

    # 测试 Orchestrator Agent
    orchestrator = create_orchestrator_agent()
    config = {"configurable": {"thread_id": "1"}}
    decision = orchestrator.invoke(
        {
            "messages": [
                HumanMessage(
                    content=f"""
当前对话历史：
1. AIMessage: 你好，请问您这次来看病，是哪里感觉不舒服呢？
2. HumanMessage: 医生，我最近总是觉得胸口疼，还有点气短
                                                
当前病例特点草稿：
{CaseFeatures(basic_info="患者，男性，45岁，吸烟史20年", present_illness="胸口疼痛，气短2周", past_history="高血压5年", physical_exam="心率90次/分，血压140/90 mmHg", aux_exam="心电图示ST段轻度抬高", complete=False)}

患者最新回答：
医生，我最近总是觉得胸口疼，还有点气短

请基于以上信息，决定需要激活哪些专科医生，并给出统一指令。
"""
                )
            ]
        },
        config=config,
    )
    print("🧠 Orchestrator Decision:")
    print(
        f"激活的专家：{decision['structured_response'].active_specialists}\n指令：{decision['structured_response'].instructions_to_specialists}"
    )

    import pdb

    pdb.set_trace()
