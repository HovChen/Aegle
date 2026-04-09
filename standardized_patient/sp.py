# Copyright (c) 2026 Huangwei Chen
# Author: Huangwei Chen

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from typing import List
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage

from utils.logging_utils import get_run_logger
from utils.utils import instantiate_chat_model
from shared.prompt_base import get_prompt


class StandardizedPatient:
    def __init__(self, patient_view: str, model_name: str = "deepseek-chat"):
        """
        初始化标准化患者。

        Args:
            patient_view: 患者可见的病例资料文本
            model_name: 使用的模型名称
        """
        self.patient_view = patient_view
        self.llm = instantiate_chat_model(model_name=model_name, temperature=0.1)
        self.system_prompt = get_prompt("sp").format(patient_view=patient_view)

    async def arespond(self, conversation_history: List[BaseMessage]) -> str:
        """
        根据对话历史生成患者的回复。

        Args:
            conversation_history: 问诊系统的消息列表 (其中 AIMessage 是医生，HumanMessage 是患者)

        Returns:
            str: 患者的回复内容
        """
        sp_messages = [SystemMessage(content=self.system_prompt)]

        for msg in conversation_history:
            if isinstance(msg, AIMessage):
                sp_messages.append(HumanMessage(content=msg.content))
            elif isinstance(msg, HumanMessage):
                sp_messages.append(AIMessage(content=msg.content))

        try:
            response = await self.llm.ainvoke(sp_messages)
            content = response.content

            logger = get_run_logger()
            if logger:
                logger.log_section(
                    title="🎭 Standardized Patient (Internal)",
                    content=f"**Generated Content:**\n{content}",
                    collapsed=True,
                )
            return content
        except Exception as e:
            print(f"❌ SP Error: {e}")
            return "标准化患者出现错误，无法回答。请稍后再试。"


if __name__ == "__main__":
    import asyncio

    async def test_sp():
        fake_case = "患者男，45岁，主诉胸痛2天。既往有高血压病史。"
        sp = StandardizedPatient(patient_view=fake_case)

        history = [AIMessage(content="你好，请问哪里不舒服？")]

        print("👨‍⚕️ 医生: 你好，请问哪里不舒服？")
        print("Patient (Thinking)...")

        response = await sp.arespond(history)
        print(f"🧑 患者: {response}")

    asyncio.run(test_sp())
