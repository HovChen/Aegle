# Copyright (c) 2026 Huangwei Chen
# Author: Huangwei Chen

import os
from dotenv import load_dotenv

load_dotenv(override=True)

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL")
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
DASHSCOPE_BASE_URL = os.getenv("DASHSCOPE_BASE_URL")
DMX_API_KEY = os.getenv("DMX_API_KEY")
DMX_BASE_URL = os.getenv("DMX_BASE_URL")


def instantiate_chat_model(model_name: str, **kwargs):
    """
    动态导入并实例化聊天模型。

    Args:
        model_name (str): 模型名称。
        *args: 位置参数，可以传入任何额外的参数。
        **kwargs: 关键字参数，可以传入任何额外的关键字参数。

    Returns:
        object: 实例化的模型对象（LangChain ChatModel）。
    """
    if "deepseek" in model_name:
        from langchain_deepseek import ChatDeepSeek

        return ChatDeepSeek(
            model=model_name,
            base_url=DEEPSEEK_BASE_URL,
            api_key=DEEPSEEK_API_KEY,
            **kwargs,
        )

    elif "qwen" in model_name:
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=model_name,
            base_url=DASHSCOPE_BASE_URL,
            api_key=DASHSCOPE_API_KEY,
            **kwargs,
        )

    else:
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=model_name, base_url=DMX_BASE_URL, api_key=DMX_API_KEY, **kwargs
        )
