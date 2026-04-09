# Copyright (c) 2026 Huangwei Chen
# Author: Huangwei Chen

import sys
import asyncio
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from typing import List, TypedDict, Annotated
from operator import add

from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, END

from shared.data_models import SOAPNote, SpecialistOutput
from consultation_system import (
    create_orchestrator_agent,
    create_specialist_agent,
    create_aggregator_agent,
)

from utils.logging_utils import get_run_logger


def get_recent_patient_responses(messages: List[BaseMessage], count: int = 3) -> str:
    patient_msgs = [m for m in messages if isinstance(m, HumanMessage)][-count:]
    if not patient_msgs:
        return "暂无患者回复记录"

    formatted_responses = []
    for i, msg in enumerate(patient_msgs, 1):
        formatted_responses.append(f"[{i}] {msg.content}")

    return "\n".join(formatted_responses)


def limit_messages_reducer(
    existing: List[BaseMessage], new: List[BaseMessage], max_size: int = 10
) -> List[BaseMessage]:
    all_messages = (existing or []) + new
    if len(all_messages) > max_size:
        return all_messages[-max_size:]
    return all_messages


def format_case_features_for_llm(case_features) -> str:
    sections = []
    sections.append(
        f"基本信息: {case_features.basic_info if case_features.basic_info.strip() else '待收集'}"
    )
    sections.append(
        f"现病史: {case_features.present_illness if case_features.present_illness.strip() else '待收集'}"
    )
    sections.append(
        f"既往史: {case_features.past_history if case_features.past_history.strip() else '待收集'}"
    )
    sections.append(
        f"体格检查: {case_features.physical_exam if case_features.physical_exam.strip() else '待收集'}"
    )
    sections.append(
        f"辅助检查: {case_features.aux_exam if case_features.aux_exam.strip() else '待收集'}"
    )
    return "\n".join(sections) if sections else "暂无病例特点信息"


def format_diagnosis_and_plan_for_llm(diagnosis_plan) -> str:
    sections = []
    if diagnosis_plan.preliminary_diagnosis.strip():
        sections.append(f"初步诊断: {diagnosis_plan.preliminary_diagnosis}")
    if diagnosis_plan.diagnosis_discussion.strip():
        sections.append(f"拟诊讨论: {diagnosis_plan.diagnosis_discussion}")
    if diagnosis_plan.treatment_plan.strip():
        sections.append(f"诊疗计划: {diagnosis_plan.treatment_plan}")
    return "\n".join(sections) if sections else "暂无诊断和计划信息"


def format_specialist_outputs_for_model(
    specialist_outputs: List[SpecialistOutput], current_phase: str
) -> str:
    if not specialist_outputs:
        return "暂无专科医生意见。"
    formatted_output = []
    if current_phase == "historytaking":
        formatted_output.append("专科医生问诊意见汇总：\n")
        for i, output in enumerate(specialist_outputs, 1):
            formatted_output.append(f"专科医生 #{i} 建议：")
            formatted_output.append(f"建议提问：{output.next_question}")
            formatted_output.append(f"病历修改建议：{output.draft_modifications}")
            formatted_output.append("---")
    elif current_phase == "diagnosis_and_plan":
        formatted_output.append("专科医生诊断意见汇总：\n")
        for i, output in enumerate(specialist_outputs, 1):
            formatted_output.append(f"专科医生 #{i} 诊断分析：")
            formatted_output.append(f"文书建议：{output.draft_modifications}")
            formatted_output.append("---")
    return "\n".join(formatted_output)


class ConsultationState(TypedDict):
    messages: Annotated[
        List[BaseMessage], lambda x, y: limit_messages_reducer(x, y, max_size=10)
    ]
    draft: SOAPNote
    active_specialists: List[str]
    history_activated_specialists: List[str]
    current_patient_response: str
    current_doctor_question: str
    suggested_question: str
    specialist_instructions: str
    specialist_outputs: List[SpecialistOutput]
    current_phase: str


class ConsultationAgents:
    def __init__(self):
        self.orchestrator_phase1 = create_orchestrator_agent(phase="historytaking")
        self.orchestrator_phase2 = create_orchestrator_agent(phase="diagnosis_and_plan")
        self.aggregator_phase1 = create_aggregator_agent(phase="historytaking")
        self.aggregator_phase2 = create_aggregator_agent(phase="diagnosis_and_plan")
        self._specialist_cache = {}

    def get_specialist(self, spec_id: str, phase: str):
        key = f"{spec_id}_{phase}"
        if key not in self._specialist_cache:
            self._specialist_cache[key] = create_specialist_agent(spec_id, phase)
        return self._specialist_cache[key]

    def get_orchestrator(self, phase: str):
        if phase == "historytaking":
            return self.orchestrator_phase1
        else:
            return self.orchestrator_phase2

    def get_aggregator(self, phase: str):
        if phase == "historytaking":
            return self.aggregator_phase1
        else:
            return self.aggregator_phase2


agents = ConsultationAgents()


async def orchestrator_node(state: ConsultationState, config: RunnableConfig):
    """
    调用 Orchestrator Agent
    """
    current_phase = state.get("current_phase", "historytaking")
    logger = get_run_logger()

    orchestrator = agents.get_orchestrator(current_phase)

    parent_thread_id = config.get("configurable", {}).get("thread_id", "default")
    sub_config = {"configurable": {"thread_id": f"{parent_thread_id}_orchestrator"}}

    if current_phase == "historytaking":
        input_data = {
            "messages": state["messages"],
            "draft": state["draft"],
            "current_patient_response": state["current_patient_response"],
        }

        orchestrator_input = HumanMessage(
            content=f"""
当前SOAP草稿:
{input_data.get("draft", {})}

患者最新回答:
{input_data.get("current_patient_response", "")}

请基于以上信息，决定是否需要激活、需要激活哪些专科医生，并给出指令。
"""
        )
        orchestrator_decision_raw = await orchestrator.ainvoke(
            {"messages": [orchestrator_input]}, config=sub_config
        )
        decision = orchestrator_decision_raw["structured_response"]
        suggested_q = getattr(decision, "suggested_question", "")

        if logger is not None:
            logger.log_orchestrator(
                phase=current_phase,
                active_specialists=decision.active_specialists,
                instructions=decision.instructions_to_specialists,
            )

        active_specialists = decision.active_specialists
        current_history = state.get("history_activated_specialists", [])
        updated_history = list(set(current_history + decision.active_specialists))

        return {
            "active_specialists": decision.active_specialists,
            "specialist_instructions": decision.instructions_to_specialists,
            "suggested_question": suggested_q,
            "history_activated_specialists": updated_history,
        }

    elif current_phase == "diagnosis_and_plan":
        history_activated = state.get("history_activated_specialists", [])
        current_active = state.get("active_specialists", [])
        active_specialists = (
            list(set(history_activated)) if history_activated else current_active
        )
        specialist_instructions = (
            "病例特点收集已完成，请基于已有信息提供初步诊断、拟诊讨论和诊疗计划"
        )

        if logger is not None:
            logger.log_orchestrator(
                phase=current_phase,
                active_specialists=active_specialists,
                instructions=specialist_instructions,
            )

        return {
            "active_specialists": active_specialists,
            "specialist_instructions": specialist_instructions,
            "history_activated_specialists": history_activated,
        }


async def _process_single_specialist_async(
    spec_id: str, state: ConsultationState, current_phase: str, parent_thread_id: str
):
    """
    异步处理单个专科医生
    """
    specialist_agent = agents.get_specialist(spec_id, current_phase)
    draft = state["draft"]

    formatted_case_features = format_case_features_for_llm(draft.case_features)
    formatted_diagnosis_plan = ""
    if draft.diagnosis_and_plan and current_phase == "diagnosis_and_plan":
        formatted_diagnosis_plan = format_diagnosis_and_plan_for_llm(
            draft.diagnosis_and_plan
        )

    recent_patient_responses = get_recent_patient_responses(state["messages"], count=3)

    diagnosis_plan_str = (
        f"\n当前SOAP草稿 - 诊断计划:\n{formatted_diagnosis_plan}\n"
        if formatted_diagnosis_plan
        else ""
    )

    specialist_input = HumanMessage(
        content=f"""
患者最近回复记录:
{recent_patient_responses}

当前SOAP草稿 - 病例特点:
{formatted_case_features}
{diagnosis_plan_str}
协调者指令:
{state["specialist_instructions"]}

请基于以上信息，提供你的专业分析和建议，输出为 JSON 格式。
"""
    )

    unique_thread_id = f"{parent_thread_id}_specialist_{spec_id}_{current_phase}"
    config = {"configurable": {"thread_id": unique_thread_id}}

    logger = get_run_logger()

    for retry in range(3):
        try:
            specialist_agent_output = await specialist_agent.ainvoke(
                {"messages": [specialist_input]},
                config=config,
            )

            output_obj = specialist_agent_output.get("structured_response")
            if output_obj is None:
                raise ValueError("specialist_agent_output 没有 structured_response")

            if logger is not None:
                logger.log_specialist(
                    input=specialist_input.content,
                    phase=current_phase,
                    spec_id=spec_id,
                    next_question=getattr(output_obj, "next_question", None),
                    draft_modifications=output_obj.draft_modifications,
                )

            return spec_id, output_obj

        except Exception as e:
            print(f"❌ {spec_id} 专科医生调用异常: {e}")
            if retry < 2:
                await asyncio.sleep(1)
            else:
                print(f"⚠️ {spec_id} 专科医生调用失败，已重试3次")

    return spec_id, None


async def specialist_node(state: ConsultationState, config: RunnableConfig):
    """
    并行调用所有激活的 Specialist Agents。
    """
    current_phase = state.get("current_phase", "historytaking")
    active_specialists = state["active_specialists"]

    if not active_specialists:
        return {"specialist_outputs": []}

    logger = get_run_logger()

    parent_thread_id = config.get("configurable", {}).get("thread_id", "default")

    messages = state["messages"]
    max_messages = 6
    if len(messages) > max_messages:
        messages = messages[-max_messages:]

    sub_state = {**state, "messages": messages}

    tasks = [
        _process_single_specialist_async(
            spec_id, sub_state, current_phase, parent_thread_id
        )
        for spec_id in active_specialists
    ]

    results = await asyncio.gather(*tasks)

    ordered_outputs = []
    result_map = {r[0]: r[1] for r in results if r[1] is not None}

    for spec_id in active_specialists:
        if spec_id in result_map:
            ordered_outputs.append(result_map[spec_id])
        else:
            print(f"⚠️ 警告: 专科医生 {spec_id} 执行失败，已跳过。")
            if logger is not None:
                logger.log_specialist(
                    input=None,
                    phase=current_phase,
                    spec_id=spec_id,
                    next_question=None,
                    draft_modifications="N/A (Failed)",
                )

    return {"specialist_outputs": ordered_outputs}


async def aggregator_node(state: ConsultationState, config: RunnableConfig):
    """
    聚合所有专科 Agent 的输出，并作出后续决策。
    """
    current_phase = state.get("current_phase", "historytaking")
    logger = get_run_logger()

    specialist_outputs = state.get("specialist_outputs", [])
    current_draft = state.get("draft", [])
    current_patient_response = state.get("current_patient_response", "")
    suggested_question = state.get("suggested_question", "")

    formatted_specialist_outputs = format_specialist_outputs_for_model(
        specialist_outputs, current_phase
    )
    aggregator_agent = agents.get_aggregator(current_phase)

    parent_thread_id = config.get("configurable", {}).get("thread_id", "default")
    sub_config = {"configurable": {"thread_id": f"{parent_thread_id}_aggregator"}}

    if current_phase == "historytaking":
        formatted_case_features = format_case_features_for_llm(
            current_draft.case_features
        )
        formatted_specialists_outputs = (
            f"专科医生意见:\n{formatted_specialist_outputs}"
            if specialist_outputs
            else ""
        )
        formatted_suggested_question = (
            f"\n协调者建议的问题: {suggested_question}\n" if suggested_question else ""
        )
        aggregator_input = HumanMessage(
            content=f"""
当前SOAP草稿 - 病例特点部分:
{formatted_case_features}

患者最新回复:
{current_patient_response}
{formatted_specialists_outputs}{formatted_suggested_question}
请基于以上信息，更新SOAP草稿并决定下接下来的问题或结束病例特点收集。
展现医生的共情，安抚患者情绪。若患者出现紧急情况，直接结束病例特点收集。
"""
        )

        aggregator_output_raw = await aggregator_agent.ainvoke(
            {"messages": [aggregator_input]}, config=sub_config
        )
        aggregator_output = aggregator_output_raw["structured_response"]
        current_draft.case_features = aggregator_output.updated_case_features

        if logger is not None:
            logger.log_aggregator_historytaking(
                input=aggregator_input.content,
                phase=current_phase,
                next_question=aggregator_output.next_question_to_patient,
                historytaking_complete=aggregator_output.historytaking_complete,
                case_features_summary=str(aggregator_output.updated_case_features),
            )

        if aggregator_output.historytaking_complete:
            return {
                "draft": current_draft,
                "messages": [],
                "current_doctor_question": "",
                "current_phase": "diagnosis_and_plan",
                "history_activated_specialists": state.get(
                    "history_activated_specialists", []
                ),
                "active_specialists": [],
                "specialist_outputs": [],
            }
        else:
            new_ai_message = AIMessage(
                content=aggregator_output.next_question_to_patient
            )
            return {
                "draft": current_draft,
                "messages": [new_ai_message],
                "current_doctor_question": aggregator_output.next_question_to_patient,
                "current_phase": "historytaking",
                "history_activated_specialists": state.get(
                    "history_activated_specialists", []
                ),
                "active_specialists": state.get("active_specialists", []),
            }

    else:
        formatted_case_features = format_case_features_for_llm(
            current_draft.case_features
        )
        aggregator_input = HumanMessage(
            content=f"""
最终确定的病例特点:
{formatted_case_features}

{formatted_specialist_outputs}

请基于以上信息，完善SOAP的初步诊断、拟诊讨论、诊疗计划部分。
"""
        )

        aggregator_output_raw = await aggregator_agent.ainvoke(
            {"messages": [aggregator_input]}, config=sub_config
        )
        aggregator_output = aggregator_output_raw["structured_response"]
        current_draft.diagnosis_and_plan = aggregator_output.diagnosis_and_plan
        current_draft.diagnosis_and_plan.complete = True

        if logger is not None:
            logger.log_aggregator_diagnosis(
                phase=current_phase,
                diagnosis_and_plan=aggregator_output.diagnosis_and_plan,
            )

        return {
            "draft": current_draft,
            "messages": state["messages"],
            "current_doctor_question": "",
            "history_activated_specialists": state.get(
                "history_activated_specialists", []
            ),
            "active_specialists": state.get("active_specialists", []),
        }


def build_consultation_graph():
    """
    构建并编译 LangGraph 状态机。
    """
    workflow = StateGraph(ConsultationState)

    workflow.add_node("orchestrator", orchestrator_node)
    workflow.add_node("specialists", specialist_node)
    workflow.add_node("aggregator", aggregator_node)

    workflow.set_entry_point("orchestrator")

    workflow.add_edge("orchestrator", "specialists")
    workflow.add_edge("specialists", "aggregator")
    workflow.add_edge("aggregator", END)

    app = workflow.compile()
    return app
