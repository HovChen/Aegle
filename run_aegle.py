# Copyright (c) 2026 Huangwei Chen
# Author: Huangwei Chen

import os
import sys
import json
import asyncio
import argparse
from typing import List, Dict, Any, Optional

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv

load_dotenv(override=True)

from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnableConfig

from consultation_system.manager import build_consultation_graph
from standardized_patient.sp import StandardizedPatient
from shared.data_models import SOAPNote
from utils.logging_utils import RunLogger, set_run_logger


def construct_patient_view(extracted: Dict[str, str]) -> str:
    """
    根据 extracted 字段构建 SP 可见的 patient_view 文本。
    """
    sections = [
        ("基本信息", extracted.get("basic_info", "")),
        ("现病史", extracted.get("present_illness", "")),
        ("既往情况", extracted.get("past_history", "")),
        ("体格检查", extracted.get("physical_exam", "")),
        ("辅助检查", extracted.get("aux_exam", "")),
    ]

    parts = ["## 一、病例特点\n"]

    cn_nums = ["一", "二", "三", "四", "五", "六", "七", "八", "九", "十"]

    for idx, (title, content) in enumerate(sections):
        if content:
            header = f"### （{cn_nums[idx]}）{title}"
            parts.append(f"{header}\n\n{content}\n")

    return "\n".join(parts)


def load_cases(
    cases_dir: str, limit: int = 0, specific_case_id: str = None
) -> List[Dict[str, Any]]:
    """
    加载病例数据。

    :param cases_dir: 病例根目录
    :param limit: 限制加载数量
    :param specific_case_id: 如果指定，只加载该 ID 的病例
    :return: 返回病例列表，每个病例包含 ID、内容和文件路径
    """
    idx_path = os.path.join(cases_dir, "index.json")
    if not os.path.exists(idx_path):
        raise FileNotFoundError(f"未找到 index.json：{idx_path}")

    with open(idx_path, "r", encoding="utf-8") as f:
        index = json.load(f) or []

    cases = []
    for rec in index:
        cid = str(rec.get("case_id", "")).strip()
        rel = rec.get("path", "")

        if not cid or not rel:
            continue

        if specific_case_id and cid != specific_case_id:
            continue

        cpath = os.path.join(cases_dir, rel)
        if not os.path.exists(cpath):
            print(f"⚠️ 警告: 索引中的文件不存在 {cpath}")
            continue

        with open(cpath, "r", encoding="utf-8") as cf:
            case_obj = json.load(cf)

        cases.append({"case_id": cid, "obj": case_obj, "file_path": cpath})

        if not specific_case_id and limit and len(cases) >= limit:
            break

    return cases


async def enter_diagnosis_and_plan_phase(
    current_state: dict,
    consultation_graph,
    config: RunnableConfig,
    verbose: bool = True,
):
    """进入诊断与计划阶段"""
    if verbose:
        print("\n📋 进入诊断与计划阶段 (Diagnosis & Plan)")

    graph_input_state = {
        **current_state,
        "current_doctor_question": "",
        "active_specialists": [],
        "specialist_instructions": "",
        "specialist_outputs": [],
        "current_phase": "diagnosis_and_plan",
    }

    output_state = await consultation_graph.ainvoke(graph_input_state, config=config)
    return output_state


def save_final_soap(draft: SOAPNote, logger: RunLogger, verbose: bool = True):
    """保存并打印最终结果"""
    if verbose:
        print("\n================ 🎉 问诊完成 ================")
        print("最终 SOAP 记录已生成。")
        if draft.diagnosis_and_plan:
            print(f"初步诊断：\n{draft.diagnosis_and_plan.preliminary_diagnosis}")
            print(f"拟诊讨论：\n{draft.diagnosis_and_plan.diagnosis_discussion}")
            print(f"诊疗计划：\n{draft.diagnosis_and_plan.treatment_plan}")

    logger.write_soap_markdown(draft)


async def run_one_case_async(
    case_data: Dict[str, Any],
    output_root: str,
    semaphore: asyncio.Semaphore,
    max_turns: int = 30,
    verbose: bool = True,
):
    """
    单个病例的异步运行逻辑。
    """
    case_id = case_data["case_id"]
    case_obj = case_data["obj"]

    case_dir = os.path.join(output_root, f"{case_id}")
    logger = RunLogger(case_dir, session_tag=f"run_{case_id}")

    set_run_logger(logger)

    try:
        if verbose:
            print(f"🚀 开始运行 Case {case_id} ...")

        extracted = case_obj.get("extracted", {})
        patient_view = construct_patient_view(extracted)
        sp_agent = StandardizedPatient(patient_view=patient_view)

        consultation_graph = build_consultation_graph()
        config = {"configurable": {"thread_id": f"case_{case_id}"}}

        first_question = (
            "你好，请先告诉我您的性别和年龄。请问您这次来看病，是哪里感觉不舒服呢？"
        )
        initial_message = AIMessage(content=first_question)

        current_state = {
            "messages": [initial_message],
            "draft": SOAPNote(),
            "active_specialists": [],
            "history_activated_specialists": [],
            "current_patient_response": "",
            "current_doctor_question": first_question,
            "specialist_instructions": "",
            "specialist_outputs": [],
            "current_phase": "historytaking",
        }

        logger.log_dialog("doctor", first_question)

        for i in range(max_turns):
            if verbose:
                print(f"\n[Case {case_id}] Round {i + 1}")

            doctor_q = current_state.get("current_doctor_question")
            if not doctor_q:
                if current_state.get("current_phase") == "diagnosis_and_plan":
                    if verbose:
                        print(f"[Case {case_id}] 阶段切换或结束检测")
                    break
                if current_state["messages"]:
                    doctor_q = current_state["messages"][-1].content

            if verbose:
                print(f"👨‍⚕️ Doctor ({case_id}): {doctor_q}")
            if i > 0:
                logger.log_dialog("doctor", doctor_q)

            patient_resp = await sp_agent.arespond(current_state["messages"])
            if verbose:
                print(f"🧑 Patient ({case_id}): {patient_resp}")
            logger.log_dialog("patient", patient_resp)

            new_msg = HumanMessage(content=patient_resp)
            current_state["messages"].append(new_msg)
            current_state["current_patient_response"] = patient_resp

            output_state = await consultation_graph.ainvoke(
                current_state, config=config
            )

            logger.log_activated_specialists_file(
                output_state.get("history_activated_specialists", [])
            )

            current_state = output_state
            draft = current_state["draft"]

            if (
                current_state["current_phase"] == "historytaking"
                and draft.case_features_complete
            ):
                current_state["current_phase"] = "diagnosis_and_plan"
                current_state["specialist_outputs"] = []

            if draft.diagnosis_complete:
                save_final_soap(draft, logger, verbose)
                break

        draft = current_state["draft"]

        if (
            current_state["current_phase"] == "diagnosis_and_plan"
            and not draft.diagnosis_complete
        ):
            current_state = await enter_diagnosis_and_plan_phase(
                current_state, consultation_graph, config, verbose
            )
            draft = current_state["draft"]
            if draft.diagnosis_complete:
                save_final_soap(draft, logger, verbose)

        if not draft.diagnosis_complete and not draft.case_features_complete:
            if verbose:
                print(f"⚠️ [Case {case_id}] 达到最大轮数，强制进入诊断阶段...")
            current_state = await enter_diagnosis_and_plan_phase(
                current_state, consultation_graph, config, verbose
            )
            draft = current_state["draft"]
            if draft.diagnosis_complete:
                save_final_soap(draft, logger)

        if verbose:
            print(f"✅ Case {case_id} 运行结束")

    except Exception as e:
        if verbose:
            print(f"❌ Case {case_id} 发生异常: {e}")
        import traceback

        traceback.print_exc()
    finally:
        logger.close()
        set_run_logger(None)


async def main():
    parser = argparse.ArgumentParser(description="Aegle 单病例运行脚本")
    parser.add_argument(
        "--cases-dir",
        type=str,
        default="./datasets/ClinicalBench/SOAP/json",
        help="病例根目录（包含 index.json）",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./output/test_run",
        help="输出目录",
    )
    parser.add_argument(
        "--case-id",
        type=str,
        default="29",
        help="要运行的病例 ID",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=15,
        help="最大问诊轮数",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="并发数",
    )
    args = parser.parse_args()

    base_cases_dir = args.cases_dir
    output_dir = args.output_dir
    target_case_id = args.case_id

    try:
        cases = load_cases(base_cases_dir, limit=1, specific_case_id=target_case_id)
    except Exception as e:
        print(f"无法加载病例: {e}")
        print("尝试寻找默认测试数据...")
        return

    if not cases:
        print(f"未找到 ID 为 {target_case_id} 的病例或 index.json 为空。")
        return

    print(f"📂 待处理病例数: {len(cases)}")

    semaphore = asyncio.Semaphore(args.concurrency)

    tasks = []
    for case_data in cases:
        task = asyncio.create_task(
            run_one_case_async(
                case_data,
                output_dir,
                semaphore,
                max_turns=args.max_turns,
            )
        )
        tasks.append(task)

    await asyncio.gather(*tasks)


if __name__ == "__main__":
    asyncio.run(main())
