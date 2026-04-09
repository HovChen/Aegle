# Copyright (c) 2026 Huangwei Chen
# Author: Huangwei Chen

import json
import os
from pathlib import Path
from typing import Optional, Dict, Any

BASE_DIR = Path(__file__).parent.parent
PROMPT_DIR = BASE_DIR / "prompts"

PHASE_RULES = {
    "historytaking": {
        "phase_name": "病史采集 (History Taking)",
        "next_question_instruction": "将需要提出的问题填入 `next_question` 以推进问诊，若没有问题，你应该说明结束。",
    },
    "diagnosis_and_plan": {
        "phase_name": "诊断讨论 (Diagnosis & Plan)",
        "next_question_instruction": "`next_question` 字段填 'N/A'，不再向患者提问。",
    },
}


def get_prompt(prompt_id: str) -> str:
    """
    加载普通的静态 prompt (如 orchestrator.prompt, aggregator.prompt, sp.prompt)。
    """
    filepath = PROMPT_DIR / f"{prompt_id}.prompt"

    try:
        if filepath.exists():
            return filepath.read_text(encoding="utf-8")

        raise FileNotFoundError(f"Prompt file not found: {filepath}")

    except Exception as e:
        print(f"Error loading prompt {prompt_id}: {e}")
        return f"Error: Could not load prompt for {prompt_id}."


def load_phase_guidance(phase: str) -> str:
    """
    加载分阶段的SOAP写作指导。
    """
    guidance_path = PROMPT_DIR / "guidance" / f"{phase}_guidance.prompt"

    try:
        if guidance_path.exists():
            return guidance_path.read_text(encoding="utf-8")

    except Exception as e:
        print(f"Error loading phase guidance for {phase}: {e}")
        return "请按照标准SOAP格式记录病历信息。"


def load_specialist_prompt(
    specialist_id: str, current_phase: str = "historytaking"
) -> str:
    """
    动态组装 Specialist System Prompt。
    使用单一的 specialist.json 作为模板，
    不再为每个 specialist_id 创建独立 JSON 文件。
    """
    template_path = PROMPT_DIR / "specialist_core.prompt"
    json_path = PROMPT_DIR / "specialist.json"

    try:
        template_content = template_path.read_text(encoding="utf-8")

        with open(json_path, "r", encoding="utf-8") as f:
            spec_data = json.load(f)

        spec_data["specialist_id"] = specialist_id

        role_desc = spec_data.get("role_description", "")
        if "{spec_id}" in role_desc:
            role_desc = role_desc.replace("{spec_id}", specialist_id)

        phase_rules = PHASE_RULES.get(current_phase, PHASE_RULES["historytaking"])
        phase_data = spec_data.get("phases", {}).get(current_phase, {})

        soap_guidance = load_phase_guidance(current_phase)

        if current_phase == "historytaking":
            phase_instructions = (
                f"请从{specialist_id}的专业角度，仔细审查当前的病例特点。"
                f"重点关注与{specialist_id}相关的实际病情信息：基本信息、现病史、既往情况、体格检查、辅助检查。"
                "如果现有信息不足以完整记录病情，请提出具体的问诊问题以补充信息；"
                "如果信息已足够，请注明无补充问题。"
                "此阶段只记录病情，不制定检查或治疗计划。"
            )
        elif current_phase == "diagnosis_and_plan":
            phase_instructions = (
                f"基于已锁定的病例特点，请提供{specialist_id}视角的专业意见。"
                "在 draft_modification 部分严格按照结构输出：初步诊断、拟诊讨论"
                "（包含诊断依据、鉴别诊断）、诊疗计划（进一步检查、药物治疗、非药物干预、随诊要求）。"
            )

        final_prompt = template_content.format(
            specialist_id=specialist_id,
            phase_name=phase_rules["phase_name"],
            role_description=role_desc,
            soap_guidance=soap_guidance,
            phase_instructions=phase_instructions,
            next_question_instruction=phase_rules["next_question_instruction"],
        )

        return final_prompt

    except Exception as e:
        print(f"🔥 Critical Error assembling prompt for {specialist_id}: {e}")
        return f"Error: Failed to assemble prompt for {specialist_id}"
