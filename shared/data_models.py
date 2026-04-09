# Copyright (c) 2026 Huangwei Chen
# Author: Huangwei Chen

from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
from pydantic import BaseModel, Field


# The "Scratchpad" for building the SOAP Note
@dataclass
class CaseFeatures:
    """SOAP首次病程录 - 病例特点"""

    basic_info: str = field(
        default="",
        metadata={
            "description": "基本信息：患者的一般信息，包括年龄、性别、职业等人口学特征，以及主诉"
        },
    )
    present_illness: str = field(
        default="",
        metadata={
            "description": "现病史：当前疾病的发生、发展、演变过程，包括起病时间、主要症状、诊治经过等"
        },
    )
    past_history: str = field(
        default="",
        metadata={
            "description": "既往史：患者过去的健康状况，包括既往疾病史、手术史、外伤史、过敏史等"
        },
    )
    physical_exam: str = field(
        default="",
        metadata={
            "description": "体格检查：医生的系统性检查发现，包括生命体征、头颈部、胸部、腹部、四肢等各系统的检查结果"
        },
    )
    aux_exam: str = field(
        default="",
        metadata={
            "description": "辅助检查：实验室检查、影像学检查、心电图等辅助诊断检查的结果"
        },
    )
    complete: bool = False


@dataclass
class DiagnosisAndPlan:
    """SOAP首次病程录 - 初步诊断、拟诊讨论、诊疗计划"""

    preliminary_diagnosis: str = field(
        default="",
        metadata={
            "description": "初步诊断：基于现有信息得出的最可能的诊断，包括疾病名称和诊断依据"
        },
    )
    diagnosis_discussion: str = field(
        default="",
        metadata={
            "description": "拟诊讨论：鉴别诊断分析，讨论其他可能的诊断及其支持或反对的理由"
        },
    )
    treatment_plan: str = field(
        default="",
        metadata={
            "description": "诊疗计划：具体的治疗和管理方案，包括进一步检查、药物治疗、手术计划、随访安排等"
        },
    )
    complete: bool = False


@dataclass
class SOAPNote:
    """SOAP首次病程录"""

    case_features: CaseFeatures = field(default_factory=CaseFeatures)
    diagnosis_and_plan: Optional[DiagnosisAndPlan] = field(
        default_factory=DiagnosisAndPlan
    )

    @property
    def case_features_complete(self) -> bool:
        return self.case_features.complete

    @property
    def diagnosis_complete(self) -> bool:
        return bool(self.diagnosis_and_plan and self.diagnosis_and_plan.complete)


# Orchestrator Agent standard output format
class OrchestratorDecision(BaseModel):
    """Orchestrator 的决策输出。"""

    active_specialists: List[str] = Field(
        description="需要激活的专科 Agent ID 列表，可以为空。 e.g., ['general_practice', 'cardiology'], []"
    )
    instructions_to_specialists: str = Field(
        description="active_specialists不为空时，本字段为给所有专科 Agent 的统一指令，告诉他们本轮的重点；active_specialists为空时，本字段为直接给Aggregator Agent 的指令。"
    )
    suggested_question: Optional[str] = Field(
        description="如果判断不需要专科医生介入（如简单追问），请在此直接填写建议问患者的问题。"
    )


# Specialist Agent standard output format
class SpecialistOutput(BaseModel):
    """每个专科 Agent 在每一轮的输出"""

    next_question: str = Field(description="我下一步想问患者的一个具体问题。")
    draft_modifications: str = Field(
        description="基于我的思考，建议对首程草稿进行的具体修改内容。"
    )


# Aggregator Agent standard output format
class AggregatorOutputPhase1(BaseModel):
    """第一阶段（问诊）Aggregator 输出：修改 CaseFeatures"""

    next_question_to_patient: str = Field(description="下一步医生要问病人的问题。")
    historytaking_complete: bool = Field(description="病例特点收集是否完成。")
    updated_case_features: CaseFeatures = Field(
        description="更新后的病例特点部分（Case Features）。"
    )


class AggregatorOutputPhase2(BaseModel):
    """第二阶段（诊断）Aggregator 输出：生成 DiagnosisAndPlan"""

    diagnosis_and_plan: DiagnosisAndPlan = Field(
        description="生成的初步诊断、拟诊讨论和诊疗计划。"
    )


# Vadlidation standard output format
@dataclass
class SpecialistEvaluation:
    """对单个专科 Agent 的评估"""

    specialist_id: str
    score: float
    contribution_rating: float
    suggestions_for_improvement: str


@dataclass
class ValidationReport:
    """对整个 Case 的最终评估报告"""

    case_id: str
    generated_soap: SOAPNote
    ground_truth_soap: SOAPNote
    overall_soap_score: float
    specialist_evals: List[SpecialistEvaluation]
