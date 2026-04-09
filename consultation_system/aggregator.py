# Copyright (c) 2026 Huangwei Chen
# Author: Huangwei Chen

import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.messages import HumanMessage
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

from shared.data_models import AggregatorOutputPhase1, AggregatorOutputPhase2, SOAPNote
from shared.prompt_base import get_prompt
from utils.utils import instantiate_chat_model

SOAP_STRUCTURE_GUIDANCE = {
    "historytaking": """
**基本信息 (basic_info)**：患者人口学特征和主诉
- 包括：年龄、性别、职业、就诊时间、主要症状等
- 要求：简洁明确，突出主要问题
- 示例：
    - 1. 基本信息：患者,男,62岁,因“左侧阵发性腰胀痛4个月余”来院就诊。

**现病史 (present_illness)**：当前疾病的发生发展过程
- 包括：起病情况、症状特点、发展演变、诊治经过、效果等
- 要求：按时间顺序，逻辑清晰
- 示例：
    - 2. 现病史：患者4月前无明显诱因下出现左侧腰部疼痛。疼痛不剧,为阵发性胀痛,可忍受,活动及休息后无明显缓解或加重,无明显放射痛,无血尿及尿路刺激症状等其他表现。无排尿疼痛、血尿、尿液异常、发热、畏寒、寒战、恶心、呕吐等伴随症状。2周前当地医院（浙江省XX市人民医院）腹部CT提示：双肾结石,建议住院治疗,患者拒绝,未行进一步诊治。患者遂来我院就诊,门诊完善检查：血常规：WBC 4.63×10⁹/L,Hb 143g/L,PLT 202×10⁹/L。尿常规：WBC 29.4cells/µl,细菌数量127.8/µl,亚硝酸盐阴性。尿培养阴性。肾功能：Cr 147µmol/L,BUN 7.4mmol/L。CT：腰椎侧弯、前凸,双侧肾脏位置向后上方移位,大小如常,形态规整,双肾结石。CT值：1200HU。现拟以“双肾结石”收治我科。

**既往史 (past_history)**：患者过去的健康状况
- 包括：既往疾病史、手术史、外伤史、过敏史、个人史、家族史等
- 要求：与当前疾病相关的历史信息
- 示例：
    - 3. 既往史：
        3.1 既往慢性疾病：高血压病史6年,间断口服替米沙坦治疗,血压控制尚可。否认糖尿病、心脏病等慢性疾病史。
        3.2 手术史外伤及输血史：55年前（7岁）高处坠落伤致胸椎及脊柱畸形。30年前腰椎脓肿2次,局部引流后治愈。10年前因“双肾结石”行体外冲击波碎石治疗,具体不详,之后未复查。脑出血病史2年,保守治疗。否认输血史。
        3.3 婚育、月经及家族史 22岁结婚,有1子,家人均体健。
        3.4 吸烟、饮酒、成瘾性药物应用史：否认吸烟饮酒、成瘾性药物应用史。
        3.5 疫苗接种史：近1年未接种疫苗。
        3.6 过敏史：否认过敏史。
        3.7 个人生活史：长期浙江省温州市居住生活,为退休公司职员,否认毒化、放射物接触史。否认疫区居留史。

**体格检查 (physical_exam)**：医生检查发现的客观体征
- 包括：生命体征、各系统检查发现
- 要求：客观描述，避免主观判断
- 示例：
    - 4.体格检查：身高186cm,体重63.8kg,体温36.6℃,心率65次/分,呼吸18次/分,血压170/102mmHg。神清,精神可,肤巩膜无黄染,浅表淋巴结未及明显肿大。心肺听诊无殊,腹平软,无压痛反跳痛,肝脾肋下未及,脊柱重度后凸、侧凸畸形。双侧肾区叩击痛阴性,双输尿管径路无压痛,膀胱耻骨上未及,患者拒绝直肠指检,双下肢无水肿,病理征未引出。

**辅助检查 (aux_exam)**：仪器和实验室检查结果
- 包括：化验、影像、心电图等客观检查数据
- 要求：包含具体数值和检查时间
- 示例：
    - 5. 辅助检查：我院,今日 血常规示WBC 4.63×10⁹/L,Hb 143g/L,PLT 202×10⁹/L；外院尿常规示WBC 29.4cells/µl,细菌数量127.8/µl,亚硝酸盐阴性；浙江省XX市人民医院,2周前尿培养阴性；肾功能示Cr 147µmol/L,BUN 7.4mmol/L；CT示腰椎侧弯、前凸,双侧肾脏位置向后上方移位,大小如常,形态规整,双肾结石,CT值1200HU。左侧肾结石约2.7cm×3.6cm,右侧肾结石约1.2×1.0cm。
""",
    "diagnosis_and_plan": """
**初步诊断 (preliminary_diagnosis)**：
    - 按常规临床格式逐行列出主要诊断，常见写法如“左输尿管结石”“泌尿系感染”。
    - 需要包含既往史和手术史里的次要诊断。
    - 按重要性或时间顺序排列。

**拟诊讨论（diagnosis_discussion）**：
    - 包括诊断依据和鉴别诊断。
    - 诊断依据:
        - 像病历小结一样，将关键信息（人群特征、起病时间、诱因、主要症状、伴随症状、体格检查阳性、关键检查提示）串成流畅段落。
        - 请在段落中**显式分句**说明：
            - 危险因素与健康问题（2）：代谢/解剖/职业暴露等（未知可写“未明”）。
            - 并发症或其他临床情况（3）：如梗阻性肾病、感染并发症、肾功能分期。
            - 依从性（4）：既往复诊/服药规律性、对方案的接受度。
            - 家庭可利用资源（5）：照护者、经济/交通条件、就医可及性。
        - 避免列未问及的阴性；不要虚构检查。

**诊疗计划（treatment_plan）**：
    - 1. 进一步诊疗计划（1）：需进行的检查及目的/时机，参考EAU指南，提及每一个决策的依据（如复查尿常规、影像学以评估）。有手术指征时，诊疗计划需要包含手术指征、手术方案及原因。
    - 2. 药物治疗（2.1）：给出**药物类属**与适应证/禁忌/监测点（避免具体剂量以免幻觉，可给首选/备选类属）。
    - 3. 非药物干预与健康教育（2.2）：饮水量目标、饮食（限盐/草酸/嘌呤等按病情取舍）、运动/体重管理、疼痛自我管理、红旗症状何时急诊。
    - 4. 随诊要求（3）：复诊时间窗（如“1–2周或症状波动时提前”）、复查项目清单、远程/线下偏好（结合依从性与家庭资源）。
""",
}

AGGREGATOR_PHASE_CONFIG = {
    "historytaking": {
        "phase_name": "病史采集",
        "task_description": "汇总专科医生的提问建议，向患者提问。当患者提及关键症状时，需要进行确认。患者是普通人，不要使用过于专业、生僻的术语。重要提醒：必须确保收集完整的病例特点，包括基本信息、现病史、既往史、体格检查和辅助检查。只有在体格检查和辅助检查都收集完成后，才能设置 historytaking_complete 为 true。",
    },
    "diagnosis_and_plan": {
        "phase_name": "诊断与计划",
        "task_description": "汇总所有专科医生的意见，生成初步诊断、拟诊讨论和诊疗计划。",
    },
}


def create_aggregator_agent(phase: str = "historytaking"):
    """
    使用 LangChain 的 create_agent() 创建 Aggregator Agent。
    """
    raw_prompt = get_prompt("aggregator")

    phase_config = AGGREGATOR_PHASE_CONFIG.get(
        phase, AGGREGATOR_PHASE_CONFIG["historytaking"]
    )
    soap_guidance = SOAP_STRUCTURE_GUIDANCE.get(
        phase, SOAP_STRUCTURE_GUIDANCE["historytaking"]
    )

    aggregator_prompt = raw_prompt.format(
        PHASE_NAME=phase_config["phase_name"],
        AGGREGATOR_TASK_DESCRIPTION=phase_config["task_description"],
        SOAP_STRUCTURE_GUIDANCE=soap_guidance,
    )

    llm = instantiate_chat_model(model_name="deepseek-chat")
    checkpointer = InMemorySaver()
    if phase == "historytaking":
        output_schema = AggregatorOutputPhase1
    else:
        output_schema = AggregatorOutputPhase2

    agent = create_agent(
        model=llm,
        system_prompt=aggregator_prompt,
        response_format=output_schema,
        checkpointer=checkpointer,
    )

    return agent


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv(override=True)
    from shared.data_models import SpecialistOutput, SOAPNote, CaseFeatures

    # 测试 Aggregator Agent
    aggregator = create_aggregator_agent()
    config = {"configurable": {"thread_id": "1"}}

    current_phase = "historytaking"
    print(f"当前阶段: {current_phase}")
    specialist_outputs = [
        SpecialistOutput(
            next_question="您的胸痛具体是什么感觉？是压迫感、紧缩感还是烧灼感？疼痛会向其他部位放射吗，比如左肩、左臂、下巴或背部？",
            draft_modifications="在现病史部分补充：患者胸痛性质待进一步明确，需详细描述疼痛特征、放射部位、持续时间、诱发因素（如活动、情绪激动等）和缓解方式（如休息、含服硝酸甘油等）。心电图ST段轻度抬高需警惕急性冠脉综合征可能。建议完善心肌酶谱、心脏超声等检查。",
        )
    ]

    draft = SOAPNote(
        case_features=CaseFeatures(
            basic_info="患者，男性，45岁，吸烟史20年",
            present_illness="胸口疼痛，气短2周",
            past_history="高血压5年",
            physical_exam="心率90次/分，血压140/90 mmHg",
            aux_exam="心电图示ST段轻度抬高",
            complete=False,
        ),
        diagnosis_and_plan=None,
    )

    if current_phase == "historytaking":
        aggregator_output_raw = aggregator.invoke(
            {
                "messages": [
                    HumanMessage(
                        content=f"""
当前SOAP草稿:
{draft}

专科医生输出:
{specialist_outputs}

请基于以上信息，更新SOAP草稿并决定接下来的问题或是否结束病例特点收集。
"""
                    )
                ]
            },
            config=config,
        )

    aggregator_output = aggregator_output_raw["structured_response"]

    next_question = aggregator_output.next_question_to_patient
    is_complete = aggregator_output.historytaking_complete
    updated_soap = aggregator_output.updated_draft

    print("🔗 Aggregator Output:")
    print(f"""
接下来的问题: {next_question}

病例特点收集完成: {is_complete}

更新后的SOAP草稿:{updated_soap}
""")
