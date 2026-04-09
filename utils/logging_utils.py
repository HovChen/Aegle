# Copyright (c) 2026 Huangwei Chen
# Author: Huangwei Chen

import os
import json
from datetime import datetime, timezone
from typing import Optional, Any
from contextvars import ContextVar

_current_logger_ctx = ContextVar("current_logger", default=None)


def set_run_logger(logger: "RunLogger"):
    """
    设置当前上下文的 Logger。
    在每个 Case 开始运行的任务中调用此函数。
    """
    _current_logger_ctx.set(logger)


def get_run_logger() -> Optional["RunLogger"]:
    """
    获取当前上下文的 Logger。
    在 Agent 工具或深层函数中调用。
    """
    return _current_logger_ctx.get()


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class RunLogger:
    def __init__(self, case_dir: str, session_tag: Optional[str] = None):
        """
        :param case_dir: 运行日志保存目录，例如 output/case_idx
        :param session_tag: 可选标签，用于区分同一 Case 的多次运行
        """
        os.makedirs(case_dir, exist_ok=True)
        self.case_dir = case_dir
        self.session_tag = session_tag

        self._dialog_fp = None  # dialog.jsonl
        self._trace_fp = None  # trace.md

        self._clear_log_files()

    def _clear_log_files(self):
        """
        清空并初始化日志文件
        """
        dialog_path = os.path.join(self.case_dir, "dialog.jsonl")
        trace_path = os.path.join(self.case_dir, "trace.md")

        # 初始化 dialog.jsonl
        with open(dialog_path, "w", encoding="utf-8") as f:
            f.write("")

        # 初始化 trace.md
        with open(trace_path, "w", encoding="utf-8") as f:
            f.write(f"# 问诊系统运行记录\n\n")
            f.write(f"**开始时间**: {now_iso()}\n")
            if self.session_tag is not None:
                f.write(f"**Session Tag**: {self.session_tag}\n")
            f.write("\n---\n\n")

    def _dialog_file(self):
        """获取对话日志文件句柄 (懒加载)"""
        if self._dialog_fp is None:
            path = os.path.join(self.case_dir, "dialog.jsonl")
            self._dialog_fp = open(path, "a", encoding="utf-8")
        return self._dialog_fp

    def _trace_file(self):
        """获取追踪日志文件句柄 (懒加载)"""
        if self._trace_fp is None:
            path = os.path.join(self.case_dir, "trace.md")
            self._trace_fp = open(path, "a", encoding="utf-8")
        return self._trace_fp

    def close(self):
        """关闭文件句柄，释放资源"""
        if self._dialog_fp is not None:
            self._dialog_fp.close()
            self._dialog_fp = None
        if self._trace_fp is not None:
            self._trace_fp.close()
            self._trace_fp = None

    def log_dialog(self, role: str, content: str, ts: Optional[str] = None):
        """记录对话交互 (User/Assistant/System)"""
        if ts is None:
            ts = now_iso()
        item = {
            "role": role,
            "content": content,
            "ts": ts,
        }
        if self.session_tag is not None:
            item["session_tag"] = self.session_tag

        fp = self._dialog_file()
        json.dump(item, fp, ensure_ascii=False)
        fp.write("\n")
        fp.flush()

    def _write_trace_block(self, header: str, body_lines: list[str]):
        """内部辅助函数：写入 Markdown 块"""
        fp = self._trace_file()
        fp.write(header + "\n")
        for line in body_lines:
            fp.write(line.rstrip() + "\n")
        fp.write("\n")
        fp.flush()

    def log_orchestrator(self, phase: str, active_specialists: Any, instructions: str):
        """记录 Orchestrator 的决策"""
        ts = now_iso()
        header = f"## [{ts}] Orchestrator ({phase})"
        body = [
            f"- 激活专科: {active_specialists}",
            "- 指令:",
            f"  {instructions.replace(chr(10), chr(10) + '  ')}",
        ]
        if self.session_tag is not None:
            body.insert(0, f"- session_tag: {self.session_tag}")
        self._write_trace_block(header, body)

    def log_activated_specialists_file(self, specialists):
        """
        更新 activated_specialists.json
        记录整个 session 中激活过的所有专家列表
        """
        unique_specs = sorted(set(specialists))
        path = os.path.join(self.case_dir, "activated_specialists.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(unique_specs, f, ensure_ascii=False, indent=2)

    def log_specialist(
        self,
        input: str,
        phase: str,
        spec_id: str,
        next_question: Optional[str],
        draft_modifications: Any,
    ):
        """记录 Specialist 的输入和输出"""
        ts = now_iso()
        header = f"## [{ts}] Specialist {spec_id} ({phase})"
        body = []
        if self.session_tag is not None:
            body.append(f"- session_tag: {self.session_tag}")
            body.append("")

        if input:
            body.append(f"- 输入内容: {input.replace(chr(10), chr(10) + '  ')}")
            body.append("")

        if next_question is not None:
            body.extend(
                [
                    "- 建议问题:",
                    f"  {next_question.replace(chr(10), chr(10) + '  ')}",
                ]
            )
            body.append("")

        body.extend(
            [
                "- 草稿修改建议:",
                f"  {str(draft_modifications).replace(chr(10), chr(10) + '  ')}",
            ]
        )
        self._write_trace_block(header, body)

    def log_tool_usage(self, tool_name: str, tool_input: str, tool_output: str):
        """
        记录工具(Tool)的调用输入和输出。
        自动截断过长的输出内容，防止日志文件爆炸。
        """
        ts = now_iso()
        header = f"## [{ts}] Tool Usage: {tool_name}"
        body = []
        if self.session_tag is not None:
            body.append(f"- session_tag: {self.session_tag}")

        body.append(f"- Input: {tool_input}")

        if len(tool_output) > 500:
            preview = tool_output[:500].replace(chr(10), chr(10) + "  ")
            body.append(
                f"- Output (Preview): {preview}...\n  (Total {len(tool_output)} chars)"
            )
        else:
            body.append(f"- Output: {tool_output.replace(chr(10), chr(10) + '  ')}")

        self._write_trace_block(header, body)

    def log_aggregator_historytaking(
        self,
        input: str,
        phase: str,
        next_question: str,
        historytaking_complete: bool,
        case_features_summary: str,
    ):
        """记录 Aggregator 在问诊阶段的输出"""
        ts = now_iso()
        header = f"## [{ts}] Aggregator ({phase})"
        body = []
        if self.session_tag is not None:
            body.append(f"- session_tag: {self.session_tag}")

        body.extend(
            [
                f"- 输入内容：{input.replace(chr(10), chr(10) + '  ')}",
            ]
        )
        body.extend(
            [
                f"- 接下来的问题: {next_question}",
                f"- 病例特点收集完成: {'是' if historytaking_complete else '否'}",
                "- 更新后的病例特点摘要:",
                f"  {case_features_summary.replace(chr(10), chr(10) + '  ')}",
            ]
        )
        self._write_trace_block(header, body)

    def log_aggregator_diagnosis(self, phase: str, diagnosis_and_plan: Any):
        """记录 Aggregator 在诊断阶段的输出"""
        ts = now_iso()
        header = f"## [{ts}] Aggregator ({phase})"
        body = []
        if self.session_tag is not None:
            body.append(f"- session_tag: {self.session_tag}")
        body.extend(
            [
                "- 诊断与计划更新:",
                f"  {str(diagnosis_and_plan).replace(chr(10), chr(10) + '  ')}",
            ]
        )
        self._write_trace_block(header, body)

    def write_soap_markdown(self, soap: Any):
        """
        写/覆盖 soap_note.md 为可读文档。
        """
        path = os.path.join(self.case_dir, "soap_note.md")
        with open(path, "w", encoding="utf-8") as f:
            # 1. 病例特点
            cf = soap.case_features
            f.write("## 一、病例特点\n\n")

            f.write("### （一）基本信息\n")
            f.write(f"{cf.basic_info if cf.basic_info else '未记录'}\n\n")

            f.write("### （二）现病史\n")
            f.write(f"{cf.present_illness if cf.present_illness else '未记录'}\n\n")

            f.write("### （三）既往情况\n")
            f.write(f"{cf.past_history if cf.past_history else '未记录'}\n\n")

            f.write("### （四）体格检查\n")
            f.write(f"{cf.physical_exam if cf.physical_exam else '未记录'}\n\n")

            f.write("### （五）辅助检查\n")
            f.write(f"{cf.aux_exam if cf.aux_exam else '未记录'}\n\n")

            # 2. 诊断与计划
            dp = soap.diagnosis_and_plan
            if dp:
                f.write("## 二、初步诊断\n")
                f.write(
                    f"{dp.preliminary_diagnosis if dp.preliminary_diagnosis else '未记录'}\n\n"
                )

                f.write("## 三、拟诊讨论\n")
                f.write(
                    f"{dp.diagnosis_discussion if dp.diagnosis_discussion else '未记录'}\n\n"
                )

                f.write("## 四、诊疗计划\n")
                f.write(f"{dp.treatment_plan if dp.treatment_plan else '未记录'}\n")
            else:
                f.write("\n*(诊断与计划部分尚未生成)*\n")

    def log_section(self, title: str, content: str, collapsed: bool = False):
        """
        记录一个通用的分节日志。
        """
        ts = now_iso()
        header = f"## [{ts}] {title}"

        body = []
        if self.session_tag is not None:
            body.append(f"- session_tag: {self.session_tag}")

        body.append(content)

        self._write_trace_block(header, body)
