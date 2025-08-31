import abc
import os
import re
from typing import Dict, Any, List

import yaml
from jinja2 import Template

from biz.llm.factory import Factory
from biz.utils.log import logger
from biz.utils.token_util import count_tokens, truncate_text_by_tokens


class BaseReviewer(abc.ABC):
    """代码审查基类"""

    def __init__(self, prompt_key: str):
        self.client = Factory().getClient()
        self.prompts = self._load_prompts(prompt_key, os.getenv("REVIEW_STYLE", "professional"))

    def _load_prompts(self, prompt_key: str, style="professional") -> Dict[str, Any]:
        """加载提示词配置"""
        prompt_templates_file = "conf/prompt_templates.yml"
        try:
            # 在打开 YAML 文件时显式指定编码为 UTF-8，避免使用系统默认的 GBK 编码。
            with open(prompt_templates_file, "r", encoding="utf-8") as file:
                prompts = yaml.safe_load(file).get(prompt_key, {})

                # 使用Jinja2渲染模板
                def render_template(template_str: str) -> str:
                    return Template(template_str).render(style=style)

                system_prompt = render_template(prompts["system_prompt"])
                user_prompt = render_template(prompts["user_prompt"])

                return {
                    "system_message": {"role": "system", "content": system_prompt},
                    "user_message": {"role": "user", "content": user_prompt},
                }
        except (FileNotFoundError, KeyError, yaml.YAMLError) as e:
            logger.error(f"加载提示词配置失败: {e}")
            raise Exception(f"提示词配置加载失败: {e}")

    def call_llm(self, messages: List[Dict[str, Any]]) -> str:
        """调用 LLM 进行代码审核"""
        logger.info(f"向 AI 发送代码 Review 请求, messages: {messages}")
        review_result = self.client.completions(messages=messages)
        logger.info(f"收到 AI 返回结果: {review_result}")
        return review_result

    @abc.abstractmethod
    def review_code(self, *args, **kwargs) -> str:
        """抽象方法，子类必须实现"""
        pass


class CodeReviewer(BaseReviewer):
    """代码 Diff 级别的审查"""

    def __init__(self):
        super().__init__("code_review_prompt")

    def review_and_strip_code(self, changes_text: str, commits_text: str = "") -> str:
        """
        Review判断changes_text超出取前REVIEW_MAX_TOKENS个token，超出则截断changes_text，
        调用review_code方法，返回review_result，如果review_result是markdown格式，则去掉头尾的```
        :param changes_text:
        :param commits_text:
        :return:
        """
        # 如果超长，取前REVIEW_MAX_TOKENS个token
        review_max_tokens = int(os.getenv("REVIEW_MAX_TOKENS", 10000))
        # 如果changes为空,打印日志
        if not changes_text:
            logger.info("代码为空, diffs_text = %", str(changes_text))
            return "代码为空"

        # 计算tokens数量，如果超过REVIEW_MAX_TOKENS，截断changes_text
        tokens_count = count_tokens(changes_text)
        if tokens_count > review_max_tokens:
            changes_text = truncate_text_by_tokens(changes_text, review_max_tokens)

        review_result = self.review_code(changes_text, commits_text).strip()
        if review_result.startswith("```markdown") and review_result.endswith("```"):
            return review_result[11:-3].strip()
        return review_result

    def review_code(self, diffs_text: str, commits_text: str = "") -> str:
        """Review 代码并返回结果"""
        # 1) 从 Diff 中提取“新增代码”
        from biz.utils.code_parser import GitDiffParser
        parser = GitDiffParser(diffs_text)
        new_code = parser.get_new_code()  # 仅对新增部分做依赖分析

        # 2) AST 解析 + 正则回退，抽取依赖关键词
        from biz.utils.ast_util import extract_dependencies_from_code
        dependencies = extract_dependencies_from_code(new_code)
        dep_display = ", ".join(dependencies[:20]) + (" ..." if len(dependencies) > 20 else "")

        # 3) 基于依赖做向量检索（带降级策略）
        related_context_blocks = []
        try:
            from biz.utils.vector_store import VectorStore
            top_k = int(os.getenv("DEP_TOP_K", 5))
            store = VectorStore()  # 默认 data/vector_store.json
            hits = store.search_similar(dependencies, top_k=top_k)

            for i, item in enumerate(hits, start=1):
                # item: {"id","name","text","score"}
                related_context_blocks.append(
                    f"[{i}] {item.get('name','')}\nscore={item.get('score',0):.4f}\n{item.get('text','')}"
                )
        except Exception as e:
            logger.warning(f"依赖向量检索失败或跳过: {e}")

        # 4) 组装“依赖上下文”并做 token 限制
        related_context = "\n\n".join(related_context_blocks).strip()
        dep_context_max_tokens = int(os.getenv("DEP_CONTEXT_MAX_TOKENS", 1024))
        if related_context:
            if count_tokens(related_context) > dep_context_max_tokens:
                related_context = truncate_text_by_tokens(related_context, dep_context_max_tokens)

        # 5) 拼接进原有 prompt（保持原有格式不变，仅追加上下文提示段）
        base_user_content = self.prompts["user_message"]["content"].format(
            diffs_text=diffs_text, commits_text=commits_text
        )

        if dependencies and related_context:
            extra = (
                "\n\n【相关依赖上下文（基于AST解析与向量检索）】\n"
                f"- 解析到的依赖: {dep_display}\n"
                "以下是与上述依赖最相关的背景信息（供审查时参考，可能包含约定、调用约束、典型用法/反例等）：\n\n"
                f"{related_context}\n"
                "请结合上述上下文进行更准确的代码审查、问题定位与建议输出。"
            )
            final_user_content = base_user_content + extra
        else:
            final_user_content = base_user_content

        messages = [
            self.prompts["system_message"],
            {"role": "user", "content": final_user_content},
        ]
        return self.call_llm(messages)

    @staticmethod
    def parse_review_score(review_text: str) -> int:
        """解析 AI 返回的 Review 结果，返回评分"""
        if not review_text:
            return 0
        match = re.search(r"总分[:：]\s*(\d+)分?", review_text)
        return int(match.group(1)) if match else 0

