import os
import json
from typing import Dict
import random

from tqdm import trange, tqdm

from ..utils.llm_output_check import parse_list
from ..base import register_tool, init_tool_instance
from ..prompts_en import question_asker_system, expert_system, \
    dlg_based_writer_system, dlg_based_writer_prompt, chapter_writer_system


def json_parse_outline(outline):
    outline = outline.strip("```json").strip("```")
    try:
        outline = json.loads(outline)
        if not isinstance(outline, dict):
            return False
        if outline.keys() != {"story_title", "story_outline"}:
            return False
        for chapter in outline["story_outline"]:
            if chapter.keys() != {"chapter_title", "chapter_summary"}:
                return False
    except json.decoder.JSONDecodeError:
        return False
    return True


# 数据驱动prompt
data_driven_writer_system = """你是一个专业的数据故事作家。你的任务是根据提供的数据内容直接生成连贯的故事页面。
数据内容可能包含各种信息，如统计数据、事件描述、用户反馈等。
请基于这些真实数据创作一个引人入胜的故事，确保故事内容与数据紧密相关，避免添加不必要的主观臆测或虚构内容。
每个故事页面应该简洁明了，直接反映数据中的关键信息。"""

data_driven_chapter_system = """你是一个专业的数据故事作家。请根据提供的数据内容，为当前数据片段生成详细的故事页面。
确保每个页面都基于真实数据，内容准确且连贯。避免添加数据之外的主观臆测。
输出格式必须是一个Python列表，每个元素是一个字符串，表示一个故事页面。"""


@register_tool("qa_outline_story_writer")
class QAOutlineStoryWriter:

    def __init__(self,
                 cfg: Dict):
        self.cfg = cfg
        self.temperature = cfg.get("temperature", 1.0)
        self.max_conv_turns = cfg.get("max_conv_turns", 3)
        self.num_outline = cfg.get("num_outline", 4)
        self.llm_type = cfg.get("llm", "qwen")

    def generate_data_summary(self, data_content):
        """从数据内容中提取关键信息并生成数据摘要"""
        data_analyzer = init_tool_instance({
            "tool": self.llm_type,
            "cfg": {
                "system_prompt": "你是一个数据分析专家。请分析提供的数据内容，提取关键信息和主要趋势。",
                "track_history": False
            }
        })
        
        analysis_prompt = f"""请分析以下数据内容，提取关键信息并生成一个简洁的数据摘要：
        
数据内容：
{data_content}

请返回一个JSON格式的摘要，包含以下字段：
- "data_key_points": 数据中的关键发现或趋势
- "main_themes": 数据涉及的主要主题
- "recommended_story_flow": 建议的故事流程（基于数据逻辑）"""

        summary, success = data_analyzer.call(analysis_prompt)
        try:
            summary = json.loads(summary.strip("```json").strip("```"))
            return summary
        except:
            # 如果解析失败，返回默认结构
            return {
                "data_key_points": ["基于提供的数据内容"],
                "main_themes": ["数据驱动的故事"],
                "recommended_story_flow": "按照数据逻辑展开"
            }

    def generate_outline(self, data_content):
        """基于数据内容生成故事结构"""
        # 生成数据摘要
        data_summary = self.generate_data_summary(data_content)
        
        # 使用数据驱动的故事生成器
        writer = init_tool_instance({
            "tool": self.llm_type,
            "cfg": {
                "system_prompt": data_driven_writer_system,
                "track_history": False
            }
        })
        
        writer_prompt = f"""基于以下数据内容生成一个故事大纲：

数据摘要：
- 关键发现：{data_summary['data_key_points']}
- 主要主题：{data_summary['main_themes']}
- 建议流程：{data_summary['recommended_story_flow']}

完整数据内容：
{data_content}

请生成一个包含{self.num_outline}个章节的故事大纲，每个章节都应该基于数据的特定部分。
返回JSON格式，包含story_title和story_outline，其中story_outline是一个列表，每个元素包含chapter_title和chapter_summary。"""

        outline, success = writer.call(writer_prompt, success_check_fn=json_parse_outline)
        if success:
            outline = json.loads(outline.strip("```json").strip("```"))
        else:
            # 如果生成失败，创建一个默认的大纲结构
            outline = {
                "story_title": "数据驱动的故事",
                "story_outline": [
                    {
                        "chapter_title": "数据概述",
                        "chapter_summary": "介绍数据的基本情况和主要发现"
                    },
                    {
                        "chapter_title": "关键分析",
                        "chapter_summary": "深入分析数据中的关键信息"
                    },
                    {
                        "chapter_title": "结论与应用",
                        "chapter_summary": "总结数据启示和实际应用"
                    }
                ]
            }
        return outline

    def generate_story_from_outline(self, outline, data_content):
        """基于大纲和数据内容生成具体的故事页面"""
        chapter_writer = init_tool_instance({
            "tool": self.llm_type,
            "cfg": {
                "system_prompt": data_driven_chapter_system,
                "track_history": False
            }
        })
        
        all_pages = []
        for idx, chapter in enumerate(tqdm(outline["story_outline"])):
            chapter_detail, success = chapter_writer.call(
                json.dumps(
                    {
                        "data_content": data_content,
                        "current_chapter": chapter,
                        "completed_story": all_pages
                    },
                    ensure_ascii=False
                ),
                success_check_fn=parse_list,
                temperature=self.temperature
            )
            
            # 如果生成失败，重试几次
            retry_count = 0
            while not success and retry_count < 3:
                chapter_detail, success = chapter_writer.call(
                    json.dumps(
                        {
                            "data_content": data_content,
                            "current_chapter": chapter,
                            "completed_story": all_pages
                        },
                        ensure_ascii=False
                    ),
                    seed=random.randint(0, 100000),
                    temperature=self.temperature,
                    success_check_fn=parse_list
                )
                retry_count += 1
            
            if success:
                try:
                    pages = eval(chapter_detail)
                    if isinstance(pages, list):
                        pages = [page.strip() for page in pages]
                        all_pages.extend(pages)
                    else:
                        # 如果返回的不是列表，创建默认页面
                        all_pages.append(f"第{idx+1}章: {chapter['chapter_title']}")
                except:
                    # 如果解析失败，创建默认页面
                    all_pages.append(f"第{idx+1}章: {chapter['chapter_title']}")
            else:
                # 如果所有重试都失败，创建默认页面
                all_pages.append(f"第{idx+1}章: {chapter['chapter_title']}")
                all_pages.append(f"基于数据的分析内容")
        
        return all_pages

    def call(self, params):
        """主调用函数，现在接受数据内容作为输入"""
        # 检查参数类型，支持字符串或字典格式的数据
        if isinstance(params, dict):
            # 如果是字典，检查是否有文件路径
            if "file_path" in params:
                # 从文件读取数据
                try:
                    with open(params["file_path"], 'r', encoding='utf-8') as f:
                        data_content = f.read()
                except Exception as e:
                    print(f"读取文件失败: {e}")
                    return ["文件读取失败，请检查文件路径"]
            else:
                # 从data_content字段获取数据
                data_content = params.get("data_content", str(params))
        else:
            # 如果是字符串，检查是否是文件路径
            if os.path.exists(params) and os.path.isfile(params):
                try:
                    with open(params, 'r', encoding='utf-8') as f:
                        data_content = f.read()
                except Exception as e:
                    print(f"读取文件失败: {e}")
                    return ["文件读取失败，请检查文件路径"]
            else:
                # 直接使用字符串作为数据内容
                data_content = str(params)
        
        # 生成故事大纲
        outline = self.generate_outline(data_content)
        
        # 基于大纲和数据生成故事页面
        pages = self.generate_story_from_outline(outline, data_content)
        
        return pages


# 数据驱动故事生成器
@register_tool("data_driven_story_writer")
class DataDrivenStoryWriter:

    def __init__(self, cfg: Dict):
        self.cfg = cfg
        self.temperature = cfg.get("temperature", 1.0)
        self.llm_type = cfg.get("llm", "qwen")
        
        # 直接生成故事的prompt
        self.direct_story_system = """你是一个专业的数据故事作家。请根据提供的数据内容直接生成一个连贯的故事。
        故事应该分为多个页面，每个页面包含一个完整的思想或数据点。
        确保故事内容真实反映数据，避免虚构和主观臆测。
        输出格式必须是一个Python列表，每个元素是一个字符串，表示一个故事页面。"""

    def call(self, params):
        """直接根据数据生成故事页面，支持文件输入"""
        # 检查参数类型，支持字符串或字典格式的数据
        if isinstance(params, dict):
            # 如果是字典，检查是否有文件路径
            if "file_path" in params:
                # 从文件读取数据
                try:
                    with open(params["file_path"], 'r', encoding='utf-8') as f:
                        data_content = f.read()
                except Exception as e:
                    print(f"读取文件失败: {e}")
                    return ["文件读取失败，请检查文件路径"]
            else:
                # 从data_content字段获取数据
                data_content = params.get("data_content", str(params))
        else:
            # 如果是字符串，检查是否是文件路径
            if os.path.exists(params) and os.path.isfile(params):
                try:
                    with open(params, 'r', encoding='utf-8') as f:
                        data_content = f.read()
                except Exception as e:
                    print(f"读取文件失败: {e}")
                    return ["文件读取失败，请检查文件路径"]
            else:
                # 直接使用字符串作为数据内容
                data_content = str(params)
        
        # 如果数据内容为空，使用默认数据
        if not data_content.strip():
            data_content = "暂无数据内容，请提供具体的数据信息"
        
        # 生成故事
        story_writer = init_tool_instance({
            "tool": self.llm_type,
            "cfg": {
                "system_prompt": self.direct_story_system,
                "track_history": False
            }
        })
        
        story_prompt = f"""请基于以下数据内容直接生成一个连贯的故事，分为多个页面：

数据内容：
{data_content}

请生成5-8个故事页面，每个页面应该：
1. 基于数据的特定部分
2. 内容简洁明了
3. 保持逻辑连贯性
4. 避免主观臆测

不要生成超过8个页面。
返回一个Python列表格式的故事页面。"""
        
        pages, success = story_writer.call(story_prompt, success_check_fn=parse_list)
        
        if success:
            try:
                pages = eval(pages)
                if isinstance(pages, list):
                    return [page.strip() for page in pages]
            except:
                pass
        
        # 如果生成失败，返回默认故事结构
        return [
            "数据概述：介绍提供的数据基本情况",
            "关键发现：分析数据中的主要趋势",
            "深度解读：深入理解数据含义",
            "实际应用：探讨数据的现实意义",
            "总结展望：总结数据启示"
        ]