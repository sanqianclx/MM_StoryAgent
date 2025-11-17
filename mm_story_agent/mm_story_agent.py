import time
import json
from pathlib import Path

import torch.multiprocessing as mp
mp.set_start_method("spawn", force=True)

from .base import init_tool_instance


class MMStoryAgent:

    def __init__(self) -> None:
        self.modalities = ["image",  "speech"]

    def call_modality_agent(self, modality, agent, params, return_dict):
        result = agent.call(params)
        return_dict[modality] = result

    def write_story(self, config):
        cfg = config["story_writer"]
        story_writer = init_tool_instance(cfg)
        pages = story_writer.call(cfg["params"])
        return pages
    
    def generate_modality_assets(self, config, pages):
        script_data = {"pages": [{"story": page} for page in pages]}
        story_dir = Path(config["story_dir"])

        for sub_dir in self.modalities:
            (story_dir / sub_dir).mkdir(exist_ok=True, parents=True)

        agents = {}
        params = {}
        enabled_modalities = []  # 添加：跟踪启用的模态
        
        # 添加：检查每个模态是否启用
        for modality in self.modalities:
            if config.get(f"enable_{modality}", True):  # 默认启用
                agents[modality] = init_tool_instance(config[modality + "_generation"])
                params[modality] = config[modality + "_generation"]["params"].copy()
                params[modality].update({
                    "pages": pages,
                    "save_path": story_dir / modality
                })
                enabled_modalities.append(modality)
            else:
                print(f"⏭️ 跳过{modality}生成")

        processes = []
        return_dict = mp.Manager().dict()

        # 修改：只处理启用的模态
        for modality in enabled_modalities:
            p = mp.Process(
                target=self.call_modality_agent,
                args=(
                    modality,
                    agents[modality],
                    params[modality],
                    return_dict)
                )
            processes.append(p)
            p.start()
        
        for p in processes:
            p.join()

        images = []
        for modality, result in return_dict.items():
            try:
                if modality == "image":
                    images = result["generation_results"]
                    for idx in range(len(pages)):
                        script_data["pages"][idx]["image_prompt"] = result["prompts"][idx]
            except Exception as e:
                print(f"Error occurred during generation: {e}")
        
        with open(story_dir / "script_data.json", "w") as writer:
            json.dump(script_data, writer, ensure_ascii=False, indent=4)
        
        return images
    
    def compose_storytelling_video(self, config, pages):
        # 添加：检查是否启用视频合成
        if not config.get("enable_video", True):
            print("⏭️ 跳过视频合成")
            return
            
        video_compose_agent = init_tool_instance(config["video_compose"])
        params = config["video_compose"]["params"].copy()
        params["pages"] = pages
        video_compose_agent.call(params)

    def call(self, config):
        # 添加：检查是否启用故事生成
        if config.get("enable_story", True):
            pages = self.write_story(config)
        else:
            print("⏭️ 跳过故事生成")
            pages = ["默认故事页面1", "默认故事页面2", "默认故事页面3"]
            
        images = self.generate_modality_assets(config, pages)
        self.compose_storytelling_video(config, pages)