import os
import json
from pathlib import Path
from typing import List, Dict
import time

from aliyunsdkcore.client import AcsClient
from aliyunsdkcore.request import CommonRequest
import nls

from mm_story_agent.base import register_tool


class StandardTTSSynthesizer:
    """使用阿里云普通语音合成服务（非CosyVoice大模型）"""

    def __init__(self, cfg=None) -> None:
        # 直接从环境变量获取Token和AppKey
        self.token = os.environ.get('ALIYUN_ACCESS_TOKEN')
        self.app_key = os.environ.get('ALIYUN_APP_KEY') or (cfg.get("app_key") if cfg else None)
        self.region = cfg.get("region", "cn-shanghai") if cfg else "cn-shanghai"
        self.sample_rate = cfg.get("sample_rate", 16000) if cfg else 16000
        
        # 验证必要的凭据
        self._validate_credentials()
        
        print(f"✅ 使用普通语音合成服务")
        print(f"✅ 使用Token: {self.token[:10]}...")
        print(f"✅ 使用AppKey: {self.app_key}")
        print(f"✅ 使用地域: {self.region}")

    def _validate_credentials(self):
        """验证凭据是否完整"""
        missing = []
        if not self.token:
            missing.append("ALIYUN_ACCESS_TOKEN")
        if not self.app_key:
            missing.append("ALIYUN_APP_KEY")
        
        if missing:
            raise ValueError(
                f"Missing required credentials: {', '.join(missing)}. "
                f"Please set these environment variables."
            )

    def call(self, save_file, transcript, voice="xiaoyun", sample_rate=16000):
        """调用普通语音合成API（非CosyVoice）"""
        try:
            # 检查文本内容是否有效
            if not transcript or len(transcript.strip()) == 0:
                print(f"⚠️  跳过空文本的语音合成: {save_file}")
                return  # 直接返回，不进行合成
            
            # 确保保存目录存在
            save_path = Path(save_file)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 确保文件扩展名为.mp3
            if not save_path.name.endswith('.mp3'):
                save_path = save_path.parent / (save_path.stem + '.mp3')
            
            writer = open(save_path, "wb")
            return_data = b''
            is_completed = False

            def on_data(data, *args):
                nonlocal return_data
                return_data += data
                if writer is not None:
                    writer.write(data)

            def on_completed(*args):
                nonlocal is_completed
                is_completed = True
                print("✅ 语音合成完成")

            def on_error(error, *args):
                raise RuntimeError(f'Synthesizing speech failed with error: {error}')

            def on_close(*args):
                if writer is not None:
                    writer.close()

            # 使用普通语音合成的端点
            endpoints = [
                "wss://nls-gateway-cn-shanghai.aliyuncs.com/ws/v1",
                "wss://nls-gateway-cn-beijing.aliyuncs.com/ws/v1",
                "wss://nls-gateway-cn-hangzhou.aliyuncs.com/ws/v1"
            ]
            
            success = False
            last_error = None
            
            for endpoint in endpoints:
                try:
                    print(f"🔧 尝试端点: {endpoint}")
                    print(f"🔊 使用发音人: {voice}")
                    print(f"🔊 生成语音: {transcript[:50]}...")
                    
                    # 使用NlsSpeechSynthesizer（普通语音合成）
                    sdk = nls.NlsSpeechSynthesizer(
                        url=endpoint,
                        token=self.token,
                        appkey=self.app_key,
                        on_data=on_data,
                        on_completed=on_completed,
                        on_error=on_error,
                        on_close=on_close,
                    )

                    # 开始语音合成 - 使用正确的参数名 aformat
                    sdk.start(text=transcript, 
                             voice=voice, 
                             aformat='mp3',  # 改为 aformat
                             sample_rate=sample_rate,
                             volume=50,
                             speech_rate=0,
                             pitch_rate=0)
                    
                    # 等待合成完成（最多30秒）
                    start_time = time.time()
                    while not is_completed and time.time() - start_time < 30:
                        time.sleep(0.1)
                    
                    if not is_completed:
                        print(f"⚠️  语音合成超时: {save_path}")
                        # 尝试关闭连接
                        try:
                            sdk.shutdown()
                        except:
                            pass
                    
                    # 检查文件是否成功生成
                    if save_path.exists() and save_path.stat().st_size > 0:
                        print(f"✅ 普通语音合成成功: {save_path}")
                        success = True
                        break
                    else:
                        print(f"❌ 语音文件生成失败: {save_path}")
                        # 记录失败原因
                        with open(save_path.parent / "synthesis_errors.log", "a") as log_file:
                            log_file.write(f"文件: {save_path.name}, 错误: 合成失败或文件为空\n")
                        
                except Exception as e:
                    last_error = e
                    print(f"❌ 端点 {endpoint} 失败: {e}")
                    continue
            
            if not success:
                if last_error:
                    raise last_error
                else:
                    raise RuntimeError("所有端点尝试都失败")
                
        except Exception as e:
            print(f"❌ 普通语音合成失败: {e}")
            raise e


@register_tool("cosyvoice_tts")
class CosyVoiceAgent:

    def __init__(self, cfg) -> None:
        self.cfg = cfg

    def call(self, params: Dict):
        """主调用函数 - 使用普通语音合成"""
        pages: List = params["pages"]
        save_path: str = params["save_path"]
        
        # 确保保存目录存在
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        try:
            # 初始化普通语音合成器
            tts_agent = StandardTTSSynthesizer(self.cfg)
            
            print(f"🎯 开始使用普通语音合成服务，共 {len(pages)} 页")
            
            # 使用普通语音合成支持的发音人
            supported_voices = ["xiaoyun", "xiaogang", "xiaowei", "xiaoxiao"]
            voice = params.get("voice", "xiaoyun")
            
            if voice not in supported_voices:
                print(f"⚠️  发音人 {voice} 可能不支持，使用默认发音人 xiaoyun")
                voice = "xiaoyun"
            
            for idx, page in enumerate(pages):
                # 检查页面文本是否有效
                if not page or len(page.strip()) == 0:
                    print(f"⚠️  跳过第 {idx+1} 页，文本内容为空")
                    continue
                
                print(f"📝 处理第 {idx+1}/{len(pages)} 页")
                tts_agent.call(
                    save_file=save_path / f"p{idx + 1}.mp3",
                    transcript=page,
                    voice=voice,
                    sample_rate=self.cfg.get("sample_rate", 16000)
                )
            
            print("✅ 所有语音文件生成完成（使用普通语音合成服务）")
            return {
                "modality": "speech",
                "status": "success",
                "generated_files": len(pages),
                "tts_type": "standard",  # 标明使用普通版
                "voice": voice
            }
            
        except Exception as e:
            print(f"❌ 语音生成过程失败: {e}")
            return {
                "modality": "speech", 
                "status": "failed",
                "error": str(e)
            }