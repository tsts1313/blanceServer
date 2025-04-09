import requests
import json
import os
import asyncio
from app.models import ChatCompletionRequest, Message  # 相对导入
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
import httpx
import logging
import base64

logger = logging.getLogger('my_logger')


@dataclass
class GeneratedText:
    text: str
    finish_reason: Optional[str] = None


class ResponseWrapper:
    def __init__(self, data: Dict[Any, Any]):  # 正确的初始化方法名
        self._data = data
        self._text = self._extract_text()
        self._finish_reason = self._extract_finish_reason()
        self._prompt_token_count = self._extract_prompt_token_count()
        self._candidates_token_count = self._extract_candidates_token_count()
        self._total_token_count = self._extract_total_token_count()
        self._thoughts = self._extract_thoughts()
        self._json_dumps = json.dumps(self._data, indent=4, ensure_ascii=False)

    def _extract_thoughts(self) -> Optional[str]:
        try:
            for part in self._data['candidates'][0]['content']['parts']:
                if 'thought' in part:
                    return part['text']
            return ""
        except (KeyError, IndexError):
            return ""

    def _extract_text(self) -> str:
        try:
            for part in self._data['candidates'][0]['content']['parts']:
                if 'thought' not in part:
                    return part['text']
            return ""
        except (KeyError, IndexError):
            return ""

    def _extract_finish_reason(self) -> Optional[str]:
        try:
            return self._data['candidates'][0].get('finishReason')
        except (KeyError, IndexError):
            return None

    def _extract_prompt_token_count(self) -> Optional[int]:
        try:
            return self._data['usageMetadata'].get('promptTokenCount')
        except (KeyError):
            return None

    def _extract_candidates_token_count(self) -> Optional[int]:
        try:
            return self._data['usageMetadata'].get('candidatesTokenCount')
        except (KeyError):
            return None

    def _extract_total_token_count(self) -> Optional[int]:
        try:
            return self._data['usageMetadata'].get('totalTokenCount')
        except (KeyError):
            return None

    @property
    def text(self) -> str:
        return self._text

    @property
    def finish_reason(self) -> Optional[str]:
        return self._finish_reason

    @property
    def prompt_token_count(self) -> Optional[int]:
        return self._prompt_token_count

    @property
    def candidates_token_count(self) -> Optional[int]:
        return self._candidates_token_count

    @property
    def total_token_count(self) -> Optional[int]:
        return self._total_token_count

    @property
    def thoughts(self) -> Optional[str]:
        return self._thoughts

    @property
    def json_dumps(self) -> str:
        return self._json_dumps


class GeminiClient:

    AVAILABLE_MODELS = []
    EXTRA_MODELS = os.environ.get("EXTRA_MODELS", "").split(",")

    def __init__(self, api_key: str):
        self.api_key = api_key

    async def stream_chat(self, request: ChatCompletionRequest, contents, request_config, system_instruction):
        logger.info("流式开始 →")
        api_version = "v1alpha" if "think" in request.model else "v1beta"
        url = f"https://generativelanguage.googleapis.com/{api_version}/models/{request.model}:streamGenerateContent?key={self.api_key}&alt=sse"
        headers = {
            "Content-Type": "application/json",
        }
        data = {
            "contents": contents,
            "generationConfig": {
                "temperature": request.temperature,
                "maxOutputTokens": request.max_tokens,
            },
            **request_config
        }
        if system_instruction:
            data["system_instruction"] = system_instruction
        
        async with httpx.AsyncClient() as client:
            async with client.stream("POST", url, headers=headers, json=data, timeout=600) as response:
                buffer = b""
                try:
                    async for line in response.aiter_lines():
                        if not line.strip():
                            continue
                        if line.startswith("data: "):
                            line = line[len("data: "):] 
                        buffer += line.encode('utf-8')
                        try:
                            data = json.loads(buffer.decode('utf-8'))
                            buffer = b""
                            if 'candidates' in data and data['candidates']:
                                candidate = data['candidates'][0]
                                if 'content' in candidate:
                                    content = candidate['content']
                                    if 'parts' in content and content['parts']:
                                        parts = content['parts']
                                        text = ""
                                        for part in parts:
                                            if 'text' in part:
                                                text += part['text']
                                        if text:
                                            yield text
                                        
                                if candidate.get("finishReason") and candidate.get("finishReason") != "STOP":
                                    # logger.warning(f"模型的响应因违反内容政策而被标记: {candidate.get('finishReason')}")
                                    raise ValueError(f"模型的响应被截断: {candidate.get('finishReason')}")
                                
                                if 'safetyRatings' in candidate:
                                    for rating in candidate['safetyRatings']:
                                        if rating['probability'] == 'HIGH':
                                            # logger.warning(f"模型的响应因高概率被标记为 {rating['category']}")
                                            raise ValueError(f"模型的响应被截断: {rating['category']}")
                        except json.JSONDecodeError:
                            # logger.debug(f"JSON解析错误, 当前缓冲区内容: {buffer}")
                            continue
                        except Exception as e:
                            # logger.error(f"流式处理期间发生错误: {e}")
                            raise e
                except Exception as e:
                    # logger.error(f"流式处理错误: {e}")
                    raise e
                finally:
                    logger.info("流式结束 ←")


    def complete_chat(self, request: ChatCompletionRequest, contents, request_config, system_instruction):
        api_version = "v1alpha" if "think" in request.model else "v1beta"
        url = f"https://generativelanguage.googleapis.com/{api_version}/models/{request.model}:generateContent?key={self.api_key}"
        headers = {
            "Content-Type": "application/json",
        }
        data = {
            "contents": contents,
            "generationConfig": {
                "temperature": request.temperature,
                "maxOutputTokens": request.max_tokens,
            },
            **request_config
        }
        
        if system_instruction:
            data["system_instruction"] = system_instruction

        #logger.info(f"完整请求payload:\n{json.dumps(data, indent=2, ensure_ascii=False)}")
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        return ResponseWrapper(response.json())

    def convert_messages(self, messages, use_system_prompt=False):
        gemini_history = []
        errors = []
        system_instruction_text = ""
        is_system_phase = use_system_prompt
        
        logger.info(f"开始转换 {len(messages)} 条消息到Gemini格式")
        
        for i, message in enumerate(messages):
            role = message.role
            content = message.content
            
            #logger.info(f"处理消息 {i}: role={role}, 内容类型={type(content).__name__}")

            # 检查是否是字符串化的JSON数组
            if isinstance(content, str) and content.startswith('[{') and content.endswith('}]'):
                #logger.info(f"检测到可能是字符串化的JSON数组，尝试解析: {content[:50]}...")
                try:
                    # 尝试将字符串解析为JSON
                    parsed_content = json.loads(content)
                    if isinstance(parsed_content, list):
                        #logger.info(f"成功解析为JSON数组，包含 {len(parsed_content)} 个项目")
                        # 将解析后的内容替换原始内容，然后按列表处理
                        content = parsed_content
                except json.JSONDecodeError as e:
                    logger.error(f"JSON解析失败: {str(e)}")
                    # 继续按字符串处理
            
            if isinstance(content, str):
                # 检查是否包含特殊格式的图文混合内容
                if "{'type': 'text'" in content and "'type': 'image_url'" in content:
                    try:
                        # 将字符串分割成单独的JSON对象
                        items = [item.strip() for item in content.split("} {")]
                        items = [item.strip("{}") for item in items]
                        
                        # 解析每个部分
                        parts = []
                        for item in items:
                            # 将字符串转换为字典
                            item_dict = eval(f"{{{item}}}")
                            
                            if item_dict.get('type') == 'text':
                                parts.append({"text": item_dict['text']})
                            elif item_dict.get('type') == 'image_url':
                                image_data = item_dict['image_url']['url']
                                if image_data.startswith('data:image/'):
                                    mime_type = image_data.split(';')[0].split(':')[1]
                                    base64_data = image_data.split(',')[1]
                                    parts.append({
                                        "inlineData": {
                                            "data": base64_data,
                                            "mimeType": mime_type
                                        }
                                    })
                        
                        if parts:
                            role_to_use = 'user' if role in ['user', 'system'] else 'model'
                            gemini_history.append({"role": role_to_use, "parts": parts})
                            continue
                            
                    except Exception as e:
                        logger.error(f"特殊格式解析失败: {str(e)}")
                        # 如果解析失败，继续按普通文本处理
                # 处理文本内容的代码保持不变
                if is_system_phase and role == 'system':
                    if system_instruction_text:
                        system_instruction_text += "\n" + content
                    else:
                        system_instruction_text = content
                else:
                    is_system_phase = False

                    if role in ['user', 'system']:
                        role_to_use = 'user'
                    elif role == 'assistant':
                        role_to_use = 'model'
                    else:
                        errors.append(f"Invalid role: {role}")
                        continue

                    if gemini_history and gemini_history[-1]['role'] == role_to_use:
                        gemini_history[-1]['parts'].append({"text": content})
                    else:
                        gemini_history.append(
                            {"role": role_to_use, "parts": [{"text": content}]})
            elif isinstance(content, list):
                parts = []
                #logger.info(f"处理多模态内容: 包含 {len(content)} 个项目")
                
                for j, item in enumerate(content):
                    # 处理嵌套列表的情况
                    if isinstance(item, list):
                        for sub_item in item:
                            if isinstance(sub_item, (dict, str)):
                                self._process_content_item(sub_item, parts, j, errors)
                    # 处理嵌套字典的情况，特别是当字典中包含text字段且text字段是列表时
                    elif isinstance(item, dict) and 'type' in item and item['type'] == 'text' and isinstance(item.get('text'), list):
                        #logger.info(f"检测到嵌套的多模态内容: {item}")
                        for sub_item in item['text']:
                            if isinstance(sub_item, (dict, str)):
                                self._process_content_item(sub_item, parts, j, errors)
                    # 处理普通字典或字符串的情况
                    elif isinstance(item, (dict, str)):
                        self._process_content_item(item, parts, j, errors)
                
                #logger.info(f"多模态内容处理完成: 生成了 {len(parts)} 个部分")
                
                if parts:
                    if role in ['user', 'system']:
                        role_to_use = 'user'
                    elif role == 'assistant':
                        role_to_use = 'model'
                    else:
                        error_msg = f"Invalid role: {role}"
                        logger.error(error_msg)
                        errors.append(error_msg)
                        continue
                    
                    if gemini_history and gemini_history[-1]['role'] == role_to_use:
                        #logger.info(f"将 {len(parts)} 个部分添加到现有的 {role_to_use} 消息中")
                        gemini_history[-1]['parts'].extend(parts)
                    else:
                        #logger.info(f"创建新的 {role_to_use} 消息，包含 {len(parts)} 个部分")
                        gemini_history.append(
                            {"role": role_to_use, "parts": parts})
        
        if errors:
            logger.error(f"转换过程中发生 {len(errors)} 个错误: {errors}")
            return errors
        else:
            logger.info(f"转换完成: 生成了 {len(gemini_history)} 个Gemini消息")
            if system_instruction_text:
                logger.info(f"系统指令: '{system_instruction_text[:50]}...'")
            return gemini_history, {"parts": [{"text": system_instruction_text}]}

    @staticmethod
    def _process_content_item(item, parts, index, errors):
        """处理单个内容项（文本或图片）"""
        # 如果是字符串，尝试解析成字典
        if isinstance(item, str):
            try:
                item = json.loads(item.replace("'", '"'))
            except json.JSONDecodeError:
                try:
                    item = eval(item)
                except Exception as e:
                    logger.error(f"解析字符串到字典失败: {str(e)}")
                    return
        
        if not isinstance(item, dict):
            errors.append(f"Invalid item type: {type(item).__name__}")
            return
            
        if item.get('type') == 'text':
            text_content = item.get('text', '')
            if isinstance(text_content, list):
                text_content = ' '.join([str(x) for x in text_content])
            parts.append({"text": text_content})
        elif item.get('type') == 'image_url':
            image_data = item.get('image_url', {}).get('url', '')
            
            if image_data.startswith('data:image/'):
                try:
                    mime_type = image_data.split(';')[0].split(':')[1]
                    base64_data = image_data.split(',')[1]
                    parts.append({
                        "inlineData": {
                            "data": base64_data,
                            "mimeType": mime_type
                        }
                    })
                except (IndexError, ValueError) as e:
                    error_msg = f"Invalid data URI for image: {image_data[:30]}..."
                    logger.error(f"处理图片 {index} 时出错: {str(e)}, {error_msg}")
                    errors.append(error_msg)
            else:
                try:
                    response = requests.get(image_data)
                    response.raise_for_status()
                    mime_type = response.headers.get('Content-Type', 'image/jpeg')
                    base64_data = base64.b64encode(response.content).decode('utf-8')
                    parts.append({
                        "inlineData": {
                            "data": base64_data,
                            "mimeType": mime_type
                        }
                    })
                except Exception as e:
                    error_msg = f"无法下载图片: {image_data[:30]}..."
                    logger.error(f"处理图片 {index} 时出错: {str(e)}, {error_msg}")
                    errors.append(error_msg)
    @staticmethod
    async def list_available_models(api_key) -> list:
        url = "https://generativelanguage.googleapis.com/v1beta/models?key={}".format(
            api_key)
        async with httpx.AsyncClient() as client:
            response = await client.get(url)
            response.raise_for_status()
            data = response.json()
            models = [model["name"] for model in data.get("models", [])]
            models.extend(GeminiClient.EXTRA_MODELS)
            return models