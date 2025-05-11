from fastapi import FastAPI, HTTPException, Request, Depends, status
from fastapi.responses import JSONResponse, StreamingResponse, HTMLResponse
from .models import ChatCompletionRequest, ChatCompletionResponse, ErrorResponse, ModelList
from .gemini import GeminiClient, ResponseWrapper
from .utils import handle_gemini_error, protect_from_abuse, APIKeyManager, test_api_key, format_log_message
import os
import json
import asyncio
from typing import Literal
import random
import requests
from datetime import datetime, timedelta
from apscheduler.schedulers.background import BackgroundScheduler
import sys
import logging

logging.getLogger("uvicorn").disabled = True
logging.getLogger("uvicorn.access").disabled = True

# 配置 logger
logger = logging.getLogger("my_logger")
logger.setLevel(logging.DEBUG)

def translate_error(message: str) -> str:
    if "quota exceeded" in message.lower():
        return "API 密钥配额已用尽"
    if "invalid argument" in message.lower():
        return "无效参数"
    if "internal server error" in message.lower():
        return "服务器内部错误"
    if "service unavailable" in message.lower():
        return "服务不可用"
    return message


def handle_exception(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.excepthook(exc_type, exc_value, exc_traceback)
        return
    error_message = translate_error(str(exc_value))
    log_msg = format_log_message('ERROR', f"未捕获的异常: %s" % error_message, extra={'status_code': 500, 'error_message': error_message})
    logger.error(log_msg)


sys.excepthook = handle_exception

app = FastAPI()

PASSWORD = os.environ.get("PASSWORD", "123")
MAX_REQUESTS_PER_MINUTE = int(os.environ.get("MAX_REQUESTS_PER_MINUTE", "60"))
MAX_REQUESTS_PER_DAY_PER_IP = int(
    os.environ.get("MAX_REQUESTS_PER_DAY_PER_IP", "2000"))
# MAX_RETRIES = int(os.environ.get('MaxRetries', '3').strip() or '3')
RETRY_DELAY = 1
MAX_RETRY_DELAY = 16
safety_settings = [
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_NONE"
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_NONE"
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_NONE"
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_NONE"
    },
    {
        "category": 'HARM_CATEGORY_CIVIC_INTEGRITY',
        "threshold": 'BLOCK_NONE'
    }
]
safety_settings_g2 = [
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "OFF"
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "OFF"
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "OFF"
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "OFF"
    },
    {
        "category": 'HARM_CATEGORY_CIVIC_INTEGRITY',
        "threshold": 'OFF'
    }
]

# 添加默认的搜索工具配置
ENABLE_SEARCH = os.environ.get("ENABLE_SEARCH", "true").lower() == "true"
tool_settings = {
    "safety_settings": safety_settings,
    "tools": [{"googleSearch": {}}] if ENABLE_SEARCH else []
}

# 新增：读取和处理 thinkingBudget 环境变量
THINKING_BUDGET_STR = os.environ.get("THINKING_BUDGET")
EFFECTIVE_THINKING_BUDGET = None
CONFIGURED_THINKING_BUDGET_DISPLAY = "未配置" # 用于在HTML页面显示
if THINKING_BUDGET_STR is not None:
    try:
        tb_value = int(THINKING_BUDGET_STR)
        if tb_value == 0:
            EFFECTIVE_THINKING_BUDGET = 0
            CONFIGURED_THINKING_BUDGET_DISPLAY = f"0 (思考已禁用)"
            log_msg = format_log_message('INFO', f"THINKING_BUDGET 配置为 0，思考功能已禁用。")
            logger.info(log_msg)
        elif 1 <= tb_value <= 1024:
            EFFECTIVE_THINKING_BUDGET = 1024
            CONFIGURED_THINKING_BUDGET_DISPLAY = f"{tb_value} (生效值为 1024)"
            log_msg = format_log_message('INFO', f"THINKING_BUDGET 配置为 {tb_value}，生效值为 1024。")
            logger.info(log_msg)
        elif 1024 < tb_value <= 24576:
            EFFECTIVE_THINKING_BUDGET = tb_value
            CONFIGURED_THINKING_BUDGET_DISPLAY = f"{tb_value}"
            log_msg = format_log_message('INFO', f"THINKING_BUDGET 配置为 {tb_value}，在有效范围内。")
            logger.info(log_msg)
        elif tb_value > 24576:
            EFFECTIVE_THINKING_BUDGET = 24576
            CONFIGURED_THINKING_BUDGET_DISPLAY = f"{tb_value} (生效值为 24576)"
            log_msg = format_log_message('INFO', f"THINKING_BUDGET 配置为 {tb_value}，超出上限，生效值为 24576。")
            logger.info(log_msg)
        else: # tb_value < 0
            log_msg = format_log_message('WARNING', f"环境变量 THINKING_BUDGET 的值 '{tb_value}' 为负数，无效，将不生效。")
            logger.warning(log_msg)
            CONFIGURED_THINKING_BUDGET_DISPLAY = f"{tb_value} (无效值，已忽略)"
    except ValueError:
        log_msg = format_log_message('WARNING', f"环境变量 THINKING_BUDGET 的值 '{THINKING_BUDGET_STR}' 不是有效的整数，将不生效。")
        logger.warning(log_msg)
        CONFIGURED_THINKING_BUDGET_DISPLAY = f"'{THINKING_BUDGET_STR}' (非整数，已忽略)"



key_manager = APIKeyManager() # 实例化 APIKeyManager，栈会在 __init__ 中初始化
current_api_key = key_manager.get_available_key()


def switch_api_key():
    global current_api_key
    key = key_manager.get_available_key() # get_available_key 会处理栈的逻辑
    if key:
        current_api_key = key
        log_msg = format_log_message('INFO', f"API key 替换为 → {current_api_key[:8]}...", extra={'key': current_api_key[:8], 'request_type': 'switch_key'})
        logger.info(log_msg)
    else:
        log_msg = format_log_message('ERROR', "API key 替换失败，所有API key都已尝试，请重新配置或稍后重试", extra={'key': 'N/A', 'request_type': 'switch_key', 'status_code': 'N/A'})
        logger.error(log_msg)


async def check_keys():
    available_keys = []
    for key in key_manager.api_keys:
        is_valid = await test_api_key(key)
        status_msg = "有效" if is_valid else "无效"
        log_msg = format_log_message('INFO', f"API Key {key[:10]}... {status_msg}.")
        logger.info(log_msg)
        if is_valid:
            available_keys.append(key)
    if not available_keys:
        log_msg = format_log_message('ERROR', "没有可用的 API 密钥！", extra={'key': 'N/A', 'request_type': 'startup', 'status_code': 'N/A'})
        logger.error(log_msg)
    return available_keys


@app.on_event("startup")
async def startup_event():
    log_msg = format_log_message('INFO', "Starting Gemini API proxy...")
    logger.info(log_msg)
    available_keys = await check_keys()
    if available_keys:
        key_manager.api_keys = available_keys
        key_manager._reset_key_stack() # 启动时也确保创建随机栈
        key_manager.show_all_keys()
        log_msg = format_log_message('INFO', f"可用 API 密钥数量：{len(key_manager.api_keys)}")
        logger.info(log_msg)
        # MAX_RETRIES = len(key_manager.api_keys)
        log_msg = format_log_message('INFO', f"最大重试次数设置为：{len(key_manager.api_keys)}") # 添加日志
        logger.info(log_msg)
        if key_manager.api_keys:
            all_models = await GeminiClient.list_available_models(key_manager.api_keys[0])
            GeminiClient.AVAILABLE_MODELS = [model.replace(
                "models/", "") for model in all_models]
            log_msg = format_log_message('INFO', "Available models loaded.")
            logger.info(log_msg)

@app.get("/v1/models", response_model=ModelList)
def list_models():
    log_msg = format_log_message('INFO', "Received request to list models", extra={'request_type': 'list_models', 'status_code': 200})
    logger.info(log_msg)
    return ModelList(data=[{"id": model, "object": "model", "created": 1678888888, "owned_by": "organization-owner"} for model in GeminiClient.AVAILABLE_MODELS])


async def verify_password(request: Request):
    if PASSWORD:
        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            raise HTTPException(
                status_code=401, detail="Unauthorized: Missing or invalid token")
        token = auth_header.split(" ")[1]
        if token != PASSWORD:
            raise HTTPException(
                status_code=401, detail="Unauthorized: Invalid token")


async def process_request(chat_request: ChatCompletionRequest, http_request: Request, request_type: Literal['stream', 'non-stream']):
    global current_api_key
    protect_from_abuse(
        http_request, MAX_REQUESTS_PER_MINUTE, MAX_REQUESTS_PER_DAY_PER_IP)
    if chat_request.model not in GeminiClient.AVAILABLE_MODELS:
        error_msg = "无效的模型"
        extra_log = {'request_type': request_type, 'model': chat_request.model, 'status_code': 400, 'error_message': error_msg}
        log_msg = format_log_message('ERROR', error_msg, extra=extra_log)
        logger.error(log_msg)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=error_msg)

    key_manager.reset_tried_keys_for_request() # 在每次请求处理开始时重置 tried_keys 集合

    contents, system_instruction = GeminiClient.convert_messages(
        GeminiClient, chat_request.messages)

    # 合并安全设置和工具配置
    request_config = {
        **tool_settings,  # 包含安全设置和工具配置
        "safety_settings": safety_settings_g2 if 'gemini-2.0-flash-exp' in chat_request.model else safety_settings
    }

    # 新增：根据模型和配置添加 thinkingConfig
    if "gemini-2.5" in chat_request.model and EFFECTIVE_THINKING_BUDGET is not None:
        request_config["thinkingConfig"] = {"thinkingBudget": EFFECTIVE_THINKING_BUDGET}
        log_msg = format_log_message('DEBUG', f"为模型 {chat_request.model} 应用 thinkingBudget: {EFFECTIVE_THINKING_BUDGET}", extra={'request_type': request_type, 'model': chat_request.model})
        logger.debug(log_msg) 

    retry_attempts = len(key_manager.api_keys) if key_manager.api_keys else 1 # 重试次数等于密钥数量，至少尝试 1 次
    for attempt in range(1, retry_attempts + 1):
        if attempt == 1:
            current_api_key = key_manager.get_available_key() # 每次循环开始都获取新的 key, 栈逻辑在 get_available_key 中处理
        
        if current_api_key is None: # 检查是否获取到 API 密钥
            log_msg_no_key = format_log_message('WARNING', "没有可用的 API 密钥，跳过本次尝试", extra={'request_type': request_type, 'model': chat_request.model, 'status_code': 'N/A'})
            logger.warning(log_msg_no_key)
            break  # 如果没有可用密钥，跳出循环

        extra_log = {'key': current_api_key[:8], 'request_type': request_type, 'model': chat_request.model, 'status_code': 'N/A', 'error_message': ''}
        log_msg = format_log_message('INFO', f"第 {attempt}/{retry_attempts} 次尝试 ... 使用密钥: {current_api_key[:8]}...", extra=extra_log)
        logger.info(log_msg)

        gemini_client = GeminiClient(current_api_key)
        try:
            if chat_request.stream:
                async def stream_generator():
                    try:
                        async for chunk in gemini_client.stream_chat(chat_request, contents, request_config, system_instruction):
                            formatted_chunk = {"id": "chatcmpl-someid", "object": "chat.completion.chunk", "created": 1234567,
                                               "model": chat_request.model, "choices": [{"delta": {"role": "assistant", "content": chunk}, "index": 0, "finish_reason": None}]}
                            yield f"data: {json.dumps(formatted_chunk)}\n\n"
                        yield "data: [DONE]\n\n"

                    except asyncio.CancelledError:
                        extra_log_cancel = {'key': current_api_key[:8], 'request_type': request_type, 'model': chat_request.model, 'error_message': '客户端已断开连接'}
                        log_msg = format_log_message('INFO', "客户端连接已中断", extra=extra_log_cancel)
                        logger.info(log_msg)
                    except Exception as e:
                        error_detail = handle_gemini_error(
                            e, current_api_key, key_manager)
                        yield f"data: {json.dumps({'error': {'message': error_detail, 'type': 'gemini_error'}})}\n\n"
                return StreamingResponse(stream_generator(), media_type="text/event-stream")
            else:
                async def run_gemini_completion():
                    try:
                        response_content = await asyncio.to_thread(gemini_client.complete_chat, chat_request, contents, request_config, system_instruction)
                        log_msg = format_log_message('INFO', f"API调用成功，返回内容：{response_content}", extra={'key': current_api_key[:8],'request_type': request_type,'model': chat_request.model})
                        logger.info(log_msg)
                        return response_content
                    except asyncio.CancelledError:
                        extra_log_gemini_cancel = {'key': current_api_key[:8], 'request_type': request_type, 'model': chat_request.model, 'error_message': '客户端断开导致API调用取消'}
                        log_msg = format_log_message('INFO', "API调用因客户端断开而取消", extra=extra_log_gemini_cancel)
                        logger.info(log_msg)
                        raise

                async def check_client_disconnect():
                    while True:
                        if await http_request.is_disconnected():
                            extra_log_client_disconnect = {'key': current_api_key[:8], 'request_type': request_type, 'model': chat_request.model, 'error_message': '检测到客户端断开连接'}
                            log_msg = format_log_message('INFO', "客户端连接已中断，正在取消API请求", extra=extra_log_client_disconnect)
                            logger.info(log_msg)
                            return True
                        await asyncio.sleep(0.5)

                gemini_task = asyncio.create_task(run_gemini_completion())
                disconnect_task = asyncio.create_task(check_client_disconnect())

                try:
                    done, pending = await asyncio.wait(
                        [gemini_task, disconnect_task],
                        return_when=asyncio.FIRST_COMPLETED
                    )

                    if disconnect_task in done:
                        gemini_task.cancel()
                        try:
                            await gemini_task
                        except asyncio.CancelledError:
                            extra_log_gemini_task_cancel = {'key': current_api_key[:8], 'request_type': request_type, 'model': chat_request.model, 'error_message': 'API任务已终止'}
                            log_msg = format_log_message('INFO', "API任务已成功取消", extra=extra_log_gemini_task_cancel)
                            logger.info(log_msg)
                        # 直接抛出异常中断循环
                        raise HTTPException(status_code=status.HTTP_408_REQUEST_TIMEOUT, detail="客户端连接已中断")

                    if gemini_task in done:
                        disconnect_task.cancel()
                        try:
                            await disconnect_task
                        except asyncio.CancelledError:
                            pass
                        response_content = gemini_task.result()
                        if response_content.text == "":
                            extra_log_empty_response = {'key': current_api_key[:8], 'request_type': request_type, 'model': chat_request.model, 'status_code': 204}
                            log_msg = format_log_message('INFO', "Gemini API 返回空响应", extra=extra_log_empty_response)
                            logger.info(log_msg)
                            # 继续循环
                            continue
                        response = ChatCompletionResponse(id="chatcmpl-someid", object="chat.completion", created=1234567890, model=chat_request.model,
                                                        choices=[{"index": 0, "message": {"role": "assistant", "content": response_content.text}, "finish_reason": "stop"}])
                        log_msg = format_log_message('INFO', "Gemini API 返回响应", extra=extra_log_empty_response)
                        logger.info(log_msg)                        
                        extra_log_success = {'key': current_api_key[:8], 'request_type': request_type, 'model': chat_request.model, 'status_code': 200}
                        log_msg = format_log_message('INFO', "请求处理成功", extra=extra_log_success)
                        logger.info(log_msg)
                        return response

                except asyncio.CancelledError:
                    extra_log_request_cancel = {'key': current_api_key[:8], 'request_type': request_type, 'model': chat_request.model, 'error_message':"请求被取消" }
                    log_msg = format_log_message('INFO', "请求取消", extra=extra_log_request_cancel)
                    logger.info(log_msg)
                    raise

        except HTTPException as e:
            if e.status_code == status.HTTP_408_REQUEST_TIMEOUT:
                extra_log = {'key': current_api_key[:8], 'request_type': request_type, 'model': chat_request.model, 
                            'status_code': 408, 'error_message': '客户端连接中断'}
                log_msg = format_log_message('ERROR', "客户端连接中断，终止后续重试", extra=extra_log)
                logger.error(log_msg)
                raise  
            else:
                raise  
        except Exception as e:
            handle_gemini_error(e, current_api_key, key_manager)
            if attempt < retry_attempts: 
                switch_api_key() 
                continue

    msg = "所有API密钥均失败,请稍后重试"
    extra_log_all_fail = {'key': "ALL", 'request_type': request_type, 'model': chat_request.model, 'status_code': 500, 'error_message': msg}
    log_msg = format_log_message('ERROR', msg, extra=extra_log_all_fail)
    logger.error(log_msg)
    raise HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=msg)


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(request: ChatCompletionRequest, http_request: Request, _: None = Depends(verify_password)):
    return await process_request(request, http_request, "stream" if request.stream else "non-stream")


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    error_message = translate_error(str(exc))
    extra_log_unhandled_exception = {'status_code': 500, 'error_message': error_message}
    log_msg = format_log_message('ERROR', f"Unhandled exception: {error_message}", extra=extra_log_unhandled_exception)
    logger.error(log_msg)
    return JSONResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, content=ErrorResponse(message=str(exc), type="internal_error").dict())


@app.get("/", response_class=HTMLResponse)
async def root():
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Gemini API 代理服务</title>
        <style>
            body {{
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
                max-width: 800px;
                margin: 0 auto;
                padding: 20px;
                line-height: 1.6;
            }}
            h1 {{
                color: #333;
                text-align: center;
                margin-bottom: 30px;
            }}
            .info-box {{
                background-color: #f8f9fa;
                border: 1px solid #dee2e6;
                border-radius: 4px;
                padding: 20px;
                margin-bottom: 20px;
            }}
            .status {{
                color: #28a745;
                font-weight: bold;
            }}
        </style>
    </head>
    <body>
        <h1>🤖 多模态 Gemini API 代理服务</h1>
        
        <div class="info-box">
            <h2>🟢 运行状态</h2>
            <p class="status">服务运行中</p>
            <p>可用API密钥数量: {len(key_manager.api_keys)}</p>
            <p>可用模型数量: {len(GeminiClient.AVAILABLE_MODELS)}</p>
            <p>搜索功能: <span class="feature">{'已启用' if ENABLE_SEARCH else '已禁用'}</span></p>
        </div>

        <div class="info-box">
            <h2>⚙️ 环境配置</h2>
            <p>每分钟请求限制: {MAX_REQUESTS_PER_MINUTE}</p>
            <p>每IP每日请求限制: {MAX_REQUESTS_PER_DAY_PER_IP}</p>
            <p>最大重试次数: {len(key_manager.api_keys)}</p>
            <p>思考预算 (thinkingBudget): <span class="feature">{CONFIGURED_THINKING_BUDGET_DISPLAY}</span> (仅对 gemini-2.5 模型生效)</p>
        </div>
    </body>
    </html>
    """
    return html_content