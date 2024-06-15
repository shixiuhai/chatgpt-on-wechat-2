# encoding:utf-8

import json
import time
from typing import List, Tuple
from config import conf
import requests

import openai
import openai.error
from broadscope_bailian import ChatQaMessage

from bot.bot import Bot
from bot.ollama.ollama_session import OllamaSession
from bot.session_manager import SessionManager
from bridge.context import ContextType
from bridge.reply import Reply, ReplyType
from common.log import logger
from common import const
from config import conf, load_config



class OllamaBot(Bot):
    def __init__(self):
        super().__init__()
        # self.api_key_expired_time = self.set_api_key()
        self.sessions = SessionManager(OllamaSession, model=conf().get("model", const.OLLAMA))
        self.ollama_model =conf().get("ollama_model") # 配置了ollama下面的model
        self.ollama_model_url=conf().get("ollama_model_url") # 配置了ollama下面的基本url
    def format_history_conversation(self, history_conversation_list:list,prompt:str):
        """
        处理一下历史对话
        """
        formatted_messages = []
        for convo in history_conversation_list:
            if convo.user != "":
                formatted_messages.append({
                    "role": "user",
                    "content": convo.user
                })
            if convo.bot != "":
                formatted_messages.append({
                    "role": "assistant",
                    "content":  convo.bot
                })
                
        formatted_messages.append({
            "role": "user",
            "content":prompt
        })
        return formatted_messages
    
    def reply(self, query, context=None):
        # acquire reply content
        if context.type == ContextType.TEXT:
            logger.info("[QWEN] query={}".format(query))

            session_id = context["session_id"]
            reply = None
            clear_memory_commands = conf().get("clear_memory_commands", ["#清除记忆"])
            if query in clear_memory_commands:
                self.sessions.clear_session(session_id)
                reply = Reply(ReplyType.INFO, "记忆已清除")
            elif query == "#清除所有":
                self.sessions.clear_all_session()
                reply = Reply(ReplyType.INFO, "所有人记忆已清除")
            elif query == "#更新配置":
                load_config()
                reply = Reply(ReplyType.INFO, "配置已更新")
            if reply:
                return reply
            session = self.sessions.session_query(query, session_id)
            logger.debug("[QWEN] session query={}".format(session.messages))

            reply_content = self.reply_text(session)
            logger.debug(
                "[QWEN] new_query={}, session_id={}, reply_cont={}, completion_tokens={}".format(
                    session.messages,
                    session_id,
                    reply_content["content"],
                    reply_content["completion_tokens"],
                )
            )
            if reply_content["completion_tokens"] == 0 and len(reply_content["content"]) > 0:
                reply = Reply(ReplyType.ERROR, reply_content["content"])
            elif reply_content["completion_tokens"] > 0:
                self.sessions.session_reply(reply_content["content"], session_id, reply_content["total_tokens"])
                reply = Reply(ReplyType.TEXT, reply_content["content"])
            else:
                reply = Reply(ReplyType.ERROR, reply_content["content"])
                logger.debug("[QWEN] reply {} used 0 tokens.".format(reply_content))
            return reply

        else:
            reply = Reply(ReplyType.ERROR, "Bot不支持处理{}类型的消息".format(context.type))
            return reply

    def reply_text(self, session: OllamaSession, retry_count=0) -> dict:
        """
        call bailian's ChatCompletion to get the answer
        :param session: a conversation session
        :param retry_count: retry count
        :return: {}
        """
        try:
            prompt, history = self.convert_messages_format(session.messages)
            data = {
                "model": self.ollama_model,
                "messages": self.format_history_conversation(history,prompt),
                "stream": False
            }
            
            completion_content= requests.post(self.ollama_model_url+"/api/chat", json=data).json().get("message", "").get("content", "")
            
            completion_tokens, total_tokens = self.calc_tokens(session.messages, completion_content)
            return {
                "total_tokens": total_tokens,
                "completion_tokens": completion_tokens,
                "content": completion_content,
            }
        except Exception as e:
            need_retry = retry_count < 2
            result = {"completion_tokens": 0, "content": "我现在有点累了，等会再来吧"}
            if isinstance(e, openai.error.RateLimitError):
                logger.warn("[QWEN] RateLimitError: {}".format(e))
                result["content"] = "提问太快啦，请休息一下再问我吧"
                if need_retry:
                    time.sleep(20)
            elif isinstance(e, openai.error.Timeout):
                logger.warn("[QWEN] Timeout: {}".format(e))
                result["content"] = "我没有收到你的消息"
                if need_retry:
                    time.sleep(5)
            elif isinstance(e, openai.error.APIError):
                logger.warn("[QWEN] Bad Gateway: {}".format(e))
                result["content"] = "请再问我一次"
                if need_retry:
                    time.sleep(10)
            elif isinstance(e, openai.error.APIConnectionError):
                logger.warn("[QWEN] APIConnectionError: {}".format(e))
                need_retry = False
                result["content"] = "我连接不到你的网络"
            else:
                logger.exception("[QWEN] Exception: {}".format(e))
                need_retry = False
                self.sessions.clear_session(session.session_id)

            if need_retry:
                logger.warn("[QWEN] 第{}次重试".format(retry_count + 1))
                return self.reply_text(session, retry_count + 1)
            else:
                return result


    def convert_messages_format(self, messages) -> Tuple[str, List[ChatQaMessage]]:
        """_summary_
        消息格式化处理
        Args:
            messages (_type_): _description_

        Raises:
            Exception: _description_

        Returns:
            Tuple[str, List[ChatQaMessage]]: _description_
        """
        history = []
        user_content = ''
        assistant_content = ''
        system_content = ''
        for message in messages:
            role = message.get('role')
            if role == 'user':
                user_content += message.get('content')
            elif role == 'assistant':
                assistant_content = message.get('content')
                history.append(ChatQaMessage(user_content, assistant_content))
                user_content = ''
                assistant_content = ''
            elif role =='system':
                system_content += message.get('content')
        if user_content == '':
            raise Exception('no user message')
        if system_content != '':
            # NOTE 模拟系统消息，测试发现人格描述以"你需要扮演ChatGPT"开头能够起作用，而以"你是ChatGPT"开头模型会直接否认
            system_qa = ChatQaMessage(system_content, '好的，我会严格按照你的设定回答问题')
            history.insert(0, system_qa)
        logger.debug("[QWEN] converted qa messages: {}".format([item.to_dict() for item in history]))
        logger.debug("[QWEN] user content as prompt: {}".format(user_content))
        return user_content, history


    def calc_tokens(self, messages, completion_content):
        """_summary_
        统计使用的tokens
        Args:
            messages (_type_): _description_
            completion_content (_type_): _description_

        Returns:
            _type_: _description_
        """
        completion_tokens = len(completion_content)
        prompt_tokens = 0
        for message in messages:
            prompt_tokens += len(message["content"])
        return completion_tokens, prompt_tokens + completion_tokens
