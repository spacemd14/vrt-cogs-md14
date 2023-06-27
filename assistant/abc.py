from abc import ABCMeta, abstractmethod
from multiprocessing.pool import Pool
from typing import Dict, Union

import discord
import tiktoken
from discord.ext.commands.cog import CogMeta
from redbot.core.bot import Red
from transformers.pipelines.question_answering import QuestionAnsweringPipeline

from .common.models import DB, CustomFunction, GuildSettings


class CompositeMetaClass(CogMeta, ABCMeta):
    """Type detection"""


class MixinMeta(metaclass=ABCMeta):
    """Type hinting"""

    bot: Red
    db: DB
    re_pool: Pool
    registry: Dict[str, Dict[str, CustomFunction]]

    openai_encoder: tiktoken.core.Encoding
    local_llm: QuestionAnsweringPipeline

    @abstractmethod
    async def get_chat_response(
        self,
        message: str,
        author: Union[discord.Member, int],
        guild: discord.Guild,
        channel: Union[discord.TextChannel, discord.Thread, discord.ForumChannel, int],
        conf: GuildSettings,
    ) -> str:
        raise NotImplementedError

    @abstractmethod
    async def save_conf(self):
        raise NotImplementedError

    @abstractmethod
    async def get_chat(
        self,
        message: str,
        author: Union[discord.Member, int],
        guild: discord.Guild,
        channel: Union[discord.TextChannel, discord.Thread, discord.ForumChannel, int],
    ) -> str:
        raise NotImplementedError

    @abstractmethod
    async def handle_message(
        self, message: discord.Message, question: str, conf: GuildSettings, listener: bool = False
    ) -> str:
        raise NotImplementedError
