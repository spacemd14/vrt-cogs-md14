import asyncio
import logging
import re
from typing import Callable, List, Optional, Tuple

import discord
from redbot.core import commands

log = logging.getLogger("red.vrt.assistant.utils")
# encoding = tiktoken.get_encoding("cl100k_base")
# encoding_local = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")


def get_attachments(message: discord.Message) -> List[discord.Attachment]:
    """Get all attachments from context"""
    attachments = []
    if message.attachments:
        direct_attachments = [a for a in message.attachments]
        attachments.extend(direct_attachments)
    if hasattr(message, "reference"):
        try:
            referenced_attachments = [a for a in message.reference.resolved.attachments]
            attachments.extend(referenced_attachments)
        except AttributeError:
            pass
    return attachments


async def wait_message(ctx: commands.Context) -> Optional[discord.Message]:
    def check(message: discord.Message):
        return message.author == ctx.author and message.channel == ctx.channel

    try:
        message = await ctx.bot.wait_for("message", timeout=600, check=check)
        if message.content == "cancel":
            await ctx.send("Canceled")
            return None
        return message
    except asyncio.TimeoutError:
        return None


async def can_use(message: discord.Message, blacklist: list, respond: bool = True) -> bool:
    allowed = True
    if message.author.id in blacklist:
        if respond:
            await message.channel.send("You have been blacklisted from using this command!")
        allowed = False
    elif any(role.id in blacklist for role in message.author.roles):
        if respond:
            await message.channel.send("You have a blacklisted role and cannot use this command!")
        allowed = False
    elif message.channel.id in blacklist:
        if respond:
            await message.channel.send("You cannot use that command in this channel!")
        allowed = False
    elif message.channel.category_id in blacklist:
        if respond:
            await message.channel.send(
                "You cannot use that command in any channels under this category"
            )
        allowed = False
    return allowed


def extract_code_blocks(content: str) -> List[str]:
    code_blocks = re.findall(r"```(?:\w+)(.*?)```", content, re.DOTALL)
    if not code_blocks:
        code_blocks = re.findall(r"```(.*?)```", content, re.DOTALL)
    return [block.strip() for block in code_blocks]


def extract_code_blocks_with_lang(content: str) -> List[Tuple[str, str]]:
    code_blocks = re.findall(r"```(\w+)(.*?)```", content, re.DOTALL)
    if not code_blocks:
        code_blocks = re.findall(r"```(.*?)```", content, re.DOTALL)
        return [("", block.strip()) for block in code_blocks]
    return [(block[0], block[1].strip()) for block in code_blocks]


def remove_code_blocks(content: str) -> str:
    content = re.sub(r"```(?:\w+)(.*?)```", "[Code Removed]", content, flags=re.DOTALL).strip()
    return re.sub(r"```(.*?)```", "[Code Removed]", content, flags=re.DOTALL).strip()


def code_string_valid(code: str) -> bool:
    # True if function is good
    if "*args, **kwargs" not in code:
        return False
    try:
        compile(code, "<string>", "exec")
        return True
    except SyntaxError:
        return False


def compile_function(function_name: str, code: str) -> Callable:
    exec(code, globals())
    return globals()[function_name]


def json_schema_invalid(schema: dict) -> str:
    # String will be empty if function is good
    missing = ""
    if "name" not in schema:
        missing += "- `name`\n"
    if "description" not in schema:
        missing += "- `description`\n"
    if "parameters" not in schema:
        missing += "- `parameters`\n"
    if "parameters" in schema:
        if "type" not in schema["parameters"]:
            missing += "- `type` in **parameters**\n"
        if "properties" not in schema["parameters"]:
            missing = "- `properties` in **parameters**\n"
        if "required" in schema["parameters"].get("properties", []):
            missing += "- `required` key needs to be outside of properties!\n"
    return missing


def compile_messages(messages: List[dict]) -> str:
    """Compile messages list into a single string"""
    text = ""
    for message in messages:
        role = message["role"]
        content = message["content"]
        text += f"{role}: {content}\n"
    text += "\n"
    return text
