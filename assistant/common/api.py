import asyncio
import inspect
import logging
import math
from typing import List, Optional, Tuple, Union

import discord
import openai
import orjson
from aiocache import cached
from openai.error import (
    APIConnectionError,
    APIError,
    RateLimitError,
    ServiceUnavailableError,
    Timeout,
)
from openai.version import VERSION
from redbot.core.utils.chat_formatting import box, humanize_number
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_delay,
    wait_random_exponential,
)

from ..abc import MixinMeta
from .constants import MODELS, SELF_HOSTED, SUPPORTS_FUNCTIONS
from .models import GuildSettings

log = logging.getLogger("red.vrt.assistant.api")


class API(MixinMeta):
    @retry(
        retry=retry_if_exception_type(
            Union[Timeout, APIConnectionError, RateLimitError, ServiceUnavailableError]
        ),
        wait=wait_random_exponential(min=1, max=5),
        stop=stop_after_delay(120),
        reraise=True,
    )
    @cached(ttl=1800)
    async def request_embedding(self, text: str, conf: GuildSettings) -> List[float]:
        response = await openai.Embedding.acreate(
            input=text, model="text-embedding-ada-002", api_key=conf.api_key, timeout=30
        )
        return response["data"][0]["embedding"]

    @retry(
        retry=retry_if_exception_type(Exception),
        wait=wait_random_exponential(min=1, max=5),
        stop=stop_after_delay(120),
        reraise=True,
    )
    @cached(ttl=1800)
    async def request_local_embedding(self, text: str, conf: GuildSettings) -> List[float]:
        def _local():
            inputs = self.local_llm.tokenizer.encode_plus(text, return_tensors="pt")
            # Get the model's output
            outputs = self.local_llm.model(**inputs)
            # Extract the embeddings (last hidden state)
            embeddings = outputs.last_hidden_state
            # Convert the tensor to a NumPy array and then to a Python list
            return embeddings.squeeze().detach().numpy().tolist()

        return await asyncio.to_thread(_local)

    @retry(
        retry=retry_if_exception_type(
            Union[Timeout, APIConnectionError, RateLimitError, APIError, ServiceUnavailableError]
        ),
        wait=wait_random_exponential(min=1, max=5),
        stop=stop_after_delay(120),
        reraise=True,
    )
    @cached(ttl=30)
    async def request_chat_response(
        messages: List[dict], conf: GuildSettings, functions: Optional[List[dict]] = []
    ) -> dict:
        if VERSION >= "0.27.6" and conf.model in SUPPORTS_FUNCTIONS:
            response = await openai.ChatCompletion.acreate(
                model=conf.model,
                messages=messages,
                temperature=conf.temperature,
                api_key=conf.api_key,
                timeout=60,
                functions=functions,
            )
        else:
            response = await openai.ChatCompletion.acreate(
                model=conf.model,
                messages=messages,
                temperature=conf.temperature,
                api_key=conf.api_key,
                timeout=60,
            )
        return response["choices"][0]["message"]

    @retry(
        retry=retry_if_exception_type(
            Union[Timeout, APIConnectionError, RateLimitError, APIError, ServiceUnavailableError]
        ),
        wait=wait_random_exponential(min=1, max=5),
        stop=stop_after_delay(120),
        reraise=True,
    )
    @cached(ttl=30)
    async def request_completion_response(self, prompt: str, conf: GuildSettings) -> str:
        response = await openai.Completion.acreate(
            model=conf.model,
            prompt=prompt,
            temperature=conf.temperature,
            api_key=conf.api_key,
            max_tokens=conf.max_tokens,
        )
        return response["choices"][0]["text"]

    @retry(
        retry=retry_if_exception_type(Exception),
        wait=wait_random_exponential(min=1, max=5),
        stop=stop_after_delay(120),
        reraise=True,
    )
    @cached(ttl=30)
    async def request_local_response(self, prompt: str, context: str) -> str:
        def _run():
            result = self.local_llm(question=prompt, context=context)
            return result["answer"] if result else ""

        return await asyncio.to_thread(_run)

    # ----------------------- HELPERS -----------------------
    def get_max_tokens(self, conf: GuildSettings, member: Optional[discord.Member]) -> int:
        return min(conf.get_user_max_tokens(member), MODELS[conf.model] - 100)

    async def cut_text_by_tokens(self, text: str, conf: GuildSettings, max_tokens: int) -> str:
        tokens = await self.get_tokens(text, conf)
        return await self.get_text(tokens[:max_tokens])

    async def get_token_count(self, text: str, conf: GuildSettings) -> int:
        return len(await self.get_tokens(text, conf))

    async def get_tokens(self, text: str, conf: GuildSettings) -> list:
        """Get token list from text"""

        def _run():
            if conf.model in SELF_HOSTED:
                return self.local_llm.tokenizer.tokenize(text)
            else:
                return self.openai_encoder.encode(text)

        if not text:
            return []
        return await asyncio.to_thread(_run)

    async def get_text(self, tokens: list, conf: GuildSettings) -> str:
        """Get text from token list"""

        def _run():
            if conf.model in SELF_HOSTED:
                return self.local_llm.tokenizer.convert_tokens_to_string(tokens)
            else:
                return self.openai_encoder.decode(tokens)

        return await asyncio.to_thread(_run)

    def degrade_conversation(
        self, messages: List[dict], function_list: List[dict], conf: GuildSettings
    ) -> Tuple[List[dict], List[dict], bool]:
        """Iteratively degrade a conversation payload, prioritizing more recent messages and critical context

        Args:
            messages (List[dict]): message entries sent to the api
            function_list (List[dict]): list of json function schemas for the model
            conf: (GuildSettings): current settings

        Returns:
            Tuple[List[dict], List[dict], bool]: updated messages list, function list, and whether the conversation was degraded
        """
        # Calculate the initial total token count
        total_tokens = sum(self.get_token_count(msg["content"], conf) for msg in messages)
        total_tokens += sum(
            self.get_token_count(orjson.dumps(func), conf) for func in function_list
        )

        # Check if the total token count is already under the max token limit
        max_tokens = min(conf.max_tokens - 100, MODELS[conf.model])
        if total_tokens <= max_tokens:
            return messages, function_list, False

        # Helper function to degrade a message
        def _degrade_message(msg: str) -> str:
            words = msg.split()
            if len(words) > 1:
                return " ".join(words[:-1])
            else:
                return ""

        log.debug(f"Removing functions (total: {total_tokens}/max: {max_tokens})")
        # Degrade function_list first
        while total_tokens > max_tokens and len(function_list) > 0:
            popped = function_list.pop(0)
            total_tokens -= self.get_token_count(orjson.dumps(popped), conf)
            if total_tokens <= max_tokens:
                return messages, function_list, True

        # Find the indices of the most recent messages for each role
        most_recent_user = most_recent_function = most_recent_assistant = -1
        for i, msg in enumerate(reversed(messages)):
            if most_recent_user == -1 and msg["role"] == "user":
                most_recent_user = len(messages) - 1 - i
            if most_recent_function == -1 and msg["role"] == "function":
                most_recent_function = len(messages) - 1 - i
            if most_recent_assistant == -1 and msg["role"] == "assistant":
                most_recent_assistant = len(messages) - 1 - i
            if (
                most_recent_user != -1
                and most_recent_function != -1
                and most_recent_assistant != -1
            ):
                break

        log.debug(f"Degrading messages (total: {total_tokens}/max: {max_tokens})")
        # Degrade the conversation except for the most recent user and function messages
        i = 0
        while total_tokens > max_tokens and i < len(messages):
            if (
                messages[i]["role"] == "system"
                or i == most_recent_user
                or i == most_recent_function
            ):
                i += 1
                continue

            degraded_content = _degrade_message(messages[i]["content"])
            if degraded_content:
                token_diff = self.get_token_count(
                    messages[i]["content"], conf
                ) - self.get_token_count(degraded_content, conf)
                messages[i]["content"] = degraded_content
                total_tokens -= token_diff
            else:
                total_tokens -= self.get_token_count(messages[i]["content"], conf)
                messages.pop(i)

            if total_tokens <= max_tokens:
                return messages, function_list, True

        # Degrade the most recent user and function messages as the last resort
        log.debug(f"Degrating user/function messages (total: {total_tokens}/max: {max_tokens})")
        for i in [most_recent_function, most_recent_user]:
            if total_tokens <= max_tokens:
                return messages, function_list, True
            while total_tokens > max_tokens:
                degraded_content = _degrade_message(messages[i]["content"])
                if degraded_content:
                    token_diff = self.get_token_count(
                        messages[i]["content"], conf
                    ) - self.get_token_count(degraded_content, conf)
                    messages[i]["content"] = degraded_content
                    total_tokens -= token_diff
                else:
                    total_tokens -= self.get_token_count(messages[i]["content"], conf)
                    messages.pop(i)
                    break

        return messages, function_list, True

    async def token_pagify(self, text: str, conf: GuildSettings):
        """Pagify a long string by tokens rather than characters"""
        token_chunks = []
        tokens = await self.get_tokens(text, conf)
        current_chunk = []

        max_tokens = min(conf.max_tokens - 100, MODELS[conf.model])
        for token in tokens:
            current_chunk.append(token)
            if len(current_chunk) == max_tokens:
                token_chunks.append(current_chunk)
                current_chunk = []

        if current_chunk:
            token_chunks.append(current_chunk)

        text_chunks = []
        for chunk in token_chunks:
            text = await self.get_text(chunk)
            text_chunks.append(text)

        return text_chunks

    async def get_function_menu_embeds(self, user: discord.Member) -> List[discord.Embed]:
        func_dump = {k: v.dict() for k, v in self.db.functions.items()}
        registry = {"Assistant": func_dump}
        for cog_name, function_schemas in self.registry.items():
            cog = self.bot.get_cog(cog_name)
            if not cog:
                continue
            for function_name, function_schema in function_schemas.items():
                function_obj = getattr(cog, function_name, None)
                if function_obj is None:
                    continue
                if cog_name not in registry:
                    registry[cog_name] = {}
                registry[cog_name][function_name] = {
                    "code": inspect.getsource(function_obj),
                    "jsonschema": function_schema,
                }

        conf = self.db.get_conf(user.guild)

        pages = sum(len(v) for v in registry.values())
        page = 1
        embeds = []
        for cog_name, functions in registry.items():
            for function_name, func in functions.items():
                embed = discord.Embed(
                    title="Custom Functions", description=function_name, color=discord.Color.blue()
                )
                if cog_name != "Assistant":
                    embed.add_field(
                        name="3rd Party",
                        value=f"This function is managed by the `{cog_name}` cog",
                        inline=False,
                    )
                schema = orjson.dumps(func["jsonschema"], indent=2)
                tokens = await self.get_token_count(schema, conf)
                schema_text = (
                    f"This function consumes `{humanize_number(tokens)}` input tokens each call\n"
                )

                if user.id in self.bot.owner_ids:
                    if len(schema) > 1000:
                        schema_text += box(schema[:1000], "py") + "..."
                    else:
                        schema_text += box(schema, "py")

                    if len(func["code"]) > 1000:
                        code_text = box(func["code"][:1000], "py") + "..."
                    else:
                        code_text = box(func["code"], "py")

                else:
                    schema_text += box(func["jsonschema"]["description"], "json")
                    code_text = box("Hidden...")

                embed.add_field(name="Schema", value=schema_text, inline=False)
                embed.add_field(name="Code", value=code_text, inline=False)

                embed.set_footer(text=f"Page {page}/{pages}")
                embeds.append(embed)
                page += 1

        if not embeds:
            embeds.append(
                discord.Embed(
                    description="No custom code has been added yet!", color=discord.Color.purple()
                )
            )
        return embeds

    async def get_embbedding_menu_embeds(
        self, conf: GuildSettings, place: int
    ) -> List[discord.Embed]:
        embeddings = sorted(conf.embeddings.items(), key=lambda x: x[0])
        embeds = []
        pages = math.ceil(len(embeddings) / 5)
        start = 0
        stop = 5
        for page in range(pages):
            stop = min(stop, len(embeddings))
            embed = discord.Embed(title="Embeddings", color=discord.Color.blue())
            embed.set_footer(text=f"Page {page + 1}/{pages}")
            num = 0
            for i in range(start, stop):
                em = embeddings[i]
                text = em[1].text
                tokens = await self.get_token_count(text, conf)
                val = f"`Tokens: `{tokens}\n```\n{text[:30]}...\n```"
                embed.add_field(
                    name=f"➣ {em[0]}" if place == num else em[0],
                    value=val,
                    inline=False,
                )
                num += 1
            embeds.append(embed)
            start += 5
            stop += 5
        if not embeds:
            embeds.append(
                discord.Embed(
                    description="No embeddings have been added!", color=discord.Color.purple()
                )
            )
        return embeds
