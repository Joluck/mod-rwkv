# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tokenization classes for RWKV."""

import os
import re
from typing import TYPE_CHECKING, List, Optional, Tuple

from transformers import AddedToken, PreTrainedTokenizer
from transformers.utils import logging


if TYPE_CHECKING:
    pass

logger = logging.get_logger(__name__)


VOCAB_FILES_NAMES = {
    "vocab_file": "wr_vocab_v20230424.txt",
}

CHAT_TEMPLATE = "{% for message in messages %}{% if message['role'] == 'user' %}{{'\x16User: ' + message['content'] + '\x17'}}{% elif message['role'] == 'system' %}{{'\x16System: ' + message['content'] + '\x17'}}{% elif message['role'] == 'assistant' %}{{'\x16Assistant: ' + message['content'] + '\x17'}}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ '\x16Assistant: <think></think>' }}{% endif %}"



def _as_vocab_key(token):
    if isinstance(token, bytes):
        return token
    if isinstance(token, str):
        return token.encode("utf-8")
    return None




class TRIE:
    __slots__ = tuple("ch,to,values,front".split(","))
    to: list
    values: set

    def __init__(self, front=None, ch=None):
        self.ch = ch
        self.to = [None for ch in range(256)]
        self.values = set()
        self.front = front

    def __repr__(self):
        fr = self
        ret = []
        while fr != None:
            if fr.ch != None:
                ret.append(fr.ch)
            fr = fr.front
        return "<TRIE %s %s>" % (ret[::-1], self.values)

    def add(self, key: bytes, idx: int = 0, val=None):
        if idx == len(key):
            if val is None:
                val = key
            self.values.add(val)
            return self
        ch = key[idx]
        if self.to[ch] is None:
            self.to[ch] = TRIE(front=self, ch=ch)
        return self.to[ch].add(key, idx=idx + 1, val=val)

    def find_longest(self, key: bytes, idx: int = 0):
        u: TRIE = self
        ch: int = key[idx]

        while u.to[ch] is not None:
            u = u.to[ch]
            idx += 1
            if u.values:
                ret = idx, u, u.values
            if idx == len(key):
                break
            ch = key[idx]
        return ret


class RWKV_TOKENIZER:
    def __init__(self, file_name):
        self.idx2token = {}
        sorted = []  # must be already sorted
        with open(file_name, "r", encoding="utf-8") as f:
            lines = f.readlines()
        for l in lines:
            idx = int(l[: l.index(" ")])
            x = eval(l[l.index(" ") : l.rindex(" ")])
            x = x.encode("utf-8") if isinstance(x, str) else x
            assert isinstance(x, bytes)

            assert len(x) == int(l[l.rindex(" ") :])
            sorted += [x]
            self.idx2token[idx] = x

        self.token2idx = {}
        for k, v in self.idx2token.items():
            self.token2idx[v] = int(k)

        self.root = TRIE()
        for t, i in self.token2idx.items():
            _ = self.root.add(t, val=(t, i))

    def encodeBytes(self, src: bytes):
        idx: int = 0
        tokens = []
        while idx < len(src):
            _idx: int = idx
            idx, _, values = self.root.find_longest(src, idx)
            assert idx != _idx
            _, token = next(iter(values))
            tokens.append(token)
        return tokens

    def decodeBytes(self, tokens):
        return b"".join(map(lambda i: self.idx2token[i], tokens))

    def encode(self, src):
        if isinstance(src, str):
            return [self.encodeBytes(src.encode("utf-8"))]
        elif isinstance(src, list):
            return [self.encodeBytes(s.encode("utf-8")) for s in src]

    def decode(self, tokens):
        return [self.decodeBytes(batch).decode("utf-8") for batch in tokens]
        # try:
        #     return self.decodeBytes(tokens).decode('utf-8')
        # except:
        #     return '\ufffd' # bad utf-8

    def printTokens(self, tokens):
        for i in tokens:
            s = self.idx2token[i]
            try:
                s = s.decode("utf-8")
            except:
                pass
            print(f"{repr(s)}{i}", end=" ")
        print()


class RwkvTokenizer(PreTrainedTokenizer):
    vocab_files_names = VOCAB_FILES_NAMES
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self, vocab_file, bos_token="<|rwkv_tokenizer_end_of_text|>", 
        eos_token="<|rwkv_tokenizer_end_of_text|>", 
        unk_token="<|rwkv_tokenizer_end_of_text|>", 
        chat_template=None,
        **kwargs
    ):
        if not os.path.isfile(vocab_file):
            raise ValueError(
                f"Can't find a vocabulary file at path '{vocab_file}'."
            )

        with open(vocab_file, "r", encoding="utf-8") as reader:
            tokens = reader.readlines()

        if "add_bos_token" in kwargs:
            self.add_bos_token = kwargs["add_bos_token"]
        else:
            self.add_bos_token = False
        self.trie_tokenizer = RWKV_TOKENIZER(vocab_file)
        vocab = self.trie_tokenizer.token2idx
        self.encoder = vocab
        self.decoder = {v: k for k, v in vocab.items()}
        self._added_tokens_decoder = {0: AddedToken(str(bos_token))}
        self.chat_template = CHAT_TEMPLATE if chat_template is None else chat_template
        super().__init__(
            bos_token=bos_token, eos_token=eos_token, unk_token=unk_token, chat_template=self.chat_template, **kwargs
        )

        # Add vision special tokens to added_tokens_encoder for proper handling
        # These tokens are already in the vocab file at specific IDs:
        # <|vision_start|>: 65530, <|vision_end|>: 65531, <|image_pad|>: 65532

        self.image_token = "<|image_pad|>"
        self.vision_start_token = "<|vision_start|>"
        self.vision_end_token = "<|vision_end|>"

        self._special_token_text = sorted(
            {
                str(self.bos_token),
                str(self.eos_token),
                str(self.unk_token),
                self.image_token,
                self.vision_start_token,
                self.vision_end_token,
            },
            key=len,
            reverse=True,
        )
        self._special_token_pattern = re.compile(
            "(" + "|".join(re.escape(token) for token in self._special_token_text) + ")"
        )


        self.image_token_id = self.convert_tokens_to_ids(self.image_token)
        self.vision_start_token_id = self.convert_tokens_to_ids(self.vision_start_token)
        self.vision_end_token_id = self.convert_tokens_to_ids(self.vision_end_token)


    @property
    def vocab_size(self):
        return len(self.encoder)

    def get_vocab(self):
        vocab = self.encoder
        vocab.update(self.added_tokens_encoder)
        vocab = dict(sorted(vocab.items(), key=lambda item: item[1]))
        return vocab

    def _split_on_special_tokens(self, text):
        if not text:
            return []
        return [part for part in self._special_token_pattern.split(text) if part]

    def _tokenize(self, text, split_special_tokens=False):
        token_ids = []
        for chunk in self._split_on_special_tokens(text):
            special_token_id = self._convert_token_to_id(chunk)
            if isinstance(special_token_id, int) and special_token_id != self.unk_token_id:
                token_ids.append(special_token_id)
                continue
            token_ids.extend(self.trie_tokenizer.encode(chunk)[0])
        return token_ids

    def _convert_token_to_id(self, token):
        if isinstance(token, int):
            return token

        vocab_key = _as_vocab_key(token)
        if vocab_key is not None and vocab_key in self.encoder:
            return self.encoder[vocab_key]

        return self.unk_token_id

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (byte) using the vocab."""
        token = self.decoder.get(index, self.unk_token)
        if isinstance(token, (bytes)):
            token = token.decode("utf-8", errors="replace")
        return token

    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (bytes) in a single string. Additional tokens are encoded to bytes"""
        out_string = b"".join(
            [k.encode(errors="replace") if isinstance(k, str) else k for k in tokens]
        ).decode("utf-8")
        return out_string

    def save_vocabulary(
        self, save_directory: str, filename_prefix: Optional[str] = None
    ) -> Tuple[str]:
        if os.path.isdir(save_directory):
            vocab_file = os.path.join(
                save_directory,
                (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"],
            )
        else:
            vocab_file = (
                filename_prefix + "-" if filename_prefix else ""
            ) + save_directory
        with open(vocab_file, "w", encoding="utf-8") as writer:
            for token_index, token in sorted(self.decoder.items()):
                if isinstance(token, str):
                    token_bytes = token.encode("utf-8")
                else:
                    token_bytes = token

                writer.write(f"{token_index} {repr(token)} {len(token_bytes)}\n")
        return (vocab_file,)

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        if self.add_bos_token:
            bos_token_ids = [self.bos_token_id]
        else:
            bos_token_ids = []

        output = bos_token_ids + token_ids_0

        if token_ids_1 is None:
            return output

        return output + bos_token_ids + token_ids_1

    def get_special_tokens_mask(
        self,
        token_ids_0: List[int],
        token_ids_1: Optional[List[int]] = None,
        already_has_special_tokens: bool = False,
    ) -> List[int]:
        """
        Retrieves sequence ids from a token list that has no special tokens added. This method is called when adding
        special tokens using the tokenizer `prepare_for_model` or `encode_plus` methods.

        Args:
            token_ids_0 (`List[int]`):
                List of IDs.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.
            already_has_special_tokens (`bool`, *optional*, defaults to `False`):
                Whether or not the token list is already formatted with special tokens for the model.

        Returns:
            `List[int]`: A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.
        """
        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0,
                token_ids_1=token_ids_1,
                already_has_special_tokens=True,
            )

        if not self.add_bos_token:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0,
                token_ids_1=token_ids_1,
                already_has_special_tokens=False,
            )

        if token_ids_1 is None:
            return [1] + ([0] * len(token_ids_0))
        return [1] + ([0] * len(token_ids_0)) + [1] + ([0] * len(token_ids_1))
    

if __name__ == "__main__":
    vocab_file = os.path.join(os.path.dirname(__file__), "wr_vocab_v20230424.txt")
    tokenizer = RwkvTokenizer(vocab_file=vocab_file)
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, how are you?"},
        {"role": "assistant", "content": "I'm good, thank you! How can I assist you today?"},
    ]

    inputs = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,return_tensors="pt")

    print(inputs.keys())

    # outputs = tokenizer.decode(inputs["input_ids"], skip_special_tokens=False)
    # print(outputs)

