# %%
import os
import tiktoken
import torch

def preview_encoding(text: str, encoding_name: str = "gpt2"):
    enc = tiktoken.get_encoding(encoding_name)
    ids = enc.encode(text)
    print(ids)
    print(enc.decode(ids))

def encode_corpus_to_pt(input_path: str, output_path: str = "novel_tokens.pt", encoding_name: str = "gpt2", chunk_size: int = 10000):
    enc = tiktoken.get_encoding(encoding_name)
    tokens_all = []
    with open(input_path, "r", encoding="utf-8") as f:
        buffer = []
        for i, line in enumerate(f):
            buffer.append(line.strip())
            if len(buffer) >= chunk_size:
                for t in buffer:
                    tokens_all.extend(enc.encode(t))
                buffer = []
                print(f"Encoded {i+1} lines...")
        if buffer:
            for t in buffer:
                tokens_all.extend(enc.encode(t))
    tokens_tensor = torch.tensor(tokens_all, dtype=torch.long)
    torch.save({"tokens": tokens_tensor}, output_path)
    print(f"✅ Saved {len(tokens_all)} tokens to {output_path}")

if __name__ == "__main__":
    sample = "我在图书馆，坐在我的女朋友旁边，使用我的电脑进行大模型相关的学习"
    preview_encoding(sample)
    # 如果需要，将语料编码为pt缓存：
    # encode_corpus_to_pt("/root/happywjk/home/jw2777/zero_to_hero/final_gpt/novel/cnnovel125k_00_01.txt")
# %%
