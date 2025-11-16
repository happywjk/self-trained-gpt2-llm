# %%
import jieba
import config
import pickle
import os
import sentencepiece as spm
# %%
import torch


def encode_large_file(sp_model, input_path, output_path, chunk_size=10000):
    """
    用SentencePiece逐行编码大文本，并流式保存为pt文件
    chunk_size：每次处理多少行（避免内存峰值）
    """
    sp = spm.SentencePieceProcessor(model_file=sp_model)
    tokens_all = []

    with open(input_path, "r", encoding="utf-8") as f:
        buffer = []
        for i, line in enumerate(f):
            buffer.append(line.strip())
            if len(buffer) >= chunk_size:
                tokens = []
                for t in buffer:
                    tokens.extend(sp.encode(t, out_type=int))
                tokens_all.extend(tokens)
                buffer = []
                print(f"Encoded {i+1} lines...")

        # 剩余部分
        if buffer:
            for t in buffer:
                tokens_all.extend(sp.encode(t, out_type=int))

    tokens_tensor = torch.tensor(tokens_all, dtype=torch.long)
    torch.save({"tokens": tokens_tensor}, output_path)
    print(f"✅ Saved {len(tokens_all)} tokens to {output_path}")
    
class dataloaderlite:
    def __init__(self, model_path):
        self.batchsize = config.batchsize
        self.blocksize = config.blocksize
        cache_path = "novel_tokens.pt"
        self.sp    = spm.SentencePieceProcessor(model_file = model_path)
        if os.path.exists(cache_path):
            print(f"loading cached tokens from {cache_path}")
            pack = torch.load(cache_path, map_location="cpu")
            self.tokens = pack["tokens"]
        else:
            print("Encoding /root/happywjk/home/jw2777/zero_to_hero/final_gpt/novel/cnnovel125k_00_01.txt using SentencePiece (stream mode)")
            encode_large_file(model_path, "/root/happywjk/home/jw2777/zero_to_hero/final_gpt/novel/cnnovel125k_00_01.txt", cache_path)
            pack = torch.load(cache_path, map_location="cpu")
            self.tokens = pack["tokens"]
        split = int(0.9*len(self.tokens))
        print(split)
        self.test   = self.tokens[:split]
        self.validation = self.tokens[split:]
        self.current_pos = {"train":0, "validation":0}
    def get_stats(self,tokens):
        out = {}
        for s in zip(tokens,tokens[1:]):
            out[s] = out.get(s,0) + 1
        return out
    def merge(self,text,pair, idx):
        newtext = []
        i = 0
        while i < len(text):
            if i < len(text) - 1 and text[i] == pair[0] and text[i+1] == pair[1]:
                newtext.append(idx)
                i +=2
            else:
                newtext.append(text[i])
                i +=1
        return newtext
    def bp_encode(self,text):
        tokens = self.encode(text)
        while len(tokens) > 2:
            out = self.get_stats(tokens)
            pair = min(out,key =lambda p: self.merges.get(p,float("inf")))
            if pair not in self.merges:
                break
            idx = self.merges[pair]
            tokens = self.merge(tokens, pair, idx)
        return tokens

    def next_batch(self,mode = "train"):
        B,T = config.batchsize,config.blocksize
        if mode == "train":
            data = self.test
        elif mode == "validation":
            data = self.validation
        if self.current_pos[mode] > len(data) -B*T-1:
            self.current_pos[mode] = 0
        token = data[self.current_pos[mode]:self.current_pos[mode]+B*T+1]
        x = token[:-1].view(B,T)
        y = token[1:].view(B,T)
        x,y = x.to(config.device),y.to(config.device)
        self.current_pos[mode] += B*T
        return x,y

