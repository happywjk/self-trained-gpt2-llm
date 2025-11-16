# %%
import os
import sentencepiece as spm
# os.environ["SPM_TRAIN_VERBOSE"] = "true"
# options = dict(
#     # input
#     input="/root/happywjk/home/jw2777/zero_to_hero/final_gpt/novel/cnnovel125k_00_01.txt",
#     input_format="text",

#     # output
#     model_prefix="tok_normal_novel_bpe",

#     # algo
#     model_type="bpe",
#     vocab_size=49152,
#     hard_vocab_limit=False,            # ← 防止达不到目标词表时报错

#     # normalization
#     normalization_rule_name="nmt_nfkc",# ← 更推荐：统一全/半角等
#     remove_extra_whitespaces=False,

#     # sampling & lengths
#     input_sentence_size=20000000,      # ← 先用 2e7 做验证，够大够快
#     max_sentence_length=4192,
#     seed_sentencepiece_size=1000000,        # ← 更正正确参数名；或直接去掉

#     shuffle_input_sentence=True,

#     # rare chars
#     character_coverage=0.9995,         # ← 搭配 byte_fallback，稳且不撑表
#     byte_fallback=True,

#     # merge/split rules
#     split_digits=False,                # ← 中文小说更建议整体数字
#     split_by_unicode_script=True,
#     split_by_whitespace=True,
#     split_by_number=True,
#     max_sentencepiece_length=16,
#     add_dummy_prefix=True,
#     allow_whitespace_only_pieces=True,

#     # special tokens
#     unk_id=0,
#     bos_id=1,
#     eos_id=2,
#     pad_id=-1,                         # 若下游需要PAD，改为具体ID（如3）
#     # user_defined_symbols=["<chapter>", "<summary>"], # 如需

#     # system
#     num_threads=8,
# )

# spm.SentencePieceTrainer.train(**options)
# # status = sorted((v,k) for k,v in status.items())
# # print(status)
# %%
sp = spm.SentencePieceProcessor()
sp.load('tok_normal_novel_bpe.model')
vocab = [[sp.id_to_piece(idx), idx] for idx in range(sp.get_piece_size())]
print(vocab[:200])
# %%
ids = sp.encode("我在图书馆，坐在我的女朋友旁边，使用我的电脑进行大模型相关的学习")
print(ids)
# %%
for i in ids:
    token = sp.id_to_piece(i)
    print(f"{i}\t{token}")
# %%
