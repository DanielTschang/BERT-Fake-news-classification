from transformers import BertTokenizer
import torch

class BertTest():
    def __init__(self,PRETRAINED_MODEL_NAME):
        self.modelname = PRETRAINED_MODEL_NAME
        self.tokenizer = BertTokenizer.from_pretrained(self.modelname)
        self.vocab = self.tokenizer.vocab

    def funcs(self):
        print("type 'vocabsize' for vocab size")
        print("type 'randomtokens' for printing random tokens and their ids")
        print("type 'ind' for ㄅ ㄆ ㄇ indices")
        print("type 'token2ids' for transform tokens to ids")
        print("type 'CloveLM' for Clove test")

    def vocabsize(self):
        print("字典大小：", len(self.vocab))
    def randomtokens(self):
        import random
        random_tokens = random.sample(list(self.vocab), 10)
        random_ids = [self.vocab[t] for t in random_tokens]

        print("{0:20}{1:15}".format("token", "index"))
        print("-" * 25)
        for t, id in zip(random_tokens, random_ids):
            print("{0:15}{1:10}".format(t, id))

    def ind(self):
        indices = list(range(647, 657))
        some_pairs = [(t, idx) for t, idx in self.vocab.items() if idx in indices]
        for pair in some_pairs:
                print(pair)
    def token2ids(self):
        text = "[CLS] 等到潮水 [MASK] 了，就知道誰沒穿褲子。"
        tokens = self.tokenizer.tokenize(text)
        ids = self.tokenizer.convert_tokens_to_ids(tokens)

        print(text)
        print(tokens[:10], '...')
        print(ids[:10], '...')

    def CloveLM(self):
        from transformers import BertForMaskedLM
        text = "[CLS] 等到潮水 [MASK] 了，就知道誰沒穿褲子。"
        tokens = self.tokenizer.tokenize(text)
        ids = self.tokenizer.convert_tokens_to_ids(tokens)
        # 除了 tokens 以外我們還需要辨別句子的 segment ids
        tokens_tensor = torch.tensor([ids])  # (1, seq_len)
        segments_tensors = torch.zeros_like(tokens_tensor)  # (1, seq_len)
        maskedLM_model = BertForMaskedLM.from_pretrained(self.modelname)

        # 使用 masked LM 估計 [MASK] 位置所代表的實際 token
        maskedLM_model.eval()
        with torch.no_grad():
            outputs = maskedLM_model(tokens_tensor, segments_tensors)
            predictions = outputs[0]
            print("output is :", predictions)
            # (1, seq_len, num_hidden_units)
        del maskedLM_model

        # 將 [MASK] 位置的機率分佈取 top k 最有可能的 tokens 出來
        masked_index = 5
        k = 3
        probs, indices = torch.topk(torch.softmax(predictions[0, masked_index], -1), k)
        predicted_tokens = self.tokenizer.convert_ids_to_tokens(indices.tolist())

        # 顯示 top k 可能的字。一般我們就是取 top 1 當作預測值
        print("輸入 tokens ：", tokens[:10], '...')
        print('-' * 50)
        for i, (t, p) in enumerate(zip(predicted_tokens, probs), 1):
            tokens[masked_index] = t
            print(t)
            print("Top {} ({:2}%)：{}".format(i, int(p.item() * 100), tokens[:10]), '...')
    def testall(self):
        self.vocabsize()
        print(("=")*45)
        self.randomtokens()
        print(("=") * 45)
        self.token2ids()
        print(("=") * 45)
        self.ind()
        print(("=") * 45)
        self.CloveLM()