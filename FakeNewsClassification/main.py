import torch
from BertTest import BertTest
from dataprocess import dataProcess
from transformers import BertTokenizer
from dataset import *

BATCH_SIZE = 64
PRETRAINED_MODEL_NAME = "bert-base-chinese"

"""
實作可以一次回傳一個 mini-batch 的 DataLoader
這個 DataLoader 吃我們上面定義的 `FakeNewsDataset`，
回傳訓練 BERT 時會需要的 4 個 tensors：
- tokens_tensors  : (batch_size, max_seq_len_in_batch)
- segments_tensors: (batch_size, max_seq_len_in_batch)
- masks_tensors   : (batch_size, max_seq_len_in_batch)
- label_ids       : (batch_size)
"""

from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence


# 這個函式的輸入 `samples` 是一個 list，裡頭的每個 element 都是
# 剛剛定義的 `FakeNewsDataset` 回傳的一個樣本，每個樣本都包含 3 tensors：
# - tokens_tensor
# - segments_tensor
# - label_tensor
# 它會對前兩個 tensors 作 zero padding，並產生前面說明過的 masks_tensors
def create_mini_batch(samples):
    tokens_tensors = [s[0] for s in samples]
    segments_tensors = [s[1] for s in samples]

    # 測試集有 labels
    if samples[0][2] is not None:
        label_ids = torch.stack([s[2] for s in samples])
    else:
        label_ids = None

    # zero pad 到同一序列長度
    tokens_tensors = pad_sequence(tokens_tensors, batch_first=True)
    segments_tensors = pad_sequence(segments_tensors, batch_first=True)

    # attention masks，將 tokens_tensors 裡頭不為 zero padding
    # 的位置設為 1 讓 BERT 只關注這些位置的 tokens
    masks_tensors = torch.zeros(tokens_tensors.shape, dtype=torch.long)
    masks_tensors = masks_tensors.masked_fill(tokens_tensors != 0, 1)

    return tokens_tensors, segments_tensors, masks_tensors, label_ids

def tensorlized(trainset) :
    # 選擇第一個樣本
    sample_idx = 0

    # 將原始文本拿出做比較
    text_a, text_b, label = trainset.df.iloc[sample_idx].values

    # 利用剛剛建立的 Dataset 取出轉換後的 id tensors
    tokens_tensor, segments_tensor, label_tensor = trainset[sample_idx]

    # 將 tokens_tensor 還原成文本
    tokens = tokenizer.convert_ids_to_tokens(tokens_tensor.tolist())
    combined_text = "".join(tokens)

    # 渲染前後差異，毫無反應就是個 print。可以直接看輸出結果
    print(f"""[原始文本]
    句子 1：{text_a}
    句子 2：{text_b}
    分類  ：{label}

    --------------------

    [Dataset 回傳的 tensors]
    tokens_tensor  ：{tokens_tensor}

    segments_tensor：{segments_tensor}

    label_tensor   ：{label_tensor}

    --------------------

    [還原 tokens_tensors]
    {combined_text}
    """)
def show(trainloader):
    data = next(iter(trainloader))

    tokens_tensors, segments_tensors, masks_tensors, label_ids = data

    print(f"""
    tokens_tensors.shape   = {tokens_tensors.shape} 
    {tokens_tensors}
    ------------------------
    segments_tensors.shape = {segments_tensors.shape}
    {segments_tensors}
    ------------------------
    masks_tensors.shape    = {masks_tensors.shape}
    {masks_tensors}
    ------------------------
    label_ids.shape        = {label_ids.shape}
    {label_ids}
    """)


def get_predictions(model, dataloader, compute_acc=False):
    predictions = None
    correct = 0
    total = 0

    with torch.no_grad():
        # 遍巡整個資料集
        for data in dataloader:
            # 將所有 tensors 移到 GPU 上
            if next(model.parameters()).is_cuda:
                data = [t.to("cuda:0") for t in data if t is not None]

            # 別忘記前 3 個 tensors 分別為 tokens, segments 以及 masks
            # 且強烈建議在將這些 tensors 丟入 `model` 時指定對應的參數名稱
            tokens_tensors, segments_tensors, masks_tensors = data[:3]
            outputs = model(input_ids=tokens_tensors,
                            token_type_ids=segments_tensors,
                            attention_mask=masks_tensors)

            logits = outputs[0]
            _, pred = torch.max(logits.data, 1)

            # 用來計算訓練集的分類準確率
            if compute_acc:
                labels = data[3]
                total += labels.size(0)
                correct += (pred == labels).sum().item()

            # 將當前 batch 記錄下來
            if predictions is None:
                predictions = pred
            else:
                predictions = torch.cat((predictions, pred))

    if compute_acc:
        acc = correct / total
        return predictions, acc
    return predictions


def buildmodel():
    # 載入一個可以做中文多分類任務的模型，n_class = 3
    from transformers import BertForSequenceClassification
    NUM_LABELS = 3

    model = BertForSequenceClassification.from_pretrained(PRETRAINED_MODEL_NAME, num_labels=NUM_LABELS)

    # high-level 顯示此模型裡的 modules
    print("""
    name            module
    ----------------------""")
    for name, module in model.named_children():
        if name == "bert":
            for n, _ in module.named_children():
                print(f"{name}:{n}")
        else:
            print("{:15} {}".format(name, module))

    return model
def train():
    # 訓練模式
    model.train()

    # 使用 Adam Optim 更新整個分類模型的參數
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    EPOCHS = 5
    for epoch in range(EPOCHS):

        running_loss = 0.0
        for data in trainloader:
            tokens_tensors, segments_tensors, \
            masks_tensors, labels = [t.to(device) for t in data]

            # 將參數梯度歸零
            optimizer.zero_grad()

            # forward pass
            outputs = model(input_ids=tokens_tensors,
                            token_type_ids=segments_tensors,
                            attention_mask=masks_tensors,
                            labels=labels)

            loss = outputs[0]
            # backward
            loss.backward()
            optimizer.step()

            # 紀錄當前 batch loss
            running_loss += loss.item()

        # 計算分類準確率
        _, acc = get_predictions(model, trainloader, compute_acc=True)

        print('[epoch %d] loss: %.3f, acc: %.3f' %
              (epoch + 1, running_loss, acc))
    return model

def eval():
    # 建立測試集。這邊我們可以用跟訓練時不同的 batch_size，看你 GPU 多大
    testset = FakeNewsDataset("test", tokenizer=tokenizer)
    testloader = DataLoader(testset, batch_size=256,
                            collate_fn=create_mini_batch)

    # 用分類模型預測測試集
    predictions = get_predictions(model, testloader)

    # 用來將預測的 label id 轉回 label 文字
    index_map = {v: k for k, v in testset.label_map.items()}

    # 生成 Kaggle 繳交檔案
    df = pd.DataFrame({"Category": predictions.tolist()})
    df['Category'] = df.Category.apply(lambda x: index_map[x])
    df_pred = pd.concat([testset.df.loc[:, ["Id"]],
                         df.loc[:, 'Category']], axis=1)
    df_pred.to_csv("output/bert_1_prec_training_samples.csv", index=False)

def save(model):
    torch.save(model.state_dict(), "output/Bert_FakenewsClassification")

if __name__ == "__main__":
    test = True
    if test:
        testBert = BertTest(PRETRAINED_MODEL_NAME)
        testBert.testall()
    else:
        data = dataProcess(traindir="data/train.csv", testdir="data/test.csv")
        data.preprocess()
        tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)
        # 初始化一個專門讀取訓練樣本的 Dataset，使用中文 BERT 斷詞
        trainset = FakeNewsDataset("train", tokenizer=tokenizer)
        tensorlized(trainset)
        # 初始化一個每次回傳 64 個訓練樣本的 DataLoader
        # 利用 `collate_fn` 將 list of samples 合併成一個 mini-batch 是關鍵

        trainloader = DataLoader(trainset, batch_size=BATCH_SIZE,
                                 collate_fn=create_mini_batch)
        model = buildmodel()
        # 讓模型跑在 GPU 上並取得訓練集的分類準確率
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("device:", device)
        model = model.to(device)
        model = train(model)
        eval(model)
        save(model)
