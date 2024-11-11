import torch
from torch import nn
import pickle

# 加载保存的预处理对象
with open('token2idx.pkl', 'rb') as f:
    token2idx = pickle.load(f)

with open('mlb.pkl', 'rb') as f:
    mlb = pickle.load(f)

with open('keywords.pkl', 'rb') as f:
    keywords = pickle.load(f)

# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 获取特殊符号的索引
unk_idx = token2idx['<unk>']
pad_idx = token2idx['<pad>']

# 定义参数（与训练时相同）
vocab_size = len(token2idx)
embed_dim = 128       # 与训练时的 embed_dim 一致
hidden_dim = 64       # 与训练时的 hidden_dim 一致
num_classes = len(mlb.classes_)
max_len = 100         # 与训练时的 max_len 一致
keyword_dim = len(keywords)


# 定义增强的 LSTM 模型（与训练时相同）
class EnhancedLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes, pad_idx, keyword_dim):
        super(EnhancedLSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(hidden_dim + keyword_dim, num_classes)

    def forward(self, x, keyword_features):
        x = self.embedding(x)  # [batch_size, max_len, embed_dim]
        x, _ = self.lstm(x)    # [batch_size, max_len, hidden_dim]
        x = x[:, -1, :]        # 取最后一个时间步的输出
        x = self.dropout(x)
        # 将 LSTM 的输出与关键词特征连接
        x = torch.cat((x, keyword_features), dim=1)
        x = torch.sigmoid(self.fc(x))
        return x

# 初始化模型
model = EnhancedLSTMClassifier(
    vocab_size=vocab_size,
    embed_dim=embed_dim,
    hidden_dim=hidden_dim,
    num_classes=num_classes,
    pad_idx=pad_idx,
    keyword_dim=keyword_dim
)
model.to(device)

# 加载模型权重
model.load_state_dict(torch.load('enhanced_model_weights.pth', map_location=device))
model.eval()


# 定义预测函数
def predict_all_labels(project_name):
    with torch.no_grad():
        # 将项目名称转换为索引序列
        seq = [token2idx.get(token, unk_idx) for token in project_name]
        if len(seq) > max_len:
            seq = seq[:max_len]
        else:
            seq += [pad_idx] * (max_len - len(seq))
        input_tensor = torch.tensor([seq], dtype=torch.long).to(device)

        # 提取关键词特征
        keyword_feature = [1 if keyword in project_name else 0 for keyword in keywords]
        keyword_tensor = torch.tensor([keyword_feature], dtype=torch.float32).to(device)

        # 模型预测
        output = model(input_tensor, keyword_tensor)
        prediction = (output > 0.5).cpu().numpy().astype(int)
        return mlb.inverse_transform(prediction)[0]


def label_predict_SQL(state):
    question = state["question"]
    # predicted_labels = str(predict_all_labels(question))
    # new_question = question + ', ' + predicted_labels
    new_question = question
    return {"question": new_question}


# 示例使用
if __name__ == '__main__':
    # 输入您想要预测的项目名称
    # test_project = "新建上海至南通铁路项目太仓至四团段（上海境内）高压燃气管线及设施搬迁工程HTIIRQQG-1标终止招标公告"
    test_project = "新建上海至杭州铁路客运专线上海南联络线等2个铁路项目建管甲供物资（通信光电缆）联合采购二次招标公告"
    # 进行预测
    predicted_labels = predict_all_labels(test_project)

    # 输出预测结果
    print("项目名称:", test_project)
    print("预测结果:", predicted_labels)
