import os
os.chdir('/home/wdy/code_LLM/code_LLM/')
import torch
import random
import argparse
import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, recall_score, f1_score
from transformers import BertTokenizer, BertPreTrainedModel, BertModel, BertConfig
from transformers.configuration_utils import PretrainedConfig

# 参数指定
parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=2e-5, help='weight_decay for Adam optimizer')
parser.add_argument('--cls_num', type=int, default=5, help='the num of classes to classify')
parser.add_argument('--bs', type=int, default=64, help='batch size')
parser.add_argument('--device', type=str, default="cuda", help='device to run')
parser.add_argument('--epochs_num', type=int, default=10, help='the train epochs')
parser.add_argument('--max_sent_len', type=int, default=256, help='maximum of sents length, tokenizer and model input')
parser.add_argument('--data_split_rate', type=float, default=0.8, help='dataset split rate')
parser.add_argument('--model', type=str, default='bert', help='which model to use')
parser.add_argument('--bert_path', type=str, default="./bert_model", 
                    help='where the bert pretrained model exists')
parser.add_argument('--bert_embedding_path', type=str, default='./bertEmbeddings_params/bertEmbeddingLayer_params.pth', 
                    help='where the bert pretrained embedding params exists')
parser.add_argument('--model_save_path', type=str, default="./output_bert", help='bert: dir, lstm: file')
parser.add_argument('--model_load_path', type=str, default="./output_bert", help='bert: dir, lstm: file')
parser.add_argument('--train', type=bool, default=False, help='if train or load from checkpoint')
args = parser.parse_args()

seed = 123456
torch.manual_seed(seed)
random.seed(seed)

# 分词器加载
tokenizer = BertTokenizer.from_pretrained(args.bert_path, local_files_only=True)

# customized data class
class MyData(Dataset):
    """
        input data file and output dataloader\n
        其中, __init__, __getitem__, __len__是继承Dataset类必须实现的三个函数\n
        Parameters:
         - data: list[Tuple(...)]
         - tokenizer: type(tokenizer)=None
         - args: type(args)=None
    """
    def __init__(self, data, tokenizer: BertTokenizer=None, args: argparse.Namespace=None) -> None:
        super().__init__()
        self.data = data
        self.tokenizer = tokenizer
        if args:
            self.data_split_rate = args.data_split_rate 
            self.bs = args.bs
            self.max_sent_len = args.max_sent_len
            self.device = args.device
        
    def __getitem__(self, index):
        return self.data[index]
    
    def __len__(self):
        return len(self.data)
    
    def build_dataLoader(self):
        """
        get list-like sources and targets, transform them into dataloader
        """
        # shuffle the data，in order that model can capture all the classes        
        random.shuffle(self.data)

        # split the data into train_set and test_set
        split_index = int(len(self.data) * self.data_split_rate)        
        dataloader_params = {"batch_size": self.bs, 
                              "shuffle": True, 
                              "collate_fn": self.collate_wrapper, 
                              "drop_last": True}
        return (DataLoader(dataset=MyData(data_split),
                    **dataloader_params)
                    for data_split in (self.data[:split_index], self.data[split_index:]))

    def collate_wrapper(self, batch):
        """
        determine the data format to be fed to model
        """
        batch_T = list(zip(*batch)) # (batch, 2) -> (2, batch)
        sents = list(batch_T[0])
        labels = batch_T[1]
        # 调用PreTrainedTokenizer的__call__(), 实现text2ids的转换
        sents_tokenized = self.tokenizer(text=sents,
                                    padding=True,
                                    truncation=True,
                                    max_length=self.max_sent_len,                                   
                                    return_attention_mask=True,
                                    return_tensors='pt', # 指定该参数以使返回pt格式的tensor，而不是数字
                                    ).to(self.device) # 指定在GPU上运行
        
        return (sents_tokenized, torch.LongTensor(labels).to(self.device))

# read data and process
df = pd.read_csv('./df_file.csv')
sources = df['Text']
targets = df['Label']
data = list(zip(sources, targets)) # to format like: [(..), (..), (..)]
train_dataLoader, test_dataLoader = MyData(data, tokenizer=tokenizer, args=args).build_dataLoader()

# build model structure
## bert-based method
class TextClassifier_bert(BertPreTrainedModel):
    """
        Bert-based model architecture\n
        directly inherit the BertPreTrainedModel, which subclass torch.nn.module\n
        Parameters:
         - config: PretrainedConfig
    """    
    def __init__(self, config: PretrainedConfig):
        super().__init__(config)        
        self.bert = BertModel(config=config)
        self.cls_layer = torch.nn.Linear(config.hidden_size, config.cls_num) # config.hidden_size即bert的最后一层pooler层输出维度
    
    def forward(self, input_ids, token_type_ids, attention_mask):        
        outputs_ = self.bert(input_ids=input_ids, token_type_ids=token_type_ids, \
                            attention_mask=attention_mask) # [batch, max_sent_len, hidden_size]
        
        # bert直接的输出返回的是BaseModelOutputWithPoolingAndCrossAttentions类，推测该类实现了__getitem__()；
        # 从而可以以下标的方式获取其中的pooler_output层（即bert的最后一层）输出；
        # 可以认为第二维度代表句嵌入向量，第三维度代表词嵌入向量，此时句子嵌入是二维的[max_sent_len, hidden_size]；
        # 然后经过pooler层，将词向量pool成句向量，相当于进行了降维，此时句子嵌入是一维的，维度是hidden_size。
        
        outputs = outputs_[1] # [batch, hidden_size]    
        logits = self.cls_layer(outputs) # [batch, cls_num]

        return logits

## lstm-based method
class TextClassifier_lstm(torch.nn.Module):
    """
        lstm-based model architecture\n
        directly employ the bert tokenizer and embedding layer\n
        Parameters:
         - args: Dict
    """
    def __init__(self, args, config: PretrainedConfig=None):
        super().__init__()
        self.embedding_layer = BertModel(config=config).embeddings
        self.lstm_layer = torch.nn.LSTM(config.hidden_size, args['hidden_size'], args['num_layers'], 
                                        batch_first=True, dropout=0.3)
        self.cls_layer = torch.nn.Linear(args['hidden_size'], args['cls_num'])

    def forward(self, input_ids, token_type_ids, attention_mask):
        embeddings = self.embedding_layer(input_ids) # [bs, max_sent_len, hidden_size]
        output, (h_n, h_c) = self.lstm_layer(embeddings, None) # output.shape: [bs, time_step, hidden_size]
        logits = self.cls_layer(output[:,-1,:]) # [bs, cls_num]
        return logits

def get_model(type: str):    
    config = BertConfig.from_pretrained(args.bert_path)
    config.max_length = args.max_sent_len # set bert的maxLen为我们数据集设定的maxLen
    config.cls_num = args.cls_num
    if type == 'bert':        
        model = TextClassifier_bert.from_pretrained(args.model_load_path, config=config)
        # freeze the bert params, only update the cls_layer
        model.bert.requires_grad_(False)
    elif args.train:
        args_lstm = {'hidden_size': 128, 'num_layers': 4, 'cls_num': 5}
        model = TextClassifier_lstm(args_lstm, config=config)
        # load the bert embedding params, do not freeze it
        model.embedding_layer.load_state_dict(torch.load(args.bert_embedding_path))        
    else:
        model = torch.load(args.model_load_path)        
    
    return model

model = get_model(args.model).to(args.device)

# criterion是损失函数，optimizer实现了梯度下降的算法；
# 二者均可选择更具体的方法，如此处用的是Adam优化算法，损失函数选择的是交叉熵损失
optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
criterion = torch.nn.CrossEntropyLoss()

def cal_metrics(labels_, preds_, loss_value):
    """
        计算评估指标值
    """
    acc = accuracy_score(labels_, preds_)
    recall = recall_score(labels_, preds_, average="macro")
    f1_micro = f1_score(labels_, preds_, average="micro")
    f1_macro = f1_score(labels_, preds_, average="macro")

    return 'loss: {:.3f}, acc: {:.3f}, recall: {:.3f}, f1_micro: {:.3f}, f1_macro: {:.3f}'\
        .format(loss_value, acc, recall, f1_micro, f1_macro)

print(args._get_args())
# train model
best_epoch = 0
loss = None
for epoch in tqdm(range(args.epochs_num), total=args.epochs_num, desc=f'training...'):
    
    # train
    preds_ = []
    labels_ = []      
    model.train() # 训练模式，使模型参数可改变
    for idx, (sents, labels) in tqdm(enumerate(train_dataLoader), total=len(train_dataLoader), \
                                     desc=f'epoch: {epoch}'):
        
        outputs = model(**sents) # [batch, cls_num]
        loss = criterion(outputs, labels) # logits: tensor, label: int
        
        # update model parameters
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        preds = torch.argmax(outputs, 1)       
        preds_ += list(preds.detach().cpu().numpy()) # tensors to list
        labels_ += list(labels.detach().cpu().numpy())
    
    print('train:', cal_metrics(labels_, preds_, loss.item()))

    # eval
    preds_ = []
    labels_ = []  
    model.eval() # 评估模式，使模型参数不改变
    for idx, (sents, labels) in enumerate(test_dataLoader):
        outputs = model(**sents) # [batch, cls_num]        
        loss = criterion(outputs, labels)
        
        # eval阶段不需要参数更新        
        # optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()

        preds = torch.argmax(outputs, 1)       
        preds_ += list(preds.detach().cpu().numpy())
        labels_ += list(labels.detach().cpu().numpy())

    print('eval:', cal_metrics(labels_, preds_, loss.item()))

    # 此轮训练效果有变好才保存模型
    acc = accuracy_score(labels_, preds_)
    if acc > best_epoch:
        best_epoch = acc
        if args.model == 'bert':
            model.save_pretrained(args.model_save_path)
        else:
            torch.save(model, args.model_save_path)