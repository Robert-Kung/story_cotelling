# story-cotelling

應用強化學習與知識圖譜於故事共述生成之研究
Story Co-telling Dialogue Generation via Reinforcement Learning and Knowledge Graph

## Environment
+ Main container: [huggingface/transformers-pytorch-gpu:4.23.1](https://hub.docker.com/layers/huggingface/transformers-pytorch-gpu/4.23.1/images/sha256-d564ba7b41309ce4ca2ff11a3d82fd37d5abb9579a8f40e3a085f17db34c8128?context=explore)

### First build docker container
```
docker pull huggingface/transformers-pytorch-gpu:4.23.1
docker run --gpus all -itd -v [ProjectPath]:/root/story-cotelling --network host --name [ContainerName] huggingface/transformers-pytorch-gpu:4.23.1
```

### Run exist container
```
docker start [Container ID/Name]
docker exec -it [Container ID/Name] bash
```

### First setup environment
```
cd /root/story-cotelling
apt update
apt install unzip tmux graphviz default-jre -y
python3 -m pip install --upgrade pip
pip install -r requirements.txt
```

### First setup corenlp
```
cd /root/story-cotelling/environment/text2kg
python3 first_run.py
```

### Download data and model

#### Data
```
gdown 1PcH43USi7MIAIm_WXx3a1zgPV0p-S5Ji
unzip data.zip
```
#### Kg2text Model
```
gdown 1ln0U7bKnQxBqyXug_TltcYUtj7wV1w_U
mv kg2text_model.pt environment/kg2text/model/
```

#### Dialogue Evalution Model
```
gdown 1au7w3N7rrc8Wckdo65SVfBOFiQl30tFd
mv dialogue_evalution_model_best.pt environment/dialogue_evalution/model/
```

## Execute
### Train
+ 只要是 `_train.py` 結尾的檔案都是訓練程式碼，以下以 `dqn_train.py` 為例
```
cd /root/story-cotelling
python3 dqn_train.py
```

### Inference
+ 只要是 `_inference.py` 結尾的檔案都是推論程式碼，以下以 `dqn_inference.py` 為例
```
cd /root/story-cotelling
python3 dqn_inference.py
```

## Folder Structure

```
├── chatgpt/                    # 使用ChatGPT產生資料的程式碼
│   ├── history/                # ChatGPT對話歷史
│   ├── mixed_plot_point/       # 不同分數的劇情重點
│   ├── plot_point/             # 使用ChatGPT產生的劇情重點
│   ├── summary/                # 使用ChatGPT產生的摘要
│   ├── generate_plot_point.py  # 使用ChatGPT產生劇情重點程式碼
│   ├── generate_summary.py     # 使用ChatGPT產生摘要程式碼
│   ├── merge_summary.py        # 合併摘要程式碼(多故事摘要合併到一個檔)
│   └── noise_function.py       # 不同分數的劇情重點程式碼
├── data/                       # 資料集
│   ├── FairytaleQA_Dataset/    # FairytaleQA原始資料集
│   │   ├── FairytaleQA_Dataset/
│   │   ├── FairytaleQA_Dataset_Sentence_Split/
│   │   └── huggingface_hub/
│   ├── kg                      # 故事知識圖譜
│   │   ├── test/               # 測試集知識圖譜
│   │   ├── test_coref/         # 測試集知識圖譜(有使用共指消解)
│   │   ├── train/              # 訓練集知識圖譜
│   │   ├── train_coref/        # 訓練集知識圖譜(有使用共指消解)
│   │   ├── val/                # 驗證集知識圖譜
│   │   └── val_coref/          # 驗證集知識圖譜(有使用共指消解)
│   ├── kg2text/                # kg2text資料集
│   │   ├── kg2text_test.json
│   │   ├── kg2text_train.json
│   │   └── kg2text_val.json
│   ├── plot_point/             # 劇情重點資料集
│   │   ├── plot_point_test.json
│   │   ├── plot_point_train.json
│   │   └── plot_point_val.json
│   └── summary/                # 摘要資料集
│       ├── summary_test.json
│       ├── summary_train.json
│       └── summary_val.json
├── environment/                # 環境
│   ├── dialogue_evalution/     # 對話評估模型與程式碼
│   │   └── model/
│   ├── kg2text                 # kg2text模型與程式碼
│   │   └── model/
│   └── text2kg                 # text2kg程式碼
├── log/                        # 訓練紀錄
├── model/                      # 模型
├── output/                     # 訓練/推論過程的對話紀錄(輸出)
├── dqn_inference.py                 # Multi-Env MARL推論程式碼
├── dqn_train.py                     # Multi-Env MARL訓練程式碼
├── dqn.py                           # Multi-Env MARL模型
├── single_dqn_inference.py          # Single-Env MARL推論程式碼
├── single_dqn_train.py              # Single-Env MARL推論程式碼
├── single_dqn.py                    # Single-Env MARL推論程式碼
├── entity_dqn_inference.py          # Multi-Env MARL(+EntityCompare)推論程式碼
├── entity_dqn_train.py              # Multi-Env MARL(+EntityCompare)推論程式碼
├── entity_dqn.py                    # Multi-Env MARL(+EntityCompare)推論程式碼
├── entity_single_dqn_inference.py   # Single-Env MARL(+EntityCompare)推論程式碼
├── entity_single_dqn_train.py       # Single-Env MARL(+EntityCompare)推論程式碼
├── entity_single_dqn.py             # Single-Env MARL(+EntityCompare)推論程式碼
├── requirements.txt                 # 環境需求
└── README.md                        # 說明文件
```