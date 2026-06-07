"""
模型工厂 — 统一创建/加载接口
model_type: "markov" | "lstm" | "transformer"
"""
from config import CFG
from tokenizer.combined import VOCAB_SIZE
from models.markov_model      import MarkovModel
from models.lstm_model        import LSTMModel
from models.transformer_model import TransformerModel


def create_model(model_type: str):
    mc = CFG.model
    if model_type == "markov":
        return MarkovModel(order=mc.markov_order, vocab_size=VOCAB_SIZE)

    elif model_type == "lstm":
        return LSTMModel(
            vocab_size=VOCAB_SIZE,
            embed_dim=mc.lstm_embed_dim,
            hidden_dim=mc.lstm_hidden_dim,
            num_layers=mc.lstm_num_layers,
            dropout=mc.lstm_dropout,
            lr=mc.learning_rate,
        )

    elif model_type == "transformer":
        return TransformerModel(
            vocab_size=VOCAB_SIZE,
            d_model=mc.tf_d_model,
            nhead=mc.tf_nhead,
            num_layers=mc.tf_num_layers,
            dim_ff=mc.tf_dim_ff,
            dropout=mc.tf_dropout,
            seq_len=mc.sequence_length,
            lr=mc.learning_rate,
        )
    else:
        raise ValueError(f"未知 model_type: {model_type}")


def load_model(model_type: str):
    path = {
        "markov":      "saved_models/markov.json",
        "lstm":        "saved_models/lstm_best.pt",
        "transformer": "saved_models/transformer_best.pt",
    }[model_type]
    if model_type == "markov":
        # classmethod：返回新实例，必须接收返回值
        return MarkovModel.load(path)
    else:
        # LSTM/Transformer：load() 是实例方法，in-place 修改
        model = create_model(model_type)
        model.load(path)
        return model
