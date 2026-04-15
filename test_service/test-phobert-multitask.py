from __future__ import annotations

import sys
from pathlib import Path
from typing import Tuple

import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer
from transformers.models.roberta.modeling_roberta import RobertaModel
from transformers import RobertaConfig


MODEL_DIR = Path(
	r"C:\Users\User\Documents\web_emotion_chat\checkpoint-3824-20260411T075134Z-3-002\checkpoint-3824"
)

COARSE_LABELS = [
	"Negative_Sad",
	"Positive",
	"Surprise",
	"Anger",
	"Fear",
	"Neutral_Other",
]

FINE_LABELS = [
	"Buồn bã",
	"Chán ghét",
	"Cô đơn",
	"Highly negative",
	"Hối tiếc",
	"Lo âu",
	"Lạc quan",
	"Ngạc nhiên",
	"Other",
	"Sợ hãi",
	"Trung lập",
	"Tức giận",
	"Vui vẻ",
]

TEST_DATA = [
	("Tui bùn mún khóc lun á, hông ai hỉu tui hết...", "Negative_Sad", "Buồn bã"),
	("Cái thói làm ăn vầy là tui chán ghét cực kì nha", "Negative_Sad", "Chán ghét"),
	("Ngồi 1 mình giữa phố đông, thấy cô đơn vcl", "Negative_Sad", "Cô đơn"),
	("Giá như hùi đó mình ko làm v, hối tiếc ghê", "Negative_Sad", "Hối tiếc"),
	("Lo wá, ko bít mai đi thi có ổn áp ko nữa", "Negative_Sad", "Lo âu"),
	("Trời ơi hnay tui zui quá xá là zui lunnn", "Positive", "Vui vẻ"),
	("Mọi chuyện rùi sẽ ổn thui, cố lên tui ơi!", "Positive", "Lạc quan"),
	("Wao, ko thể tin đc lun, bất ngờ thực sự!", "Surprise", "Ngạc nhiên"),
	("Đm làm ăn như cc, bực cả mình, mún đấm ghê", "Anger", "Tức giận"),
	("Sợ vcl, chỗ này nhìn âm u ghê rợn quá", "Fear", "Sợ hãi"),
	("Cũng bth thui, ko có gì đặc biệt lắm", "Neutral_Other", "Trung lập"),
	("Ủa alo, cái gì đang xảy ra v mọi người?", "Neutral_Other", "Other"),
]


def load_tokenizer(model_dir: Path):
	try:
		return AutoTokenizer.from_pretrained(model_dir)
	except Exception:
		return AutoTokenizer.from_pretrained("vinai/phobert-large")


def build_config_from_weights(weights) -> RobertaConfig:
	hidden_size = weights["encoder.embeddings.word_embeddings.weight"].shape[1]
	vocab_size = weights["encoder.embeddings.word_embeddings.weight"].shape[0]
	max_position_embeddings = weights["encoder.embeddings.position_embeddings.weight"].shape[0]
	num_layers = len(
		{
			int(key.split(".")[3])
			for key in weights.keys()
			if key.startswith("encoder.encoder.layer.")
		}
	)

	return RobertaConfig(
		vocab_size=vocab_size,
		hidden_size=hidden_size,
		num_hidden_layers=num_layers,
		num_attention_heads=hidden_size // 64,
		intermediate_size=hidden_size * 4,
		hidden_act="gelu",
		hidden_dropout_prob=0.1,
		attention_probs_dropout_prob=0.1,
		max_position_embeddings=max_position_embeddings,
		type_vocab_size=1,
		layer_norm_eps=1e-5,
		pad_token_id=1,
		bos_token_id=0,
		eos_token_id=2,
	)


class PhoBertMultiTaskClassifier(torch.nn.Module):
	def __init__(self, config: RobertaConfig, coarse_labels: int, fine_labels: int):
		super().__init__()
		self.encoder = RobertaModel(config, add_pooling_layer=True)
		self.dense = torch.nn.Linear(config.hidden_size, config.hidden_size)
		self.classifier_coarse = torch.nn.Linear(config.hidden_size, coarse_labels)
		self.classifier_fine = torch.nn.Linear(config.hidden_size, fine_labels)
		self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)

	def forward(self, input_ids=None, attention_mask=None, token_type_ids=None):
		outputs = self.encoder(
			input_ids=input_ids,
			attention_mask=attention_mask,
			token_type_ids=token_type_ids,
		)
		cls_state = outputs.pooler_output if outputs.pooler_output is not None else outputs.last_hidden_state[:, 0]
		hidden = self.dense(cls_state)
		hidden = torch.tanh(hidden)
		hidden = self.dropout(hidden)
		coarse_logits = self.classifier_coarse(hidden)
		fine_logits = self.classifier_fine(hidden)
		return coarse_logits, fine_logits


def resolve_weight_file(model_dir: Path) -> Path:
	safetensors_path = model_dir / "model.safetensors"
	pytorch_path = model_dir / "pytorch_model.bin"

	if safetensors_path.exists():
		return safetensors_path
	if pytorch_path.exists():
		return pytorch_path

	raise FileNotFoundError(f"Khong tim thay weight file trong {model_dir}")


def load_model(model_dir: Path):
	weight_file = resolve_weight_file(model_dir)
	if weight_file.suffix == ".safetensors":
		weights = load_file(str(weight_file))
	else:
		weights = torch.load(weight_file, map_location="cpu")

	config = build_config_from_weights(weights)
	coarse_labels = weights["classifier_coarse.bias"].shape[0]
	fine_labels = weights["classifier_fine.bias"].shape[0]

	model = PhoBertMultiTaskClassifier(config, coarse_labels, fine_labels)
	missing, unexpected = model.load_state_dict(weights, strict=False)
	if missing:
		print(f"Missing keys: {missing}")
	if unexpected:
		print(f"Unexpected keys: {unexpected}")

	model.eval()
	return model, config, coarse_labels, fine_labels


def predict_batch(model: PhoBertMultiTaskClassifier, tokenizer, text: str):
	device = torch.device("cpu")
	encoded = tokenizer(
		text,
		return_tensors="pt",
		truncation=True,
		max_length=256,
		padding=False,
	)
	encoded = {key: value.to(device) for key, value in encoded.items()}
	with torch.no_grad():
		coarse_logits, fine_logits = model(**encoded)
	coarse_id = int(coarse_logits.argmax(dim=-1).item())
	fine_id = int(fine_logits.argmax(dim=-1).item())
	return coarse_id, fine_id


def predict(text: str) -> Tuple[int, int, torch.Tensor, torch.Tensor]:
	tokenizer = load_tokenizer(MODEL_DIR)
	model, _, _, _ = load_model(MODEL_DIR)
	device = torch.device("cpu")
	model.to(device)

	encoded = tokenizer(
		text,
		return_tensors="pt",
		truncation=True,
		max_length=256,
		padding=False,
	)

	encoded = {key: value.to(device) for key, value in encoded.items()}

	with torch.no_grad():
		coarse_logits, fine_logits = model(**encoded)

	coarse_id = int(coarse_logits.argmax(dim=-1).item())
	fine_id = int(fine_logits.argmax(dim=-1).item())
	return coarse_id, fine_id, coarse_logits.squeeze(0), fine_logits.squeeze(0)


def main() -> None:
	text = " ".join(sys.argv[1:]).strip()
	if text == "--eval" or text == "eval":
		run_evaluation()
		return
	if not text:
		text = "Tui cảm thấy hôm nay khá vui và nhẹ nhõm."

	model, config, coarse_labels, fine_labels = load_model(MODEL_DIR)
	tokenizer = load_tokenizer(MODEL_DIR)
	device = torch.device("cpu")
	model.to(device)

	encoded = tokenizer(
		text,
		return_tensors="pt",
		truncation=True,
		max_length=256,
	)
	encoded = {key: value.to(device) for key, value in encoded.items()}

	with torch.no_grad():
		coarse_logits, fine_logits = model(**encoded)

	coarse_id = int(coarse_logits.argmax(dim=-1).item())
	fine_id = int(fine_logits.argmax(dim=-1).item())

	print(f"Model path: {MODEL_DIR}")
	print(f"Hidden size: {config.hidden_size}")
	print(f"Layers: {config.num_hidden_layers}")
	print(f"Coarse classes: {coarse_labels}")
	print(f"Fine classes: {fine_labels}")
	print(f"Input: {text}")
	print(f"Pred coarse id: {coarse_id}")
	print(f"Pred fine id: {fine_id}")
	print(f"Pred coarse label: {COARSE_LABELS[coarse_id] if 0 <= coarse_id < len(COARSE_LABELS) else 'Unknown'}")
	print(f"Pred fine label: {FINE_LABELS[fine_id] if 0 <= fine_id < len(FINE_LABELS) else 'Unknown'}")


def run_evaluation() -> None:
	model, _, _, _ = load_model(MODEL_DIR)
	tokenizer = load_tokenizer(MODEL_DIR)
	device = torch.device("cpu")
	model.to(device)

	correct_coarse = 0
	correct_fine = 0
	total = len(TEST_DATA)

	print(f"Model path: {MODEL_DIR}")
	print(f"Evaluating {total} samples\n")
	print(f"{'Text':<48} | {'Pred coarse / fine':<42} | {'Gold coarse / fine'}")
	print("-" * 120)

	for text, gold_coarse, gold_fine in TEST_DATA:
		coarse_id, fine_id = predict_batch(model, tokenizer, text)
		pred_coarse = COARSE_LABELS[coarse_id] if 0 <= coarse_id < len(COARSE_LABELS) else f"ID_{coarse_id}"
		pred_fine = FINE_LABELS[fine_id] if 0 <= fine_id < len(FINE_LABELS) else f"ID_{fine_id}"

		if pred_coarse == gold_coarse:
			correct_coarse += 1
		if pred_fine == gold_fine:
			correct_fine += 1

		print(f"{text[:45]:<48} | {pred_coarse:<20} / {pred_fine:<18} | {gold_coarse:<16} / {gold_fine}")

	coarse_acc = correct_coarse / total * 100
	fine_acc = correct_fine / total * 100

	print("-" * 120)
	print(f"Coarse accuracy: {correct_coarse}/{total} = {coarse_acc:.2f}%")
	print(f"Fine accuracy:   {correct_fine}/{total} = {fine_acc:.2f}%")


if __name__ == "__main__":
	run_evaluation()

