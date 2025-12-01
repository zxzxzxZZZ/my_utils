


import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import (
    AutoProcessor, AutoTokenizer, AutoModel, TrainingArguments, Trainer
)
from PIL import Image
from collections import defaultdict
from tqdm import tqdm
import numpy as np
import argparse

# 设置可见 GPU（根据你的环境）
os.environ["CUDA_VISIBLE_DEVICES"] = "6,7"

# ===============================
# 预处理与评估函数
# ===============================
def _preprocess_text(text):
    text = text.lower().replace("“", '"').replace("”", '"')
    return text

def get_data(data_file):
    with open(data_file, "r", encoding="utf-8") as f:
        if data_file.endswith(".json"):
            data_list = json.load(f)
            lines = [json.dumps(item) for item in data_list]
        else:  # .jsonl
            lines = f.readlines()
    img2txt, txt2img = defaultdict(list), defaultdict(list)
    texts, images = [], []
    text_ids, image_ids = {}, {}
    for i, line in enumerate(lines):
        data = json.loads(line.strip())
        img = data["path"]
        cap = data["label"]
        images.append(img)
        image_ids[img] = i
        for j in range(len(cap)):
            cap[j] = _preprocess_text(cap[j])
            img2txt[img].append(cap[j])
            txt2img[cap[j]].append(img)
            texts.append(cap[j])
            text_ids[cap[j]] = len(texts) - 1

    img2txt_gt = np.zeros((len(images), len(texts)), dtype=np.float32)
    txt2img_gt = np.zeros((len(texts), len(images)), dtype=np.float32)

    for i, img in enumerate(images):
        for txt in img2txt[img]:
            img2txt_gt[i][text_ids[txt]] = 1.0
    for i, txt in enumerate(texts):
        for img in txt2img[txt]:
            txt2img_gt[i][image_ids[img]] = 1.0

    return texts, images, txt2img_gt, img2txt_gt


def extract_feats(model, processor, texts, images, device="cuda", batch_size=16, root_dir=""):
    # 文本特征
    txt_feats = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Encoding texts"):
        batch_texts = texts[i:i + batch_size]
        inputs = processor(
            text=batch_texts,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=64
        ).to(device)
        with torch.no_grad():
            feats = model.text_model(**inputs).pooler_output
            feats = F.normalize(feats, dim=-1)
        txt_feats.append(feats.cpu().numpy())
    txt_feats = np.vstack(txt_feats)

    # 图像特征
    img_feats = []
    for i in tqdm(range(0, len(images), batch_size), desc="Encoding images"):
        batch_imgs = []
        for p in images[i:i + batch_size]:
            full_path = os.path.join(root_dir, p) if root_dir else p
            try:
                img = Image.open(full_path).convert("RGB")
                batch_imgs.append(img)
            except Exception as e:
                print(f"⚠️ 跳过损坏图像 {full_path}: {e}")
                batch_imgs.append(Image.new("RGB", (224, 224)))
        inputs = processor(images=batch_imgs, return_tensors="pt").to(device)
        with torch.no_grad():
            feats = model.vision_model(**inputs).pooler_output
            feats = F.normalize(feats, dim=-1)
        img_feats.append(feats.cpu().numpy())
    img_feats = np.vstack(img_feats)

    return txt_feats, img_feats


def calu_recall(txt_feats, img_feats, txt2img_gt, img2txt_gt):
    t2i_mat = txt_feats @ img_feats.T
    t2i_idx = np.argsort(-t2i_mat, axis=1)
    i2t_mat = img_feats @ txt_feats.T
    i2t_idx = np.argsort(-i2t_mat, axis=1)

    def topk_calc(gt, idx):
        pred = np.zeros((len(idx), 10))
        for i in range(len(idx)):
            for j in range(10):
                if gt[i][idx[i][j]] == 1:
                    pred[i][j] = 1
        pred = np.cumsum(pred, axis=1)
        topk = [0] * 10
        for i in range(len(pred)):
            for j in range(10):
                if pred[i][j] > 0:
                    topk[j] += 1
        return np.asarray(topk) / len(pred)

    t2i_topk = topk_calc(txt2img_gt, t2i_idx)
    i2t_topk = topk_calc(img2txt_gt, i2t_idx)

    results = {
        "I2T_R@1": float(round(i2t_topk[0], 4)),
        "I2T_R@5": float(round(i2t_topk[4], 4)),
        "I2T_R@10": float(round(i2t_topk[9], 4)),
        "T2I_R@1": float(round(t2i_topk[0], 4)),
        "T2I_R@5": float(round(t2i_topk[4], 4)),
        "T2I_R@10": float(round(t2i_topk[9], 4)),
    }
    return results


# =======================================
# Dataset
# =======================================
class ImageTextDataset(Dataset):
    def __init__(self, data, processor, tokenizer=None, max_length=128, transform=None, root_dir=None):
        self.samples = []
        self.processor = processor
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.transform = transform
        self.root_dir = root_dir

        unique_group_ids = sorted(set(item['path'] for item in data))
        self.group_id_to_int = {gid: i for i, gid in enumerate(unique_group_ids)}

        for item in data:
            img_path = item['path']
            labels = item['label']
            group_int = self.group_id_to_int[img_path]
            for label in labels:
                self.samples.append({
                    'path': img_path,
                    'label': label,
                    'group_id': group_int
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        img_path = os.path.join(self.root_dir, item['path'])
        label = item['label']
        group_id = item['group_id']

        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        text_tokenized = self.processor(
            text=label,
            images=image,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )

        if self.tokenizer:
            gte_tokenized = self.tokenizer(
                label,
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt'
            )
            gte_input_ids = gte_tokenized.input_ids.squeeze(0)
            gte_attention_mask = gte_tokenized.attention_mask.squeeze(0)
        else:
            gte_input_ids = torch.tensor([])
            gte_attention_mask = torch.tensor([])

        return {
            "pixel_values": text_tokenized.pixel_values.squeeze(0),
            "input_ids": text_tokenized.input_ids.squeeze(0),
            "gte_input_ids": gte_input_ids,
            "gte_attention_mask": gte_attention_mask,
            "group_id": group_id
        }


def collate_fn(batch):
    pixel_values = torch.stack([x["pixel_values"] for x in batch])
    input_ids = torch.stack([x["input_ids"] for x in batch])
    gte_input_ids = torch.stack([x["gte_input_ids"] for x in batch])
    gte_attention_mask = torch.stack([x["gte_attention_mask"] for x in batch])
    group_ids = torch.tensor([x["group_id"] for x in batch], dtype=torch.long)
    return {
        "pixel_values": pixel_values,
        "input_ids": input_ids,
        "gte_input_ids": gte_input_ids,
        "gte_attention_mask": gte_attention_mask,
        "group_ids": group_ids
    }


# =======================================
# Loss Function
# =======================================
def multi_positive_contrastive_loss(image_embeds, text_embeds, gte_text_embeds, group_ids, temperature=0.07):
    device = image_embeds.device
    B = image_embeds.size(0)

    if not isinstance(group_ids, torch.Tensor):
        group_ids_tensor = torch.tensor(group_ids, device=device, dtype=torch.long)
    else:
        group_ids_tensor = group_ids.to(device)

    if group_ids_tensor.size(0) != B:
        group_ids_tensor = group_ids_tensor[:B]

    pos_mask = (group_ids_tensor.unsqueeze(1) == group_ids_tensor.unsqueeze(0)).float()
    neg_mask = 1.0 - pos_mask
    # print("pos_mask.shape, neg_mask.shape:", pos_mask.shape, neg_mask.shape)
    num_pos = pos_mask.sum(dim=1).clamp(min=1.0)
    num_neg = neg_mask.sum(dim=1).clamp(min=1.0) /25
    # print(num_pos, num_neg)

    logits_per_image = torch.matmul(image_embeds, text_embeds.t()) / temperature
    # print("logits_per_image.shape:", logits_per_image.shape)
    with torch.no_grad():
        gte_sim = torch.matmul(gte_text_embeds, gte_text_embeds.t())
        gte_sim_clamped = torch.clamp(gte_sim, min=-1.0, max=1.0)

    # positive_log_probs = (F.logsigmoid(logits_per_image) * pos_mask) / num_pos.unsqueeze(1)
    positive_log_probs = (F.logsigmoid(logits_per_image) * pos_mask)
    # print("positive_log_probs.shape:", positive_log_probs.shape)
    # print("positive_log_probs:", positive_log_probs)

    positive_log_probs = positive_log_probs.sum(dim=1)
    # print("positive_log_probs.shape:", positive_log_probs.shape)
    # print("positive_log_probs:", positive_log_probs)
    adjusted_neg_logits = logits_per_image - (1.0 - gte_sim_clamped) * neg_mask
    # negative_log_probs = (F.logsigmoid(-adjusted_neg_logits) * neg_mask) / num_neg.unsqueeze(1)
    negative_log_probs = (F.logsigmoid(-adjusted_neg_logits) * neg_mask)
    # print("negative_log_probs.shape:", negative_log_probs.shape)
    # print("negative_log_probs:", negative_log_probs)
    negative_log_probs = negative_log_probs.sum(dim=1)
    # print("negative_log_probs.shape:", negative_log_probs.shape)
    # print("negative_log_probs:", negative_log_probs)


    loss = - (positive_log_probs + negative_log_probs).mean()


    pos_loss = - positive_log_probs.mean()
    neg_loss = - negative_log_probs.mean()
    total_loss = pos_loss + neg_loss

    # print(f"pos_loss: {pos_loss.item():.6f}")
    # print(f"neg_loss: {neg_loss.item():.6f}")
    # print(f"total_loss: {total_loss.item():.6f}")

    print("正样本贡献比例:", pos_loss.item() / total_loss.item())
    print("负样本贡献比例:", neg_loss.item() / total_loss.item())
        # print(loss)
    return loss


# =======================================
# Model
# =======================================
class SigLIPTrainerModel(nn.Module):
    def __init__(self, vision_model_name, gte_model_name):
        super().__init__()
        self.vision_model = AutoModel.from_pretrained(vision_model_name).vision_model
        self.text_model = AutoModel.from_pretrained(vision_model_name).text_model
        self.gte_model = AutoModel.from_pretrained(gte_model_name, trust_remote_code=True, local_files_only=True)
        for p in self.gte_model.parameters():
            p.requires_grad = False
        self.logit_scale = nn.Parameter(torch.ones([]) * 1 / 0.07)

    def forward(self, pixel_values, input_ids, gte_input_ids, group_ids, gte_attention_mask=None, return_loss=False):
        vision_outputs = self.vision_model(pixel_values=pixel_values)
        text_outputs = self.text_model(input_ids=input_ids)

        image_embeds = F.normalize(vision_outputs.pooler_output, dim=-1)
        text_embeds = F.normalize(text_outputs.pooler_output, dim=-1)

        logits_per_image = self.logit_scale.exp() * torch.matmul(image_embeds, text_embeds.t())

        loss = None
        if return_loss:
            with torch.no_grad():
                gte_outputs = self.gte_model(input_ids=gte_input_ids, attention_mask=gte_attention_mask)
                gte_text_embeds = F.normalize(gte_outputs.last_hidden_state[:, 0], dim=-1)

            loss = multi_positive_contrastive_loss(
                image_embeds=image_embeds,
                text_embeds=text_embeds,
                gte_text_embeds=gte_text_embeds,
                group_ids=group_ids,
                temperature=0.07
            )

        return {"loss": loss, "logits": logits_per_image}


# =======================================
# Custom Trainer
# =======================================
class SigLIPHuggingFaceTrainer(Trainer):
    def __init__(self, *args, eval_data_file=None, root_dir="", **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_data_file = eval_data_file
        self.eval_root_dir = root_dir

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(
            pixel_values=inputs["pixel_values"],
            input_ids=inputs["input_ids"],
            gte_input_ids=inputs["gte_input_ids"],
            gte_attention_mask=inputs["gte_attention_mask"],
            group_ids=inputs["group_ids"],
            return_loss=True
        )
        loss = outputs["loss"]
        return (loss, outputs) if return_outputs else loss

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        assert self.eval_data_file is not None, "请传入 eval_data_file"

        device = self.args.device
        model = self.model
        model.eval()

        texts, images, txt2img_gt, img2txt_gt = get_data(self.eval_data_file)
        processor = self.tokenizer
        batch_size = self.args.per_device_eval_batch_size or 16

        txt_feats, img_feats = extract_feats(
            model, processor, texts, images,
            device=device,
            batch_size=batch_size,
            root_dir=self.eval_root_dir
        )

        results = calu_recall(txt_feats, img_feats, txt2img_gt, img2txt_gt)

        metrics = {
            f"{metric_key_prefix}_I2T_R@1": results["I2T_R@1"],
            f"{metric_key_prefix}_I2T_R@5": results["I2T_R@5"],
            f"{metric_key_prefix}_I2T_R@10": results["I2T_R@10"],
            f"{metric_key_prefix}_T2I_R@1": results["T2I_R@1"],
            f"{metric_key_prefix}_T2I_R@5": results["T2I_R@5"],
            f"{metric_key_prefix}_T2I_R@10": results["T2I_R@10"],
        }

        self.log(metrics)
        return metrics


# =======================================
# Dummy Dataset (绕过 HF 检查)
# =======================================
class DummyDataset(Dataset):
    def __init__(self, max_length=64):
        self.max_length = max_length
    def __len__(self):
        return 1
    def __getitem__(self, idx):
        return {
            "pixel_values": torch.zeros(3, 224, 224),
            "input_ids": torch.zeros(self.max_length, dtype=torch.long),
            "gte_input_ids": torch.zeros(self.max_length, dtype=torch.long),
            "gte_attention_mask": torch.zeros(self.max_length, dtype=torch.long),
            "group_id": 0
        }


# =======================================
# Main
# =======================================
def main(args):
    with open(args.train_json_path, 'r', encoding='utf-8') as f:
        train_list = json.load(f)

    processor = AutoProcessor.from_pretrained(args.vision_model)
    tokenizer = AutoTokenizer.from_pretrained(args.gte_model, trust_remote_code=True, local_files_only=True)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    train_dataset = ImageTextDataset(
        train_list, processor, tokenizer, transform=transform,
        max_length=args.max_length, root_dir=args.root_dir
    )

    dummy_eval_dataset = DummyDataset(max_length=args.max_length)
    model = SigLIPTrainerModel(args.vision_model, args.gte_model)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        logging_steps=50,
        save_strategy="steps",
        save_steps=500,
        eval_strategy="steps",
        eval_steps=500,
        per_device_eval_batch_size=64,
        save_total_limit=3,
        fp16=True,
        remove_unused_columns=False,
        report_to="none",
        dataloader_drop_last=True
    )

    trainer = SigLIPHuggingFaceTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dummy_eval_dataset,
        tokenizer=processor,
        data_collator=collate_fn,
        eval_data_file=args.val_json_path,
        root_dir=args.root_dir
    )

    trainer.train()

    final_metrics = trainer.evaluate()
    print("\n✅ Final Evaluation Results:")
    for k, v in final_metrics.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vision_model", type=str, default="/home/zx/workspace/Siglip/models/ckpt_52_1812_eval_loss_1.377303")
    parser.add_argument("--gte_model", type=str, default="/home/zx/workspace/Siglip/models/gte")
    parser.add_argument("--train_json_path", type=str, default="data/f30k-en_zh_train.json")
    parser.add_argument("--val_json_path", type=str, default="data/f30k-cn_test.json")
    parser.add_argument("--root_dir", type=str, default="")
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--max_length", type=int, default=64)
    args = parser.parse_args()
    main(args)
