
# Fine-tunes a RoBERTa transformer for phishing text detection
# Handles class imbalance using WEIGHTED LOSS (no SMOTE)
# Adds phrase-level context (2–4 word n-grams)
# Keeps social media context (emojis, hashtags)
# =========================================================

# ---------------------------------------------------------
# 1️⃣ INSTALL DEPENDENCIES
# ---------------------------------------------------------
!pip install emoji transformers torch scikit-learn pandas numpy tqdm -q

import re, emoji, numpy as np, pandas as pd, torch, warnings
from torch.utils.data import Dataset, DataLoader
from torch.nn import CrossEntropyLoss
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import precision_recall_fscore_support, classification_report, confusion_matrix, roc_auc_score
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ---------------------------------------------------------
# 2️⃣ CONFIGURATION
# ---------------------------------------------------------
MODEL_NAME = "roberta-base"
CSV_PATH = "/content/Dataset_5971.csv"   # Upload to Colab Files or generate in /content
MAX_LEN = 128
BATCH_SIZE = 16
EPOCHS = 4
LR = 2e-5
RANDOM_SEED = 42
USE_WEIGHTED_LOSS = True   # we still use weighted loss to handle imbalance
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

print("🚀 Device:", DEVICE)
!nvidia-smi

# ---------------------------------------------------------
# 3️⃣ UPLOAD YOUR DATASET (if not already in /content)
# ---------------------------------------------------------
from google.colab import files
print("📂 If you haven't uploaded social_posts.csv yet, upload it now (or skip if file exists).")
uploaded = files.upload()  # run once, then you can comment out when file is present

# ---------------------------------------------------------
# 4️⃣ PREPROCESSING
# ---------------------------------------------------------
def preprocess_text(text):
    text = str(text).strip()
    text = re.sub(r"https?://\S+", " <URL> ", text)
    text = re.sub(r"@\w+", " <USER> ", text)
    text = emoji.replace_emoji(text, replace=lambda e, _: f" {e} ")
    return re.sub(r"\s+", " ", text).strip()

print("📄 Loading dataset...")
df = pd.read_csv(CSV_PATH)
df = df.dropna(subset=["text", "label"])
df["text"] = df["text"].apply(preprocess_text)
df["label"] = df["label"].astype(int)

print("✅ Loaded", len(df), "rows")
print(df["label"].value_counts())

# ---------------------------------------------------------
# 5️⃣ PHRASE EXTRACTION (TF-IDF n-grams)
# ---------------------------------------------------------
def extract_phishing_phrases(texts, labels, top_n=200):
    print("\n📊 Extracting 2–4 word phishing phrases...")
    vec = TfidfVectorizer(ngram_range=(2, 4), min_df=3, max_df=0.8, token_pattern=r'\b\w+\b', lowercase=True)
    X = vec.fit_transform(texts)
    labels = np.array(labels)
    phishing_mean = X[labels==1].mean(axis=0).A1
    normal_mean = X[labels==0].mean(axis=0).A1
    diff = phishing_mean - normal_mean
    top_idx = np.argsort(diff)[-top_n:]
    feats = vec.get_feature_names_out()
    phrases = {feats[i]: float(diff[i]) for i in top_idx if diff[i] > 0}
    maxv = max(phrases.values()) if phrases else 1.0
    phrases = {k: v/maxv for k,v in phrases.items()}
    print("✅ Extracted", len(phrases), "phrases")
    return phrases

def phrase_score(text, phrases):
    return min(sum(w for p,w in phrases.items() if p in text.lower()), 1.0)

train_df, test_df = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=RANDOM_SEED)
phrases = extract_phishing_phrases(train_df["text"], train_df["label"])

train_scores = [phrase_score(t, phrases) for t in train_df["text"]]
test_scores  = [phrase_score(t, phrases) for t in test_df["text"]]

# ---------------------------------------------------------
# 6️⃣ TOKENIZATION
# ---------------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_with_context(texts, scores):
    enhanced = [f"[SCORE:{s:.2f}] {t}" for t,s in zip(texts, scores)]
    return tokenizer(enhanced, truncation=True, padding="max_length", max_length=MAX_LEN, return_tensors="pt")

train_enc = tokenize_with_context(train_df["text"], train_scores)
test_enc  = tokenize_with_context(test_df["text"], test_scores)

train_labels = train_df["label"].values
test_labels  = test_df["label"].values

# ---------------------------------------------------------
# 7️⃣ PREPARE DATALOADERS (NO SMOTE)
# ---------------------------------------------------------
train_ids, train_masks = train_enc["input_ids"], train_enc["attention_mask"]
test_ids, test_masks = test_enc["input_ids"], test_enc["attention_mask"]

class PhishDataset(Dataset):
    def __init__(self, ids, masks, labels):
        self.ids, self.masks, self.labels = ids, masks, labels
    def __len__(self): return len(self.labels)
    def __getitem__(self, i):
        return {"input_ids": self.ids[i], "attention_mask": self.masks[i],
                "labels": torch.tensor(self.labels[i], dtype=torch.long)}

train_loader = DataLoader(PhishDataset(train_ids, train_masks, train_labels), batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(PhishDataset(test_ids, test_masks, test_labels), batch_size=BATCH_SIZE)

print(f"✅ Final training set size: {len(train_loader.dataset)} samples")

# ---------------------------------------------------------
# 8️⃣ MODEL + TRAINING SETUP
# ---------------------------------------------------------
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2).to(DEVICE)
opt = AdamW(model.parameters(), lr=LR)
total_steps = len(train_loader)*EPOCHS
sched = get_linear_schedule_with_warmup(opt, int(0.1*total_steps), total_steps)

if USE_WEIGHTED_LOSS:
    unique, counts = np.unique(train_labels, return_counts=True)
    # If a label is missing in training (unlikely), handle safely
    if len(counts) < 2:
        counts = np.append(counts, 1)
    w = torch.tensor([len(train_labels)/(2*counts[0]), len(train_labels)/(2*counts[1])], dtype=torch.float32).to(DEVICE)
    loss_fn = CrossEntropyLoss(weight=w)
    print(f"⚖️ Using weighted loss: class weights = {w.cpu().numpy()}")
else:
    loss_fn = CrossEntropyLoss()

# ---------------------------------------------------------
# 9️⃣ TRAINING + EVAL FUNCTIONS
# ---------------------------------------------------------
def train_epoch():
    model.train(); total_loss=correct=total=0
    for b in tqdm(train_loader, desc="Training", ncols=100):
        opt.zero_grad()
        ids = b["input_ids"].to(DEVICE)
        msk = b["attention_mask"].to(DEVICE)
        lbl = b["labels"].to(DEVICE)
        out = model(input_ids=ids, attention_mask=msk)
        loss = loss_fn(out.logits, lbl)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        total_loss += loss.item()
        pred = torch.argmax(out.logits,1)
        correct += (pred==lbl).sum().item()
        total += lbl.size(0)
    return total_loss/len(train_loader), correct/total

def evaluate():
    model.eval(); preds, labels, probs = [], [], []
    with torch.no_grad():
        for b in tqdm(test_loader, desc="Evaluating", ncols=100):
            ids = b["input_ids"].to(DEVICE)
            msk = b["attention_mask"].to(DEVICE)
            lbl = b["labels"].to(DEVICE)
            out = model(input_ids=ids, attention_mask=msk)
            p = torch.softmax(out.logits,1)
            preds.extend(torch.argmax(p,1).cpu().numpy())
            labels.extend(lbl.cpu().numpy())
            probs.extend(p[:,1].cpu().numpy())
    pr, rc, f1, _ = precision_recall_fscore_support(labels, preds, average="binary", zero_division=0)
    try:
        auc = roc_auc_score(labels, probs)
    except:
        auc = 0.0
    print(classification_report(labels, preds, target_names=["Normal","Phishing"], digits=4))
    print("Confusion Matrix:\n", confusion_matrix(labels,preds))
    return pr, rc, f1, auc

# ---------------------------------------------------------
# 🔟 TRAINING LOOP
# ---------------------------------------------------------
best_f1 = 0.0
for ep in range(EPOCHS):
    print(f"\n📍 Epoch {ep+1}/{EPOCHS}")
    loss, acc = train_epoch()
    pr, rc, f1, auc = evaluate()
    print(f"Loss={loss:.4f}  Acc={acc:.4f}  F1={f1:.4f}  AUC={auc:.4f}")
    if f1 > best_f1:
        best_f1 = f1
        model.save_pretrained("/content/best_roberta_phishing")
        tokenizer.save_pretrained("/content/best_roberta_phishing")
        print("✅ Saved new best model!")

print(f"\n🎉 Training complete! Best F1 = {best_f1:.4f}")
