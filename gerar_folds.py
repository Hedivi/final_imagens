import os
import tarfile
import re
from collections import defaultdict, Counter
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

# === CONFIGURAÇÕES ===
tar_path = "lung_blocks.tar.gz"
extract_path = "lung_blocks_extracted"
output_path = "folds_stratified_groupk"
num_folds = 5
os.makedirs(output_path, exist_ok=True)

# === 1. EXTRAÇÃO DO ARQUIVO ===
if not os.path.exists(extract_path):
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(extract_path)

base_dir = os.path.join(extract_path, "lung_blocks")

# === 2. COLETAR INFORMAÇÕES DOS ARQUIVOS ===
pattern = re.compile(r"_(\\d)\\.dcm$")
patient_data = defaultdict(list)
total_files = 0

print("🔍 Lendo arquivos DICOM por paciente...")

for patient_id in os.listdir(base_dir):
    patient_path = os.path.join(base_dir, patient_id)
    if os.path.isdir(patient_path):
        dcm_found = 0
        for fname in os.listdir(patient_path):
            if fname.endswith(".dcm"):
                match = re.search(r"_(\d)\.dcm$", fname)
                if match:
                    label = int(match.group(1))
                    filepath = os.path.join(patient_path, fname)
                    patient_data[patient_id].append((filepath, label))
                    dcm_found += 1
        print(f"Paciente {patient_id} → {dcm_found} arquivos")

print(f"Total de pacientes válidos: {len(patient_data)}")

# === 3. PREPARAR LISTAS PARA StratifiedGroupKFold ===
X, y, groups = [], [], []
for patient_id, samples in patient_data.items():
    if samples:
        labels = [lbl for _, lbl in samples]
        most_common_label = Counter(labels).most_common(1)[0][0]
        X.append(patient_id)  # Placeholder
        y.append(most_common_label)
        groups.append(patient_id)

print(f"Pacientes com dados válidos para split: {len(groups)}")

# === 4. GERAR OS FOLDS ===
sgkf = StratifiedGroupKFold(n_splits=num_folds, shuffle=True, random_state=42)
folds = list(sgkf.split(X=X, y=y, groups=groups))

# === 5. SALVAR CSVs DE CADA FOLD ===
for fold_idx, (train_idx, val_idx) in enumerate(folds):
    train_patients = [groups[i] for i in train_idx]
    val_patients = [groups[i] for i in val_idx]

    for split_name, patient_list in [("train", train_patients), ("val", val_patients)]:
        rows = []
        for pid in patient_list:
            for filepath, label in patient_data[pid]:
                rows.append({
                    "patient_id": pid,
                    "filepath": filepath,
                    "class": label
                })
        df = pd.DataFrame(rows)
        csv_path = os.path.join(output_path, f"fold{fold_idx+1}_{split_name}.csv")
        df.to_csv(csv_path, index=False)

print(f"Folds gerados com sucesso em: {output_path}")
