# STANDARD LIBRARIES
import os
import json
import gc


# NUMPY / PANDAS
import numpy as np
import pandas as pd


# PLOTTING
import matplotlib.pyplot as plt
import seaborn as sns


# MACHINE LEARNING METRICS
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)

# PYTORCH
import torch
import torch.nn as nn
import torch.optim as optim


# PROGRESS BAR
from tqdm import tqdm

# OPTUNA (HYPERPARAMETER OPTIMIZATION)
import optuna

# CUSTOM MODULES (Your Project)
from models.resnet50.multimodal_resnet50 import MultiModalResNet
from src.modeling.resnet50.datasets.lung_cancer_multimodal import create_dataloader


def compute_class_weights_tensor(labels, expected_num_classes, device):
    unique_classes = np.unique(labels)
    weights_present = compute_class_weight(class_weight='balanced', classes=unique_classes, y=labels)
    full_weights = np.zeros(expected_num_classes, dtype=np.float32)
    
    for idx, cls in enumerate(unique_classes):
        full_weights[int(cls)] = weights_present[idx]

    return torch.tensor(full_weights, dtype=torch.float).to(device)

def save_cm_and_report(y_true, y_pred, labels, target_name, trial_number, results_dir):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    report = classification_report(y_true, y_pred, labels=labels, output_dict=True)

    report_path = os.path.join(results_dir, f"trial_{trial_number + 1}_{target_name}_report.json")
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=4)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix - {target_name}")
    cm_path = os.path.join(results_dir, f"trial_{trial_number + 1}_{target_name}_confusion_matrix.png")
    plt.savefig(cm_path)
    plt.close()

    return report, cm

def objective(trial):
    gc.collect()
    torch.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    learning_rate = trial.suggest_float('learning_rate', 1e-6, 5e-4, log=True)
    batch_size = trial.suggest_categorical('batch_size', [16, 32])
    weight_decay = trial.suggest_float('weight_decay', 1e-4, 1e-1, log=True)
    trainable_layers = trial.suggest_int('trainable_layers', 1, 4)
    dropout_rate = trial.suggest_float('dropout_rate', 0.3, 0.9)
    dropout_rate_RN50 = trial.suggest_float('dropout_rate_RN50', 0.1, 0.5)
    optimizer_choice = trial.suggest_categorical('optimizer', ['Adam', 'SGD', 'RMSprop'])
    scheduler_type = trial.suggest_categorical('scheduler', ['StepLR', 'CosineAnnealingLR', 'ReduceLROnPlateau', 'OneCycleLR'])
    label_smoothing = trial.suggest_float('label_smoothing', 0.1, 0.3)
    hidden_units = trial.suggest_categorical('hidden_units', [64, 128, 256])
    num_fc_layers = trial.suggest_int('num_fc_layers', 1, 3)

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')

    model = MultiModalResNet(
        num_classes_T=4, num_classes_N=4, num_classes_M=2,
        metadata_dim=5,
        dropout_rate=dropout_rate,
        dropout_rate_RN50=dropout_rate_RN50,
        trainable_layers=trainable_layers,
        hidden_units=hidden_units,
        num_fc_layers=num_fc_layers
    ).to(device)

    # Otimizador
    if optimizer_choice == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_choice == 'SGD':
        momentum = trial.suggest_float('momentum', 0.7, 0.99)
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    else:
        optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    train_loader = create_dataloader(train_dataset, batch_size)
    val_loader = create_dataloader(val_dataset, batch_size)
    
    # Scheduler
    if scheduler_type == 'StepLR':
        step_size = trial.suggest_int('step_size', 1, 6)
        gamma = trial.suggest_float('gamma', 0.1, 0.9)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif scheduler_type == 'CosineAnnealingLR':
        T_max = trial.suggest_int('T_max', 5, 15)
        eta_min = trial.suggest_float('eta_min', 1e-6, 1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)
    elif scheduler_type == 'ReduceLROnPlateau':
        factor = trial.suggest_float('factor', 0.1, 0.5)
        patience = trial.suggest_int('patience', 2, 6)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=factor, patience=patience)
    elif scheduler_type == 'OneCycleLR':
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=learning_rate, steps_per_epoch=len(train_loader), epochs=40
        )

    t_all = [sample[2]['T'].item() for sample in train_dataset]
    n_all = [sample[2]['N'].item() for sample in train_dataset]
    m_all = [sample[2]['M'].item() for sample in train_dataset]

    weights_T = compute_class_weights_tensor(t_all, 4, device)
    weights_N = compute_class_weights_tensor(n_all, 4, device)
    weights_M = compute_class_weights_tensor(m_all, 2, device)

    criterion_T = nn.CrossEntropyLoss(weight=weights_T, label_smoothing=label_smoothing)
    criterion_N = nn.CrossEntropyLoss(weight=weights_N, label_smoothing=label_smoothing)
    criterion_M = nn.CrossEntropyLoss(weight=weights_M, label_smoothing=label_smoothing)

    trial_results = {"train_metrics": [], "val_metrics": [], "hyperparameters": trial.params,
                     "class_weights": {"T": weights_T.tolist(), "N": weights_N.tolist(), "M": weights_M.tolist()}}

    val_f1_scores = []
    
    train_f1_buffer = []
    early_stopping_patience = 8
    best_val_f1 = -np.inf
    epochs_without_improvement = 0


    for epoch in range(40):
        model.train()
        total_loss = 0


        # Inicializar listas para treino
        preds_train_T, preds_train_N, preds_train_M = [], [], []
        labels_train_T, labels_train_N, labels_train_M = [], [], []

        with tqdm(train_loader, desc=f"[Trial {trial.number}] Training Epoch {epoch+1}", leave=False) as pbar:
            for images, metadata, targets in pbar:
                images, metadata = images.to(device), metadata.to(device)
                t_true = targets['T'].to(device)
                n_true = targets['N'].to(device)
                m_true = targets['M'].to(device)

                optimizer.zero_grad()
                output = model(images, metadata)

                loss_t = criterion_T(output['T'], t_true)
                loss_n = criterion_N(output['N'], n_true)
                loss_m = criterion_M(output['M'], m_true)
                loss = loss_t + loss_n + loss_m
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

                # Armazenar previsões
                preds_train_T.extend(torch.argmax(output['T'], 1).cpu().numpy())
                preds_train_N.extend(torch.argmax(output['N'], 1).cpu().numpy())
                preds_train_M.extend(torch.argmax(output['M'], 1).cpu().numpy())

                labels_train_T.extend(t_true.cpu().numpy())
                labels_train_N.extend(n_true.cpu().numpy())
                labels_train_M.extend(m_true.cpu().numpy())

        model.eval()
        preds_T, preds_N, preds_M = [], [], []
        labels_T, labels_N, labels_M = [], [], []

        with torch.no_grad():
            for images, metadata, targets in tqdm(val_loader, desc=f"[Trial {trial.number}] Validation Epoch {epoch+1}", leave=False):
                images, metadata = images.to(device), metadata.to(device)
                t_true = targets['T'].to(device)
                n_true = targets['N'].to(device)
                m_true = targets['M'].to(device)

                outputs = model(images, metadata)
                preds_T.extend(torch.argmax(outputs['T'], 1).cpu().numpy())
                preds_N.extend(torch.argmax(outputs['N'], 1).cpu().numpy())
                preds_M.extend(torch.argmax(outputs['M'], 1).cpu().numpy())

                labels_T.extend(t_true.cpu().numpy())
                labels_N.extend(n_true.cpu().numpy())
                labels_M.extend(m_true.cpu().numpy())

        # Val metrics
        acc_T = accuracy_score(labels_T, preds_T)
        acc_N = accuracy_score(labels_N, preds_N)
        acc_M = accuracy_score(labels_M, preds_M)

        precision_T, recall_T, f1_T, _ = precision_recall_fscore_support(labels_T, preds_T, average='macro')
        precision_N, recall_N, f1_N, _ = precision_recall_fscore_support(labels_N, preds_N, average='macro')
        precision_M, recall_M, f1_M, _ = precision_recall_fscore_support(labels_M, preds_M, average='macro')

        mean_f1 = (f1_T + f1_N + f1_M) / 3
        val_f1_scores.append(mean_f1)

        if scheduler_type == 'ReduceLROnPlateau':
            scheduler.step(mean_f1)
        else:
            scheduler.step() 
            
        trial.report(mean_f1, epoch)
        
        if trial.should_prune():
            print(f"[⚡] Trial {trial.number} pruned at epoch {epoch+1} (Mean F1={mean_f1:.4f})")
            raise optuna.TrialPruned()
            
        if mean_f1 > best_val_f1:
            best_val_f1 = mean_f1
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            print(f"  [⏸] No improvement in val F1 for {epochs_without_improvement} epoch(s)")

        if epochs_without_improvement >= early_stopping_patience:
            print(f"[🏁] Early stopping triggered after {early_stopping_patience} epochs without improvement (Trial {trial.number})")
            break
            
        # Training metrics
        acc_train_T = accuracy_score(labels_train_T, preds_train_T)
        acc_train_N = accuracy_score(labels_train_N, preds_train_N)
        acc_train_M = accuracy_score(labels_train_M, preds_train_M)

        prec_train_T, rec_train_T, f1_train_T, _ = precision_recall_fscore_support(labels_train_T, preds_train_T, average='macro')
        prec_train_N, rec_train_N, f1_train_N, _ = precision_recall_fscore_support(labels_train_N, preds_train_N, average='macro')
        prec_train_M, rec_train_M, f1_train_M, _ = precision_recall_fscore_support(labels_train_M, preds_train_M, average='macro')

        mean_train_f1 = (f1_train_T + f1_train_N + f1_train_M) / 3
        
        # Guarda o F1 no buffer
        train_f1_buffer.append(mean_train_f1)
        if len(train_f1_buffer) > 3:
            train_f1_buffer.pop(0)

        # Verifica condição de early stopping baseado no treino
        if len(train_f1_buffer) == 3 and all(f1 >= 0.99 for f1 in train_f1_buffer):
            print(f"[🏁] Early stopping: Train F1 ≥ 0.99 for 3 consecutive epochs (Trial {trial.number})")
            raise optuna.TrialPruned()

        print(f"Epoch {epoch+1}:")
        print(f"  Training Metrics:")
        print(f"    T: Acc={acc_train_T:.4f}, Prec={prec_train_T:.4f}, Rec={rec_train_T:.4f}, F1={f1_train_T:.4f}")
        print(f"    N: Acc={acc_train_N:.4f}, Prec={prec_train_N:.4f}, Rec={rec_train_N:.4f}, F1={f1_train_N:.4f}")
        print(f"    M: Acc={acc_train_M:.4f}, Prec={prec_train_M:.4f}, Rec={rec_train_M:.4f}, F1={f1_train_M:.4f}")
        print(f"    Mean Train F1 Score: {mean_train_f1:.4f}")

        print(f"  Training Loss: {total_loss / len(train_loader.dataset):.4f}")
        print(f"  Validation Metrics:")
        print(f"    T: Acc={acc_T:.4f}, Prec={precision_T:.4f}, Rec={recall_T:.4f}, F1={f1_T:.4f}")
        print(f"    N: Acc={acc_N:.4f}, Prec={precision_N:.4f}, Rec={recall_N:.4f}, F1={f1_N:.4f}")
        print(f"    M: Acc={acc_M:.4f}, Prec={precision_M:.4f}, Rec={recall_M:.4f}, F1={f1_M:.4f}")
        print(f"    Mean F1 Score: {mean_f1:.4f}")

    report_T, _ = save_cm_and_report(labels_T, preds_T, list(range(4)), "T", trial.number, RESULTS_DIR)
    report_N, _ = save_cm_and_report(labels_N, preds_N, list(range(4)), "N", trial.number, RESULTS_DIR)
    report_M, _ = save_cm_and_report(labels_M, preds_M, list(range(2)), "M", trial.number, RESULTS_DIR)

    trial_results["classification_reports"] = {"T": report_T, "N": report_N, "M": report_M}
    trial_results["final_metrics"] = {"val_f1_T": f1_T, "val_f1_N": f1_N, "val_f1_M": f1_M, "val_f1_mean": mean_f1}

    trial_file = os.path.join(RESULTS_DIR, f"trial_{trial.number + 1}.json")
    with open(trial_file, 'w') as f:
        json.dump(trial_results, f, indent=4)

    gc.collect()
    return mean_f1

def progress_callback(study, trial):
    # Filtra apenas os trials que não foram pruned e têm .value definido
    valid_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None]

    if not valid_trials:
        print("There's still no trial concluded to display results.")
        return

    # Usa trial com maior valor de F1
    best_trial = max(valid_trials, key=lambda t: t.value)

    print("\nBest Trial So Far:")
    print(f"  Best Trial Number : {best_trial.number + 1}")
    print(f"  Best F1-Score     : {best_trial.value:.4f}")
    print("  Best Hyperparameters:")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")
    
    print('_______________________________________________________________________________________________________')
