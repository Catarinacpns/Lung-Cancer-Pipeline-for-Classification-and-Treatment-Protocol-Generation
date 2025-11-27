# STANDARD LIBRARIES
import os
import csv
import gc


# PYTORCH
import torch
import torch.nn as nn
import torch.optim as optim

# TORCHVISION (Backbone likely used inside MultiModalResNet)
from torchvision import models

# DATA & EVALUATION UTILITIES
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)

import pandas as pd
from tqdm import tqdm

# CUSTOM PROJECT MODULES
from src.modeling.resnet50.datasets.lung_cancer_multimodal import (
    create_dataloader
)

# Reproducibility utilities
from optimization.utils.reproducibility import seed_worker

# Your model architecture
from models.resnet50.multimodal_resnet50 import MultiModalResNet


def train_ResNet50_optuna_hyperparams(best_params, train_dataset, val_dataset, model_save_path, metrics_file, weights_T, weights_N, weights_M):
    torch.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

    # Hyperparameters
    lr = best_params['learning_rate']
    batch_size = best_params['batch_size']
    weight_decay = best_params['weight_decay']
    trainable_layers = best_params['trainable_layers']
    dropout_rate = best_params['dropout_rate']
    dropout_rate_RN50 = best_params['dropout_rate_RN50']  
    hidden_units = best_params['hidden_units']            
    num_fc_layers = best_params['num_fc_layers']          
    optimizer_choice = best_params['optimizer']
    scheduler_type = best_params['scheduler']
    label_smoothing = best_params['label_smoothing']

    momentum = best_params.get('momentum')
    step_size = best_params.get('step_size')
    gamma = best_params.get('gamma')
    factor = best_params.get('factor')
    patience = best_params.get('patience')
    T_max = best_params.get('T_max')
    eta_min = best_params.get('eta_min')

    train_loader = create_dataloader(train_dataset, batch_size, worker_init_fn=seed_worker)
    val_loader = create_dataloader(val_dataset, batch_size, worker_init_fn=seed_worker)

    # Modelo
    #model = MultiModalResNet(num_classes_T=4, num_classes_N=4, num_classes_M=2, metadata_dim=5,  dropout_rate=dropout_rate).to(device)
    model = MultiModalResNet(
        num_classes_T=4, num_classes_N=4, num_classes_M=2,
        metadata_dim=5,
        dropout_rate=dropout_rate,
        dropout_rate_RN50=dropout_rate_RN50,
        trainable_layers=trainable_layers,
        hidden_units=hidden_units,
        num_fc_layers=num_fc_layers
    ).to(device)

    for param in model.backbone.parameters():
        param.requires_grad = False
    for layer in list(model.backbone.children())[-trainable_layers:]:
        for param in layer.parameters():
            param.requires_grad = True

    # Loss
    criterion_T = torch.nn.CrossEntropyLoss(weight=weights_T, label_smoothing=label_smoothing)
    criterion_N = torch.nn.CrossEntropyLoss(weight=weights_N, label_smoothing=label_smoothing)
    criterion_M = torch.nn.CrossEntropyLoss(weight=weights_M, label_smoothing=label_smoothing)

    # Otimizador
    if optimizer_choice == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_choice == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif optimizer_choice == 'RMSprop':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Scheduler
    if scheduler_type == 'StepLR':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif scheduler_type == 'ReduceLROnPlateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=factor, patience=patience)
    elif scheduler_type == 'CosineAnnealingLR':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)

    # Métricas
    os.makedirs(os.path.dirname(metrics_file), exist_ok=True)
    with open(metrics_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            "Epoch", "Train Loss", "Val Loss",
            "Train Acc", "Val Acc",
            "Train Precision", "Val Precision",
            "Train Recall", "Val Recall",
            "Train F1", "Val F1",
            # Métricas por alvo - Treino
            "Train_T_Acc", "Train_T_Prec", "Train_T_Rec", "Train_T_F1",
            "Train_N_Acc", "Train_N_Prec", "Train_N_Rec", "Train_N_F1",
            "Train_M_Acc", "Train_M_Prec", "Train_M_Rec", "Train_M_F1",
            # Métricas por alvo - Validação
            "Val_T_Acc", "Val_T_Prec", "Val_T_Rec", "Val_T_F1",
            "Val_N_Acc", "Val_N_Prec", "Val_N_Rec", "Val_N_F1",
            "Val_M_Acc", "Val_M_Prec", "Val_M_Rec", "Val_M_F1"
        ])


    best_val_f1 = 0
    epochs_without_improvement = 0
    max_patience = 30

    for epoch in range(200):
        print(f"\nEpoch {epoch+1}")
        model.train()
        train_loss = 0

        preds_train_T, preds_train_N, preds_train_M = [], [], []
        labels_train_T, labels_train_N, labels_train_M = [], [], []

        for images, metadata, targets in tqdm(train_loader, desc="Train"):
            images, metadata = images.to(device), metadata.to(device)
            t_true, n_true, m_true = targets['T'].to(device), targets['N'].to(device), targets['M'].to(device)

            optimizer.zero_grad()
            outputs = model(images, metadata)

            loss_t = criterion_T(outputs['T'], t_true)
            loss_n = criterion_N(outputs['N'], n_true)
            loss_m = criterion_M(outputs['M'], m_true)

            loss = loss_t + loss_n + loss_m
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)

            # Preds + labels
            preds_train_T.extend(outputs['T'].argmax(1).cpu().numpy())
            preds_train_N.extend(outputs['N'].argmax(1).cpu().numpy())
            preds_train_M.extend(outputs['M'].argmax(1).cpu().numpy())

            labels_train_T.extend(t_true.cpu().numpy())
            labels_train_N.extend(n_true.cpu().numpy())
            labels_train_M.extend(m_true.cpu().numpy())


        # Val
        model.eval()
        val_loss = 0
        preds_val, labels_val = [], []
        preds_T, preds_N, preds_M = [], [], []
        labels_T, labels_N, labels_M = [], [], []

        with torch.no_grad():
            for images, metadata, targets in tqdm(val_loader, desc="Val"):
                images, metadata = images.to(device), metadata.to(device)
                t_true, n_true, m_true = targets['T'].to(device), targets['N'].to(device), targets['M'].to(device)

                outputs = model(images, metadata)

                loss_t = criterion_T(outputs['T'], t_true)
                loss_n = criterion_N(outputs['N'], n_true)
                loss_m = criterion_M(outputs['M'], m_true)

                loss = loss_t + loss_n + loss_m
                val_loss += loss.item() * images.size(0)

                preds_val.extend(outputs['T'].argmax(1).cpu().numpy())
                labels_val.extend(t_true.cpu().numpy())

                preds_T.extend(outputs['T'].argmax(1).cpu().numpy())
                preds_N.extend(outputs['N'].argmax(1).cpu().numpy())
                preds_M.extend(outputs['M'].argmax(1).cpu().numpy())

                labels_T.extend(t_true.cpu().numpy())
                labels_N.extend(n_true.cpu().numpy())
                labels_M.extend(m_true.cpu().numpy())

        # Métricas
        acc_T = accuracy_score(labels_T, preds_T)
        acc_N = accuracy_score(labels_N, preds_N)
        acc_M = accuracy_score(labels_M, preds_M)

        precision_T, recall_T, f1_T, _ = precision_recall_fscore_support(labels_T, preds_T, average='macro')
        precision_N, recall_N, f1_N, _ = precision_recall_fscore_support(labels_N, preds_N, average='macro')
        precision_M, recall_M, f1_M, _ = precision_recall_fscore_support(labels_M, preds_M, average='macro')

        val_f1_mean = (f1_T + f1_N + f1_M) / 3
        val_precision_mean = (precision_T + precision_N + precision_M) / 3
        val_recall_mean = (recall_T + recall_N + recall_M) / 3
        val_acc_mean = (acc_T + acc_N + acc_M) / 3        
        
        precision_train_T, recall_train_T, f1_train_T, _ = precision_recall_fscore_support(labels_train_T, preds_train_T, average='macro')
        precision_train_N, recall_train_N, f1_train_N, _ = precision_recall_fscore_support(labels_train_N, preds_train_N, average='macro')
        precision_train_M, recall_train_M, f1_train_M, _ = precision_recall_fscore_support(labels_train_M, preds_train_M, average='macro')

        train_f1_mean = (f1_train_T + f1_train_N + f1_train_M) / 3
        train_precision_mean = (precision_train_T + precision_train_N + precision_train_M) / 3
        train_recall_mean = (recall_train_T + recall_train_N + recall_train_M) / 3
        train_acc_mean = (
            accuracy_score(labels_train_T, preds_train_T) +
            accuracy_score(labels_train_N, preds_train_N) +
            accuracy_score(labels_train_M, preds_train_M)
        ) / 3

        print(f"Epoch {epoch+1}:")
        print(f"  Validation Metrics:")
        print(f"    T: Acc={acc_T:.4f}, Prec={precision_T:.4f}, Rec={recall_T:.4f}, F1={f1_T:.4f}")
        print(f"    N: Acc={acc_N:.4f}, Prec={precision_N:.4f}, Rec={recall_N:.4f}, F1={f1_N:.4f}")
        print(f"    M: Acc={acc_M:.4f}, Prec={precision_M:.4f}, Rec={recall_M:.4f}, F1={f1_M:.4f}")
        print(f"  Train F1 Mean: {train_f1_mean:.4f} | Val F1 Mean: {val_f1_mean:.4f}")
        print(f"Train Loss: {train_loss/len(train_loader.dataset):.4f} | Val Loss: {val_loss/len(val_loader.dataset):.4f} | Val F1: {val_f1_mean:.4f}")

        with open(metrics_file, mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([
                epoch + 1,
                train_loss / len(train_loader.dataset),
                val_loss / len(val_loader.dataset),
                train_acc_mean, val_acc_mean,
                train_precision_mean, val_precision_mean,
                train_recall_mean, val_recall_mean,
                train_f1_mean, val_f1_mean,
                acc_T, precision_T, recall_T, f1_T,
                acc_N, precision_N, recall_N, f1_N,
                acc_M, precision_M, recall_M, f1_M
            ])

        if scheduler_type == 'ReduceLROnPlateau':
            scheduler.step(val_f1_mean)
        else:
            scheduler.step()

        if val_f1_mean > best_val_f1:
            best_val_f1 = val_f1_mean
            epochs_without_improvement = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "hyperparameters": best_params,
                "class_weights": {"T": weights_T.tolist(), "N": weights_N.tolist(), "M": weights_M.tolist()}
            }, model_save_path)
            print(f"Saved new best model at {model_save_path}")
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= max_patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

        gc.collect()

    # ===== End of training - saving final results ===== #
    results_dir = os.path.dirname(model_save_path)
    def save_confusion_matrix_and_report(y_true, y_pred, label):
        cm = confusion_matrix(y_true, y_pred)
        report = classification_report(y_true, y_pred, output_dict=True)

        cm_df = pd.DataFrame(cm)
        report_df = pd.DataFrame(report).transpose()

        cm_df.to_csv(os.path.join(results_dir, f"{label}_confusion_matrix.csv"), index=False)
        report_df.to_csv(os.path.join(results_dir, f"{label}_classification_report.csv"))

    save_confusion_matrix_and_report(labels_T, preds_T, "T")
    save_confusion_matrix_and_report(labels_N, preds_N, "N")
    save_confusion_matrix_and_report(labels_M, preds_M, "M")

    print("\nConfusion matricx and classification report - saved.")