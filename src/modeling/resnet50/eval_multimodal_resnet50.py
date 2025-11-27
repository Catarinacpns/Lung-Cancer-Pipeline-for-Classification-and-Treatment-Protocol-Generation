import os
import csv
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
from torch.utils.data import DataLoader
import torch.nn as nn


def evaluate_multimodal_resnet50(checkpoint_path, test_dataset, batch_size=32, save_path='test_metrics.csv'):
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    best_params = checkpoint['hyperparameters']
    print(best_params)

    trainable_layers = best_params['trainable_layers']
    dropout_rate = best_params['dropout_rate']
    dropout_rate_RN50 = best_params['dropout_rate_RN50']  
    hidden_units = best_params['hidden_units']            
    num_fc_layers = best_params['num_fc_layers']          

    model = MultiModalResNet(
        num_classes_T=4, num_classes_N=4, num_classes_M=2,
        metadata_dim=5,
        dropout_rate=dropout_rate,
        dropout_rate_RN50=dropout_rate_RN50,
        trainable_layers=trainable_layers,
        hidden_units=hidden_units,
        num_fc_layers=num_fc_layers
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    weights_T = torch.tensor(checkpoint['class_weights']['T']).to(device)
    weights_N = torch.tensor(checkpoint['class_weights']['N']).to(device)
    weights_M = torch.tensor(checkpoint['class_weights']['M']).to(device)

    criterion_T = nn.CrossEntropyLoss(weight=weights_T)
    criterion_N = nn.CrossEntropyLoss(weight=weights_N)
    criterion_M = nn.CrossEntropyLoss(weight=weights_M)

    total_loss = 0
    preds_T, preds_N, preds_M = [], [], []
    labels_T, labels_N, labels_M = [], [], []
    correct_all_targets = []
    image_paths = list(test_dataset.valid_paths)

    with torch.no_grad():
        idx_global = 0
        for images, metadata, targets in tqdm(test_loader, desc="Evaluating Test Set", unit="batch"):
            images, metadata = images.to(device), metadata.to(device)
            t_true, n_true, m_true = targets['T'].to(device), targets['N'].to(device), targets['M'].to(device)

            outputs = model(images, metadata)

            t_pred = outputs['T'].argmax(1)
            n_pred = outputs['N'].argmax(1)
            m_pred = outputs['M'].argmax(1)

            for i in range(len(images)):
                if (
                    t_pred[i].item() == t_true[i].item() and
                    n_pred[i].item() == n_true[i].item() and
                    m_pred[i].item() == m_true[i].item()
                ):
                    correct_all_targets.append(image_paths[idx_global + i])

            preds_T.extend(t_pred.cpu().numpy())
            preds_N.extend(n_pred.cpu().numpy())
            preds_M.extend(m_pred.cpu().numpy())

            labels_T.extend(t_true.cpu().numpy())
            labels_N.extend(n_true.cpu().numpy())
            labels_M.extend(m_true.cpu().numpy())

            loss_t = criterion_T(outputs['T'], t_true)
            loss_n = criterion_N(outputs['N'], n_true)
            loss_m = criterion_M(outputs['M'], m_true)

            loss = loss_t + loss_n + loss_m
            total_loss += loss.item() * images.size(0)

            idx_global += len(images)

    test_loss = total_loss / len(test_loader.dataset)

    def compute_metrics(y_true, y_pred):
        acc = accuracy_score(y_true, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')
        return acc, prec, rec, f1

    acc_T, prec_T, rec_T, f1_T = compute_metrics(labels_T, preds_T)
    acc_N, prec_N, rec_N, f1_N = compute_metrics(labels_N, preds_N)
    acc_M, prec_M, rec_M, f1_M = compute_metrics(labels_M, preds_M)

    mean_f1 = (f1_T + f1_N + f1_M) / 3

    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Target T - Acc: {acc_T:.4f}, Prec: {prec_T:.4f}, Rec: {rec_T:.4f}, F1: {f1_T:.4f}")
    print(f"Target N - Acc: {acc_N:.4f}, Prec: {prec_N:.4f}, Rec: {rec_N:.4f}, F1: {f1_N:.4f}")
    print(f"Target M - Acc: {acc_M:.4f}, Prec: {prec_M:.4f}, Rec: {rec_M:.4f}, F1: {f1_M:.4f}")
    print(f"Mean F1: {mean_f1:.4f}")

    output_dir = os.path.dirname(checkpoint_path)
    os.makedirs(output_dir, exist_ok=True)  # Garante que o diretório existe

    metrics_file = os.path.join(output_dir, save_path)

    with open(metrics_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Target", "Acc", "Prec", "Rec", "F1"])
        writer.writerow(["T", acc_T, prec_T, rec_T, f1_T])
        writer.writerow(["N", acc_N, prec_N, rec_N, f1_N])
        writer.writerow(["M", acc_M, prec_M, rec_M, f1_M])
        writer.writerow(["MEAN", "-", "-", "-", mean_f1])

    print(f"\nMetrics saved to {metrics_file}")

    # Save correct predictions
    correct_path = os.path.join(os.path.dirname(checkpoint_path), "correct_all_targets.csv")
    pd.DataFrame({"Correctly Classified Image": correct_all_targets}).to_csv(correct_path, index=False)
    print(f"\n {len(correct_all_targets)} imagens classificadas corretamente em T, N e M.")
    print(f"Lista salva em: {correct_path}")

    print("\n____________Classification Report____________\n")
    print("\nClassification Report T:\n")
    print(classification_report(labels_T, preds_T))

    print("\nClassification Report N:\n")
    print(classification_report(labels_N, preds_N))

    print("\nClassification Report M:\n")
    print(classification_report(labels_M, preds_M))

    print("\n____________Confusion Matrix____________\n")
    print("\nConfusion Matrix T:\n")
    print(confusion_matrix(labels_T, preds_T))

    print("\nConfusion Matrix N:\n")
    print(confusion_matrix(labels_N, preds_N))

    print("\nConfusion Matrix M:\n")
    print(confusion_matrix(labels_M, preds_M))
