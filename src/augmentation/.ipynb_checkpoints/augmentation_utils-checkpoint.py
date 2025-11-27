import os
import re
import random
from glob import glob
from collections import defaultdict
import cv2
import numpy as np
import pandas as pd

# Albumentations augmentations
from albumentations import (
    Compose,
    HorizontalFlip,
    RandomBrightnessContrast,
    Affine,
    GaussianBlur,
    CLAHE,
    RandomGamma
)

def remove_majority_combination_images(df, image_dir, t_major='0', n_major='3', m_major='0', target_to_remove=100):
    """
    Remove imagens da combinação T0-N3-M0 após augmentação, priorizando imagens aumentadas e pacientes com mais imagens.

    Args:
        df (pd.DataFrame): DataFrame com info dos pacientes.
        image_dir (str): Diretório onde estão as imagens (e.g., .../images/train).
        t_major (str): Classe majoritária de T-Stage.
        n_major (str): Classe majoritária de N-Stage.
        m_major (str): Classe majoritária de M-Stage.
        target_to_remove (int): Nº total de imagens a remover.
    """
    print(f"\n Initiating removal for majority class combo: T={t_major}, N={n_major}, M={m_major}")
    removed_count = 0

    # Mapear imagens por paciente
    image_paths = glob(os.path.join(image_dir, "*.jpg"))
    patient_to_images = defaultdict(list)
    for img_path in image_paths:
        fname = os.path.basename(img_path)
        pid = extract_patient_id(fname)
        if pid:
            patient_to_images[pid].append(img_path)

    # Identificar pacientes da classe majoritária combinada
    majority_patients = df[
        (df['T-Stage'].astype(str) == t_major) &
        (df['N-Stage'].astype(str) == n_major) &
        (df['M-Stage'].astype(str) == m_major)
    ]['PatientID'].tolist()

    print(f"  Found {len(majority_patients)} patients with majority combo")

    # Ordenar por número de imagens (priorizar quem tem mais)
    sorted_patients = sorted(
        majority_patients,
        key=lambda pid: len(patient_to_images.get(pid, [])),
        reverse=True
    )

    for pid in sorted_patients:
        imgs = patient_to_images.get(pid, [])
        if not imgs:
            continue

        # Priorizar imagens aumentadas
        aug_imgs = [img for img in imgs if '_aug_' in os.path.basename(img)]
        orig_imgs = [img for img in imgs if '_aug_' not in os.path.basename(img)]

        for img_path in aug_imgs + orig_imgs:
            if removed_count >= target_to_remove:
                break
            os.remove(img_path)
            removed_count += 1

        if removed_count >= target_to_remove:
            break

    print(f" Removed {removed_count} images from majority class combination.")

def augment_specific_class_combination(
    input_dir,
    df,
    target_class=('1', '3', '0'),
    num_aug_images=100
):
    augmentation = Compose([
        HorizontalFlip(p=0.3),
        RandomBrightnessContrast(p=0.4),
        Affine(translate_percent=(0.05, 0.1), scale=(0.85, 1.2), rotate=(-15, 15), shear=(-5, 5), p=0.7),
        GaussianBlur(blur_limit=(3, 7), p=0.3),
        CLAHE(clip_limit=(1.0, 5.0), p=0.4),
        RandomGamma(p=0.3),
    ])

    df = df[df['PatientID'].notna()].copy()
    for col in ['T-Stage', 'N-Stage', 'M-Stage']:
        df[col] = df[col].astype(str)
    df['class_tuple'] = list(zip(df['T-Stage'], df['N-Stage'], df['M-Stage']))

    df_class = df[df['class_tuple'] == target_class]
    if df_class.empty:
        print(f" No patients found for class {target_class}")
        return

    image_paths = glob(os.path.join(input_dir, '*.jpg'))
    patient_to_imgs = defaultdict(list)

    for path in image_paths:
        pid = extract_patient_id(os.path.basename(path))
        if pid in df_class['PatientID'].values:
            patient_to_imgs[pid].append(path)

    patient_aug_count = defaultdict(int)
    all_augmented = 0

    print(f"\n Augmenting {num_aug_images} images for class {target_class} in {input_dir}...\n")

    while all_augmented < num_aug_images:
        eligible = sorted(patient_to_imgs.items(), key=lambda kv: len(kv[1]) + patient_aug_count[kv[0]])
        augmented_this_round = False

        for pid, imgs in eligible:
            if all_augmented >= num_aug_images:
                break

            if not imgs:
                continue

            img_path = random.choice(imgs)
            img = cv2.imread(img_path)
            if img is None:
                continue

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            aug_img = augmentation(image=img)['image']
            aug_img = cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR)

            base = os.path.splitext(os.path.basename(img_path))[0]
            aug_name = f"{base}_aug_{random.randint(1000, 9999)}.jpg"
            aug_path = os.path.join(input_dir, aug_name)

            cv2.imwrite(aug_path, aug_img)
            patient_to_imgs[pid].append(aug_path)
            patient_aug_count[pid] += 1
            all_augmented += 1
            augmented_this_round = True
            print(f"  [+] Augmented for patient {pid} ({all_augmented}/{num_aug_images})")
            break  # apenas 1 por iteração p/ equilíbrio

        if not augmented_this_round:
            print(" No more augmentable patients/images found.")
            break

    print("\n Augmentation completed.")
    print(f" Final augmented counts per patient (class {target_class}):")
    for pid, count in sorted(patient_aug_count.items(), key=lambda x: -x[1]):
        print(f"  {pid}: {count} augmented")

        
        
def extract_patient_id(filename):
    name = os.path.splitext(os.path.basename(filename))[0]
    match = re.match(r'^([ABEG]\d{3,4})_', name)
    if match:
        return match.group(1)
    match = re.match(r'^[EG]_(LUNG\d+-\d+)_', name)
    if match:
        return match.group(1)
    return None