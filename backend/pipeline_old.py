#!/usr/bin/env python3
"""
This script is a direct, modular version of the original clust_nn_pipeline_sklearn.py.
It contains all original logic for training, clustering, prediction, and report generation,
structured to be imported and called by a web server like FastAPI.
"""
import os
import re
import itertools
import joblib
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from Bio import SeqIO
from tqdm import tqdm

# Scikit-learn imports
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Matplotlib configuration for server-side use (no GUI)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------
# Helpers from original script
# -------------------------
def ensure_fasta_headers(in_file, tmp_file="tmp_with_headers.fasta"):
    """Ensures FASTA file has headers; creates a temporary file if not."""
    # Check if the file is empty first
    if os.path.getsize(in_file) == 0:
        return in_file
    with open(in_file, "r") as f:
        first_line = f.readline()
    if first_line.startswith(">"):
        return in_file
    
    with open(in_file, "r") as f_in, open(tmp_file, "w") as f_out:
        for i, line in enumerate(f_in):
            line = line.strip()
            if line:
                f_out.write(f">seq{i+1}\n{line}\n")
    return tmp_file

def read_fasta_to_list(fasta_path):
    """Reads sequences and IDs from a FASTA file."""
    seqs, ids = [], []
    # Use a temporary file path within the same directory to avoid permission issues
    temp_fasta_path = Path(fasta_path).parent / (Path(fasta_path).name + ".tmp")
    processed_fasta_path = ensure_fasta_headers(fasta_path, str(temp_fasta_path))
    try:
        for rec in SeqIO.parse(processed_fasta_path, "fasta"):
            seq = str(rec.seq).upper().replace("U", "T")
            if seq:
                seqs.append(seq)
                ids.append(rec.id)
    finally:
        if processed_fasta_path != fasta_path and os.path.exists(processed_fasta_path):
            os.remove(processed_fasta_path)
    return ids, seqs

# -------------------------
# K-mer vectorizer from original script
# -------------------------
class KmerVectorizer:
    def __init__(self, k=4):
        self.k = int(k)
        self.vocab = [''.join(p) for p in itertools.product('ATGC', repeat=self.k)]
        self.vocab_size = len(self.vocab)

    def transform_sequence(self, seq):
        seq = seq.upper()
        counts = Counter()
        for i in range(len(seq) - self.k + 1):
            kmer = seq[i:i+self.k]
            if 'N' in kmer or any(ch not in 'ATGC' for ch in kmer):
                continue
            counts[kmer] += 1
        total = max(sum(counts.values()), 1)
        return np.array([counts[k]/total for k in self.vocab], dtype=np.float32)

    def transform(self, sequences):
        X = np.zeros((len(sequences), self.vocab_size), dtype=np.float32)
        for i, seq in enumerate(tqdm(sequences, desc="Vectorizing (k-mer)")):
            X[i,:] = self.transform_sequence(seq)
        return X

# -------------------------
# Dimensionality reduction from original script
# -------------------------
def make_reducer(X, n_components=50):
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    Xr = svd.fit_transform(X)
    return svd, Xr

# -------------------------
# Clustering from original script
# -------------------------
def cluster_dbscan(Xr, eps=0.5, min_samples=5):
    db = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean', n_jobs=-1)
    labels = db.fit_predict(Xr)
    return labels

def cluster_kmeans(Xr, n_clusters=10):
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    labels = km.fit_predict(Xr)
    return labels

# -------------------------
# Medoid selection from original script
# -------------------------
def cluster_medoid_indices(X, labels):
    medoids = {}
    unique = [l for l in set(labels) if l != -1]
    for lab in unique:
        idxs = np.where(labels == lab)[0]
        if len(idxs) == 0:
            continue
        
        # For single-point clusters, the point itself is the medoid
        if len(idxs) == 1:
            medoids[lab] = idxs[0]
            continue

        centroid = X[idxs].mean(axis=0)
        dists = np.linalg.norm(X[idxs] - centroid, axis=1)
        med_idx = idxs[np.argmin(dists)]
        medoids[lab] = med_idx
    return medoids

# -------------------------
# sklearn MLP training from original script
# -------------------------
def train_mlp_sklearn(X, y, epochs=50):
    mlp = MLPClassifier(hidden_layer_sizes=(512,256,128),
                        activation='relu',
                        solver='adam',
                        alpha=1e-4,
                        batch_size=64,
                        learning_rate_init=1e-3,
                        max_iter=epochs,
                        verbose=True,
                        random_state=42)
    mlp.fit(X, y)
    return mlp

# -------------------------
# Main Training Function (adapted from original script)
# -------------------------
def train_from_labeled_fasta(train_fasta, outdir, k=4, pca_comp=50, epochs=50, centroid_percentile=95.0):
    outdir_path = Path(outdir)
    outdir_path.mkdir(exist_ok=True, parents=True)
    
    ids, seqs = read_fasta_to_list(train_fasta)
    
    labels = []
    for header in ids:
        parts = header.split()
        lab = parts[0] if len(parts)>=1 else header
        lab = re.sub(r'[^A-Za-z0-9_]', '_', lab)
        labels.append(lab)
        
    vec = KmerVectorizer(k=k)
    X = vec.transform(seqs)
    
    reducer, Xr = make_reducer(X, n_components=min(pca_comp, X.shape[1]-1))
    
    scaler = StandardScaler()
    Xr_s = scaler.fit_transform(Xr)
    
    le = LabelEncoder()
    y = le.fit_transform(labels)
    
    counts = Counter(y)
    bad_classes = [lab for lab,c in counts.items() if c < 2]
    if bad_classes:
        print("Warning: following classes have <2 samples and will be removed:", [le.classes_[b] for b in bad_classes])
        keep_mask = np.isin(y, [lab for lab in counts if counts[lab]>=2])
        Xr_s = Xr_s[keep_mask]
        
        kept_labels = [labels[i] for i in range(len(labels)) if keep_mask[i]]
        le = LabelEncoder().fit(kept_labels)
        y = le.transform(kept_labels)

    if len(np.unique(y)) < 2:
        raise ValueError("Training requires at least two classes with at least two samples each after filtering.")

    stratify_param = y if min(Counter(y).values()) >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(Xr_s, y, test_size=0.2, random_state=42, stratify=stratify_param)
    
    mlp = train_mlp_sklearn(X_train, y_train, epochs=epochs)
    
    preds = mlp.predict(X_test)
    print("Classification report on test set:")
    print(classification_report(y_test, preds, zero_division=0, target_names=le.inverse_transform(np.unique(y_test))))
    
    # Save artifacts
    joblib.dump(vec, outdir_path / "kmer_vectorizer.joblib")
    joblib.dump(reducer, outdir_path / "reducer.joblib")
    joblib.dump(scaler, outdir_path / "scaler.joblib")
    joblib.dump(le, outdir_path / "label_encoder.joblib")
    joblib.dump(mlp, outdir_path / "mlp_model.joblib")
    
    centroids = {}
    thresholds = {}
    for class_idx, class_label in enumerate(le.classes_):
        mask = (y == class_idx)
        if mask.sum() == 0:
            continue
        Xc = Xr_s[mask]
        centroid = Xc.mean(axis=0)
        dists = np.linalg.norm(Xc - centroid, axis=1)
        centroids[class_label] = centroid
        thresholds[class_label] = float(np.percentile(dists, float(centroid_percentile)))
    joblib.dump({'centroids': centroids, 'thresholds': thresholds}, outdir_path / 'class_centroids.joblib')
    
    print(f"[{outdir_path.name}] Saved all model artifacts.")
    return str(outdir_path / "mlp_model.joblib")

# -------------------------
# Main Prediction Function (adapted from original script)
# -------------------------
def cluster_and_predict(raw_fasta, model_dir, outdir, k=4, pca_comp=50, 
                        cluster_method="kmeans", dbscan_eps=0.5, dbscan_min=5, 
                        kmeans_n=10, threshold=0.7):

    outdir_path = Path(outdir)
    outdir_path.mkdir(exist_ok=True, parents=True)
    model_dir_path = Path(model_dir)

    ids, seqs = read_fasta_to_list(raw_fasta)
    
    vec = joblib.load(model_dir_path / "kmer_vectorizer.joblib")
    X = vec.transform(seqs)
    
    reducer = joblib.load(model_dir_path / "reducer.joblib")
    Xr = reducer.transform(X)
    
    scaler = joblib.load(model_dir_path / "scaler.joblib")
    Xr_s = scaler.transform(Xr)
    
    if cluster_method.lower() == "dbscan":
        labels = cluster_dbscan(Xr_s, eps=dbscan_eps, min_samples=dbscan_min)
    else:
        n_clusters = min(kmeans_n, len(Xr_s))
        if n_clusters < 1: n_clusters = 1
        labels = cluster_kmeans(Xr_s, n_clusters=n_clusters)
        
    labels = np.array(labels)
    print("Clusters found:", len(set(labels)) - (1 if -1 in labels else 0), "plus noise (-1):", sum(labels==-1))
    
    medoids = cluster_medoid_indices(Xr_s, labels)
    
    mlp = joblib.load(model_dir_path / "mlp_model.joblib")
    le = joblib.load(model_dir_path / "label_encoder.joblib")
    
    centroids_path = model_dir_path / 'class_centroids.joblib'
    centroids_data = joblib.load(centroids_path) if centroids_path.exists() else None
    
    medoid_rows = []
    for cl, midx in medoids.items():
        x = X[midx].reshape(1, -1)
        xr = reducer.transform(x)
        xr_s = scaler.transform(xr)
        probs = mlp.predict_proba(xr_s)[0]
        
        top_idx = probs.argsort()[::-1][:3]
        top_probs = [(le.classes_[i], float(probs[i])) for i in top_idx]
        top_labels = [le.inverse_transform([int(c)])[0] for c,_ in top_probs]
        
        maxp = float(np.max(probs))
        pred_numeric = mlp.classes_[int(np.argmax(probs))]
        pred_label = le.inverse_transform([pred_numeric])[0] if maxp >= threshold else "Unknown"
        
        if pred_label != "Unknown" and centroids_data is not None:
            centroids = centroids_data.get('centroids', {})
            thresholds = centroids_data.get('thresholds', {})
            if pred_label in centroids:
                centroid = centroids[pred_label]
                dist = float(np.linalg.norm(xr_s.ravel() - centroid))
                if pred_label in thresholds and dist > thresholds[pred_label]:
                    pred_label = "Unknown"
                    maxp = 0.0
                    
        medoid_rows.append((cl, midx, ids[midx], pred_label, maxp, top_probs, top_labels))
        
    medoid_df = pd.DataFrame(medoid_rows, columns=["cluster","medoid_index","medoid_id","predicted_species","confidence","top_probs","top_labels"])
    medoid_df.to_csv(outdir_path / "cluster_medoid_predictions.csv", index=False)
    
    cluster_assignments = []
    for i, lab in enumerate(labels):
        assigned = medoid_df.loc[medoid_df['cluster'] == lab, 'predicted_species'].values
        assigned_conf = medoid_df.loc[medoid_df['cluster'] == lab, 'confidence'].values
        assigned_species = "Unknown" if len(assigned) == 0 else assigned[0]
        conf = 0.0 if len(assigned_conf) == 0 else assigned_conf[0]
        cluster_assignments.append((ids[i], lab, assigned_species, conf))
        
    assign_df = pd.DataFrame(cluster_assignments, columns=["sequence_id","cluster","predicted_species","confidence"])
    assign_df.to_csv(outdir_path / "sequence_cluster_assignments.csv", index=False)
    print("Saved sequence assignments.")

    # Abundance and plots
    try:
        total_reads = len(assign_df)
        species_counts = assign_df.groupby("predicted_species").size().reset_index(name="n_reads")
        species_counts["percentage"] = 100.0 * species_counts["n_reads"] / total_reads
        species_counts = species_counts.sort_values("n_reads", ascending=False).reset_index(drop=True)
        species_counts.to_csv(outdir_path / "species_abundance_by_reads.csv", index=False)
        print("Saved species_abundance_by_reads.csv")

        cluster_sizes = assign_df.groupby("cluster").size().reset_index(name="cluster_size")
        cluster_info = assign_df.merge(cluster_sizes, on="cluster").drop_duplicates(subset=["cluster"])
        cluster_counts = cluster_info.groupby("predicted_species")["cluster_size"].sum().reset_index(name="n_reads_via_clusters")
        cluster_counts["percentage_via_clusters"] = 100.0 * cluster_counts["n_reads_via_clusters"] / total_reads
        cluster_counts = cluster_counts.sort_values("n_reads_via_clusters", ascending=False)
        cluster_counts.to_csv(outdir_path / "species_abundance_by_clusters.csv", index=False)

        # Bar plot
        plt.figure(figsize=(8, max(4, 0.4 * min(20, len(species_counts)))))
        topN = min(20, len(species_counts))
        ax = sns.barplot(x="percentage", y="predicted_species", data=species_counts.head(topN), orient="h")
        plt.xlabel("Percentage of reads (%)")
        plt.ylabel("Predicted species")
        plt.title(f"Species abundance (top {topN}) by reads")
        for p in ax.patches:
            width = p.get_width()
            ax.text(width + 0.5, p.get_y() + p.get_height() / 2, f"{width:.1f}%", va='center')
        plt.tight_layout()
        plt.savefig(outdir_path / "species_abundance_bar.png")
        plt.close()
        print("Saved species abundance bar chart.")

        # Pie chart
        pie_df = species_counts.copy()
        if len(pie_df) > topN:
            top_df = pie_df.head(topN).copy()
            others_pct = pie_df['percentage'].iloc[topN:].sum()
            others_row = pd.DataFrame([{'predicted_species': 'Others', 'n_reads': 0, 'percentage': others_pct}])
            top_df = pd.concat([top_df, others_row], ignore_index=True)
        else:
            top_df = pie_df
        labels = top_df['predicted_species'].tolist()
        sizes = top_df['percentage'].tolist()
        fig, ax = plt.subplots(figsize=(6,6))
        ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, pctdistance=0.8)
        ax.axis('equal')
        plt.title('Species composition (by reads)')
        plt.savefig(outdir_path / 'species_composition_pie.png')
        plt.close()
        print('Saved species composition pie chart.')
    
        # Scatter plot
        fig, ax = plt.subplots(figsize=(8,6))
        unique = sorted(set(labels))
        palette = sns.color_palette("hls", n_colors=len(unique))
        for idx, lab in enumerate(unique):
            mask = labels == lab
            ax.scatter(Xr[mask,0], Xr[mask,1], s=10, color=palette[idx], label=str(lab))
        if len(unique) < 20: # Only show legend if not too cluttered
            ax.legend(bbox_to_anchor=(1.05,1), loc="upper left", fontsize='small')
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_title("Clustering scatter (first 2 reduced components)")
        plt.tight_layout()
        plt.savefig(outdir_path / "cluster_scatter.png")
        plt.close()
        print("Saved cluster scatter plot.")
    except Exception as e:
        print(f"Could not compute/save species abundance plots: {e}")

    return assign_df, medoid_df, labels

