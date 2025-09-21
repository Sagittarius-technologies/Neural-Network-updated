#!/usr/bin/env python3
"""
Corrected pipeline.py

This script is a modular version of the original clust_nn_pipeline_sklearn.py.
It contains all original logic for training, clustering, prediction, and report generation,
structured to be imported and called by a web server like FastAPI.

Fixed issues:
- Corrected function signatures to match main.py calls
- Added proper error handling
- Fixed parameter mismatches
- Added validation functions
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
# Validation Functions
# -------------------------
def validate_fasta_file(fasta_path):
    """Validate FASTA file exists and has content."""
    path = Path(fasta_path)
    if not path.exists():
        raise FileNotFoundError(f"FASTA file not found: {fasta_path}")
    
    if path.stat().st_size == 0:
        raise ValueError(f"FASTA file is empty: {fasta_path}")
    
    return True

def validate_sequences(sequences):
    """Validate that sequences contain valid nucleotides."""
    if not sequences:
        raise ValueError("No sequences found in FASTA file")
    
    valid_chars = set('ATGCNRYSWKMBDHVNU')  # Include ambiguous nucleotides
    for i, seq in enumerate(sequences):
        if not seq:
            raise ValueError(f"Empty sequence found at position {i}")
        
        invalid_chars = set(seq.upper()) - valid_chars
        if invalid_chars:
            print(f"Warning: Sequence {i} contains unusual characters: {invalid_chars}")
    
    return True

# -------------------------
# Helpers from original script (with improvements)
# -------------------------
def ensure_fasta_headers(in_file, tmp_file=None):
    """Ensures FASTA file has headers; creates a temporary file if not."""
    in_path = Path(in_file)
    
    # Generate temporary file name if not provided
    if tmp_file is None:
        tmp_file = in_path.parent / f"tmp_with_headers_{in_path.stem}.fasta"
    
    # Check if the file is empty first
    if in_path.stat().st_size == 0:
        return str(in_path)
    
    try:
        with open(in_path, "r") as f:
            first_line = f.readline().strip()
        
        # If already has headers, return original file
        if first_line.startswith(">"):
            return str(in_path)
        
        # Create file with headers
        with open(in_path, "r") as f_in, open(tmp_file, "w") as f_out:
            for i, line in enumerate(f_in):
                line = line.strip()
                if line:
                    f_out.write(f">seq{i+1}\n{line}\n")
        
        return str(tmp_file)
    
    except Exception as e:
        raise RuntimeError(f"Error processing FASTA headers: {e}")

def read_fasta_to_list(fasta_path):
    """Reads sequences and IDs from a FASTA file."""
    validate_fasta_file(fasta_path)
    
    seqs, ids = [], []
    temp_fasta_path = None
    
    try:
        # Ensure FASTA has proper headers
        processed_fasta_path = ensure_fasta_headers(fasta_path)
        if processed_fasta_path != fasta_path:
            temp_fasta_path = processed_fasta_path
        
        # Read sequences
        for rec in SeqIO.parse(processed_fasta_path, "fasta"):
            seq = str(rec.seq).upper().replace("U", "T")  # Convert RNA to DNA
            if seq:
                seqs.append(seq)
                ids.append(rec.id)
        
        if not seqs:
            raise ValueError("No valid sequences found in FASTA file")
        
        # Validate sequences
        validate_sequences(seqs)
        
        return ids, seqs
    
    finally:
        # Clean up temporary file
        if temp_fasta_path and Path(temp_fasta_path).exists():
            try:
                os.remove(temp_fasta_path)
            except:
                pass

# -------------------------
# K-mer vectorizer from original script (with improvements)
# -------------------------
class KmerVectorizer:
    """K-mer vectorizer for DNA sequences."""
    
    def __init__(self, k=4):
        if k < 1 or k > 10:
            raise ValueError("K-mer size must be between 1 and 10")
        
        self.k = int(k)
        self.vocab = [''.join(p) for p in itertools.product('ATGC', repeat=self.k)]
        self.vocab_size = len(self.vocab)
        print(f"Initialized K-mer vectorizer with k={self.k}, vocabulary size={self.vocab_size}")

    def transform_sequence(self, seq):
        """Transform a single sequence to k-mer frequency vector."""
        seq = seq.upper()
        counts = Counter()
        
        # Count k-mers
        for i in range(len(seq) - self.k + 1):
            kmer = seq[i:i+self.k]
            # Skip k-mers with ambiguous nucleotides
            if 'N' in kmer or any(ch not in 'ATGC' for ch in kmer):
                continue
            counts[kmer] += 1
        
        # Normalize by total count
        total = max(sum(counts.values()), 1)
        return np.array([counts[k]/total for k in self.vocab], dtype=np.float32)

    def transform(self, sequences):
        """Transform multiple sequences to k-mer frequency matrix."""
        if not sequences:
            raise ValueError("No sequences provided for transformation")
        
        X = np.zeros((len(sequences), self.vocab_size), dtype=np.float32)
        
        for i, seq in enumerate(tqdm(sequences, desc="Vectorizing k-mers")):
            if not seq:
                print(f"Warning: Empty sequence at index {i}")
                continue
            X[i,:] = self.transform_sequence(seq)
        
        print(f"Vectorization complete: {X.shape[0]} sequences, {X.shape[1]} features")
        return X

# -------------------------
# Dimensionality reduction from original script
# -------------------------
def make_reducer(X, n_components=50):
    """Create and fit TruncatedSVD reducer."""
    if X.shape[0] < n_components:
        n_components = X.shape[0] - 1
        print(f"Adjusted PCA components to {n_components} due to small sample size")
    
    if X.shape[1] < n_components:
        n_components = X.shape[1] - 1
        print(f"Adjusted PCA components to {n_components} due to feature dimensionality")
    
    n_components = max(1, n_components)  # Ensure at least 1 component
    
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    Xr = svd.fit_transform(X)
    
    explained_var = svd.explained_variance_ratio_.sum()
    print(f"Dimensionality reduction: {X.shape[1]} -> {n_components} components "
          f"(explained variance: {explained_var:.3f})")
    
    return svd, Xr

# -------------------------
# Clustering from original script (with improvements)
# -------------------------
def cluster_dbscan(Xr, eps=0.5, min_samples=5):
    """Perform DBSCAN clustering."""
    print(f"Running DBSCAN clustering (eps={eps}, min_samples={min_samples})")
    
    db = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean', n_jobs=-1)
    labels = db.fit_predict(Xr)
    
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = sum(labels == -1)
    
    print(f"DBSCAN results: {n_clusters} clusters, {n_noise} noise points")
    
    return labels

def cluster_kmeans(Xr, n_clusters=10):
    """Perform K-means clustering."""
    # Adjust n_clusters if it exceeds the number of samples
    max_clusters = min(n_clusters, Xr.shape[0])
    if max_clusters != n_clusters:
        print(f"Adjusted K-means clusters from {n_clusters} to {max_clusters}")
        n_clusters = max_clusters
    
    print(f"Running K-means clustering with {n_clusters} clusters")
    
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    labels = km.fit_predict(Xr)
    
    print(f"K-means results: {len(set(labels))} clusters")
    
    return labels

# -------------------------
# Medoid selection from original script (with improvements)
# -------------------------
def cluster_medoid_indices(X, labels):
    """Find medoid (most central point) for each cluster."""
    medoids = {}
    unique_labels = [l for l in set(labels) if l != -1]
    
    print(f"Finding medoids for {len(unique_labels)} clusters")
    
    for lab in unique_labels:
        idxs = np.where(labels == lab)[0]
        if len(idxs) == 0:
            continue
        
        # For single-point clusters, the point itself is the medoid
        if len(idxs) == 1:
            medoids[lab] = idxs[0]
            continue

        # Find centroid and closest point to it
        cluster_points = X[idxs]
        centroid = cluster_points.mean(axis=0)
        dists = np.linalg.norm(cluster_points - centroid, axis=1)
        med_idx = idxs[np.argmin(dists)]
        medoids[lab] = med_idx
    
    print(f"Found {len(medoids)} medoids")
    return medoids

# -------------------------
# sklearn MLP training from original script (with improvements)
# -------------------------
def train_mlp_sklearn(X, y, epochs=50):
    """Train MLP classifier."""
    print(f"Training MLP with {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Classes: {len(np.unique(y))}, Epochs: {epochs}")
    
    mlp = MLPClassifier(
        hidden_layer_sizes=(512, 256, 128),
        activation='relu',
        solver='adam',
        alpha=1e-4,
        batch_size=min(64, X.shape[0]//2),  # Adjust batch size for small datasets
        learning_rate_init=1e-3,
        max_iter=epochs,
        verbose=True,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=10
    )
    
    mlp.fit(X, y)
    
    print(f"MLP training completed. Final loss: {mlp.loss_:.4f}")
    
    return mlp

# -------------------------
# Main Training Function (corrected and improved)
# -------------------------
def train_from_labeled_fasta(train_fasta, outdir, k=4, pca_comp=50, epochs=50, centroid_percentile=95.0):
    """
    Train model from labeled FASTA file.
    
    Args:
        train_fasta: Path to labeled FASTA file
        outdir: Output directory for model artifacts
        k: K-mer size (default 4)
        pca_comp: Number of PCA components (default 50)
        epochs: Training epochs (default 50)
        centroid_percentile: Percentile for distance thresholds (default 95.0)
    
    Returns:
        Path to saved MLP model
    """
    print(f"Starting training from {train_fasta}")
    print(f"Parameters: k={k}, pca_comp={pca_comp}, epochs={epochs}, centroid_percentile={centroid_percentile}")
    
    # Create output directory
    outdir_path = Path(outdir)
    outdir_path.mkdir(exist_ok=True, parents=True)
    
    # Read and validate input
    ids, seqs = read_fasta_to_list(train_fasta)
    print(f"Loaded {len(seqs)} sequences")
    
    # Extract labels from FASTA headers
    labels = []
    for header in ids:
        parts = header.split()
        lab = parts[0] if len(parts) >= 1 else header
        # Clean label to be filesystem safe
        lab = re.sub(r'[^A-Za-z0-9_]', '_', lab)
        labels.append(lab)
    
    print(f"Extracted labels: {len(set(labels))} unique classes")
    
    # K-mer vectorization
    vec = KmerVectorizer(k=k)
    X = vec.transform(seqs)
    
    # Check for zero variance features
    feature_var = np.var(X, axis=0)
    zero_var_features = np.sum(feature_var == 0)
    if zero_var_features > 0:
        print(f"Warning: {zero_var_features} features have zero variance")
    
    # Dimensionality reduction
    reducer, Xr = make_reducer(X, n_components=min(pca_comp, X.shape[1]-1, X.shape[0]-1))
    
    # Standardize features
    scaler = StandardScaler()
    Xr_s = scaler.fit_transform(Xr)
    
    # Encode labels
    le = LabelEncoder()
    y = le.fit_transform(labels)
    
    # Check class distribution and remove classes with insufficient samples
    class_counts = Counter(y)
    min_samples = 2
    insufficient_classes = [lab for lab, count in class_counts.items() if count < min_samples]
    
    if insufficient_classes:
        insufficient_names = [le.classes_[i] for i in insufficient_classes]
        print(f"Warning: Removing {len(insufficient_classes)} classes with <{min_samples} samples: {insufficient_names}")
        
        # Filter out insufficient classes
        sufficient_mask = np.isin(y, [lab for lab in class_counts if class_counts[lab] >= min_samples])
        Xr_s = Xr_s[sufficient_mask]
        
        # Re-encode labels
        kept_labels = [labels[i] for i in range(len(labels)) if sufficient_mask[i]]
        le = LabelEncoder().fit(kept_labels)
        y = le.transform(kept_labels)
    
    # Final validation
    final_class_count = len(np.unique(y))
    if final_class_count < 2:
        raise ValueError(f"Training requires at least 2 classes with at least {min_samples} samples each. "
                        f"Only {final_class_count} classes remain after filtering.")
    
    print(f"Final dataset: {len(y)} samples, {final_class_count} classes")
    
    # Train-test split with proper size calculation
    class_counts = Counter(y)
    min_class_size = min(class_counts.values())
    n_classes = len(class_counts)
    
    # Calculate appropriate test size
    # Need at least 1 sample per class in test set when stratifying
    if min_class_size >= 2:
        # Can stratify - ensure test set has at least n_classes samples
        min_test_size = max(n_classes, int(0.1 * len(y)))  # At least 10% or n_classes
        max_test_size = int(0.3 * len(y))  # At most 30%
        test_size = min(max_test_size, max(min_test_size, int(0.2 * len(y))))
        stratify_param = y
    else:
        # Cannot stratify - use simple split
        test_size = max(1, int(0.2 * len(y)))
        stratify_param = None
    
    print(f"Train-test split: test_size={test_size}, stratify={'yes' if stratify_param is not None else 'no'}")
    
    X_train, X_test, y_train, y_test = train_test_split(
        Xr_s, y, test_size=test_size, random_state=42, stratify=stratify_param
    )
    
    print(f"Train-test split: {len(X_train)} train, {len(X_test)} test samples")
    
    # Train model
    mlp = train_mlp_sklearn(X_train, y_train, epochs=epochs)
    
    # Evaluate model
    if len(X_test) > 0:
        preds = mlp.predict(X_test)
        print("\nClassification report on test set:")
        try:
            unique_test_labels = np.unique(y_test)
            target_names = le.inverse_transform(unique_test_labels)
            print(classification_report(y_test, preds, zero_division=0, target_names=target_names))
        except Exception as e:
            print(f"Could not generate detailed classification report: {e}")
            accuracy = np.mean(preds == y_test)
            print(f"Test accuracy: {accuracy:.3f}")
    
    # Save model artifacts
    artifacts = {
        "kmer_vectorizer.joblib": vec,
        "reducer.joblib": reducer,
        "scaler.joblib": scaler,
        "label_encoder.joblib": le,
        "mlp_model.joblib": mlp
    }
    
    for filename, obj in artifacts.items():
        filepath = outdir_path / filename
        joblib.dump(obj, filepath)
        print(f"Saved {filename}")
    
    # Compute and save class centroids and distance thresholds
    centroids = {}
    thresholds = {}
    
    for class_idx, class_label in enumerate(le.classes_):
        class_mask = (y == class_idx)
        if class_mask.sum() == 0:
            continue
        
        class_points = Xr_s[class_mask]
        centroid = class_points.mean(axis=0)
        distances = np.linalg.norm(class_points - centroid, axis=1)
        
        centroids[class_label] = centroid
        thresholds[class_label] = float(np.percentile(distances, centroid_percentile))
    
    centroids_data = {'centroids': centroids, 'thresholds': thresholds}
    joblib.dump(centroids_data, outdir_path / 'class_centroids.joblib')
    
    print(f"Saved class centroids and thresholds (percentile: {centroid_percentile})")
    print(f"Training completed successfully. Model artifacts saved to: {outdir_path}")
    
    return str(outdir_path / "mlp_model.joblib")

# -------------------------
# Main Prediction Function (corrected to match main.py calls)
# -------------------------
def cluster_and_predict(raw_fasta, model_dir, outdir, k=4, pca_comp=50, 
                        cluster_method="kmeans", dbscan_eps=0.5, dbscan_min=5, 
                        kmeans_n=10, threshold=0.7):
    """
    Cluster raw sequences and predict species using trained model.
    
    Args:
        raw_fasta: Path to raw FASTA file
        model_dir: Directory containing trained model artifacts
        outdir: Output directory for results
        k: K-mer size (should match training)
        pca_comp: PCA components (should match training)
        cluster_method: "kmeans" or "dbscan"
        dbscan_eps: DBSCAN epsilon parameter
        dbscan_min: DBSCAN min_samples parameter
        kmeans_n: Number of clusters for K-means
        threshold: Confidence threshold for predictions
    
    Returns:
        Tuple of (assignments_df, medoid_df, cluster_labels)
    """
    print(f"Starting prediction from {raw_fasta}")
    print(f"Model directory: {model_dir}")
    print(f"Parameters: cluster_method={cluster_method}, threshold={threshold}")
    
    # Create output directory
    outdir_path = Path(outdir)
    outdir_path.mkdir(exist_ok=True, parents=True)
    model_dir_path = Path(model_dir)
    
    # Validate model directory
    if not model_dir_path.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
    
    # Read raw sequences
    ids, seqs = read_fasta_to_list(raw_fasta)
    print(f"Loaded {len(seqs)} sequences for prediction")
    
    # Load model artifacts
    try:
        vec = joblib.load(model_dir_path / "kmer_vectorizer.joblib")
        reducer = joblib.load(model_dir_path / "reducer.joblib")
        scaler = joblib.load(model_dir_path / "scaler.joblib")
        le = joblib.load(model_dir_path / "label_encoder.joblib")
        mlp = joblib.load(model_dir_path / "mlp_model.joblib")
        print("Loaded all model artifacts successfully")
    except Exception as e:
        raise RuntimeError(f"Failed to load model artifacts: {e}")
    
    # Transform sequences using trained pipeline
    print("Transforming sequences...")
    X = vec.transform(seqs)
    Xr = reducer.transform(X)
    Xr_s = scaler.transform(Xr)
    
    # Clustering
    if cluster_method.lower() == "dbscan":
        labels = cluster_dbscan(Xr_s, eps=dbscan_eps, min_samples=dbscan_min)
    else:
        # Ensure we don't request more clusters than we have samples
        n_clusters = min(kmeans_n, len(Xr_s))
        n_clusters = max(1, n_clusters)  # At least 1 cluster
        labels = cluster_kmeans(Xr_s, n_clusters=n_clusters)
    
    labels = np.array(labels)
    unique_labels = set(labels)
    n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
    n_noise = sum(labels == -1)
    
    print(f"Clustering results: {n_clusters} clusters, {n_noise} noise points")
    
    # Find cluster medoids
    medoids = cluster_medoid_indices(Xr_s, labels)
    
    # Load centroids and thresholds for distance-based rejection
    centroids_path = model_dir_path / 'class_centroids.joblib'
    centroids_data = None
    if centroids_path.exists():
        centroids_data = joblib.load(centroids_path)
        print("Loaded class centroids for distance-based validation")
    
    # Predict species for each medoid
    medoid_rows = []
    for cluster_label, medoid_idx in medoids.items():
        # Get medoid sequence data
        medoid_seq = seqs[medoid_idx]
        medoid_id = ids[medoid_idx]
        
        # Transform medoid for prediction
        x_medoid = X[medoid_idx].reshape(1, -1)
        xr_medoid = reducer.transform(x_medoid)
        xr_s_medoid = scaler.transform(xr_medoid)
        
        # Get prediction probabilities
        probs = mlp.predict_proba(xr_s_medoid)[0]
        
        # Get top 3 predictions for reporting
        top_indices = probs.argsort()[::-1][:3]
        top_probs = [(mlp.classes_[i], float(probs[i])) for i in top_indices]
        top_labels = [le.inverse_transform([int(c)])[0] for c, _ in top_probs]
        
        # Determine final prediction
        max_prob = float(np.max(probs))
        pred_class_numeric = mlp.classes_[int(np.argmax(probs))]
        pred_species = le.inverse_transform([pred_class_numeric])[0] if max_prob >= threshold else "Unknown"
        
        # Apply distance-based validation if available
        if pred_species != "Unknown" and centroids_data is not None:
            centroids = centroids_data.get('centroids', {})
            thresholds = centroids_data.get('thresholds', {})
            
            if pred_species in centroids and pred_species in thresholds:
                centroid = centroids[pred_species]
                distance = float(np.linalg.norm(xr_s_medoid.ravel() - centroid))
                
                if distance > thresholds[pred_species]:
                    print(f"Medoid {medoid_idx} rejected by distance threshold "
                          f"(dist={distance:.3f} > threshold={thresholds[pred_species]:.3f})")
                    pred_species = "Unknown"
                    max_prob = 0.0
        
        medoid_rows.append((
            cluster_label, medoid_idx, medoid_id, pred_species, max_prob, top_probs, top_labels
        ))
    
    # Create medoid predictions DataFrame
    medoid_df = pd.DataFrame(medoid_rows, columns=[
        "cluster", "medoid_index", "medoid_id", "predicted_species", 
        "confidence", "top_probs", "top_labels"
    ])
    medoid_df.to_csv(outdir_path / "cluster_medoid_predictions.csv", index=False)
    print("Saved medoid predictions")
    
    # Propagate predictions to all sequences
    cluster_assignments = []
    for i, cluster_label in enumerate(labels):
        # Find prediction for this cluster
        cluster_prediction = medoid_df.loc[medoid_df['cluster'] == cluster_label]
        
        if len(cluster_prediction) == 0:
            # No medoid found (noise points)
            assigned_species = "Unknown"
            confidence = 0.0
        else:
            assigned_species = cluster_prediction['predicted_species'].iloc[0]
            confidence = cluster_prediction['confidence'].iloc[0]
        
        cluster_assignments.append((ids[i], cluster_label, assigned_species, confidence))
    
    # Create assignments DataFrame
    assign_df = pd.DataFrame(cluster_assignments, columns=[
        "sequence_id", "cluster", "predicted_species", "confidence"
    ])
    assign_df.to_csv(outdir_path / "sequence_cluster_assignments.csv", index=False)
    print("Saved sequence assignments")
    
    # Generate abundance reports and visualizations
    try:
        generate_abundance_reports(assign_df, outdir_path)
        generate_visualizations(assign_df, Xr, labels, outdir_path)
    except Exception as e:
        print(f"Warning: Could not generate all reports and visualizations: {e}")
    
    print(f"Prediction completed successfully. Results saved to: {outdir_path}")
    
    return assign_df, medoid_df, labels

# -------------------------
# Abundance Reports Generation
# -------------------------
def generate_abundance_reports(assign_df, outdir_path):
    """Generate species abundance reports."""
    total_reads = len(assign_df)
    
    # Abundance by reads
    species_counts = assign_df.groupby("predicted_species").size().reset_index(name="n_reads")
    species_counts["percentage"] = 100.0 * species_counts["n_reads"] / total_reads
    species_counts = species_counts.sort_values("n_reads", ascending=False).reset_index(drop=True)
    species_counts.to_csv(outdir_path / "species_abundance_by_reads.csv", index=False)
    print("Saved species abundance by reads")
    
    # Abundance by clusters (alternative calculation)
    cluster_sizes = assign_df.groupby("cluster").size().reset_index(name="cluster_size")
    cluster_info = assign_df.merge(cluster_sizes, on="cluster").drop_duplicates(subset=["cluster"])
    cluster_counts = cluster_info.groupby("predicted_species")["cluster_size"].sum().reset_index(name="n_reads_via_clusters")
    cluster_counts["percentage_via_clusters"] = 100.0 * cluster_counts["n_reads_via_clusters"] / total_reads
    cluster_counts = cluster_counts.sort_values("n_reads_via_clusters", ascending=False)
    cluster_counts.to_csv(outdir_path / "species_abundance_by_clusters.csv", index=False)
    print("Saved species abundance by clusters")

# -------------------------
# Visualization Generation
# -------------------------
def generate_visualizations(assign_df, Xr, labels, outdir_path):
    """Generate visualization plots."""
    try:
        # Read abundance data for plotting
        species_counts = assign_df.groupby("predicted_species").size().reset_index(name="n_reads")
        total_reads = len(assign_df)
        species_counts["percentage"] = 100.0 * species_counts["n_reads"] / total_reads
        species_counts = species_counts.sort_values("n_reads", ascending=False).reset_index(drop=True)
        
        # Bar plot
        plt.figure(figsize=(10, max(6, 0.4 * min(20, len(species_counts)))))
        top_n = min(20, len(species_counts))
        
        if len(species_counts) > 0:
            ax = sns.barplot(x="percentage", y="predicted_species", 
                           data=species_counts.head(top_n), orient="h")
            plt.xlabel("Percentage of reads (%)")
            plt.ylabel("Predicted species")
            plt.title(f"Species abundance (top {top_n}) by reads")
            
            # Add percentage annotations
            for p in ax.patches:
                width = p.get_width()
                if width > 0:
                    ax.text(width + 0.5, p.get_y() + p.get_height() / 2, 
                           f"{width:.1f}%", va='center')
            
            plt.tight_layout()
            plt.savefig(outdir_path / "species_abundance_bar.png", dpi=300, bbox_inches='tight')
            plt.close()
            print("Saved species abundance bar chart")
        
        # Pie chart
        if len(species_counts) > 0:
            pie_df = species_counts.copy()
            
            # Aggregate low-abundance species if there are many
            if len(pie_df) > top_n:
                top_df = pie_df.head(top_n).copy()
                others_pct = pie_df['percentage'].iloc[top_n:].sum()
                if others_pct > 0:
                    others_row = pd.DataFrame([{
                        'predicted_species': 'Others', 
                        'n_reads': 0, 
                        'percentage': others_pct
                    }])
                    top_df = pd.concat([top_df, others_row], ignore_index=True)
            else:
                top_df = pie_df
            
            if len(top_df) > 0 and top_df['percentage'].sum() > 0:
                fig, ax = plt.subplots(figsize=(8, 8))
                labels_pie = top_df['predicted_species'].tolist()
                sizes = top_df['percentage'].tolist()
                
                # Only show labels for slices > 2%
                labels_display = [label if size >= 2 else '' for label, size in zip(labels_pie, sizes)]
                
                wedges, texts, autotexts = ax.pie(sizes, labels=labels_display, 
                                                 autopct=lambda pct: f'{pct:.1f}%' if pct >= 1 else '',
                                                 startangle=90, pctdistance=0.85)
                ax.axis('equal')
                plt.title('Species composition (by reads)')
                plt.savefig(outdir_path / 'species_composition_pie.png', dpi=300, bbox_inches='tight')
                plt.close()
                print('Saved species composition pie chart')
        
        # Clustering scatter plot
        if Xr.shape[1] >= 2:
            fig, ax = plt.subplots(figsize=(10, 8))
            unique_labels = sorted(set(labels))
            
            # Use a color palette that works well for many clusters
            if len(unique_labels) <= 10:
                palette = sns.color_palette("tab10", n_colors=len(unique_labels))
            else:
                palette = sns.color_palette("hls", n_colors=len(unique_labels))
            
            for idx, lab in enumerate(unique_labels):
                mask = labels == lab
                color = palette[idx % len(palette)]
                label_name = f"Cluster {lab}" if lab != -1 else "Noise"
                ax.scatter(Xr[mask, 0], Xr[mask, 1], s=20, color=color, 
                          label=label_name, alpha=0.7)
            
            # Only show legend if not too many clusters
            if len(unique_labels) <= 15:
                ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize='small')
            
            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")
            ax.set_title("Clustering scatter plot (first 2 principal components)")
            plt.tight_layout()
            plt.savefig(outdir_path / "cluster_scatter.png", dpi=300, bbox_inches='tight')
            plt.close()
            print("Saved cluster scatter plot")
        
    except Exception as e:
        print(f"Error generating visualizations: {e}")
        import traceback
        traceback.print_exc()

# -------------------------
# Utility Functions
# -------------------------
def get_model_info(model_dir):
    """Get information about a trained model."""
    model_dir_path = Path(model_dir)
    
    if not model_dir_path.exists():
        return None
    
    info = {"model_dir": str(model_dir_path), "files_present": {}}
    
    # Check for required files
    required_files = [
        "kmer_vectorizer.joblib",
        "reducer.joblib",
        "scaler.joblib", 
        "label_encoder.joblib",
        "mlp_model.joblib"
    ]
    
    for filename in required_files:
        filepath = model_dir_path / filename
        info["files_present"][filename] = filepath.exists()
    
    # Try to load model parameters
    try:
        if info["files_present"]["kmer_vectorizer.joblib"]:
            vec = joblib.load(model_dir_path / "kmer_vectorizer.joblib")
            info["k_mer_size"] = vec.k
            info["vocabulary_size"] = vec.vocab_size
        
        if info["files_present"]["label_encoder.joblib"]:
            le = joblib.load(model_dir_path / "label_encoder.joblib")
            info["num_classes"] = len(le.classes_)
            info["class_names"] = le.classes_.tolist()
        
        if info["files_present"]["reducer.joblib"]:
            reducer = joblib.load(model_dir_path / "reducer.joblib")
            info["pca_components"] = reducer.n_components
    
    except Exception as e:
        info["parameter_loading_error"] = str(e)
    
    return info

if __name__ == "__main__":
    print("This is a pipeline module. Import and use the functions in your main application.")