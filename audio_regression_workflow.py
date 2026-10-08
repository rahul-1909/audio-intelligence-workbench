"""Train and evaluate a generic WAV-audio regression model.

Expected input layout:
  <data-dir>/train.csv      with identifier and numeric target columns
  <data-dir>/test.csv       with an identifier column
  <data-dir>/train/         audio files named by the identifier column
  <data-dir>/test/          audio files named by the identifier column
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import make_pipeline


def summarize(values: np.ndarray, prefix: str, features: dict[str, float]) -> None:
    values = np.atleast_2d(np.asarray(values, dtype=np.float64))
    for index, row in enumerate(values):
        features[f"{prefix}_{index:02d}_mean"] = float(np.nanmean(row))
        features[f"{prefix}_{index:02d}_std"] = float(np.nanstd(row))


def extract_features(audio_path: Path, sample_rate: int, max_duration: float | None) -> dict[str, float]:
    audio, rate = librosa.load(audio_path, sr=sample_rate, mono=True, duration=max_duration)
    if audio.size == 0:
        raise ValueError(f"Empty audio file: {audio_path}")

    features: dict[str, float] = {
        "duration_seconds": float(audio.size / rate),
        "amplitude_rms": float(np.sqrt(np.mean(audio**2))),
        "amplitude_abs_mean": float(np.mean(np.abs(audio))),
        "silence_fraction": float(np.mean(np.abs(audio) < 1e-4)),
    }
    summarize(librosa.feature.mfcc(y=audio, sr=rate, n_mfcc=20), "mfcc", features)
    summarize(librosa.feature.spectral_centroid(y=audio, sr=rate), "centroid", features)
    summarize(librosa.feature.spectral_bandwidth(y=audio, sr=rate), "bandwidth", features)
    summarize(librosa.feature.spectral_rolloff(y=audio, sr=rate), "rolloff", features)
    summarize(librosa.feature.spectral_contrast(y=audio, sr=rate), "contrast", features)
    summarize(librosa.feature.zero_crossing_rate(audio), "zcr", features)
    summarize(librosa.feature.rms(y=audio), "rms", features)
    summarize(librosa.feature.chroma_stft(y=audio, sr=rate), "chroma", features)

    pitch = librosa.yin(audio, fmin=65, fmax=500, sr=rate)
    voiced_pitch = pitch[np.isfinite(pitch) & (pitch > 0)]
    features["pitch_median_hz"] = float(np.median(voiced_pitch)) if voiced_pitch.size else np.nan
    features["pitch_std_hz"] = float(np.std(voiced_pitch)) if voiced_pitch.size else np.nan
    return features


def build_feature_frame(manifest: pd.DataFrame, audio_dir: Path, id_column: str, sample_rate: int, max_duration: float | None) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for filename in manifest[id_column]:
        path = audio_dir / filename
        if not path.is_file():
            raise FileNotFoundError(f"Audio referenced by manifest is missing: {path}")
        rows.append({id_column: filename, **extract_features(path, sample_rate, max_duration)})
    return pd.DataFrame(rows).set_index(id_column)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a generic WAV-audio regression workflow.")
    parser.add_argument("--data-dir", type=Path, required=True, help="Directory containing CSV manifests and train/test audio folders.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"), help="Directory for predictions and metrics.")
    parser.add_argument("--id-column", default="filename", help="Audio-file identifier column in both manifests.")
    parser.add_argument("--target-column", default="label", help="Numeric training target column.")
    parser.add_argument("--folds", type=int, default=5, help="Number of shuffled cross-validation folds.")
    parser.add_argument("--seed", type=int, default=2026, help="Random seed.")
    parser.add_argument("--sample-rate", type=int, default=16000, help="Audio decoding sample rate.")
    parser.add_argument("--max-duration", type=float, default=None, help="Optional maximum decoded seconds per recording.")
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    train_csv, test_csv = args.data_dir / "train.csv", args.data_dir / "test.csv"
    train_audio, test_audio = args.data_dir / "train", args.data_dir / "test"
    for path in (train_csv, test_csv, train_audio, test_audio):
        if not path.exists():
            raise FileNotFoundError(f"Required input is missing: {path}")

    train_manifest = pd.read_csv(train_csv)
    test_manifest = pd.read_csv(test_csv)
    required_train = {args.id_column, args.target_column}
    if missing := required_train - set(train_manifest.columns):
        raise ValueError(f"train.csv is missing columns: {sorted(missing)}")
    if args.id_column not in test_manifest.columns:
        raise ValueError(f"test.csv is missing identifier column: {args.id_column}")
    if train_manifest[args.id_column].isna().any() or test_manifest[args.id_column].isna().any():
        raise ValueError("Audio identifiers cannot be empty")
    if train_manifest[args.id_column].duplicated().any() or test_manifest[args.id_column].duplicated().any():
        raise ValueError("Audio identifiers must be unique within each manifest")

    targets = pd.to_numeric(train_manifest[args.target_column], errors="raise").to_numpy(dtype=float)
    if not np.isfinite(targets).all():
        raise ValueError("Training targets must be finite numeric values")

    print("Extracting training audio features...")
    train_features = build_feature_frame(train_manifest, train_audio, args.id_column, args.sample_rate, args.max_duration)
    print("Extracting prediction audio features...")
    test_features = build_feature_frame(test_manifest, test_audio, args.id_column, args.sample_rate, args.max_duration)
    feature_columns = train_features.columns.tolist()
    if feature_columns != test_features.columns.tolist():
        raise ValueError("Training and prediction feature columns do not match")

    model = make_pipeline(
        SimpleImputer(strategy="median"),
        ExtraTreesRegressor(n_estimators=600, min_samples_leaf=2, max_features=0.8, random_state=args.seed, n_jobs=-1),
    )
    cv = KFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    oof_predictions = cross_val_predict(model, train_features, targets, cv=cv, n_jobs=1)
    oof_rmse = float(np.sqrt(mean_squared_error(targets, oof_predictions)))
    oof_pearson = float(pearsonr(targets, oof_predictions).statistic) if np.std(oof_predictions) > 0 else None

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics = {"oof_rmse": oof_rmse, "oof_pearson": oof_pearson, "folds": args.folds, "seed": args.seed, "features": len(feature_columns), "training_rows": len(train_manifest), "prediction_rows": len(test_manifest)}
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))

    model.fit(train_features, targets)
    final_estimator = model.named_steps["extratreesregressor"]
    feature_importance = pd.DataFrame({"feature": feature_columns, "importance": final_estimator.feature_importances_}).sort_values("importance", ascending=False)
    feature_importance.to_csv(args.output_dir / "feature_importance.csv", index=False)

    predictions = model.predict(test_features)
    pd.DataFrame({args.id_column: test_manifest[args.id_column].to_numpy(), args.target_column: predictions}).to_csv(args.output_dir / "predictions.csv", index=False)
    print(f"Wrote predictions to {args.output_dir / 'predictions.csv'}")


if __name__ == "__main__":
    main()
