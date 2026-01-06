import os
import re
import glob
import sys
import shutil
import argparse
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from configs import CNENHPS
from models import BetaVAEVC
from audio import TestUtils, Audio


os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")


@dataclass(frozen=True)
class PairItem:
    src_id: str
    ref_id: str
    # output path relative to output root
    out_rel_path: str


def extract_mel(wav_f: str, audio_processor: Audio) -> np.ndarray:
    wav_arr = audio_processor.load_wav(wav_f)
    wav_arr = audio_processor.trim_silence_by_trial(wav_arr, top_db=20.0, lower_db=25.0)
    wav_arr = wav_arr / max(0.01, np.max(np.abs(wav_arr)))
    wav_arr = audio_processor.preemphasize(wav_arr)
    mel = audio_processor.melspectrogram(wav_arr).T
    return mel


def read_mels(wav_list_f: str, audio_processor: Audio) -> Tuple[List[np.ndarray], List[str]]:
    """Backward-compatible: read wav paths from a list file and extract mels."""
    mels: List[np.ndarray] = []
    mel_names: List[str] = []
    with open(wav_list_f, "r", encoding="utf-8") as f:
        for line in f:
            wav_path = line.strip()
            if not wav_path:
                continue
            mel = extract_mel(wav_path, audio_processor).astype(np.float32)
            mels.append(mel)
            name = os.path.basename(wav_path).split(".")[0]
            mel_names.append(name)
    return mels, mel_names


def _normalize_rel_path(p: str) -> str:
    p = p.strip().replace("\\", "/")
    # If the txt accidentally contains absolute paths, only keep the basename.
    if os.path.isabs(p):
        p = os.path.basename(p)
    # prevent '../' escaping
    p = re.sub(r"^\./", "", p)
    while p.startswith("../"):
        p = p[3:]
    return p


def parse_pairs_txt(pairs_txt: str) -> List[PairItem]:
    """Parse a txt file whose each line contains something like:
    seen_P00003A/BAC009S0002W0351_to_20170001P00003A0075.wav

    Returns list of (src_id, ref_id, out_filename). (Txt subdirectories are ignored for output layout.)
    """
    pairs: List[PairItem] = []
    with open(pairs_txt, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.split("#", 1)[0].strip()
            if not line:
                continue
            rel = _normalize_rel_path(line)
            base = os.path.basename(rel)
            stem, ext = os.path.splitext(base)
            if ext and ext.lower() != ".wav":
                # allow non-wav extension in txt, but we'll output wav
                stem = stem
            # Expect something like <src>_to_<ref>
            if "_to_" not in stem:
                raise ValueError(f"Bad line (missing '_to_'): {raw.strip()}")
            src_id, ref_id = stem.split("_to_", 1)
            out_rel_path = f"{src_id}_to_{ref_id}.wav"
            pairs.append(PairItem(src_id=src_id, ref_id=ref_id, out_rel_path=out_rel_path))
    return pairs


def build_wav_index(root: str) -> Dict[str, str]:
    """Build {filename.wav -> full_path} by scanning root recursively.

    Uses glob2 if available (requested), otherwise falls back to Python's glob.
    If duplicate filenames exist under different subdirs, the first hit wins and
    duplicates are reported to stderr.
    """
    index: Dict[str, str] = {}
    dups: Dict[str, List[str]] = {}

    # Prefer glob2 (supports ** patterns)
    try:
        import glob2  # type: ignore
        wav_paths = glob2.glob(os.path.join(root, "**", "*.wav"))
    except Exception:
        wav_paths = glob.glob(os.path.join(root, "**", "*.wav"), recursive=True)

    for p in wav_paths:
        fn = os.path.basename(p)
        if not fn.lower().endswith(".wav"):
            continue
        if fn in index and os.path.abspath(index[fn]) != os.path.abspath(p):
            dups.setdefault(fn, [index[fn]]).append(p)
            continue
        index[fn] = p

    if dups:
        # Avoid printing an enormous wall of text
        sample = list(dups.items())[:10]
        print(
            f"[WARN] Found duplicate wav basenames under {root}. "
            f"Using the first hit for each name. Sample duplicates: "
            + "; ".join([f"{k} -> {len(v)} paths" for k, v in sample]),
            file=sys.stderr,
        )

    return index



def resolve_wav(root: str, filename: str, index: Optional[Dict[str, str]] = None) -> str:
    # If we already have an index, use it.
    if index is not None:
        hit = index.get(filename)
        if hit is not None:
            return hit

    # Fast path: file exists directly under root
    cand = os.path.join(root, filename)
    if os.path.isfile(cand):
        return cand

    # Lazy fallback: build an index via glob2/glob and retry once.
    lazy_index = build_wav_index(root)
    hit = lazy_index.get(filename)
    if hit is not None:
        return hit

    raise FileNotFoundError(f"Cannot find wav: {filename} under {root}")


def _glob_wavs(root: str) -> List[str]:
    return glob.glob(os.path.join(root, "**", "*.wav"), recursive=True)


def _ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent and not os.path.isdir(parent):
        os.makedirs(parent, exist_ok=True)


def _pick_generated_wav(tmp_root: str, before: set, keyword: str) -> Optional[str]:
    after = set(_glob_wavs(tmp_root))
    new_files = list(after - before)
    if not new_files:
        # fallback: try keyword search
        cands = glob.glob(os.path.join(tmp_root, "**", f"*{keyword}*.wav"), recursive=True)
        if not cands:
            return None
        return max(cands, key=os.path.getmtime)

    # Prefer files containing the keyword
    kw = [p for p in new_files if keyword in os.path.basename(p)]
    if kw:
        return max(kw, key=os.path.getmtime)
    return max(new_files, key=os.path.getmtime)


def _setup_model_and_vc(hparams, ckpt_path: str):
    model = BetaVAEVC(hparams)
    checkpoint = tf.train.Checkpoint(model=model)
    checkpoint.restore(ckpt_path).expect_partial()

    @tf.function(
        input_signature=[
            tf.TensorSpec(shape=[None, None, hparams.Audio.num_mels], dtype=tf.float32),
            tf.TensorSpec(shape=[None, None, hparams.Audio.num_mels], dtype=tf.float32),
            tf.TensorSpec(shape=[None], dtype=tf.int32),
        ]
    )
    def vc(mels, mel_ext, m_lengths):
        out, _ = model.post_inference(mels, m_lengths, mel_ext)
        return out

    return model, vc


def synthesize_from_pairs(args) -> None:
    ckpt_path = args.ckpt_path
    ckpt_step = ckpt_path.split("-")[-1]

    assert os.path.isfile(args.pairs_txt), f"pairs_txt not found: {args.pairs_txt}"

    out_root = args.out_dir or args.test_dir
    if not out_root:
        raise ValueError("You must provide --out_dir (or --test_dir).")
    os.makedirs(out_root, exist_ok=True)

    # We generate into a tmp folder then move/rename to exactly what you want.
    tmp_root = os.path.join(out_root, "_tmp_vc")
    os.makedirs(tmp_root, exist_ok=True)

    hparams = CNENHPS()
    tester = TestUtils(hparams, tmp_root)
    audio_processor = Audio(hparams.Audio)

    _, vc = _setup_model_and_vc(hparams, ckpt_path)

    pairs = parse_pairs_txt(args.pairs_txt)

    src_root = args.src_root or args.wav_root
    ref_root = args.ref_root or args.wav_root
    if not src_root or not ref_root:
        raise ValueError("You must provide --wav_root (or --src_root/--ref_root).")

        # Your wavs may live under many subdirectories. Build an index once via glob2/glob.
    # (Even if you don't pass --recursive, we still do this to avoid repeated disk scans.)
    src_index = build_wav_index(src_root)
    ref_index = src_index if os.path.abspath(src_root) == os.path.abspath(ref_root) else build_wav_index(ref_root)


    # Cache mels to avoid recomputing
    src_mel_cache: Dict[str, np.ndarray] = {}
    ref_mel_cache: Dict[str, np.ndarray] = {}

    has_wav_writer = hasattr(tester, "synthesize_and_save_wavs")

    for item in tqdm(pairs, desc="VC pairs"):
        src_wav = resolve_wav(src_root, f"{item.src_id}.wav", src_index)
        ref_wav = resolve_wav(ref_root, f"{item.ref_id}.wav", ref_index)

        # src mel
        if src_wav not in src_mel_cache:
            src_mel_cache[src_wav] = extract_mel(src_wav, audio_processor).astype(np.float32)
        src_mel = src_mel_cache[src_wav]

        # ref mel (padded/truncated)
        if ref_wav not in ref_mel_cache:
            ref_mel = extract_mel(ref_wav, audio_processor).astype(np.float32)
            while ref_mel.shape[0] < hparams.Dataset.chunk_size:
                ref_mel = np.concatenate([ref_mel, ref_mel], axis=0)
            ref_mel_cache[ref_wav] = ref_mel[: hparams.Dataset.chunk_size, :]
        ref_mel = ref_mel_cache[ref_wav]

        if src_mel.shape[1] != hparams.Audio.num_mels:
            raise ValueError(
                f"Bad mel shape for {src_wav}: {src_mel.shape}, expected num_mels={hparams.Audio.num_mels}"
            )

        src_mel_batch = tf.constant(np.expand_dims(src_mel, axis=0), dtype=tf.float32)
        ref_mel_batch = tf.constant(np.expand_dims(ref_mel, axis=0), dtype=tf.float32)
        mel_len_batch = tf.constant([src_mel.shape[0]], dtype=tf.int32)

        # Keep ids clean: no extension.
        id_base = f"{item.src_id}_to_{item.ref_id}"

        prediction = vc(src_mel_batch, ref_mel_batch, mel_len_batch)

        final_out = os.path.join(out_root, item.out_rel_path)
        _ensure_parent_dir(final_out)

        if os.path.exists(final_out) and not args.overwrite:
            continue

        if has_wav_writer:
            before = set(_glob_wavs(tmp_root))
            # Most implementations will save something like: <prefix>-<id>.wav or <id>.wav.
            tester.synthesize_and_save_wavs(
                ckpt_step,
                prediction.numpy(),
                mel_len_batch.numpy(),
                [id_base],
                prefix=args.save_prefix,
            )
            gen = _pick_generated_wav(tmp_root, before, id_base)
            if gen is None:
                raise RuntimeError(
                    f"synthesize_and_save_wavs() finished but no wav file was found under {tmp_root} (id={id_base})."
                )
            # Move/rename to the exact expected path
            if os.path.abspath(gen) != os.path.abspath(final_out):
                if os.path.exists(final_out) and args.overwrite:
                    os.remove(final_out)
                shutil.move(gen, final_out)
        else:
            # Fallback: write mel if wav synthesis isn't available in your TestUtils.
            tester.write_mels(
                ckpt_step,
                prediction.numpy(),
                mel_len_batch.numpy(),
                [id_base],
                prefix=args.save_prefix,
            )

    # Optional cleanup (keep tmp folder if it still contains files)
    try:
        if os.path.isdir(tmp_root) and not _glob_wavs(tmp_root):
            shutil.rmtree(tmp_root, ignore_errors=True)
    except Exception:
        pass


def synthesize_from_lists(args) -> None:
    """Original behavior: read two wav-list txt files and convert all src x ref."""
    ckpt_path = args.ckpt_path
    ckpt_step = ckpt_path.split("-")[-1]
    assert os.path.isfile(args.src_wavs), f"src_wavs list not found: {args.src_wavs}"
    assert os.path.isfile(args.ref_wavs), f"ref_wavs list not found: {args.ref_wavs}"

    out_root = args.out_dir or args.test_dir
    if not out_root:
        raise ValueError("You must provide --out_dir (or --test_dir).")
    os.makedirs(out_root, exist_ok=True)

    tmp_root = os.path.join(out_root, "_tmp_vc")
    os.makedirs(tmp_root, exist_ok=True)

    hparams = CNENHPS()
    tester = TestUtils(hparams, tmp_root)
    audio_processor = Audio(hparams.Audio)

    _, vc = _setup_model_and_vc(hparams, ckpt_path)

    src_mels, src_names = read_mels(args.src_wavs, audio_processor)
    ref_mels, ref_names = read_mels(args.ref_wavs, audio_processor)

    has_wav_writer = hasattr(tester, "synthesize_and_save_wavs")

    for src_mel, src_name in tqdm(list(zip(src_mels, src_names)), desc="VC all-pairs"):
        for ref_mel, ref_name in zip(ref_mels, ref_names):
            ref_mel2 = ref_mel
            while ref_mel2.shape[0] < hparams.Dataset.chunk_size:
                ref_mel2 = np.concatenate([ref_mel2, ref_mel2], axis=0)
            ref_mel2 = ref_mel2[: hparams.Dataset.chunk_size, :]

            assert src_mel.shape[1] == hparams.Audio.num_mels
            src_mel_batch = tf.constant(np.expand_dims(src_mel, axis=0), dtype=tf.float32)
            ref_mel_batch = tf.constant(np.expand_dims(ref_mel2, axis=0), dtype=tf.float32)
            mel_len_batch = tf.constant([src_mel.shape[0]], dtype=tf.int32)

            id_base = f"{src_name}_to_{ref_name}"
            prediction = vc(src_mel_batch, ref_mel_batch, mel_len_batch)

            final_out = os.path.join(out_root, f"{id_base}.wav")
            if os.path.exists(final_out) and not args.overwrite:
                continue

            if has_wav_writer:
                before = set(_glob_wavs(tmp_root))
                tester.synthesize_and_save_wavs(
                    ckpt_step,
                    prediction.numpy(),
                    mel_len_batch.numpy(),
                    [id_base],
                    prefix=args.save_prefix,
                )
                gen = _pick_generated_wav(tmp_root, before, id_base)
                if gen is None:
                    raise RuntimeError(
                        f"synthesize_and_save_wavs() finished but no wav file was found under {tmp_root} (id={id_base})."
                    )
                if os.path.abspath(gen) != os.path.abspath(final_out):
                    if os.path.exists(final_out) and args.overwrite:
                        os.remove(final_out)
                    shutil.move(gen, final_out)
            else:
                tester.write_mels(
                    ckpt_step,
                    prediction.numpy(),
                    mel_len_batch.numpy(),
                    [id_base],
                    prefix=args.save_prefix,
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser("BetaVAE-VC inference from wav")

    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="/root/BetaVAE_VC/checkpoints/BetaVAE-VC",
        help="path to the model ckpt",
    )
    # Output folders
    parser.add_argument("--out_dir", type=str, default="/root/autodl-tmp/unseen_to_unseen", help="output directory for converted wavs")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="overwrite existing outputs (default: skip if exists)",
    )

    # Mode A (your requested): pairs txt like *_to_*.wav
    parser.add_argument("--pairs_txt", type=str, default="/root/autodl-tmp/openvoicev2.txt", help="txt file listing '<src>_to_<ref>.wav'")
    parser.add_argument(
        "--wav_root",
        type=str,
        default="/root/autodl-tmp/test_set",
        help="root directory containing both source and target wavs (used when src_root/ref_root not set)",
    )

    parser.add_argument(
        "--recursive",
        action="store_true",
        help="recursively search wav_root/src_root/ref_root using glob2/glob (build an index once). If not set, script may still build a lazy index when needed.",
    )

    # Mode B (original): two wav-list txt files


    parser.add_argument(
        "--save_prefix",
        type=str,
        default="test",
        help="prefix passed to TestUtils writer (only used internally; final name will still be '<src>_to_<ref>.wav')",
    )
    # 不用填
    parser.add_argument("--src_root", type=str, default=None, help="root directory to find <src>.wav")
    parser.add_argument("--ref_root", type=str, default=None, help="root directory to find <ref>.wav")
    parser.add_argument("--test_dir", type=str, default=None, help="(legacy) directory for test artifacts")
    main_args = parser.parse_args()

    if main_args.pairs_txt:
        synthesize_from_pairs(main_args)
    else:
        if not main_args.src_wavs or not main_args.ref_wavs:
            raise ValueError("Provide either --pairs_txt OR both --src_wavs and --ref_wavs")
        synthesize_from_lists(main_args)
