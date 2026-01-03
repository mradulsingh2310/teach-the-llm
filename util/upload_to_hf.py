"""
Upload fine-tuned Gemma model to Hugging Face Hub.

Usage:
    python -m util.upload_to_hf --repo-name your-model-name
    python -m util.upload_to_hf --repo-name your-model-name --token hf_xxx
    python -m util.upload_to_hf --repo-name your-model-name --private
"""

import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi, login

# Default path to the fine-tuned Gemma model
DEFAULT_MODEL_PATH = Path(__file__).parent.parent / "gemma" / "fine-tune" / "functiongemma-finetuned"


def upload_model(
    model_path: Path,
    repo_name: str,
    token: str | None = None,
    private: bool = False,
    commit_message: str = "Upload fine-tuned Gemma model",
) -> str:
    """
    Upload a model folder to Hugging Face Hub.

    Args:
        model_path: Path to the model folder
        repo_name: Repository name (e.g., 'username/model-name' or just 'model-name')
        token: Hugging Face token (optional if already logged in)
        private: Whether to make the repo private
        commit_message: Commit message for the upload

    Returns:
        URL of the uploaded model
    """
    if not model_path.exists():
        raise FileNotFoundError(f"Model path does not exist: {model_path}")

    if token:
        login(token=token)
    else:
        login()

    api = HfApi()

    # Get username if not provided in repo_name
    if "/" not in repo_name:
        user_info = api.whoami()
        username = user_info["name"]
        repo_id = f"{username}/{repo_name}"
    else:
        repo_id = repo_name

    # Create repo if it doesn't exist
    api.create_repo(
        repo_id=repo_id,
        repo_type="model",
        private=private,
        exist_ok=True,
    )

    print(f"Uploading model from {model_path} to {repo_id}...")

    # Upload the folder (exclude checkpoints to save space)
    api.upload_folder(
        folder_path=str(model_path),
        repo_id=repo_id,
        repo_type="model",
        commit_message=commit_message,
        ignore_patterns=["checkpoint-*"],  # Exclude checkpoint folders
    )

    url = f"https://huggingface.co/{repo_id}"
    print(f"Model uploaded successfully: {url}")

    return url


def main():
    parser = argparse.ArgumentParser(description="Upload Gemma model to Hugging Face Hub")
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help=f"Path to the model folder (default: {DEFAULT_MODEL_PATH})",
    )
    parser.add_argument(
        "--repo-name",
        type=str,
        required=True,
        help="Repository name (e.g., 'functiongemma-finetuned' or 'username/functiongemma-finetuned')",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Hugging Face token (or set HF_TOKEN env var)",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make the repository private",
    )
    parser.add_argument(
        "--commit-message",
        type=str,
        default="Upload fine-tuned Gemma model",
        help="Commit message for the upload",
    )
    parser.add_argument(
        "--include-checkpoints",
        action="store_true",
        help="Include checkpoint folders in upload (excluded by default)",
    )

    args = parser.parse_args()

    # Use default model path if not provided
    model_path = Path(args.model_path) if args.model_path else DEFAULT_MODEL_PATH

    # Check for token in environment if not provided
    token = args.token or os.environ.get("HF_TOKEN")

    upload_model(
        model_path=model_path,
        repo_name=args.repo_name,
        token=token,
        private=args.private,
        commit_message=args.commit_message,
    )


if __name__ == "__main__":
    main()
