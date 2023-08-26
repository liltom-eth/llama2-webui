import os
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo_id",
        type=str,
        default="",
        required=True,
        help="Repo ID like 'TheBloke/Llama-2-7B-Chat-GGML' ",
    )
    parser.add_argument(
        "--filename",
        type=str,
        default=None,
        help="Filename like llama-2-7b-chat.ggmlv3.q4_0.bin",
    )
    parser.add_argument(
        "--save_dir", type=str, default="./models", help="Directory to save models"
    )

    args = parser.parse_args()

    repo_id = args.repo_id
    save_dir = args.save_dir

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if args.filename:
        filename = args.filename
        from huggingface_hub import hf_hub_download

        print(f"Start downloading model {repo_id} {filename} to: {save_dir}")

        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=save_dir,
        )
    else:
        repo_name = repo_id.split("/")[1]
        save_path = os.path.join(save_dir, repo_name)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        print(f"Start downloading model {repo_id} to: {save_path}")

        from huggingface_hub import snapshot_download

        snapshot_download(
            repo_id=repo_id,
            local_dir=save_path,
        )


if __name__ == "__main__":
    main()
