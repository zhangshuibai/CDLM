#!/usr/bin/env python3
"""
复制 HuggingFace 模型到新的 repo

使用方法:
    python copy_hf_model.py \
        --source_repo Shuibai12138/Open-Dcoder-0.5B-mixture-mdm-step2000 \
        --target_repo your-username/new-model-name \
        --private  # 可选，创建 private repo
"""

import argparse
from huggingface_hub import snapshot_download, upload_folder
import tempfile
from pathlib import Path


def copy_model_to_new_repo(source_repo: str, target_repo: str, private: bool = False):
    """
    将模型从一个 HuggingFace repo 复制到另一个
    
    Args:
        source_repo: 源 repo ID
        target_repo: 目标 repo ID
        private: 是否创建为 private repo
    """
    print("="*60)
    print(f"复制模型: {source_repo} -> {target_repo}")
    print("="*60)
    
    # 创建临时目录
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        model_path = tmp_path / "model"
        
        print(f"\n步骤 1/2: 正在从 {source_repo} 下载模型...")
        try:
            # 下载整个 repo（包括所有文件）
            snapshot_download(
                repo_id=source_repo,
                local_dir=str(model_path),
                local_dir_use_symlinks=False,  # 不使用符号链接，确保文件都在
            )
            print(f"✓ 模型已下载到临时目录")
            
            # 显示下载的文件
            files = list(model_path.rglob("*"))
            files = [f for f in files if f.is_file()]
            print(f"  下载了 {len(files)} 个文件")
            print(f"  主要文件:")
            for f in sorted(files)[:10]:  # 显示前10个文件
                print(f"    - {f.relative_to(model_path)}")
            if len(files) > 10:
                print(f"    ... 还有 {len(files) - 10} 个文件")
                
        except Exception as e:
            print(f"❌ 下载失败: {e}")
            return False
        
        print(f"\n步骤 2/2: 正在上传到 {target_repo}...")
        try:
            # 上传到新 repo
            upload_folder(
                folder_path=str(model_path),
                repo_id=target_repo,
                repo_type="model",
                private=private,
            )
            print(f"\n✓ 成功！模型已复制到: https://huggingface.co/{target_repo}")
            return True
            
        except Exception as e:
            print(f"❌ 上传失败: {e}")
            print(f"\n提示: 确保你已经登录 HuggingFace (运行: huggingface-cli login)")
            return False


def main():
    parser = argparse.ArgumentParser(description="复制 HuggingFace 模型到新的 repo")
    parser.add_argument(
        "--source_repo",
        type=str,
        required=True,
        help="源 repo ID (例如: Shuibai12138/Open-Dcoder-0.5B-mixture-mdm-step2000)"
    )
    parser.add_argument(
        "--target_repo",
        type=str,
        required=True,
        help="目标 repo ID (例如: your-username/new-model-name)"
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="创建 private repo (默认: public)"
    )
    
    args = parser.parse_args()
    
    success = copy_model_to_new_repo(
        args.source_repo,
        args.target_repo,
        args.private
    )
    
    exit(0 if success else 1)


if __name__ == "__main__":
    main()

