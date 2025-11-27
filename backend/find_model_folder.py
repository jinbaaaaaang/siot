# -*- coding: utf-8 -*-
"""
Colab에서 모델 폴더를 찾는 헬퍼 스크립트

Colab 셀에서 실행:
exec(open('find_model_folder.py').read())
"""

from pathlib import Path
import os

def find_model_folders():
    """모든 가능한 모델 폴더 찾기 (20251109_08로 시작하는 모델만)"""
    print("🔍 모델 폴더 검색 중... (20251109_08로 시작하는 모델만)\n")
    
    current_dir = Path(".")
    print(f"현재 디렉토리: {current_dir.absolute()}\n")
    
    # 필터링할 날짜/시간 패턴
    target_prefix = "20251109_08"
    
    # 방법 1: kogpt2 관련 폴더 찾기 (20251109_08로 시작하는 것만)
    kogpt2_folders = []
    for item in current_dir.iterdir():
        if item.is_dir() and "kogpt2" in item.name.lower():
            # 폴더 이름에 20251109_08이 포함되어 있는지 확인
            if target_prefix in item.name:
                kogpt2_folders.append(item)
    
    # 방법 2: 직접 fold 폴더 찾기 (현재 디렉토리에 바로 있는 경우, 20251109_08로 시작하는 것만)
    direct_fold_folders = []
    for item in current_dir.iterdir():
        if item.is_dir() and "_fold" in item.name and "kogpt2" in item.name.lower():
            # 폴더 이름에 20251109_08이 포함되어 있는지 확인
            if target_prefix in item.name:
                direct_fold_folders.append(item)
    
    print(f"📁 검색 결과:\n")
    
    # kogpt2 관련 폴더 확인
    if kogpt2_folders:
        print(f"✅ {len(kogpt2_folders)}개의 kogpt2 관련 폴더를 찾았습니다:\n")
        for i, folder in enumerate(kogpt2_folders, 1):
            # fold 폴더 개수 확인
            fold_count = 0
            fold_names = []
            if folder.exists():
                for subfolder in folder.iterdir():
                    if subfolder.is_dir() and "_fold" in subfolder.name and target_prefix in subfolder.name:
                        fold_count += 1
                        fold_names.append(subfolder.name)
            
            print(f"  {i}. {folder.name}")
            print(f"     경로: {folder.absolute()}")
            print(f"     Fold 개수: {fold_count}")
            if fold_names:
                print(f"     Fold 목록:")
                for fold_name in sorted(fold_names)[:5]:  # 최대 5개만 표시
                    print(f"       - {fold_name}")
                if len(fold_names) > 5:
                    print(f"       ... 외 {len(fold_names) - 5}개")
            print()
    
    # 직접 fold 폴더 확인
    if direct_fold_folders:
        print(f"✅ {len(direct_fold_folders)}개의 직접 fold 폴더를 찾았습니다:\n")
        for i, folder in enumerate(direct_fold_folders, 1):
            print(f"  {i}. {folder.name}")
            print(f"     경로: {folder.absolute()}")
        print()
    
    # 가장 가능성 높은 폴더 추천
    best_folder = None
    
    # 먼저 직접 fold 폴더가 3개 이상인 경우
    if len(direct_fold_folders) >= 3:
        print(f"💡 추천: 현재 디렉토리에 직접 fold 폴더들이 있습니다.")
        print(f"   Fold 개수: {len(direct_fold_folders)}")
        print(f"\n📝 평가 스크립트에서 사용:")
        print(f"   ```python")
        print(f"   import evaluate_folds_colab")
        print(f"   # 현재 디렉토리를 base_dir로 사용")
        print(f"   evaluate_folds_colab.find_best_fold_model(base_dir='.')")
        print(f"   ```")
        return
    
    # kogpt2 폴더 내에서 찾기 (20251109_08로 시작하는 fold만)
    for folder in kogpt2_folders:
        fold_count = sum(1 for f in folder.iterdir() if f.is_dir() and "_fold" in f.name and target_prefix in f.name)
        if fold_count >= 3:  # 최소 3개 이상의 fold가 있으면
            best_folder = folder
            break
    
    if best_folder:
        print(f"💡 추천 폴더: {best_folder.name}")
        print(f"   경로: {best_folder.absolute()}")
        print(f"\n📝 평가 스크립트에서 사용:")
        print(f"   ```python")
        print(f"   import evaluate_folds_colab")
        print(f"   evaluate_folds_colab.find_best_fold_model(base_dir='{best_folder}')")
        print(f"   ```")
    else:
        print("⚠️ fold 모델이 충분히 있는 폴더를 찾을 수 없습니다.")
        print("\n💡 수동으로 확인:")
        print("   1. 학습 스크립트에서 모델이 저장된 경로 확인")
        print("   2. 해당 경로를 base_dir로 지정하여 평가 실행")
        print("\n📝 예시:")
        print("   ```python")
        print("   import evaluate_folds_colab")
        print("   evaluate_folds_colab.find_best_fold_model(base_dir='./실제폴더경로')")
        print("   ```")
    
    if not kogpt2_folders and not direct_fold_folders:
        print("\n❌ kogpt2 관련 폴더를 찾을 수 없습니다.")
        print("\n📁 현재 디렉토리의 모든 폴더:")
        for item in current_dir.iterdir():
            if item.is_dir():
                print(f"  - {item.name}")
        
        print("\n💡 수동으로 폴더 경로를 확인하세요:")
        print("   ```python")
        print("   from pathlib import Path")
        print("   folder = Path('폴더명')")
        print("   if folder.exists():")
        print("       print('✅ 폴더 존재')")
        print("   ```")


if __name__ == "__main__":
    find_model_folders()

