#!/usr/bin/env python3
"""
Database Permission Fix Utility
This script helps fix common database permission issues with ChromaDB
"""

import os
import shutil
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def fix_permissions():
    """Fix database permissions for ChromaDB and storage directories"""
    
    # Directories to fix
    directories = [
        "./embeddings",
        "./embeddings/video_index", 
        "./embeddings/chroma_db",
        "./storage",
        "./chroma_db"
    ]
    
    print("🔧 Fixing database permissions...")
    
    for dir_path in directories:
        path = Path(dir_path)
        
        if path.exists():
            print(f"📁 Processing {dir_path}...")
            
            try:
                # Fix directory permissions
                os.chmod(path, 0o755)
                
                # Fix file permissions recursively
                for root, dirs, files in os.walk(path):
                    for d in dirs:
                        try:
                            os.chmod(os.path.join(root, d), 0o755)
                        except Exception as e:
                            print(f"⚠️  Could not fix directory {d}: {e}")
                    
                    for f in files:
                        try:
                            os.chmod(os.path.join(root, f), 0o644)
                        except Exception as e:
                            print(f"⚠️  Could not fix file {f}: {e}")
                
                print(f"✅ Fixed permissions for {dir_path}")
                
            except Exception as e:
                print(f"❌ Error fixing {dir_path}: {e}")
        else:
            print(f"⏭️  Skipping {dir_path} (doesn't exist)")
    
    print("\n🧹 Clearing corrupt database files...")
    
    # Clean up potentially corrupt database files
    corrupt_patterns = [
        "*.db-shm",
        "*.db-wal", 
        "*.db-journal"
    ]
    
    for pattern in corrupt_patterns:
        for dir_path in directories:
            path = Path(dir_path)
            if path.exists():
                for file in path.rglob(pattern):
                    try:
                        file.unlink()
                        print(f"🗑️  Removed {file}")
                    except Exception as e:
                        print(f"⚠️  Could not remove {file}: {e}")

def clean_databases():
    """Completely remove existing databases to start fresh"""
    
    print("🧹 Cleaning existing databases...")
    
    directories_to_remove = [
        "./embeddings/chroma_db",
        "./chroma_db",
        "./embeddings/video_index",
        "./storage"
    ]
    
    for dir_path in directories_to_remove:
        path = Path(dir_path)
        if path.exists():
            try:
                # Try to fix permissions first
                for root, dirs, files in os.walk(path):
                    for d in dirs:
                        try:
                            os.chmod(os.path.join(root, d), 0o755)
                        except:
                            pass
                    for f in files:
                        try:
                            os.chmod(os.path.join(root, f), 0o644)
                        except:
                            pass
                
                shutil.rmtree(path)
                print(f"✅ Removed {dir_path}")
                
            except Exception as e:
                print(f"❌ Could not remove {dir_path}: {e}")
        else:
            print(f"⏭️  {dir_path} doesn't exist")

def main():
    """Main function with user choices"""
    
    print("🎥 Video Search Database Permission Fixer")
    print("=" * 40)
    
    print("\nChoose an option:")
    print("1. Fix permissions only (recommended)")
    print("2. Clean databases and start fresh")
    print("3. Both - clean and fix permissions")
    print("4. Exit")
    
    choice = input("\nEnter your choice (1-4): ").strip()
    
    if choice == "1":
        fix_permissions()
        print("\n✅ Permission fix completed!")
        
    elif choice == "2":
        confirm = input("⚠️  This will delete all search indexes. Continue? (y/N): ").strip().lower()
        if confirm == 'y':
            clean_databases()
            print("\n✅ Database cleanup completed!")
        else:
            print("❌ Operation cancelled")
            
    elif choice == "3":
        confirm = input("⚠️  This will delete all search indexes. Continue? (y/N): ").strip().lower()
        if confirm == 'y':
            clean_databases()
            fix_permissions()
            print("\n✅ Cleanup and permission fix completed!")
        else:
            print("❌ Operation cancelled")
            
    elif choice == "4":
        print("👋 Goodbye!")
        
    else:
        print("❌ Invalid choice")
    
    print("\n💡 After running this script:")
    print("   1. Restart your Streamlit app")
    print("   2. Re-initialize the system")
    print("   3. Process your uploaded videos again if needed")

if __name__ == "__main__":
    main() 