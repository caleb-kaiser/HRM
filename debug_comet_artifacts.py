#!/usr/bin/env python3
"""
Debug Comet ML Artifacts

This script helps troubleshoot Comet ML artifact upload/download issues.
"""

import argparse
from pathlib import Path


def list_artifacts(project_name: str, workspace: str = None):
    """List all artifacts in a project."""
    try:
        import comet_ml
        
        api = comet_ml.API()
        
        print(f"🔍 Listing artifacts in project: {project_name}")
        if workspace:
            print(f"   Workspace: {workspace}")
        
        # Get artifacts
        artifacts = api.get_artifacts(workspace=workspace, project_name=project_name)
        
        if not artifacts:
            print("❌ No artifacts found in this project")
            return
        
        print(f"✅ Found {len(artifacts)} artifacts:")
        print("-" * 50)
        
        for artifact in artifacts:
            print(f"📦 {artifact.name}")
            print(f"   Type: {artifact.artifact_type}")
            print(f"   Latest Version: {artifact.latest_version}")
            print(f"   Versions: {len(artifact.versions)}")
            print(f"   Created: {artifact.created_at}")
            
            # List aliases for latest version
            if hasattr(artifact, 'aliases') and artifact.aliases:
                print(f"   Aliases: {list(artifact.aliases)}")
            
            print()
            
    except ImportError:
        print("❌ comet_ml not installed. Run: pip install comet_ml")
    except Exception as e:
        print(f"❌ Error listing artifacts: {e}")


def check_experiment_artifacts(experiment_key: str):
    """Check artifacts for a specific experiment."""
    try:
        import comet_ml
        
        api = comet_ml.API()
        
        print(f"🔍 Checking artifacts for experiment: {experiment_key}")
        
        # Get experiment
        experiment = api.get_experiment_by_key(experiment_key)
        
        print(f"📊 Experiment: {experiment.name}")
        print(f"   Project: {experiment.project_name}")
        print(f"   Workspace: {experiment.workspace}")
        print(f"   URL: {experiment.url}")
        
        # Get artifacts
        artifacts = experiment.get_output_artifacts()
        
        if not artifacts:
            print("❌ No output artifacts found for this experiment")
        else:
            print(f"✅ Found {len(artifacts)} output artifacts:")
            for artifact in artifacts:
                print(f"   📦 {artifact.name} v{artifact.version}")
                print(f"      Aliases: {list(artifact.aliases) if artifact.aliases else 'None'}")
        
        # Get input artifacts
        input_artifacts = experiment.get_input_artifacts()
        if input_artifacts:
            print(f"📥 Found {len(input_artifacts)} input artifacts:")
            for artifact in input_artifacts:
                print(f"   📦 {artifact.name} v{artifact.version}")
        
    except ImportError:
        print("❌ comet_ml not installed. Run: pip install comet_ml")
    except Exception as e:
        print(f"❌ Error checking experiment: {e}")


def test_artifact_access(project_name: str, artifact_name: str, version_or_alias: str = "latest", workspace: str = None):
    """Test if an artifact can be accessed."""
    try:
        import comet_ml
        
        print(f"🧪 Testing artifact access:")
        print(f"   Project: {project_name}")
        print(f"   Artifact: {artifact_name}")
        print(f"   Version/Alias: {version_or_alias}")
        if workspace:
            print(f"   Workspace: {workspace}")
        
        # Create a temporary experiment to test access
        experiment = comet_ml.Experiment(project_name=project_name, workspace=workspace)
        
        try:
            artifact = experiment.get_artifact(artifact_name, version_or_alias=version_or_alias)
            print(f"✅ Artifact found!")
            print(f"   Name: {artifact.name}")
            print(f"   Version: {artifact.version}")
            print(f"   Type: {artifact.artifact_type}")
            print(f"   Size: {artifact.size} bytes")
            print(f"   Assets: {len(artifact.assets)}")
            
            if artifact.aliases:
                print(f"   Aliases: {list(artifact.aliases)}")
            
            if artifact.metadata:
                print(f"   Metadata keys: {list(artifact.metadata.keys())}")
            
        except Exception as e:
            print(f"❌ Cannot access artifact: {e}")
            print(f"   This could mean:")
            print(f"   1. Artifact doesn't exist")
            print(f"   2. Wrong project/workspace")
            print(f"   3. Artifact is still processing")
            print(f"   4. Permission issues")
        
        experiment.end()
        
    except ImportError:
        print("❌ comet_ml not installed. Run: pip install comet_ml")
    except Exception as e:
        print(f"❌ Error testing artifact access: {e}")


def wait_for_artifact(project_name: str, artifact_name: str, version_or_alias: str = "latest", 
                     workspace: str = None, max_wait: int = 60):
    """Wait for an artifact to become available."""
    try:
        import comet_ml
        import time
        
        print(f"⏳ Waiting for artifact to become available...")
        print(f"   Project: {project_name}")
        print(f"   Artifact: {artifact_name}")
        print(f"   Version/Alias: {version_or_alias}")
        print(f"   Max wait: {max_wait} seconds")
        
        experiment = comet_ml.Experiment(project_name=project_name, workspace=workspace)
        
        start_time = time.time()
        while time.time() - start_time < max_wait:
            try:
                artifact = experiment.get_artifact(artifact_name, version_or_alias=version_or_alias)
                elapsed = int(time.time() - start_time)
                print(f"✅ Artifact found after {elapsed} seconds!")
                print(f"   Version: {artifact.version}")
                print(f"   Size: {artifact.size} bytes")
                experiment.end()
                return True
                
            except Exception:
                print(".", end="", flush=True)
                time.sleep(2)
        
        print(f"\n❌ Artifact not found after {max_wait} seconds")
        experiment.end()
        return False
        
    except ImportError:
        print("❌ comet_ml not installed. Run: pip install comet_ml")
        return False
    except Exception as e:
        print(f"❌ Error waiting for artifact: {e}")
        return False


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Debug Comet ML artifacts")
    parser.add_argument("--list", help="List artifacts in project", metavar="PROJECT_NAME")
    parser.add_argument("--check-experiment", help="Check artifacts for experiment", metavar="EXPERIMENT_KEY")
    parser.add_argument("--test-access", help="Test artifact access", metavar="PROJECT_NAME")
    parser.add_argument("--artifact-name", help="Artifact name for testing", default="hrm-sudoku-traces")
    parser.add_argument("--version", help="Version or alias", default="latest")
    parser.add_argument("--workspace", help="Comet workspace name")
    parser.add_argument("--wait", help="Wait for artifact", metavar="PROJECT_NAME")
    parser.add_argument("--max-wait", type=int, default=60, help="Max wait time in seconds")
    
    args = parser.parse_args()
    
    try:
        import comet_ml
        if not comet_ml.config.get_api_key():
            print("⚠️  No Comet API key found. Please set COMET_API_KEY environment variable")
            return 1
    except ImportError:
        print("❌ comet_ml not installed. Run: pip install comet_ml")
        return 1
    
    if args.list:
        list_artifacts(args.list, args.workspace)
    elif args.check_experiment:
        check_experiment_artifacts(args.check_experiment)
    elif args.test_access:
        test_artifact_access(args.test_access, args.artifact_name, args.version, args.workspace)
    elif args.wait:
        success = wait_for_artifact(args.wait, args.artifact_name, args.version, args.workspace, args.max_wait)
        return 0 if success else 1
    else:
        print("❓ No action specified. Use --help for options")
        print("\nQuick examples:")
        print("  python debug_comet_artifacts.py --list hrm-traces")
        print("  python debug_comet_artifacts.py --test-access hrm-traces --artifact-name hrm-sudoku-traces")
        print("  python debug_comet_artifacts.py --wait hrm-traces --artifact-name hrm-sudoku-traces")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 