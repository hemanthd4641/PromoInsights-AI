import sys
import subprocess
from pathlib import Path

def main():
    """
    Entry point to run the Promotion Analytics AI Assistant.
    Starts the Streamlit application.
    """
    project_root = Path(__file__).resolve().parent
    app_path = project_root / "app" / "streamlit_app.py"
    
    if not app_path.exists():
        print(f"Error: Could not find the streamlit app at {app_path}", file=sys.stderr)
        sys.exit(1)
        
    print("Starting Promotion Analytics AI Assistant...")
    print("Press Ctrl+C to stop.")
    
    try:
        # Run streamlit in a subprocess
        subprocess.run(
            [sys.executable, "-m", "streamlit", "run", str(app_path)],
            cwd=str(project_root),
            check=True
        )
    except KeyboardInterrupt:
        print("\nShutting down...")
    except subprocess.CalledProcessError as e:
        print(f"\nError running the application: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
