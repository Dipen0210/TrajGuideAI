from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent / "vehicle-trajectory-agent"

DIRECTORIES = [
    "data/raw",
    "data/processed",
    "data/external",
    "model",
    "agent/llm",
    "agent/tools",
    "agent/rag/processed_docs",
    "backend",
    "frontend/components",
    "notebooks",
    "tests",
]

FILES = [
    "model/dataset.py",
    "model/lstm_model.py",
    "model/train.py",
    "model/inference.py",
    "model/utils.py",
    "agent/llm/llama3_client.py",
    "agent/llm/llama3_config.yaml",
    "agent/tools/prediction_tool.py",
    "agent/tools/explanation_tool.py",
    "agent/tools/risk_assessment_tool.py",
    "agent/tools/context_query_tool.py",
    "agent/rag/build_vectorstore.py",
    "agent/rag/rag_chain.py",
    "agent/agent_pipeline.py",
    "backend/main.py",
    "backend/schemas.py",
    "frontend/app.py",
    "notebooks/EDA.ipynb",
    "notebooks/Trajectory_Visualization.ipynb",
    "notebooks/LSTM_Training.ipynb",
    "notebooks/Agent_Demo.ipynb",
    "tests/test_model.py",
    "tests/test_tools.py",
    "tests/test_agent.py",
    "README.md",
    "requirements.txt",
    "LICENSE",
]


def create_directories(base_path: Path) -> None:
    for relative in DIRECTORIES:
        directory = base_path / relative
        directory.mkdir(parents=True, exist_ok=True)


def create_files(base_path: Path) -> None:
    for relative in FILES:
        file_path = base_path / relative
        if not file_path.parent.exists():
            file_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            file_path.touch(exist_ok=False)
        except FileExistsError:
            continue


def main() -> None:
    PROJECT_ROOT.mkdir(exist_ok=True)
    create_directories(PROJECT_ROOT)
    create_files(PROJECT_ROOT)


if __name__ == "__main__":
    main()
