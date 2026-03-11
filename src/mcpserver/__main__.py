from pathlib import Path
from mcpserver.server import mcp


def main():
    mcp.run(transport="sse")
    #mcp.run(transport="stdio")


def load_prompt(name: str) -> str:
    return (Path(__file__).parent / "promts" / f"{name}.md").read_text()


if __name__ == "__main__":
    main()
    
