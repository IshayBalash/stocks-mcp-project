from server import mcp # type: ignore



   
def main():
     mcp.run(transport="sse")



if __name__ == "__main__":
    main()
