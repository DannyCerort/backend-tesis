import asyncio
from fastmcp.client import Client

async def main():
    async with Client("http://127.0.0.1:8000/mcp/") as client:
        tools = await client.list_tools()
        print("TOOLS:")
        for t in tools:
            print("-", t.name)

        result = await client.call_tool("forecast_options", {})
        print(result)

asyncio.run(main())