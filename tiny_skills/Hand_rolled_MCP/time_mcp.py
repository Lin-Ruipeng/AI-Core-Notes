from mcp.server.mcpserver import MCPServer
import datetime

# 创建mcp实例
mcp = MCPServer("time-server")


# 注册tool
@mcp.tool()
def get_system_time(format: str = "iso") -> str:
    """获取本机当前的系统时间。format='iso'(默认)或者'human'"""
    now = datetime.datetime.now()
    return (
        now.isoformat(timespec="seconds")
        if format == "iso"
        else now.strftime("%Y年%m月%d日 %H:%M:%S")
    )


if __name__ == "__main__":
    mcp.run()  # 默认stdio格式
