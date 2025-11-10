# 配置管理模块
import os
from dotenv import load_dotenv

# 加载.env文件中的环境变量
load_dotenv()

# 火山引擎配置
VOLC_API_KEY = os.getenv("VOLC_API_KEY", "")
# V3 API需要使用接入点ID作为model参数
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "ep-m-20250718174411-j9zsb")

# DeepSeek配置
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_API_URL = os.getenv("DEEPSEEK_API_URL", "https://api.deepseek.com/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-chat")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.1"))

# 向量数据库配置
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db")

# Neo4j知识图谱配置
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "12345678")

# 文本分割配置
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))

# 应用配置
APP_NAME = "民法万事通"
APP_ICON = "📚"
APP_DESCRIPTION = "基于PDF文档的法律知识库问答系统"

# 检查必要的配置是否存在
def check_config():
    """检查关键配置是否存在"""
    missing = []
    if not VOLC_API_KEY:
        missing.append("VOLC_API_KEY")
    if not DEEPSEEK_API_KEY:
        missing.append("DEEPSEEK_API_KEY")
    return missing