# LLM工具模块
import requests
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DeepSeekLLM:
    """
    DeepSeek语言模型封装类
    """
    
    def __init__(self, api_key, api_base, model_name="deepseek-chat", temperature=0.1, timeout=60):
        """
        初始化DeepSeek模型
        
        Args:
            api_key: DeepSeek API密钥
            api_base: DeepSeek API基础URL
            model_name: 模型名称
            temperature: 生成温度参数
            timeout: 请求超时时间（秒）
        """
        self.api_key = api_key
        self.api_base = api_base
        self.model_name = model_name
        self.temperature = temperature
        self.timeout = timeout
    
    def invoke(self, prompt):
        """
        调用DeepSeek API生成文本
        
        Args:
            prompt: 提示词
        
        Returns:
            str: 生成的文本
        """
        try:
            # 设置HTTP请求头
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            
            # 构建请求负载
            payload = {
                "model": self.model_name,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "temperature": self.temperature,
                "stream": False
            }
            
            # 发送POST请求到DeepSeek API
            response = requests.post(
                f"{self.api_base}/chat/completions",
                headers=headers,
                json=payload,
                timeout=self.timeout
            )
            
            # 检查响应状态码
            if response.status_code == 200:
                result = response.json()
                return result['choices'][0]['message']['content']
            else:
                logger.error(f"DeepSeek API错误: {response.status_code} - {response.text}")
                return f"抱歉，生成答案时出错: {response.text}"
                
        except requests.exceptions.Timeout:
            logger.error("DeepSeek API请求超时")
            return "抱歉，请求超时，请稍后重试"
        except requests.exceptions.ConnectionError:
            logger.error("DeepSeek API连接错误")
            return "抱歉，网络连接错误，请检查网络连接"
        except Exception as e:
            logger.error(f"调用DeepSeek API时出错: {str(e)}")
            return f"抱歉，生成答案时出错: {str(e)}"


def create_rag_prompt(context, question, system_prompt=None):
    """
    创建RAG提示词
    
    Args:
        context: 检索到的上下文信息
        question: 用户问题
        system_prompt: 系统提示词（可选）
    
    Returns:
        str: 格式化的提示词
    """
    # 默认系统提示词
    default_system = "你是一个专业的知识问答助手，擅长基于提供的上下文信息回答问题。"
    
    # 使用提供的系统提示词或默认值
    system = system_prompt if system_prompt else default_system
    
    # 构建提示词模板
    prompt_template = f"""{system}

请根据以下上下文信息回答问题。如果上下文信息不足以回答问题，请回答"知识库中未找到相关信息"。

上下文信息：
{context}

问题：
{question}

请提供准确、简洁的回答：
"""
    
    return prompt_template


def create_general_prompt(question):
    """
    创建通用提示词（当没有向量数据库时使用）
    
    Args:
        question: 用户问题
    
    Returns:
        str: 格式化的提示词
    """
    prompt_template = f"""
请回答以下问题，提供准确、简洁的信息：

问题：
{question}

请提供专业、客观的回答：
"""
    return prompt_template


def validate_api_key(api_key, provider="deepseek"):
    """
    验证API密钥是否有效（简单验证）
    
    Args:
        api_key: API密钥
        provider: 提供商名称
    
    Returns:
        bool: 是否有效
    """
    # 简单的格式验证
    if provider.lower() == "deepseek":
        # DeepSeek API密钥通常以"sk-"开头
        return api_key.startswith("sk-") and len(api_key) >= 30
    elif provider.lower() == "volc":
        # 火山引擎API密钥格式验证
        return len(api_key) >= 30  # 简单验证长度
    return False