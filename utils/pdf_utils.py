# PDF处理工具模块
import os
import tempfile
from PyPDF2 import PdfReader

def validate_pdf(file_path):
    """
    验证文件是否为有效的PDF文件
    """
    try:
        # 直接使用PyPDF2尝试读取文件，这是最可靠的验证方法
        with open(file_path, 'rb') as f:
            # 尝试创建PdfReader对象，如果不是PDF文件会抛出异常
            PdfReader(f)
            return True
    except:
        # 如果读取失败或不是PDF格式，返回False
        return False

def extract_text_from_pdf(pdf_file):
    """
    从PDF文件中提取文本内容
    
    Args:
        pdf_file: PDF文件对象（Streamlit上传的文件）
    
    Returns:
        str: 提取的文本内容，如果失败则返回空字符串
    """
    try:
        # 创建临时文件
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(pdf_file.getvalue())
            tmp_file_path = tmp_file.name
        
        # 验证文件是否为有效的PDF
        if not validate_pdf(tmp_file_path):
            os.unlink(tmp_file_path)
            return "", "无效的PDF文件格式"
        
        # 读取PDF文件内容
        text = ""
        with open(tmp_file_path, 'rb') as f:
            pdf_reader = PdfReader(f)
            total_pages = len(pdf_reader.pages)
            
            for page_num, page in enumerate(pdf_reader.pages):
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        
        # 清理临时文件
        os.unlink(tmp_file_path)
        
        return text, f"成功提取 {total_pages} 页文本"
        
    except Exception as e:
        # 确保临时文件被清理
        if 'tmp_file_path' in locals() and os.path.exists(tmp_file_path):
            os.unlink(tmp_file_path)
        return "", f"PDF处理错误: {str(e)}"

def clean_text(text):
    """
    清理提取的文本，去除多余的空白字符等
    """
    # 去除多余的空白字符
    import re
    text = re.sub(r'\s+', ' ', text)
    # 去除首尾空白
    text = text.strip()
    return text

def estimate_token_count(text):
    """
    估算文本的token数量（粗略计算）
    """
    # 简单估算：英文按空格分割，中文按字符分割
    import re
    # 英文单词计数
    english_words = len(re.findall(r'\b\w+\b', text))
    # 中文字符计数
    chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
    # 粗略估算：英文单词每个约1.3token，中文字符每个约1token
    return int(english_words * 1.3 + chinese_chars)