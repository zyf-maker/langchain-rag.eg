# 向量存储工具模块
import os
import sys
import logging

# 导入混合检索模块
try:
    from utils.retrieval_utils import initialize_hybrid_retriever, hybrid_search, get_hybrid_retriever, close_hybrid_retriever
except ImportError:
    logging.warning("无法导入混合检索模块，将在需要时动态导入")

# 配置基本日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 确保使用官方的RecursiveCharacterTextSplitter，不再使用自定义实现
try:
    # 尝试从langchain-text-splitters导入（适用于新版langchain）
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        logger.info("成功从langchain_text_splitters导入RecursiveCharacterTextSplitter")
    except ImportError:
        # 尝试从langchain导入（适用于旧版langchain）
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        logger.info("成功从langchain导入RecursiveCharacterTextSplitter")
except ImportError as e:
    logger.error(f"导入文本分割器失败: {e}")
    logger.error("请安装必要的包: pip install langchain langchain-text-splitters")
    # 临时提供一个简单的实现作为备选，确保程序能运行
    logger.warning("创建临时的文本分割器实现")
    class RecursiveCharacterTextSplitter:
        def __init__(self, chunk_size=500, chunk_overlap=50, length_function=len, separators=None):
            self.chunk_size = chunk_size
            self.chunk_overlap = chunk_overlap
            self.length_function = length_function if length_function else len
            self.separators = separators if separators else ["\n\n", "\n", " ", ""]
            logger.info(f"临时分割器初始化: chunk_size={chunk_size}, chunk_overlap={chunk_overlap}")
        
        def split_text(self, text):
            chunks = []
            start = 0
            text_length = self.length_function(text)
            
            while start < text_length:
                end = min(start + self.chunk_size, text_length)
                
                # 尝试在分隔符处分割
                for separator in self.separators:
                    if separator:
                        sep_idx = text.rfind(separator, start, end)
                        if sep_idx != -1 and sep_idx > start:
                            end = sep_idx + len(separator)
                            break
                
                chunks.append(text[start:end])
                start = end - self.chunk_overlap
                
                # 防止无限循环
                if start >= end:
                    start = end
                    
            logger.info(f"临时分割器处理完成，生成 {len(chunks)} 个块")
            return chunks

try:
    # 尝试导入向量存储
    from langchain_community.vectorstores import Chroma
    logger.info("成功导入Chroma")
except ImportError as e:
    logger.error(f"导入Chroma失败: {e}")
    # 创建一个简单的模拟类作为备选
    class MockChroma:
        def __init__(self, persist_directory=None):
            self.persist_directory = persist_directory
            self.documents = []
            self.embeddings = []
            self._collection = type('obj', (object,), {'count': lambda: len(self.documents)})
        
        @staticmethod
        def from_documents(documents, embedding, persist_directory=None):
            mock = MockChroma(persist_directory)
            mock.documents = documents
            return mock
        
        def as_retriever(self, search_type="similarity", search_kwargs=None):
            return self
        
        def invoke(self, query):
            return []
    
    Chroma = MockChroma
    logger.info("使用自定义的MockChroma")


def create_text_chunks(text, chunk_size=500, chunk_overlap=50):
    """
    将文本分割成多个块，针对大文件进行优化
    
    Args:
        text: 原始文本
        chunk_size: 每个块的大小（字符数）
        chunk_overlap: 块之间的重叠字符数
    
    Returns:
        list: 文本块列表
    """
    logger.info(f"开始处理文本，总长度: {len(text)} 字符")
    
    # 验证参数
    if not text:
        logger.warning("空文本，返回空列表")
        return []
    
    # 确保块大小合理
    if chunk_size < 100:
        chunk_size = 100
        logger.warning(f"块大小过小，调整为: {chunk_size}")
    
    # 确保重叠大小合理
    if chunk_overlap > chunk_size // 2:
        chunk_overlap = chunk_size // 2
        logger.warning(f"重叠过大，调整为: {chunk_overlap}")
    
    try:
        # 根据文本大小动态调整处理策略
        text_length = len(text)
        
        # 对于超大文本（>10MB），采用更稳健的分阶段处理
        if text_length > 10000000:
            logger.info(f"检测到超大文本({text_length/1000000:.1f}MB)，采用分阶段稳健处理策略")
            
            # 1. 先按章节/大段落分割
            major_sections = []
            current_section = []
            section_size = 0
            max_section_size = 5000000  # 每个主段落不超过5MB
            
            # 先按换行分割
            lines = text.split('\n')
            for line in lines:
                line_size = len(line) + 1  # +1 for the newline
                if section_size + line_size > max_section_size and current_section:
                    major_sections.append('\n'.join(current_section))
                    current_section = []
                    section_size = 0
                current_section.append(line)
                section_size += line_size
            
            if current_section:
                major_sections.append('\n'.join(current_section))
            
            logger.info(f"已将文本分割为 {len(major_sections)} 个主要段落")
            
            # 2. 对每个主要段落单独进行精细化分割
            all_chunks = []
            total_sections = len(major_sections)
            
            for i, section in enumerate(major_sections):
                logger.info(f"处理主要段落 {i+1}/{total_sections}，大小: {len(section)} 字符")
                
                # 对子段落使用官方分割器
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    length_function=len,
                    separators=["\n\n", "\n", " ", ""]
                )
                
                # 限制单次处理的文本大小
                if len(section) > 2000000:
                    logger.info("进一步分割超大段落")
                    # 将段落分成更小的部分
                    sub_sections = []
                    current_sub = []
                    sub_size = 0
                    sub_lines = section.split('\n')
                    
                    for line in sub_lines:
                        line_size = len(line) + 1
                        if sub_size + line_size > 1000000 and current_sub:
                            sub_sections.append('\n'.join(current_sub))
                            current_sub = []
                            sub_size = 0
                        current_sub.append(line)
                        sub_size += line_size
                    
                    if current_sub:
                        sub_sections.append('\n'.join(current_sub))
                    
                    # 处理每个子段落
                    for sub in sub_sections:
                        try:
                            sub_chunks = text_splitter.split_text(sub)
                            all_chunks.extend(sub_chunks)
                        except Exception as e:
                            logger.error(f"处理子段落时出错: {e}")
                            # 对子段落进行应急处理
                            all_chunks.extend(emergency_split(sub, chunk_size, chunk_overlap))
                else:
                    # 直接处理常规大小的段落
                    try:
                        section_chunks = text_splitter.split_text(section)
                        all_chunks.extend(section_chunks)
                    except Exception as e:
                        logger.error(f"处理段落 {i+1} 时出错: {e}")
                        # 对整个段落进行应急处理
                        all_chunks.extend(emergency_split(section, chunk_size, chunk_overlap))
                
                logger.info(f"已处理 {i+1}/{total_sections} 个主要段落，当前总块数: {len(all_chunks)}")
            
            logger.info(f"分阶段分割完成，共生成 {len(all_chunks)} 个块")
            return all_chunks
        
        # 对于中等大小的文本（1-10MB）
        elif text_length > 1000000:
            logger.info(f"检测到大文本({text_length/1000000:.1f}MB)，采用分段落处理")
            
            # 先按段落分割
            paragraphs = text.split('\n\n')
            all_chunks = []
            total_paras = len(paragraphs)
            
            # 对每个段落单独处理
            for i, para in enumerate(paragraphs):
                if len(para) > 0:  # 跳过空段落
                    # 为每个段落创建独立的分割器
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                        length_function=len,
                        separators=["\n\n", "\n", " ", ""]
                    )
                    
                    try:
                        para_chunks = text_splitter.split_text(para)
                        all_chunks.extend(para_chunks)
                    except Exception as e:
                        logger.error(f"处理段落时出错: {e}")
                        # 对失败的段落进行应急处理
                        all_chunks.extend(emergency_split(para, chunk_size, chunk_overlap))
                
                # 记录进度
                if (i + 1) % 200 == 0:
                    logger.info(f"已处理 {i + 1}/{total_paras} 个段落，当前总块数: {len(all_chunks)}")
            
            logger.info(f"分段落分割完成，共生成 {len(all_chunks)} 个块")
            return all_chunks
        
        # 常规文本直接使用官方分割器
        else:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len,
                separators=["\n\n", "\n", " ", ""]
            )
            
            chunks = text_splitter.split_text(text)
            logger.info(f"文本分割成功，生成 {len(chunks)} 个块")
            return chunks
            
    except MemoryError as e:
        logger.error(f"内存不足错误: {e}")
        # 应急处理：使用非常小的块大小
        return emergency_split(text, 200, 20)
    
    except Exception as e:
        logger.error(f"文本分割过程中发生错误: {e}")
        # 使用应急分割方法
        return emergency_split(text, chunk_size, chunk_overlap)

def emergency_split(text, chunk_size, chunk_overlap):
    """
    应急文本分割函数，当官方分割器失败时使用
    
    Args:
        text: 要分割的文本
        chunk_size: 块大小
        chunk_overlap: 重叠大小
    
    Returns:
        list: 分割后的文本块列表
    """
    logger.info(f"执行应急分割，块大小: {chunk_size}, 重叠: {chunk_overlap}")
    
    chunks = []
    text_length = len(text)
    start = 0
    
    # 安全的循环，防止无限循环
    max_iterations = (text_length // chunk_size) + 10  # 额外10次循环作为安全边界
    iteration = 0
    
    while start < text_length and iteration < max_iterations:
        end = min(start + chunk_size, text_length)
        
        # 尝试在自然边界分割
        if end < text_length:
            # 限制搜索范围，提高效率
            search_end = min(end + 20, text_length)
            for i in range(end - 1, max(start, end - 40), -1):
                if text[i:i+1] in ['.', '。', '!', '！', '?', '？', '\n', ' ']:
                    end = i + 1
                    break
        
        # 添加当前块
        try:
            chunk = text[start:end]
            if chunk.strip():  # 只添加非空块
                chunks.append(chunk)
        except Exception as e:
            logger.error(f"添加文本块失败: {e}")
        
        # 更新起始位置
        next_start = end - chunk_overlap
        # 防止无限循环
        if next_start >= start or next_start >= text_length:
            start = end  # 强制前进
        else:
            start = next_start
        
        iteration += 1
    
    # 安全检查
    if iteration >= max_iterations:
        logger.warning("应急分割达到最大迭代次数，可能未完成完整分割")
    
    logger.info(f"应急分割完成，生成 {len(chunks)} 个块")
    return chunks

def create_vector_store(texts, embeddings, persist_directory=None, metadatas=None, neo4j_uri=None, neo4j_user=None, neo4j_password=None):
    """
    创建向量数据库
    
    Args:
        texts: 文本块列表
        embeddings: 嵌入模型实例
        persist_directory: 持久化目录路径
        metadatas: 元数据列表
        neo4j_uri: Neo4j数据库URI
        neo4j_user: Neo4j用户名
        neo4j_password: Neo4j密码
    
    Returns:
        Chroma: 向量存储实例
    """
    # 动态导入混合检索模块
    try:
        from utils.retrieval_utils import initialize_hybrid_retriever, get_hybrid_retriever
        from langchain_core.documents import Document
        hybrid_available = True
    except ImportError as e:
        logging.warning(f"混合检索模块导入失败: {e}")
        hybrid_available = False
    try:
        # 如果没有提供元数据，创建默认元数据
        if metadatas is None:
            metadatas = [{"source": f"chunk_{i}", "chunk_id": i} for i in range(len(texts))]
        
        # 创建向量存储
        vector_store = Chroma.from_texts(
            texts=texts,
            embedding=embeddings,
            metadatas=metadatas,
            persist_directory=persist_directory
        )
        
        # 如果指定了持久化目录，执行持久化
        if persist_directory:
            vector_store.persist()
        
        # 初始化并更新混合检索器
        if hybrid_available:
            try:
                # 创建Document对象列表
                documents = []
                for i, text in enumerate(texts):
                    metadata = metadatas[i] if metadatas and i < len(metadatas) else {}
                    document = Document(page_content=text, metadata=metadata)
                    documents.append(document)
                
                # 初始化混合检索器（支持Neo4j）
                neo4j_config = None
                if neo4j_uri and neo4j_user and neo4j_password:
                    neo4j_config = {
                        'uri': neo4j_uri,
                        'user': neo4j_user,
                        'password': neo4j_password
                    }
                    logging.info("Neo4j配置已提供，将使用Neo4j知识图谱")
                
                retriever = initialize_hybrid_retriever(vector_store, documents=documents, neo4j_config=neo4j_config)
                logging.info("成功更新混合检索器")
            except Exception as e:
                logging.warning(f"更新混合检索器失败: {e}")
        
        return vector_store, f"成功创建向量数据库，包含 {len(texts)} 个文档片段"
        
    except Exception as e:
        return None, f"创建向量数据库失败: {str(e)}"

def load_vector_store(embeddings, persist_directory):
    """
    加载现有的向量数据库
    
    Args:
        embeddings: 嵌入模型实例
        persist_directory: 持久化目录路径
    
    Returns:
        Chroma: 向量存储实例
    """
    try:
        # 检查目录是否存在且不为空
        if not os.path.exists(persist_directory):
            return None, "向量数据库目录不存在"
        
        if not os.listdir(persist_directory):
            return None, "向量数据库目录为空"
        
        # 加载向量存储
        vector_store = Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings
        )
        
        # 获取文档数量
        doc_count = vector_store._collection.count()
        if doc_count == 0:
            return None, "向量数据库中没有文档"
        
        return vector_store, f"成功加载向量数据库，包含 {doc_count} 个文档片段"
        
    except Exception as e:
        return None, f"加载向量数据库失败: {str(e)}"

def search_vector_store(vector_store, query, k=3, use_hybrid=True, weights=None):
    """
    在向量数据库中搜索相关文档
    
    Args:
        vector_store: 向量存储实例
        query: 查询文本
        k: 返回结果数量
        use_hybrid: 是否使用混合检索
        weights: 混合检索权重配置（仅在use_hybrid=True时有效）
    
    Returns:
        list: 相关文档列表
    """
    # 检查是否使用混合检索
    if use_hybrid:
        try:
            from utils.retrieval_utils import hybrid_search
            # 使用混合检索
            results, message = hybrid_search(
                query=query,
                vector_k=k,
                keyword_k=k,
                kg_k=k,
                rerank_top_k=k,
                weights=weights
            )
            return results, message
        except ImportError as e:
            logging.warning(f"混合检索不可用，回退到向量检索: {e}")
        except Exception as e:
            logging.warning(f"混合检索失败，回退到向量检索: {e}")
    
    # 默认使用向量检索
    try:
        # 使用相似度搜索
        results = vector_store.similarity_search(query, k=k)
        return results, f"成功检索到 {len(results)} 个相关文档"
        
    except Exception as e:
        return [], f"检索失败: {str(e)}"

def delete_vector_store(persist_directory):
    """
    删除向量数据库（删除持久化目录）
    
    Args:
        persist_directory: 持久化目录路径
    
    Returns:
        bool: 是否删除成功
    """
    try:
        import shutil
        if os.path.exists(persist_directory):
            shutil.rmtree(persist_directory)
            return True, "向量数据库已删除"
        return True, "向量数据库不存在"
    except Exception as e:
        return False, f"删除向量数据库失败: {str(e)}"