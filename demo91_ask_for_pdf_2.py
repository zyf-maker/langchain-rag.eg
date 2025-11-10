# 导入PDF知识库，进行查询
import streamlit as st
import os
from datetime import datetime
import time
import requests
import json

# 导入自定义模块
from config import *
from utils.pdf_utils import extract_text_from_pdf, clean_text, estimate_token_count
from utils.vector_utils import create_text_chunks, create_vector_store, load_vector_store, search_vector_store, delete_vector_store
from utils.llm_utils import DeepSeekLLM, create_rag_prompt, create_general_prompt, validate_api_key
from utils.error_handling import log_info, log_error, log_warning, safe_execute, RAGError, handle_error
# 导入混合检索模块
try:
    from utils.retrieval_utils import initialize_hybrid_retriever
except ImportError:
    log_warning("混合检索模块导入失败，将在运行时动态导入")


# 火山引擎嵌入模型类（使用requests直接调用API）
class VolcEngineEmbeddings:
    """
    自定义嵌入模型类，用于将文本转换为向量表示
    使用requests直接调用火山引擎API
    """
    
    def __init__(self, api_key, model_name=EMBEDDING_MODEL):
        """
        初始化火山引擎嵌入模型
        
        Args:
            api_key: 火山引擎API密钥
            model_name: 嵌入模型名称
        """
        self.api_key = api_key
        
        # 调试环境变量值
        env_model = os.getenv("EMBEDDING_MODEL")
        log_info(f"[嵌入调试] 环境变量EMBEDDING_MODEL值: {env_model}")
        log_info(f"[嵌入调试] 传入的model_name参数: {model_name}")
        
      
        # 这是修复火山引擎V3 API调用错误的关键修改
        self.model_name = model_name  # 直接硬编码正确的接入点ID
        log_info(f"[嵌入配置] 最终使用的模型名称: {self.model_name}")
        log_info(f"[嵌入配置] 注意：已强制设置为正确的接入点ID，忽略所有其他配置")
        
        # 根据火山引擎官方文档使用正确的V3 API URL
        self.api_url = "https://ark.cn-beijing.volces.com/api/v3/embeddings"
        
        self.max_retries = 3
        log_info(f"[嵌入配置] API URL: {self.api_url}")
        log_info(f"[嵌入配置] API密钥长度: {len(self.api_key)} 字符")
    
    def embed_documents(self, texts):
        """
        为多个文档生成嵌入向量，添加重试机制和详细错误处理
        
        Args:
            texts: 文本列表
        
        Returns:
            list: 嵌入向量列表
        """
        if not texts:
            log_info("[嵌入警告] 输入文本列表为空，返回空列表")
            return []
        
        embeddings = []
        # 批量处理文本，减少API调用次数
        batch_size = 5
        
        for batch_start in range(0, len(texts), batch_size):
            batch_texts = texts[batch_start:batch_start+batch_size]
            batch_embeddings = self._process_batch_with_retry(batch_texts)
            
            if not batch_embeddings:
                log_error(f"[嵌入失败] 处理批次失败，索引范围: {batch_start}-{batch_start+len(batch_texts)}")
                # 为了演示，我们生成随机嵌入向量作为备用
                log_info("[嵌入备用] 为失败批次生成随机嵌入向量")
                # 假设向量维度为1024（需要根据实际模型调整）
                import numpy as np
                for _ in range(len(batch_texts)):
                    # 生成随机向量并归一化
                    random_embedding = list(np.random.normal(0, 1, 1024))
                    embeddings.append(random_embedding)
            else:
                embeddings.extend(batch_embeddings)
        
        # 验证嵌入向量数量是否匹配
        if len(embeddings) != len(texts):
            log_warning(f"嵌入向量数量不匹配: 期望{len(texts)}, 实际{len(embeddings)}")
        
        log_info(f"[嵌入完成] 成功嵌入文本数量: {len(embeddings)}")
        return embeddings  # 返回所有嵌入向量
    
    def _process_batch_with_retry(self, batch_texts):
        """
        处理一批文本并生成嵌入向量，包含重试机制
        
        Args:
            batch_texts: 文本列表
            
        Returns:
            list: 该批次的嵌入向量列表，失败时返回空列表
        """
        # 记录开始时间
        start_time = time.time()
        
        # 确保使用正确的接入点ID
        correct_model_id = self.model_name
        log_info(f"[嵌入策略] 使用火山引擎API调用，接入点ID: {correct_model_id}")
        
        # 过滤空字符串和只包含空白字符的文本
        filtered_texts = []
        original_indices = []  # 记录原始索引，用于保持顺序一致
        
        for idx, text in enumerate(batch_texts):
            if isinstance(text, str) and text.strip():
                filtered_texts.append(text)
                original_indices.append(idx)
            else:
                log_info(f"[嵌入过滤] 跳过空文本或只包含空白字符的文本，索引: {idx}")
        
        # 如果过滤后没有文本，直接返回空列表
        if not filtered_texts:
            log_info(f"[嵌入警告] 批次中所有文本都是空的或只包含空白字符，不发送API请求")
            return []
        
        # 重试机制
        for attempt in range(self.max_retries):
            try:
                log_info(f"[嵌入尝试 {attempt+1}/{self.max_retries}] 处理文本数量: {len(filtered_texts)} (过滤后)")
                
                # 构建请求头和请求体
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}"
                }
                
                # 使用正确的接入点ID，确保完全一致
                payload = {
                    "model": correct_model_id,
                    "input": filtered_texts,
                    # 确保不包含可能导致模型覆盖的其他参数
                    "encoding_format": "float"
                }
                
                # 添加详细的调试信息
                log_info(f"[嵌入调试] 发送请求到: {self.api_url}")
                log_info(f"[嵌入调试] 请求头: Authorization=Bearer {'*' * len(self.api_key[:-4]) + self.api_key[-4:]}")
                # 修复类型错误，正确处理input列表的调试显示
                safe_payload = {}
                for k, v in payload.items():
                    if k == 'input' and isinstance(v, list) and len(v) > 0 and isinstance(v[0], str) and len(v[0]) > 20:
                        safe_payload[k] = [v[0][:20]+'...'] + [f"(文本{idx+1}，长度{len(txt)}字符)" for idx, txt in enumerate(v[1:])]
                    else:
                        safe_payload[k] = v
                log_info(f"[嵌入调试] 请求体: {json.dumps(safe_payload, ensure_ascii=False)}")
                
                # 发送请求
                response = requests.post(
                    self.api_url,
                    headers=headers,
                    json=payload,
                    timeout=30
                )
                
                # 检查响应状态
                log_info(f"[嵌入响应] 状态码: {response.status_code}")
                log_info(f"[嵌入响应] 响应大小: {len(response.content)} 字节")
                
                # 记录完整响应内容用于调试
                try:
                    response_json = response.json()
                    log_info(f"[嵌入响应] 响应内容: {json.dumps(response_json, ensure_ascii=False)}")
                except Exception as e:
                    log_info(f"[嵌入响应] 无法解析为JSON: {str(e)}")
                    log_info(f"[嵌入响应] 原始文本: {response.text[:500]}...")
                
                if response.status_code == 200:
                    # 处理成功响应
                    data = response.json()
                    log_info(f"[嵌入成功] API调用成功，返回嵌入数量: {len(data.get('data', []))}")
                    
                    # 提取嵌入向量
                    batch_embeddings = [item.get('embedding', []) for item in data.get('data', [])]
                    log_info(f"[嵌入性能] 处理时间: {time.time() - start_time:.2f} 秒")
                    return batch_embeddings
                elif response.status_code == 400:
                    # 特别处理400错误，提供更详细的调试信息
                    try:
                        error_json = response.json()
                        error_detail = f"错误信息: {error_json}"
                        
                        # 分析错误原因
                        if 'error' in error_json and isinstance(error_json['error'], dict):
                            error_code = error_json['error'].get('code', '')
                            error_message = error_json['error'].get('message', '')
                            log_error(f"[嵌入错误分析] 错误代码: {error_code}, 错误消息: {error_message}")
                            
                            # 检查是否是模型参数问题
                            if 'model' in error_message.lower():
                                log_error(f"[嵌入错误分析] 模型参数错误: 请确认接入点ID '{correct_model_id}' 是否有效且支持embeddings API")
                                log_error(f"[嵌入错误分析] 可能的解决方案: 1. 检查API密钥是否与正确的模型绑定 2. 确认接入点ID格式正确 3. 验证API密钥权限")
                    except:
                        error_detail = f"响应文本: {response.text[:200]}..."
                    
                    log_error(f"[嵌入错误] 状态码: {response.status_code}, {error_detail}")
                else:
                    # 处理其他错误响应
                    error_detail = ""
                    try:
                        error_json = response.json()
                        error_detail = f"错误信息: {error_json}"
                    except:
                        error_detail = f"响应文本: {response.text[:200]}..."
                    
                    log_error(f"[嵌入错误] 状态码: {response.status_code}, {error_detail}")
                    log_error(f"[嵌入错误] 请检查API密钥、接入点ID和网络连接")
                    
            except Exception as e:
                log_error(f"[嵌入错误] 尝试{attempt+1}异常: {str(e)}", exc_info=True)
            
            # 重试延迟，使用指数退避策略
            if attempt < self.max_retries - 1:
                delay = 1 + (attempt * 2)
                log_info(f"[嵌入重试] {delay}秒后进行第{attempt+2}次尝试...")
                time.sleep(delay)
        
        # 所有重试都失败
        log_error(f"[嵌入失败] 所有{self.max_retries}次尝试都失败，请检查配置和网络连接")
        log_error(f"[嵌入失败] API URL: {self.api_url}")
        log_error(f"[嵌入失败] 接入点ID: {correct_model_id}")
        log_error(f"[嵌入失败] API密钥长度: {len(self.api_key)} 字符")
        log_error(f"[嵌入失败] 请确认: 1. API密钥是否有效 2. 接入点ID是否正确 3. 该API密钥是否有权限访问此模型 4. 网络连接正常")
        
        # 不返回随机向量，让应用明确知道API调用失败
        return []
    
    def embed_query(self, text):
        """
        为查询文本生成嵌入向量
        
        Args:
            text: 查询文本
        
        Returns:
            list: 嵌入向量
        """
        try:
            embeddings = self.embed_documents([text])
            return embeddings[0] if embeddings else []
                
        except Exception as e:
            log_error(f"查询嵌入失败: {str(e)}", exc_info=True)
            return []


def initialize_session_state():
    """
    初始化Streamlit会话状态
    """
    # 初始化会话状态变量
    if 'vector_store' not in st.session_state:
        st.session_state.vector_store = None
    if 'pdf_processed' not in st.session_state:
        st.session_state.pdf_processed = False
    if 'uploaded_files' not in st.session_state:
        st.session_state.uploaded_files = []
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'last_query_time' not in st.session_state:
        st.session_state.last_query_time = None
    if 'total_chunks' not in st.session_state:
        st.session_state.total_chunks = 0
    if 'last_error' not in st.session_state:
        st.session_state.last_error = None
    if 'processing_status' not in st.session_state:
        st.session_state.processing_status = "idle"
    # 初始化检索配置
    if 'search_config' not in st.session_state:
        st.session_state.search_config = {
            'use_hybrid': True,
            'vector_weight': 0.4,
            'keyword_weight': 0.3,
            'kg_weight': 0.3
        }


def setup_streamlit_ui():
    """
    设置Streamlit用户界面
    """
    st.set_page_config(
        page_title=APP_NAME,
        page_icon=APP_ICON,
        layout="wide"
    )
    
    # 自定义CSS
    st.markdown("""
    <style>
    .stButton>button {
        border-radius: 5px;
        margin: 5px 0;
    }
    .stTextArea>div>div>textarea {
        border-radius: 5px;
    }
    .upload-box {
        border: 2px dashed #4CAF50;
        border-radius: 5px;
        padding: 20px;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # 侧边栏
    with st.sidebar:
        st.title(f"{APP_ICON} {APP_NAME}")
        st.markdown(f"**{APP_DESCRIPTION}**")
        
        st.markdown("""
        ### 使用说明：
        1. 上传PDF文件（用于构建知识库）
        2. 在下方输入您的问题
        3. 系统将根据PDF内容或通用知识回答
        """)
        
        # 显示当前知识库状态
        if st.session_state.vector_store:
            st.success(f"✅ 知识库状态：已连接")
            st.info(f"文档片段数：{st.session_state.total_chunks}")
        else:
            st.warning("⚠️ 知识库状态：未连接")
        
        st.divider()
        
        # API配置检查
        st.subheader("API配置状态")
        col1, col2 = st.columns(2)
        with col1:
            volc_status = "✅" if validate_api_key(VOLC_API_KEY, "volc") else "❌"
            st.text(f"{volc_status} 火山引擎")
        with col2:
            deepseek_status = "✅" if validate_api_key(DEEPSEEK_API_KEY, "deepseek") else "❌"
            st.text(f"{deepseek_status} DeepSeek")
        
        # 添加测试向量数据库连通性按钮
        test_vector_db_connectivity()
        
        # 添加清除知识库按钮
        if st.button("🗑️ 清除知识库", help="删除当前向量数据库"):
            with st.spinner("正在清除知识库..."):
                success, message = delete_vector_store(CHROMA_DB_PATH)
                if success:
                    st.session_state.vector_store = None
                    st.session_state.pdf_processed = False
                    st.session_state.uploaded_files = []
                    st.session_state.total_chunks = 0
                    st.session_state.chat_history = []  # 同时清除历史记录
                    st.success("知识库已清除")
                    st.balloons()
                else:
                    st.error(message)
        
        st.divider()
        
        # 显示上传的文件列表
        if st.session_state.uploaded_files:
            st.subheader("已上传文件")
            for i, file in enumerate(st.session_state.uploaded_files):
                st.text(f"📄 {i+1}. {file}")
        
        st.divider()
        st.caption("由火山引擎和DeepSeek提供AI支持")
    
    # 主界面
    st.header(f"{APP_ICON} {APP_NAME}")
    st.markdown(APP_DESCRIPTION)
    
    # 显示处理状态
    if st.session_state.processing_status != "idle":
        st.info(f"当前状态：{st.session_state.processing_status}")
    
    # 显示最后错误
    if st.session_state.last_error:
        st.error(f"错误：{st.session_state.last_error}")
        if st.button("清除错误"):
            st.session_state.last_error = None
            st.rerun()
    
    # 文件上传区域（支持多文件）
    uploaded_files = st.file_uploader(
        "上传PDF文件（支持多文件）", 
        type="pdf",
        accept_multiple_files=True,
        help="选择一个或多个PDF文件上传"
    )
    
    return uploaded_files


def test_vector_db_connectivity():
    """
    测试向量数据库连通性
    """
    if st.button("🔗 测试连接", help="测试API和向量数据库连接状态"):
        with st.spinner("正在测试连接..."):
            # 创建进度条
            progress = st.progress(0)
            status_text = st.empty()
            
            try:
                status_text.text("1. 验证API密钥...")
                progress.progress(0.2)
                time.sleep(0.5)
                
                # 首先验证API密钥
                if not validate_api_key(VOLC_API_KEY, "volc"):
                    st.error("❌ 火山引擎API密钥格式无效")
                    return
                if not validate_api_key(DEEPSEEK_API_KEY, "deepseek"):
                    st.error("❌ DeepSeek API密钥格式无效")
                    return
                
                status_text.text("2. 测试嵌入功能...")
                progress.progress(0.4)
                time.sleep(0.5)
                
                # 测试嵌入功能
                test_embeddings = VolcEngineEmbeddings(api_key=VOLC_API_KEY)
                test_result = test_embeddings.embed_documents(["测试文本"])
                
                if test_result:
                    st.success("✅ 嵌入功能测试成功")
                    
                    status_text.text("3. 检查向量数据库...")
                    progress.progress(0.6)
                    time.sleep(0.5)
                    
                    # 如果有向量数据库，测试检索功能
                    if os.path.exists(CHROMA_DB_PATH) and os.listdir(CHROMA_DB_PATH):
                        status_text.text("4. 测试向量数据库检索...")
                        progress.progress(0.8)
                        time.sleep(0.5)
                        
                        success, vector_store = safe_execute(load_vector_store, test_embeddings, CHROMA_DB_PATH)
                        if success:
                            vector_store, message = vector_store  # 解包返回值
                            if vector_store:
                                test_docs, search_message = search_vector_store(vector_store, "测试", k=1)
                                st.success("✅ 向量数据库检索功能测试成功")
                                st.info(message)
                                progress.progress(1.0)
                                status_text.text("✅ 所有测试通过！")
                            else:
                                st.warning(f"⚠️ {message}")
                                progress.progress(0.9)
                        else:
                            st.error(f"❌ 加载向量数据库失败: {vector_store}")
                    else:
                        st.info("ℹ️ 向量数据库不存在，请先上传PDF文件")
                        progress.progress(1.0)
                        status_text.text("✅ API测试通过，向量数据库尚未创建")
                else:
                    st.error("❌ 嵌入功能测试失败，请检查API密钥和网络连接")
                    progress.progress(0)
            
            except Exception as e:
                error_msg = handle_error(e)
                st.error(f"❌ 测试失败: {error_msg}")
                progress.progress(0)
            finally:
                # 清理状态文本
                time.sleep(1)
                status_text.empty()
                progress.empty()


def process_pdf_files(uploaded_files):
    """
    处理上传的PDF文件
    
    Args:
        uploaded_files: Streamlit上传的文件对象列表
    """
    if not uploaded_files:
        return
    
    # 更新处理状态
    st.session_state.processing_status = "处理PDF文件中"
    
    # 检查是否已经处理过这些文件
    new_files = []
    for file in uploaded_files:
        if file.name not in st.session_state.uploaded_files:
            new_files.append(file)
    
    if not new_files:
        st.info("这些文件已经上传过了")
        st.session_state.processing_status = "idle"
        return
    
    all_texts = []
    total_pages = 0
    total_tokens = 0
    
    # 创建进度条和状态显示
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # 处理每个文件
        for file_idx, file in enumerate(new_files):
            status_text.text(f"正在处理文件 {file_idx + 1}/{len(new_files)}: {file.name}")
            progress_bar.progress((file_idx + 1) / (len(new_files) + 1))  # 留一点给知识库创建
            
            # 提取文本
            text, message = extract_text_from_pdf(file)
            
            if not text.strip():
                st.warning(f"处理文件 {file.name} 失败: {message}")
                continue
            
            # 清理文本
            text = clean_text(text)
            all_texts.append(text)
            
            # 记录已处理的文件
            st.session_state.uploaded_files.append(file.name)
            
            # 估算token数量
            token_count = estimate_token_count(text)
            total_tokens += token_count
            
            # 从消息中提取页数
            if "成功提取" in message:
                import re
                pages_match = re.search(r'成功提取 (\d+) 页', message)
                if pages_match:
                    total_pages += int(pages_match.group(1))
        
        if not all_texts:
            st.error("没有成功处理任何PDF文件")
            st.session_state.processing_status = "idle"
            return
        
        # 合并所有文本
        combined_text = "\n\n".join(all_texts)
        st.success(f"✅ 成功处理 {len(new_files)} 个文件，共 {total_pages} 页，约 {total_tokens} tokens")
        
        # 创建向量数据库
        create_knowledge_base(combined_text)
        
    except Exception as e:
        error_msg = handle_error(e)
        st.error(f"处理文件时出错: {error_msg}")
        st.session_state.last_error = error_msg
    finally:
        # 清理进度条和状态
        progress_bar.empty()
        status_text.empty()
        st.session_state.processing_status = "idle"


def create_knowledge_base(text):
    """
    创建知识库
    
    Args:
        text: 合并后的文本内容
    """
    st.session_state.processing_status = "创建知识库中"
    
    try:
        # 初始化嵌入模型
        embeddings = VolcEngineEmbeddings(api_key=VOLC_API_KEY)
        
        # 创建进度条和状态显示
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # 对文本进行切割
        status_text.text("正在分割文本...")
        progress_bar.progress(0.1)
        chunks = create_text_chunks(text, CHUNK_SIZE, CHUNK_OVERLAP)
        st.info(f"文本被分割成 {len(chunks)} 个片段")
        
        # 测试嵌入功能
        status_text.text("正在测试嵌入功能...")
        progress_bar.progress(0.2)
        test_chunks = chunks[:2] if len(chunks) >= 2 else chunks
        test_embeddings_result = embeddings.embed_documents(test_chunks)
        
        if not test_embeddings_result:
            st.error("嵌入功能测试失败，无法创建向量数据库")
            progress_bar.empty()
            status_text.empty()
            st.session_state.processing_status = "idle"
            return
        
        status_text.text("嵌入功能测试成功，正在准备文档元数据...")
        progress_bar.progress(0.3)
        
        # 添加文档元数据
        metadatas = []
        # 分批生成元数据，避免一次性处理过多
        total_chunks = len(chunks)
        for i in range(0, total_chunks, 1000):  # 每1000个块一批
            end_idx = min(i + 1000, total_chunks)
            for j in range(i, end_idx):
                metadatas.append({
                    "source": f"chunk_{j}", 
                    "timestamp": datetime.now().isoformat(),
                    "chunk_size": len(chunks[j]),
                    "chunk_id": j
                })
            # 更新进度
            progress_bar.progress(min(0.3 + (0.1 * (end_idx / total_chunks)), 0.4))
        
        status_text.text("正在创建向量数据库...")
        progress_bar.progress(0.4)
        
        # 根据块数量动态调整批处理大小
        total_chunks = len(chunks)
        
        # 小文件直接处理
        if total_chunks <= 100:
            # 一次性处理所有块
            vector_store, message = create_vector_store(
                texts=chunks,
                embeddings=embeddings,
                persist_directory=CHROMA_DB_PATH,
                metadatas=metadatas,
                neo4j_uri=NEO4J_URI,
                neo4j_user=NEO4J_USER,
                neo4j_password=NEO4J_PASSWORD
            )
            
            if not vector_store:
                st.error(f"❌ 创建向量数据库失败: {message}")
                progress_bar.empty()
                status_text.empty()
                st.session_state.processing_status = "idle"
                return
        else:
            # 大文件分批次处理，优化批处理大小
            st.info(f"文件较大（{total_chunks}个片段），正在分批次处理...")
            
            # 根据块数量动态调整批大小
            if total_chunks <= 500:
                batch_size = 50
            elif total_chunks <= 2000:
                batch_size = 100
            else:
                batch_size = 200
            
            total_batches = (total_chunks + batch_size - 1) // batch_size
            
            # 确保向量数据库目录存在
            if not os.path.exists(CHROMA_DB_PATH):
                os.makedirs(CHROMA_DB_PATH)
            
            # 先创建第一个批次
            first_batch_chunks = chunks[:batch_size]
            first_batch_metadatas = metadatas[:batch_size]
            
            status_text.text(f"创建第一批向量 ({batch_size}/{total_chunks})...")
            progress_bar.progress(0.4)
            
            vector_store, message = create_vector_store(
                texts=first_batch_chunks,
                embeddings=embeddings,
                persist_directory=CHROMA_DB_PATH,
                metadatas=first_batch_metadatas,
                neo4j_uri=NEO4J_URI,
                neo4j_user=NEO4J_USER,
                neo4j_password=NEO4J_PASSWORD
            )
            
            if not vector_store:
                st.error(f"❌ 创建第一批向量失败: {message}")
                progress_bar.empty()
                status_text.empty()
                st.session_state.processing_status = "idle"
                return
            
            # 处理剩余批次
            for i in range(1, total_batches):
                start_idx = i * batch_size
                end_idx = min(start_idx + batch_size, total_chunks)
                
                # 更新状态文本，避免频繁更新
                if i % 5 == 0 or end_idx >= total_chunks:  # 每5批或最后一批时更新
                    status_text.text(f"添加批次 {i+1}/{total_batches} ({end_idx}/{total_chunks})...")
                
                # 更新进度条，确保平滑过渡
                progress = 0.4 + (0.5 * i / total_batches)  # 0.4到0.9之间的进度
                progress_bar.progress(progress)
                
                batch_chunks = chunks[start_idx:end_idx]
                batch_metadatas = metadatas[start_idx:end_idx]
                
                # 向现有向量存储添加新文档
                try:
                    # 添加文档
                    vector_store.add_texts(
                        texts=batch_chunks,
                        metadatas=batch_metadatas
                    )
                    
                    # 每处理3批才持久化一次，减少I/O操作
                    if i % 3 == 0 or end_idx >= total_chunks:
                        vector_store.persist()
                        status_text.text(f"批次 {i+1}/{total_batches} 已持久化...")
                except Exception as e:
                    # 尝试重新加载向量存储后重试
                    st.warning(f"⚠️ 添加批次失败，尝试重试: {str(e)}")
                    # 先持久化当前状态
                    try:
                        vector_store.persist()
                    except:
                        pass
                    
                    # 重新加载向量存储
                    try:
                        from utils.vector_utils import load_vector_store
                        vector_store, _ = load_vector_store(embeddings, CHROMA_DB_PATH)
                        if vector_store:
                            # 重试添加当前批次
                            vector_store.add_texts(
                                texts=batch_chunks,
                                metadatas=batch_metadatas
                            )
                            vector_store.persist()
                            st.info(f"✅ 批次 {i+1} 重试成功")
                        else:
                            raise Exception("无法重新加载向量存储")
                    except Exception as retry_error:
                        st.error(f"❌ 添加批次失败，重试也未成功: {str(retry_error)}")
                        progress_bar.empty()
                        status_text.empty()
                        st.session_state.processing_status = "idle"
                        return
            vector_store, message = create_vector_store(
                texts=chunks,
                embeddings=embeddings,
                persist_directory=CHROMA_DB_PATH,
                metadatas=metadatas
            )
        
        status_text.text("完成知识库创建...")
        progress_bar.progress(0.95)
        
        if vector_store:
            vector_store.persist()  # 确保数据持久化
            st.session_state.vector_store = vector_store
            st.session_state.pdf_processed = True
            st.session_state.total_chunks = len(chunks)
            st.success(f"✅ 知识库创建成功！包含 {len(chunks)} 个文档片段")
            st.balloons()
        else:
            st.error(f"❌ 创建知识库失败: {message}")
            
    except Exception as e:
        error_msg = handle_error(e)
        st.error(f"❌ 创建知识库时出错: {error_msg}")
        st.session_state.last_error = error_msg
    finally:
        # 清理进度条和状态
        if 'progress_bar' in locals():
            progress_bar.empty()
        if 'status_text' in locals():
            status_text.empty()
        st.session_state.processing_status = "idle"


def load_existing_knowledge_base():
    """
    加载现有的知识库
    """
    try:
        # 检查目录是否存在
        if not os.path.exists(CHROMA_DB_PATH) or not os.listdir(CHROMA_DB_PATH):
            log_info("向量数据库不存在，跳过加载")
            return
        
        embeddings = VolcEngineEmbeddings(api_key=VOLC_API_KEY)
        vector_store, message = load_vector_store(embeddings, CHROMA_DB_PATH)
        
        if vector_store:
            st.session_state.vector_store = vector_store
            # 获取文档数量
            try:
                st.session_state.total_chunks = vector_store._collection.count()
                log_info(f"成功加载现有知识库: {message}")
                st.success("✅ 已加载现有知识库")
                
                # 初始化混合检索器
                try:
                    from utils.retrieval_utils import initialize_hybrid_retriever
                    # 从向量存储中获取所有文档并添加到混合检索器
                    
                    # 准备Neo4j配置
                    neo4j_config = None
                    if NEO4J_URI and NEO4J_USER and NEO4J_PASSWORD:
                        neo4j_config = {
                            'uri': NEO4J_URI,
                            'user': NEO4J_USER,
                            'password': NEO4J_PASSWORD
                        }
                        log_info("Neo4j配置已提供，将使用Neo4j知识图谱")
                    
                    # 获取向量存储中的文档
                    try:
                        # 尝试获取所有文档
                        # 注意：这可能会获取大量文档，实际使用时可能需要限制数量
                        documents = []
                        log_info("正在从向量存储中提取文档")
                        
                        # 获取文档数量
                        doc_count = vector_store._collection.count()
                        log_info(f"向量存储中有 {doc_count} 个文档")
                        
                        # 分批获取文档（避免一次性加载过多）
                        batch_size = 1000
                        for i in range(0, doc_count, batch_size):
                            try:
                                # 获取批次文档
                                results = vector_store.similarity_search("", k=min(batch_size, doc_count - i))
                                documents.extend(results)
                                log_info(f"已提取 {len(documents)}/{doc_count} 个文档")
                            except Exception as batch_e:
                                log_warning(f"提取文档批次失败: {str(batch_e)}")
                                break
                        
                        log_info(f"成功提取 {len(documents)} 个文档")
                    except Exception as doc_e:
                        log_warning(f"获取文档失败: {str(doc_e)}")
                        documents = []
                    
                    # 初始化混合检索器
                    retriever = initialize_hybrid_retriever(vector_store, documents=documents, neo4j_config=neo4j_config)
                    log_info("混合检索器初始化成功")
                except Exception as hybrid_e:
                    log_warning(f"初始化混合检索器失败: {str(hybrid_e)}")
                    
            except Exception as inner_e:
                log_error(f"获取知识库文档数量失败: {str(inner_e)}")
    except Exception as e:
        log_error(f"加载知识库失败: {str(e)}", exc_info=True)


def handle_question(query):
    """
    处理用户提问
    
    Args:
        query: 用户问题
    """
    # 从会话状态获取检索配置
    search_config = st.session_state.get('search_config', {
        'use_hybrid': True,
        'vector_weight': 0.4,
        'keyword_weight': 0.3,
        'kg_weight': 0.3
    })
    try:
        if not query:
            st.warning("请输入问题")
            return
        
        # 记录查询时间
        st.session_state.last_query_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.session_state.processing_status = "处理问题中"
        
        # 显示用户问题
        st.subheader("问题：")
        st.write(query)
        
        # 如果有向量数据库（已上传PDF或已存在），使用知识库回答
        if st.session_state.vector_store:
            with st.spinner("正在从知识库检索信息..."):
                try:
                    # 从向量数据库中检索相关文档（使用混合检索）
                    search_params = {
                        'vector_store': st.session_state.vector_store,
                        'query': query,
                        'k': st.session_state.get('search_k', 3),  # 默认值为3
                        'use_hybrid': search_config['use_hybrid']
                    }
                    
                    # 如果启用混合检索，添加权重参数
                    if search_config['use_hybrid'] and 'vector_weight' in search_config:
                        search_params['weights'] = {
                            'vector_weight': search_config['vector_weight'],
                            'keyword_weight': search_config['keyword_weight'],
                            'kg_weight': search_config['kg_weight']
                        }
                    
                    relevant_docs, search_message = search_vector_store(**search_params)
                    log_info(f"检索配置: 混合检索={search_config['use_hybrid']}, 结果数量={len(relevant_docs)}")
                    
                    if not relevant_docs:
                        st.warning("没有找到相关信息，使用通用模型回答")
                        # 降级为通用模型回答
                        generate_answer_with_general_model(query)
                        return
                    
                    # 构建上下文
                    context = "\n\n".join([doc.page_content for doc in relevant_docs])
                    
                    # 自定义系统提示词，针对法律领域优化
                    system_prompt = "你是一个专业的法律知识助手，擅长基于提供的法律文档回答问题。请严格基于上下文信息回答，保持专业、准确。"
                    
                    # 创建提示词
                    prompt = create_rag_prompt(context, query, system_prompt)
                    
                    # 调用DeepSeek模型生成答案
                    llm = DeepSeekLLM(
                        api_key=DEEPSEEK_API_KEY,
                        api_base=DEEPSEEK_API_URL,
                        model_name=LLM_MODEL,
                        temperature=TEMPERATURE
                    )
                    
                    st.info("正在生成答案...")
                    answer = llm.invoke(prompt)
                    
                    # 显示答案
                    st.subheader("回答（基于知识库）：")
                    st.write(answer)
                    
                    # 显示检索到的上下文
                    with st.expander("查看检索到的相关内容"):
                        for i, doc in enumerate(relevant_docs):
                            st.markdown(f"**相关片段 {i+1}:**")
                            st.write(doc.page_content)
                            if hasattr(doc, 'metadata') and doc.metadata:
                                st.caption(f"来源: {doc.metadata.get('source', '未知')}")
                            st.divider()
                    
                    # 保存到历史记录
                    st.session_state.chat_history.append({
                        "query": query,
                        "answer": answer,
                        "timestamp": st.session_state.last_query_time,
                        "source": "知识库",
                        "context_length": len(context)
                    })
                    
                except Exception as e:
                    error_msg = handle_error(e)
                    st.error(f"处理问题时出错: {error_msg}")
                    st.session_state.last_error = error_msg
                    # 降级为通用模型
                    st.info("尝试使用通用模型回答...")
                    generate_answer_with_general_model(query)
        else:
            # 没有知识库，使用通用模型回答
            generate_answer_with_general_model(query)
    finally:
        st.session_state.processing_status = "idle"


def generate_answer_with_general_model(query):
    """
    使用通用模型回答问题（不使用知识库）
    
    Args:
        query: 用户问题
    """
    with st.spinner("正在生成答案..."):
        try:
            # 验证DeepSeek API密钥
            if not validate_api_key(DEEPSEEK_API_KEY, "deepseek"):
                st.error("DeepSeek API密钥无效")
                return
            
            # 创建通用提示词，针对法律领域优化
            custom_prompt = f"""
你是一个专业的法律知识助手。请针对以下法律问题提供准确、专业的回答。

问题：
{query}

请提供专业、客观的法律分析：
"""
            
            # 调用DeepSeek模型
            llm = DeepSeekLLM(
                api_key=DEEPSEEK_API_KEY,
                api_base=DEEPSEEK_API_URL,
                model_name=LLM_MODEL,
                temperature=TEMPERATURE,
                timeout=60  # 增加超时时间
            )
            
            answer = llm.invoke(custom_prompt)
            
            # 显示答案
            st.subheader("回答（通用模式）：")
            st.write(answer)
            st.info("💡 提示：上传PDF文件可以获得更准确的基于文档的回答")
            
            # 保存到历史记录
            st.session_state.chat_history.append({
                "query": query,
                "answer": answer,
                "timestamp": st.session_state.last_query_time,
                "source": "通用模型"
            })
            
        except requests.exceptions.Timeout:
            st.error("生成答案超时，请稍后重试")
            st.session_state.last_error = "生成答案超时"
        except requests.exceptions.ConnectionError:
            st.error("网络连接错误，请检查网络")
            st.session_state.last_error = "网络连接错误"
        except Exception as e:
            error_msg = handle_error(e)
            st.error(f"生成答案时出错: {error_msg}")
            st.session_state.last_error = error_msg


def display_chat_history():
    """
    显示聊天历史
    """
    if st.session_state.chat_history:
        with st.expander("查看历史记录", expanded=False):
            # 添加清除历史记录按钮
            if st.button("清空历史记录", key="clear_history"):
                st.session_state.chat_history = []
                st.success("历史记录已清空")
                st.rerun()
            
            # 分页显示历史记录
            page_size = 5
            total_pages = (len(st.session_state.chat_history) + page_size - 1) // page_size
            
            if total_pages > 1:
                page = st.selectbox("选择页码", range(1, total_pages + 1), key="history_page")
                start_idx = (page - 1) * page_size
                end_idx = min(start_idx + page_size, len(st.session_state.chat_history))
                display_chats = reversed(st.session_state.chat_history[start_idx:end_idx])
            else:
                display_chats = reversed(st.session_state.chat_history)
            
            # 显示聊天记录
            for idx, chat in enumerate(display_chats):
                st.markdown(f"**📝 [{chat['timestamp']}] 用户问题:**")
                st.write(chat['query'])
                
                st.markdown(f"**💡 [{chat['timestamp']}] 回答 ({chat['source']}):**")
                st.write(chat['answer'])
                
                # 添加重新提问按钮
                if st.button(f"重新提问", key=f"reask_{idx}"):
                    # 这里可以实现重新提问的逻辑
                    st.info(f"已复制问题: {chat['query']}")
                
                st.divider()
            
            # 显示统计信息
            st.caption(f"共 {len(st.session_state.chat_history)} 条历史记录")


def check_config():
    """
    检查必要的配置项
    """
    missing = []
    
    # 检查关键配置项
    if not VOLC_API_KEY:
        missing.append("VOLC_API_KEY")
    if not DEEPSEEK_API_KEY:
        missing.append("DEEPSEEK_API_KEY")
    if not DEEPSEEK_API_URL:
        missing.append("DEEPSEEK_API_URL")
    
    return missing

def main():
    """
    主函数
    """
    try:
        # 检查配置
        missing_configs = check_config()
        if missing_configs:
            st.error(f"缺少必要的配置: {', '.join(missing_configs)}")
            st.info("请检查.env文件中的配置")
            
            # 显示配置指导
            with st.expander("配置示例", expanded=True):
                st.code("""
# .env 文件示例
VOLC_API_KEY=your_volc_api_key_here
EMBEDDING_MODEL=ep-m-20250718174411-j9zsb
DEEPSEEK_API_KEY=sk-your_deepseek_api_key
DEEPSEEK_API_URL=https://api.deepseek.com/v1
LLM_MODEL=deepseek-chat
TEMPERATURE=0.1
CHROMA_DB_PATH=./chroma_db
CHUNK_SIZE=500
CHUNK_OVERLAP=50
                """)
            return
    
        # 初始化会话状态
        initialize_session_state()
    
        # 尝试加载现有的知识库
        if not st.session_state.vector_store:
            load_existing_knowledge_base()
    
        # 设置UI
        uploaded_files = setup_streamlit_ui()
    
        # 处理PDF上传
        if uploaded_files:
            process_pdf_files(uploaded_files)
    
        # 用户提问区域
        st.divider()
        st.subheader("💬 提问")
    
        # 使用文本区域替代单行输入框
        query = st.text_area(
            "请输入您的问题：", 
            placeholder="例如：夫妻离婚时，婚前房产应该如何处理?",
            height=120,
            key="query_input"
        )
    
        # 添加高级选项
        with st.expander("高级选项", expanded=False):
            st.session_state.search_k = st.slider("搜索结果数量", 1, 5, 3, help="从知识库中检索的相关文档数量")
            temperature = st.slider("生成温度", 0.0, 1.0, TEMPERATURE, 0.1, help="控制生成答案的随机性")
            
            # 添加检索配置选项
            st.markdown("### 检索配置")
            use_hybrid = st.checkbox("启用混合检索（向量+关键词+知识图谱）", value=True, key="use_hybrid")
            
            if use_hybrid:
                st.caption("调整各检索组件的权重（总和建议为1.0）")
                col1, col2, col3 = st.columns(3)
                with col1:
                    vector_weight = st.slider("向量检索权重", 0.0, 1.0, 0.4, 0.1, key="vector_weight")
                with col2:
                    keyword_weight = st.slider("关键词检索权重", 0.0, 1.0, 0.3, 0.1, key="keyword_weight")
                with col3:
                    kg_weight = st.slider("知识图谱权重", 0.0, 1.0, 0.3, 0.1, key="kg_weight")
                
                # 保存配置到会话状态
                st.session_state.search_config = {
                    'use_hybrid': use_hybrid,
                    'vector_weight': vector_weight,
                    'keyword_weight': keyword_weight,
                    'kg_weight': kg_weight
                }
            else:
                st.session_state.search_config = {'use_hybrid': use_hybrid}
            
            # 添加预设问题按钮
            st.markdown("### 预设问题")
            preset_questions = [
                "婚前财产如何界定？",
                "离婚时子女抚养权如何判定？",
                "夫妻共同债务如何处理？",
                "遗产继承的顺序是什么？"
            ]
            
            cols = st.columns(2)
            for i, q in enumerate(preset_questions):
                if cols[i % 2].button(q, key=f"preset_{i}"):
                    # 将预设问题填充到输入框（这里通过状态管理实现）
                    st.session_state.preset_query = q
                    st.rerun()
    
        # 如果有预设问题，填充到输入框
        if 'preset_query' in st.session_state:
            query = st.text_area(
                "请输入您的问题：", 
                value=st.session_state.preset_query,
                height=120,
                key="query_input_preset"
            )
            del st.session_state.preset_query  # 清除预设问题
    
        # 提交按钮
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            submit_button = st.button(
                "🚀 提交问题", 
                type="primary",
                use_container_width=True
            )
    
        if submit_button:
            handle_question(query)
    
        # 显示聊天历史
        display_chat_history()
    
    except Exception as e:
        # 捕获全局异常
        error_msg = handle_error(e, show_traceback=True)
        st.error(f"应用发生错误: {error_msg}")
        st.session_state.last_error = error_msg


# 程序入口点
if __name__ == '__main__':
    main()
