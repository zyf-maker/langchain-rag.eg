# 高级检索工具模块
import os
import re
import logging
import numpy as np
from collections import defaultdict, Counter
from neo4j import GraphDatabase
import uuid

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 知识图谱类 - 基于Neo4j实现
class Neo4jKnowledgeGraph:
    """
    基于Neo4j的知识图谱实现，用于存储实体和它们之间的关系
    """
    
    def __init__(self, uri="neo4j://localhost:7687", user="neo4j", password="12345678"):
        """
        初始化Neo4j连接
        
        Args:
            uri: Neo4j数据库URI
            user: Neo4j用户名
            password: Neo4j密码
        """
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self._initialize_schema()
    
    def _initialize_schema(self):
        """
        初始化Neo4j数据库模式
        """
        with self.driver.session() as session:
            # 创建索引以提高查询性能
            session.run("CREATE INDEX IF NOT EXISTS FOR (e:Entity) ON (e.name)")
            session.run("CREATE INDEX IF NOT EXISTS FOR (d:Document) ON (d.id)")
            print("Neo4j知识图谱数据库模式初始化完成")
    
    def add_document(self, doc_id=None, text=""):
        """
        从文本中提取实体和关系并添加到Neo4j知识图谱
        
        Args:
            doc_id: 文档ID，如果为None则自动生成
            text: 文档文本
        """
        if not text:
            return
        
        # 生成文档ID
        if doc_id is None:
            doc_id = str(uuid.uuid4())
        
        # 提取实体
        entities = self._extract_entities(text)
        
        with self.driver.session() as session:
            # 创建文档节点
            session.run(
                "MERGE (d:Document {id: $doc_id, content: $content})",
                doc_id=doc_id, content=text
            )
            
            # 创建实体节点并建立与文档的关系
            for entity in entities:
                # 创建实体节点
                session.run(
                    "MERGE (e:Entity {name: $entity_name})",
                    entity_name=entity
                )
                
                # 建立文档与实体的关系
                session.run(
                    "MATCH (d:Document {id: $doc_id}), (e:Entity {name: $entity_name}) "
                    "MERGE (d)-[:MENTIONS]->(e)",
                    doc_id=doc_id, entity_name=entity
                )
            
            # 建立实体之间的关系
            for i, entity1 in enumerate(entities):
                for entity2 in entities[i+1:]:
                    # 创建实体之间的相关关系
                    session.run(
                        "MATCH (e1:Entity {name: $entity1}), (e2:Entity {name: $entity2}) "
                        "MERGE (e1)-[:RELATED_TO]->(e2)",
                        entity1=entity1, entity2=entity2
                    )
    
    def _extract_entities(self, text):
        """
        从文本中提取实体
        
        Args:
            text: 输入文本
            
        Returns:
            list: 实体列表
        """
        # 使用简单的规则提取实体（可以替换为更复杂的NLP技术）
        entities = []
        
        # 提取法律术语（这里使用一些常见的法律术语作为示例）
        legal_terms = [
            "合同", "协议", "婚姻", "离婚", "财产", "继承", "债务", "赔偿",
            "抚养权", "监护权", "遗嘱", "遗产", "婚前财产", "共同财产",
            "侵权", "违约", "诉讼", "仲裁", "证据", "证人"
        ]
        
        for term in legal_terms:
            if term in text:
                entities.append(term)
        
        # 提取可能的人名（简化版）
        chinese_names = re.findall(r'[\u4e00-\u9fff]{2,4}(?:先生|女士|律师|法官)', text)
        entities.extend(chinese_names)
        
        # 去重
        return list(set(entities))
    
    def get_related_docs(self, query_entities, top_k=3):
        """
        根据查询中的实体获取相关文档
        
        Args:
            query_entities: 查询中的实体列表
            top_k: 返回的文档数量
            
        Returns:
            list: 文档ID列表
        """
        if not query_entities:
            return []
        
        with self.driver.session() as session:
            # 构建查询语句
            query = """
            MATCH (d:Document)-[:MENTIONS]->(e:Entity)
            WHERE e.name IN $entities
            WITH d, count(e) as score
            ORDER BY score DESC
            LIMIT $top_k
            RETURN d.id as doc_id, score
            """
            
            result = session.run(query, entities=query_entities, top_k=top_k)
            return [record["doc_id"] for record in result]
    
    def extract_query_entities(self, query):
        """
        从查询中提取实体
        
        Args:
            query: 查询文本
            
        Returns:
            list: 实体列表
        """
        return self._extract_entities(query)
    
    def search(self, query, k=3):
        """
        搜索与查询相关的文档
        
        Args:
            query: 查询文本
            k: 返回结果数量
            
        Returns:
            list: 文档列表
        """
        # 从查询中提取实体
        query_entities = self.extract_query_entities(query)
        
        # 获取相关文档ID
        doc_ids = self.get_related_docs(query_entities, top_k=k)
        
        # 获取文档内容
        with self.driver.session() as session:
            docs = []
            for doc_id in doc_ids:
                result = session.run(
                    "MATCH (d:Document {id: $doc_id}) RETURN d.content as content",
                    doc_id=doc_id
                )
                for record in result:
                    # 创建简单的Document对象以保持兼容性
                    class SimpleDocument:
                        def __init__(self, content):
                            self.page_content = content
                            self.metadata = {"doc_id": doc_id}
                    
                    docs.append(SimpleDocument(record["content"]))
            
            return docs
    
    def close(self):
        """
        关闭Neo4j连接
        """
        if self.driver:
            self.driver.close()

# 创建一个兼容原接口的别名
SimpleKnowledgeGraph = Neo4jKnowledgeGraph

# 关键词检索类
class KeywordRetriever:
    """
    基于关键词的检索实现
    """
    
    def __init__(self):
        # 词项文档矩阵 {term: [doc_ids]}
        self.term_doc_matrix = defaultdict(list)
        # 文档ID到文档内容的映射
        self.docs = {}
    
    def add_document(self, doc_id, text):
        """
        添加文档到关键词检索索引
        
        Args:
            doc_id: 文档ID
            text: 文档文本
        """
        self.docs[doc_id] = text
        
        # 提取关键词并构建倒排索引
        terms = self._extract_terms(text)
        for term in terms:
            if doc_id not in self.term_doc_matrix[term]:
                self.term_doc_matrix[term].append(doc_id)
    
    def _extract_terms(self, text):
        """
        从文本中提取关键词
        
        Args:
            text: 输入文本
            
        Returns:
            list: 关键词列表
        """
        # 移除标点符号
        text = re.sub(r'[^\w\u4e00-\u9fff\s]', ' ', text)
        # 分词（简化版，中英文混合处理）
        terms = []
        
        # 提取中文词（每个汉字作为单独的词，或者可以使用更复杂的分词方法）
        chinese_chars = re.findall(r'[\u4e00-\u9fff]', text)
        terms.extend(chinese_chars)
        
        # 提取英文词和数字
        english_terms = re.findall(r'\b\w+\b', text)
        terms.extend(english_terms)
        
        return terms
    
    def search(self, query, top_k=3):
        """
        基于关键词搜索文档
        
        Args:
            query: 查询文本
            top_k: 返回的文档数量
            
        Returns:
            list: 文档ID和分数的列表 [(doc_id, score)]
        """
        query_terms = self._extract_terms(query)
        
        # 计算文档的词频-逆文档频率分数（简化版）
        doc_scores = defaultdict(float)
        term_counts = Counter(query_terms)
        
        for term, count in term_counts.items():
            if term in self.term_doc_matrix:
                doc_freq = len(self.term_doc_matrix[term])
                idf = np.log(len(self.docs) / (doc_freq + 1)) + 1  # 平滑处理
                
                for doc_id in self.term_doc_matrix[term]:
                    # 计算词频（简化版，使用是否出现）
                    tf = 1
                    doc_scores[doc_id] += tf * idf * count
        
        # 按分数排序并返回前top_k个文档
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_docs[:top_k]

# 混合检索器类
class HybridRetriever:
    """
    混合检索器，结合向量检索、关键词检索和Neo4j知识图谱检索
    """
    def __init__(self, vector_store=None, keyword_retriever=None, knowledge_graph=None):
        """
        初始化混合检索器
        
        Args:
            vector_store: 向量存储
            keyword_retriever: 关键词检索器
            knowledge_graph: Neo4j知识图谱实例
        """
        self.vector_store = vector_store
        self.keyword_retriever = keyword_retriever
        self.knowledge_graph = knowledge_graph if knowledge_graph else SimpleKnowledgeGraph()
        
    def search(self, query, k=3, weights=None):
        """
        执行混合检索 - 采用粗排+知识图谱增强+精排的策略
        
        Args:
            query: 查询文本
            k: 返回结果数量
            weights: 权重配置字典
            
        Returns:
            排序后的文档列表
        """
        # 使用固定的向量:关键词=6:4权重进行粗排
        rough_weights = {
            'vector_weight': 0.6,
            'keyword_weight': 0.4,
            'kg_weight': 0.0  # 粗排阶段不使用知识图谱权重
        }
        
        # 标准化权重
        total_weight = sum(rough_weights.values())
        if total_weight > 0:
            rough_weights = {key: val/total_weight for key, val in rough_weights.items()}
        
        # 1. 执行向量检索
        vector_results = []
        try:
            vector_results = self.vector_store.similarity_search(query, k=k*3)  # 获取更多结果用于粗排
            logger.debug(f"向量检索返回 {len(vector_results)} 个结果")
        except Exception as e:
            logger.error(f"向量检索失败: {str(e)}")
        
        # 2. 执行关键词检索
        keyword_results = []
        try:
            # 获取关键词检索结果
            keyword_search_results = self.keyword_retriever.search(query, k=k*3)
            # 将(doc_id, score)格式转换为Document对象列表
            for doc_id, _ in keyword_search_results:
                if hasattr(self.keyword_retriever, 'docs') and doc_id in self.keyword_retriever.docs:
                    # 创建简单的Document对象
                    class SimpleDocument:
                        def __init__(self, content):
                            self.page_content = content
                            self.metadata = {"doc_id": doc_id}
                    keyword_results.append(SimpleDocument(self.keyword_retriever.docs[doc_id]))
            logger.debug(f"关键词检索返回 {len(keyword_results)} 个结果")
        except Exception as e:
            logger.error(f"关键词检索失败: {str(e)}")
        
        # 3. 粗排：合并向量和关键词检索结果
        rough_results = self.rerank_results(
            query, vector_results, keyword_results, [],  # 粗排阶段知识图谱结果为空
            rough_weights['vector_weight'],
            rough_weights['keyword_weight'],
            0.0,  # 粗排阶段不使用知识图谱权重
            k*2  # 保留更多结果用于知识图谱增强
        )
        
        # 4. 对粗排结果进行知识图谱检索
        kg_match_results = []
        try:
            # 从查询中提取实体
            query_entities = self.knowledge_graph.extract_query_entities(query)
            logger.debug(f"从查询中提取到 {len(query_entities)} 个实体")
            
            if query_entities:
                # 获取与实体相关的文档ID
                related_doc_ids = self.knowledge_graph.get_related_docs(query_entities, top_k=k*3)
                logger.debug(f"知识图谱检索返回 {len(related_doc_ids)} 个相关文档ID")
                
                # 找出粗排结果中与知识图谱匹配的文档
                for doc in rough_results:
                    # 检查文档是否在知识图谱检索结果中
                    # 使用文档内容的前100个字符作为标识进行匹配
                    doc_content = doc.page_content[:100] if hasattr(doc, 'page_content') else str(doc)
                    
                    # 获取文档在知识图谱中的ID
                    doc_in_kg = False
                    try:
                        # 尝试通过知识图谱搜索找出是否包含该文档
                        kg_search_results = self.knowledge_graph.search(query, k=k*3)
                        for kg_doc in kg_search_results:
                            kg_content = kg_doc.page_content[:100] if hasattr(kg_doc, 'page_content') else str(kg_doc)
                            if doc_content == kg_content:
                                doc_in_kg = True
                                break
                    except Exception as inner_e:
                        logger.warning(f"检查文档是否在知识图谱中时出错: {str(inner_e)}")
                    
                    if doc_in_kg:
                        kg_match_results.append(doc)
                
                logger.debug(f"粗排结果中有 {len(kg_match_results)} 个文档与知识图谱匹配")
        except Exception as e:
            logger.error(f"知识图谱增强检索失败: {str(e)}")
        
        # 5. 精排：提升知识图谱匹配文档的排名
        # 创建精排权重
        final_weights = {
            'rough_weight': 0.6,  # 粗排结果的权重
            'kg_boost_weight': 0.4  # 知识图谱匹配的提升权重
        }
        
        # 构建文档分数映射
        doc_scores = {}
        doc_map = {}
        
        # 为粗排结果分配基础分数
        for i, doc in enumerate(rough_results):
            content_key = doc.page_content[:100] if hasattr(doc, 'page_content') else str(doc)
            # 基础分数：位置越靠前，分数越高
            base_score = final_weights['rough_weight'] * (1.0 / (i + 1))
            
            # 如果文档在知识图谱匹配结果中，提升其分数
            if doc in kg_match_results:
                kg_boost = final_weights['kg_boost_weight']
                base_score += kg_boost
                logger.debug(f"文档 {i+1} 获得知识图谱提升，总分数: {base_score}")
            
            doc_scores[content_key] = base_score
            doc_map[content_key] = doc
        
        # 按分数排序
        sorted_docs = sorted(
            [(score, doc) for content_key, score in doc_scores.items() for doc in [doc_map[content_key]]],
            key=lambda x: x[0],
            reverse=True
        )
        
        # 返回前k个文档
        final_results = [doc for _, doc in sorted_docs[:k]]
        logger.debug(f"最终精排后返回 {len(final_results)} 个结果")
        
        return final_results
        
    def rerank_results(self, query, vector_results, keyword_results, kg_results,
                      vector_weight, keyword_weight, kg_weight, k):
        """
        对混合检索结果进行重排序
        
        Args:
            query: 查询文本
            vector_results: 向量检索结果
            keyword_results: 关键词检索结果
            kg_results: 知识图谱检索结果
            vector_weight: 向量检索权重
            keyword_weight: 关键词检索权重
            kg_weight: 知识图谱检索权重
            k: 返回结果数量
            
        Returns:
            排序后的文档列表
        """
        # 为每个文档计算综合分数
        doc_scores = {}
        doc_map = {}
        
        # 计算向量检索分数
        if vector_results and vector_weight > 0:
            for i, doc in enumerate(vector_results):
                # 使用文档内容作为唯一标识
                content_key = doc.page_content[:100] if hasattr(doc, 'page_content') else str(doc)
                # 使用递减的分数权重，靠前的结果分数更高
                doc_scores[content_key] = doc_scores.get(content_key, 0) + vector_weight * (1.0 / (i + 1))
                doc_map[content_key] = doc
        
        # 计算关键词检索分数
        if keyword_results and keyword_weight > 0:
            for i, doc in enumerate(keyword_results):
                content_key = doc.page_content[:100] if hasattr(doc, 'page_content') else str(doc)
                doc_scores[content_key] = doc_scores.get(content_key, 0) + keyword_weight * (1.0 / (i + 1))
                doc_map[content_key] = doc
        
        # 计算知识图谱检索分数
        if kg_results and kg_weight > 0:
            for i, doc in enumerate(kg_results):
                content_key = doc.page_content[:100] if hasattr(doc, 'page_content') else str(doc)
                doc_scores[content_key] = doc_scores.get(content_key, 0) + kg_weight * (1.0 / (i + 1))
                doc_map[content_key] = doc
        
        # 特殊情况：如果所有检索都失败，尝试只使用向量检索作为后备
        if not doc_scores and hasattr(self.vector_store, 'similarity_search'):
            try:
                fallback_results = self.vector_store.similarity_search(query, k=k)
                return fallback_results
            except Exception as e:
                logger.error(f"后备向量检索失败: {str(e)}")
        
        # 按分数排序
        sorted_docs = sorted(
            [(score, doc) for content_key, score in doc_scores.items() for doc in [doc_map[content_key]]],
            key=lambda x: x[0],
            reverse=True
        )
        
        # 返回前k个文档
        return [doc for _, doc in sorted_docs[:k]]

# 全局混合检索器实例
hybrid_retriever = None

def initialize_hybrid_retriever(vector_store, documents=None, neo4j_config=None, neo4j_uri="neo4j://localhost:7687", neo4j_user="neo4j", neo4j_password="12345678"):
    """
    初始化混合检索器
    
    Args:
        vector_store: 向量存储
        documents: 文档列表，用于初始化关键词检索器和知识图谱
        neo4j_config: Neo4j配置字典（包含uri、user、password）
        neo4j_uri: Neo4j数据库URI（当neo4j_config未提供时使用）
        neo4j_user: Neo4j用户名（当neo4j_config未提供时使用）
        neo4j_password: Neo4j密码（当neo4j_config未提供时使用）
        
    Returns:
        HybridRetriever实例
    """
    global hybrid_retriever
    
    # 获取所有文档以初始化关键词检索器和知识图谱
    try:
        # 如果没有提供文档，尝试从向量存储中获取
        all_docs = []
        if documents:
            all_docs = documents
        else:
            try:
                # 对于ChromaDB，获取集合中的所有文档
                if hasattr(vector_store, '_collection'):
                    # 获取所有文档ID
                    result = vector_store._collection.get()
                    if 'documents' in result and isinstance(result['documents'], list):
                        all_docs = result['documents']
                    else:
                        # 尝试通过范围查询获取文档
                        try:
                            all_docs = vector_store.similarity_search("查询所有文档", k=1000)
                        except:
                            pass
                else:
                    # 尝试通过查询获取文档（可能不完美）
                    try:
                        all_docs = vector_store.similarity_search("", k=1000)
                    except:
                        pass
            except Exception as e:
                logger.error(f"获取文档失败: {str(e)}")
        
        # 如果没有获取到文档，至少添加一个空文档以确保检索器可以初始化
        if not all_docs:
            logger.warning("未获取到文档，使用空文档初始化检索器")
            all_docs = ["初始化文档"]
        
        # 初始化关键词检索器
        keyword_retriever = KeywordRetriever()
        for idx, doc in enumerate(all_docs):
            try:
                # 确保doc是字符串类型
                doc_id = f"doc_{idx}"
                if isinstance(doc, str):
                    keyword_retriever.add_document(doc_id, doc)
                elif hasattr(doc, 'page_content'):
                    keyword_retriever.add_document(doc_id, doc.page_content)
            except Exception as doc_e:
                logger.error(f"添加文档到关键词检索器失败: {str(doc_e)}")
        
        # 初始化Neo4j知识图谱
        try:
            # 优先使用neo4j_config参数
            if neo4j_config and isinstance(neo4j_config, dict):
                knowledge_graph = SimpleKnowledgeGraph(
                    uri=neo4j_config.get('uri', neo4j_uri),
                    user=neo4j_config.get('user', neo4j_user),
                    password=neo4j_config.get('password', neo4j_password)
                )
            else:
                knowledge_graph = SimpleKnowledgeGraph(uri=neo4j_uri, user=neo4j_user, password=neo4j_password)
        except Exception as e:
            logger.warning(f"Neo4j知识图谱初始化失败: {str(e)}，将使用空知识图谱继续")
            knowledge_graph = None
            
        if knowledge_graph:
            for idx, doc in enumerate(all_docs):
                try:
                    # 确保doc是字符串类型
                    doc_id = f"doc_{idx}"
                    if isinstance(doc, str):
                        knowledge_graph.add_document(doc_id, doc)
                    elif hasattr(doc, 'page_content'):
                        knowledge_graph.add_document(doc_id, doc.page_content)
                    logger.debug(f"文档 {doc_id} 已添加到Neo4j知识图谱")
                except Exception as doc_e:
                    logger.error(f"添加文档到Neo4j知识图谱失败: {str(doc_e)}")
        
        # 创建并返回混合检索器
        hybrid_retriever = HybridRetriever(vector_store, keyword_retriever, knowledge_graph)
        logger.info(f"混合检索器初始化成功，处理文档数: {len(all_docs)}")
        return hybrid_retriever
    except Exception as e:
        logger.error(f"初始化混合检索器失败: {str(e)}")
        # 返回一个基础的混合检索器，可能无法正常工作
        return HybridRetriever(vector_store, KeywordRetriever(), SimpleKnowledgeGraph())

def get_hybrid_retriever():
    """
    获取混合检索器实例
    
    Returns:
        HybridRetriever: 混合检索器实例
    """
    global hybrid_retriever
    return hybrid_retriever

def hybrid_search(query, k=3, weights=None):
    """
    执行混合检索的便捷函数
    
    Args:
        query: 查询文本
        k: 返回结果数量
        weights: 权重配置字典，包含vector_weight, keyword_weight, kg_weight
        
    Returns:
        检索结果列表
    """
    retriever = get_hybrid_retriever()
    if retriever:
        return retriever.search(query, k=k, weights=weights)
    else:
        logger.warning("混合检索器未初始化")
        return []

def close_hybrid_retriever():
    """
    关闭混合检索器，释放资源
    """
    global hybrid_retriever
    if hybrid_retriever and hasattr(hybrid_retriever, 'knowledge_graph') and hasattr(hybrid_retriever.knowledge_graph, 'close'):
        try:
            hybrid_retriever.knowledge_graph.close()
            logger.info("Neo4j知识图谱连接已关闭")
        except Exception as e:
            logger.error(f"关闭Neo4j连接失败: {str(e)}")
    hybrid_retriever = None