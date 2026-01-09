"""
Symbol indexing interface for different storage backends.

This module provides an abstract interface for symbol caching and querying,
allowing different storage implementations (Kuzu, DuckDB, etc.) to be used
interchangeably.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Optional, Dict, List

from solidlsp import ls_types

log = logging.getLogger(__name__)


class SymbolIndex(ABC):
    """
    Abstract interface for symbol indexing and caching.
    
    Provides unified API for different storage backends (Kuzu, DuckDB, etc.)
    to store and query symbol information.
    
    Implementations must be thread-safe and handle the hierarchical nature
    of symbol trees, including circular references (parent-child relationships).
    """
    
    # ==================== 生命周期管理 ====================
    
    @abstractmethod
    def start(self) -> None:
        """
        启动索引连接并初始化存储结构。
        
        此方法应该：
        - 建立数据库连接
        - 初始化必要的表/模式
        - 准备好接收数据
        
        如果已经启动，应该是幂等操作（直接返回）。
        """
        pass
    
    @abstractmethod
    def stop(self) -> None:
        """
        停止索引连接并释放资源。
        
        此方法应该：
        - 关闭数据库连接
        - 释放内存资源
        - 清理临时文件（如果有）
        
        如果已经停止，应该是幂等操作（直接返回）。
        """
        pass
    
    @abstractmethod
    def is_started(self) -> bool:
        """
        检查索引是否已启动。
        
        :return: True 如果索引已启动并可用，否则 False
        """
        pass
    
    # ==================== 核心存储操作 ====================
    
    @abstractmethod
    def store_doc_symbols(
        self,
        relative_path: str,
        content_hash: str,
        root_symbols: List[ls_types.UnifiedSymbolInformation]
    ) -> None:
        """
        存储文档符号到索引。
        
        此方法应该：
        - 删除该文档的旧符号（如果存在）
        - 存储新的符号树
        - 记录内容哈希用于缓存验证
        - 以原子方式完成操作（事务支持）
        
        :param relative_path: 文档的相对路径
        :param content_hash: 文档内容的 MD5 哈希
        :param root_symbols: 根级别符号列表（包含完整的子树）
        :raises: 存储失败时抛出异常
        """
        pass
    
    @abstractmethod
    def get_doc_symbols(
        self,
        relative_path: str,
        content_hash: str
    ) -> Optional[Dict[str, Any]]:
        """
        获取文档符号（带缓存验证）。
        
        此方法应该：
        - 检查文档是否存在于缓存
        - 验证内容哈希是否匹配
        - 如果有效，返回完整的符号树
        - 如果无效或不存在，返回 None
        
        :param relative_path: 文档的相对路径
        :param content_hash: 当前文档内容的 MD5 哈希
        :return: 字典包含 'file_symbol' 和 'content_hash'，或 None
        
        返回格式:
        {
            'file_symbol': {
                'id': str,
                'name': str,
                'kind': int,
                'relativePath': str,
                'children': List[...]  # 完整的符号树
            },
            'content_hash': str
        }
        """
        pass
    
    @abstractmethod
    def is_doc_cached(self, relative_path: str, content_hash: str) -> bool:
        """
        检查文档是否已缓存且有效。
        
        这是 get_doc_symbols 的轻量级版本，只检查缓存状态
        而不加载实际数据。
        
        :param relative_path: 文档的相对路径
        :param content_hash: 当前文档内容的 MD5 哈希
        :return: True 如果文档已缓存且哈希匹配，否则 False
        """
        pass
    
    # ==================== 查询操作 ====================
    
    @abstractmethod
    def query_symbols(
        self,
        name_path_regex: str,
        include_kinds: Optional[List[int]] = None,
        exclude_kinds: Optional[List[int]] = None,
        relative_path_regex: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Query symbols using regex pattern matching.
        
        :param name_path_regex: Regular expression to match symbol name_path
        :param include_kinds: Optional list of symbol kinds to include
        :param exclude_kinds: Optional list of symbol kinds to exclude
        :param relative_path_regex: Optional regex to match file relative_path
        :return: List of document symbols matching the criteria
        """
        pass
    

    
    # ==================== 缓存管理 ====================
    
    @abstractmethod
    def invalidate_doc(self, relative_path: str) -> None:
        """
        使指定文档的缓存失效（删除）。
        
        此方法应该删除该文档的所有符号和关系。
        
        :param relative_path: 要失效的文档相对路径
        """
        pass
    
    @abstractmethod
    def clear_all(self) -> None:
        """
        清除所有缓存数据。
        
        此方法应该删除所有文档和符号，但保留存储结构。
        """
        pass

    
    @abstractmethod
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        获取缓存统计信息。
        
        :return: 统计信息字典
        
        建议返回格式:
        {
            'total_docs': int,       # 缓存的文档总数
            'total_symbols': int,    # 符号总数
            'storage_size': int,     # 存储大小（字节）
            'backend': str           # 后端类型（如 "kuzu", "duckdb"）
        }
        """
        pass
    
    @abstractmethod
    def get_all_cached_docs(self) -> List[str]:
        """
        获取所有已缓存文档的相对路径列表。
        
        :return: 文档相对路径的列表
        """
        pass
    
    @abstractmethod
    def create_appender(self, batch_mode: bool = False) -> 'SymbolAppender':
        """
        创建一个符号追加器用于批量索引操作。
        
        :param batch_mode: 是否使用批量模式（True=批量累积后提交，False=每次立即提交）
        :return: SymbolAppender 实例
        
        批量模式通过累积多个文档的数据并一次性提交来提高性能，
        适用于初始化索引或批量更新的场景。
        
        非批量模式每次调用 append() 立即持久化，适用于增量更新。
        """
        pass


# ======================== Symbol Appender Interface ========================


class SymbolAppender(ABC):
    """
    Abstract interface for batch symbol appending.
    
    Provides a way to accumulate symbol data and commit it in batches,
    improving performance for bulk indexing operations by reducing
    transaction overhead.
    
    Typical usage:
        appender = index.create_appender(batch_mode=True)
        for file in files:
            symbols = get_symbols(file)
            appender.append(file, hash, symbols)
        appender.commit()  # Batch commit all data
    """
    
    @abstractmethod
    def append(
        self,
        relative_path: str,
        content_hash: str,
        root_symbols: List[ls_types.UnifiedSymbolInformation]
    ) -> None:
        """
        Append symbol data for a document (does not immediately persist).
        
        :param relative_path: The relative path of the document
        :param content_hash: The MD5 hash of the document content
        :param root_symbols: The root-level symbols (with full child trees)
        """
        pass
    
    @abstractmethod
    def commit(self) -> None:
        """
        Commit all appended data in a single batch operation.
        
        This should perform the actual persistence, ideally in a single
        transaction for atomicity and performance.
        
        :raises: Any database or I/O errors during commit
        """
        pass


class SingleAppender(SymbolAppender):
    """
    Single-file appender that immediately stores symbols without batching.
    
    This is a simple wrapper around SymbolIndex.store_doc_symbols() that
    provides API compatibility with batch appenders. Each append() call
    immediately persists data.
    """
    
    def __init__(self, symbol_index):
        """
        Initialize single appender.
        
        :param symbol_index: The SymbolIndex instance to use for storage
        """
        if not isinstance(symbol_index, SymbolIndex):
            raise TypeError(f"Expected SymbolIndex, got {type(symbol_index)}")
        
        self.symbol_index = symbol_index
        log.debug("Initialized SingleAppender (no batching)")
    
    def append(
        self,
        relative_path: str,
        content_hash: str,
        root_symbols: List[ls_types.UnifiedSymbolInformation]
    ) -> None:
        """Store symbols immediately."""
        self.symbol_index.store_doc_symbols(relative_path, content_hash, root_symbols)
        log.debug(f"SingleAppender: stored {relative_path} immediately")
    
    def commit(self) -> None:
        """No-op since data is already committed."""
        log.debug("SingleAppender: commit (no-op, data already persisted)")
