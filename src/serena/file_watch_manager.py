import logging
import os
import queue
import threading
from typing import Any

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from solidlsp import SolidLanguageServer

log = logging.getLogger(__name__)


class FileChangeHandler(FileSystemEventHandler):
    
    def __init__(self, file_queue: queue.Queue, root_path: str, get_language_server):
        super().__init__()
        self._queue = file_queue
        self._root_path = root_path
        self._get_language_server = get_language_server
    
    def _should_process_file(self, relative_path: str) -> bool:
        return self._get_language_server(relative_path) is not None
    
    def on_created(self, event):
        if not event.is_directory:
            relative_path = os.path.relpath(event.src_path, self._root_path)
            # Normalize path separators to forward slashes for consistent cross-platform handling
            relative_path = relative_path.replace(os.path.sep, "/")
            if self._should_process_file(relative_path):
                self._queue.put((relative_path, "created"))
    
    def on_modified(self, event):
        if not event.is_directory:
            relative_path = os.path.relpath(event.src_path, self._root_path)
            # Normalize path separators to forward slashes for consistent cross-platform handling
            relative_path = relative_path.replace(os.path.sep, "/")
            if self._should_process_file(relative_path):
                self._queue.put((relative_path, "modified"))
    
    def on_deleted(self, event):
        if not event.is_directory:
            relative_path = os.path.relpath(event.src_path, self._root_path)
            # Normalize path separators to forward slashes for consistent cross-platform handling
            relative_path = relative_path.replace(os.path.sep, "/")
            if self._should_process_file(relative_path):
                self._queue.put((relative_path, "deleted"))


class FileWatchManager:
    """
    管理文件监听、队列和符号缓存工作线程
    
    工作流程：
    1. watchdog监听文件系统变化
    2. FileChangeHandler将变化事件加入队列
    3. 后台工作线程从队列取出事件
    4. 调用对应LSP的request_document_symbols自动缓存到Kùzu
    """
    
    def __init__(self, root_path: str, language_server_manager: Any):
        """
        :param root_path: 项目根目录
        :param language_server_manager: LanguageServerManager实例，用于获取LSP
        """
        self._root_path = root_path
        self._ls_manager = language_server_manager
        self._file_change_queue: queue.Queue = queue.Queue()
        self._cache_worker_thread: threading.Thread | None = None
        self._file_observer: Observer | None = None
        self._stop_worker = threading.Event()
        self._running = False
    
    def start(self):
        """启动文件监听和缓存工作线程"""
        if self._running:
            log.warning("FileWatchManager already running")
            return
        
        log.info(f"Starting FileWatchManager for {self._root_path}")
        
        # 启动后台缓存工作线程
        self._stop_worker.clear()
        self._cache_worker_thread = threading.Thread(
            target=self._cache_worker,
            daemon=True,
            name="FileWatchCacheWorker"
        )
        self._cache_worker_thread.start()
        
        # 启动watchdog文件监听
        handler = FileChangeHandler(self._file_change_queue, self._root_path, self._get_language_server)
        self._file_observer = Observer()
        self._file_observer.schedule(handler, self._root_path, recursive=True)
        self._file_observer.start()
        
        self._running = True
        log.info("FileWatchManager started successfully")
    
    def stop(self):
        """停止文件监听和工作线程"""
        if not self._running:
            return
        
        log.info("Stopping FileWatchManager...")
        
        # 停止watchdog
        if self._file_observer:
            self._file_observer.stop()
            self._file_observer.join(timeout=2.0)
            self._file_observer = None
        
        # 停止工作线程
        self._stop_worker.set()
        if self._cache_worker_thread:
            self._cache_worker_thread.join(timeout=3.0)
            self._cache_worker_thread = None
        
        # 清空队列
        while not self._file_change_queue.empty():
            try:
                self._file_change_queue.get_nowait()
            except queue.Empty:
                break
        
        self._running = False
        log.info("FileWatchManager stopped")
    
    def _cache_worker(self):
        """
        后台工作线程：从队列中取出文件变化事件并批量缓存符号
        """
        # 批量处理的文件集合 - 按语言服务器分组（使用 set 去重）
        batch_files: dict[SolidLanguageServer, set[str]] = {}
        batch_timeout = 2.0  # 批量处理超时时间（秒）
        
        while not self._stop_worker.is_set():
            try:
                # 从队列获取文件变化事件（带超时）
                try:
                    relative_path, event_type = self._file_change_queue.get(timeout=batch_timeout)
                    
                    if event_type == "deleted":
                        # 文件删除：立即处理
                        self._delete_cached_symbols(relative_path)
                    else:
                        # 文件创建或修改：加入批处理队列（自动去重）
                        ls = self._get_language_server(relative_path)
                        if ls is not None:
                            if ls not in batch_files:
                                batch_files[ls] = set()
                            batch_files[ls].add(relative_path)  # set 自动去重
                    
                except queue.Empty:
                    # 队列为空，继续等待
                    pass
                
                # 批量处理条件：队列为空 且 有待处理文件
                if self._file_change_queue.empty() and batch_files:
                    for ls, files_set in batch_files.items():
                        if not files_set:
                            continue
                        
                        try:
                            log.debug(f"Batch indexing {len(files_set)} unique files with {ls.__class__.__name__}")
                            
                            # 创建批量索引器
                            appender = ls.create_build_appender()
                            
                            # 批量处理所有文件（已去重）
                            for relative_path in files_set:
                                try:
                                    ls.build_index(relative_path, appender, rebuild=False)
                                except Exception as e:
                                    log.error(f"❌ Failed to index {relative_path}: {e}")
                            
                            # 提交批量索引
                            appender.commit()
                            log.debug(f"✅ Batch indexed {len(files_set)} files successfully")
                            
                        except Exception as e:
                            log.error(f"❌ Failed to batch process files: {e}")
                    
                    # 清空批处理队列
                    batch_files.clear()
                
            except Exception as e:
                log.error(f"Unexpected error in cache worker: {e}")
    
    def _get_language_server(self, relative_path: str) -> SolidLanguageServer | None:
        """
        获取处理指定文件的语言服务器
        
        :param relative_path: 文件相对路径
        :return: 匹配的SolidLanguageServer实例，如果没有匹配返回None
        """
        try:
            for ls in self._ls_manager._language_servers.values():
                if not ls.is_ignored_path(relative_path, ignore_unsupported_files=True):
                    return ls
            
            return None
        except Exception:
            return None
    
    def _delete_cached_symbols(self, relative_path: str) -> None:
        """
        删除指定文件的Kùzu缓存符号
        
        :param relative_path: 文件相对路径
        """
        ls = self._get_language_server(relative_path)
        if ls is not None:
            ls.kuzu_cache.invalidate_doc(relative_path)
    
    def is_running(self) -> bool:
        """返回监听器是否正在运行"""
        return self._running    
