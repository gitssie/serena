"""
Handler for LSP publishDiagnostics notifications.

This module provides a reusable handler for language servers that use push-based
diagnostics via textDocument/publishDiagnostics notifications instead of pull-based
textDocument/diagnostic requests (LSP 3.17).
"""

import logging
import os
import pathlib
import threading
import time
import uuid
from pathlib import PurePath
from urllib.parse import unquote, urlparse

from solidlsp import ls_types
from solidlsp.ls import SolidLanguageServer

log = logging.getLogger(__name__)


def normalize_uri(uri: str) -> str:
    """
    规范化 URI 以解决跨平台和编码问题
    
    在 Windows 上会出现以下问题：
    - 盘符大小写不一致：E:/ vs e:/
    - 盘符编码不一致：E:/ vs e%3A/
    - 路径分隔符不一致：/ vs \\
    
    这个函数将所有 URI 统一规范化为小写、解码的格式
    
    :param uri: 原始 URI
    :return: 规范化后的 URI
    """
    if not uri or not uri.startswith('file:///'):
        return uri
    
    # 解析 URI
    parsed = urlparse(uri)
    
    # URL 解码路径部分（处理 %3A 等编码）
    decoded_path = unquote(parsed.path)
    
    # 在 Windows 上，统一转换为小写（Windows 文件系统不区分大小写）
    if os.name == 'nt':
        decoded_path = decoded_path.lower()
    
    # 重构规范化的 URI
    # file:/// + 规范化路径
    normalized = f"file://{decoded_path}"
    
    return normalized


class PublishDiagnosticsHandler:
    """
    处理语言服务器的 publishDiagnostics 通知机制
    
    某些语言服务器（如 JDTLS）不支持 textDocument/diagnostic 请求（LSP 3.17），
    而是通过 textDocument/publishDiagnostics 通知推送诊断信息。
    这个类封装了等待和处理这些通知的完整逻辑。
    
    注意：语言服务器可能会对同一文件发送多次 publishDiagnostics 通知：
    - 第一次可能是空数组（清空旧诊断）
    - 后续通知包含实际的诊断信息
    
    Usage:
        # In language server __init__:
        self._publish_diagnostics_handler = PublishDiagnosticsHandler(self)
        
        # In _start_server:
        self.server.on_notification(
            "textDocument/publishDiagnostics",
            self._publish_diagnostics_handler.on_publish_diagnostics
        )
        
        # In request_text_document_diagnostics:
        return self._publish_diagnostics_handler.wait_for_diagnostics(
            relative_file_path, timeout
        )
    """
    
    def __init__(self, language_server: SolidLanguageServer, wait_for_stable: bool = False, stability_delay: float = 3.0):
        """
        初始化处理器
        
        :param language_server: 语言服务器实例，用于访问 repository_root_path 等属性
        :param wait_for_stable: 是否等待诊断稳定（防止接收到空诊断就立即返回）
        :param stability_delay: 收到最后一次通知后等待多久认为诊断已稳定（秒）
        """
        self._ls = language_server
        # 改为支持多个等待者：uri -> list of waiters
        # 每个 waiter 有唯一 ID 以支持同一文件的多个并发请求
        self._waiters: dict[str, list[dict]] = {}
        self._lock = threading.RLock()  # 使用可重入锁防止死锁
        
        # 诊断稳定性配置
        self._wait_for_stable = wait_for_stable
        self._stability_delay = stability_delay
        
        # Timer管理：uri -> Timer对象（按文件粒度）
        self._timers: dict[str, threading.Timer] = {}
    
    def wait_for_diagnostics (
        self, 
        relative_file_path: str, 
        timeout: float = 15.0,
        filter_func: callable = None
    ) -> list[ls_types.Diagnostic]:
        """
        等待并获取指定文件的诊断信息
        
        这是主要的对外接口，封装了完整的等待、接收、过滤逻辑
        支持多个线程同时等待同一文件的诊断信息
        
        :param relative_file_path: 相对文件路径
        :param timeout: 超时时间（秒）
        :param filter_func: 可选的自定义过滤函数，接收 (diagnostics: list[dict], uri: str) 返回过滤后的列表
        :return: 诊断信息列表
        """
        uri = pathlib.Path(str(PurePath(self._ls.repository_root_path, relative_file_path))).as_uri()
        # 规范化 URI 以处理大小写和编码问题
        normalized_uri = normalize_uri(uri)
        
        # 注册等待者（返回 waiter_id 和 waiter）
        waiter_id, waiter = self._register_waiter(normalized_uri)
        
        try:
            # 等待通知
            if waiter['event'].wait(timeout):
                diagnostics = waiter['diagnostics']
                
                if diagnostics is None:
                    return self._create_info_diagnostic(
                        uri, "Diagnostics check completed successfully - no issues found"
                    )
                
                # 过滤和转换诊断信息
                if filter_func:
                    filtered = filter_func(diagnostics, uri)
                else:
                    filtered = self._default_filter_diagnostics(diagnostics, uri)
                
                if not filtered:
                    return self._create_info_diagnostic(
                        uri, "Diagnostics check completed successfully - no issues found"
                    )
                
                return filtered
            else:
                log.warning(f"Timeout ({timeout}s) waiting for diagnostics for {relative_file_path}")
                return self._create_warning_diagnostic(
                    uri, f"Timeout ({timeout}s) waiting for diagnostics"
                )
        finally:
            self._unregister_waiter(normalized_uri, waiter_id)
    
    def on_publish_diagnostics(self, params: dict) -> None:
        """
        处理 textDocument/publishDiagnostics 通知
        
        这个方法会被注册为 LSP 通知回调
        通知所有等待该 URI 的请求
        
        智能通知策略：
        - 如果收到非空诊断 → 立即通知（这是最终结果）
        - 如果收到空诊断 → 等待稳定延迟（可能还有后续诊断）
        
        :param params: 通知参数（包含 uri 和 diagnostics）
        """
        
        uri = params.get('uri')
        diagnostics = params.get('diagnostics', [])
        
        # 规范化 URI 以处理大小写和编码问题
        normalized_uri = normalize_uri(uri)
        
        has_diagnostics = len(diagnostics) > 0
        
        if has_diagnostics:
            log.debug(f"Received textDocument/publishDiagnostics: uri={uri} (normalized={normalized_uri}), diagnostics_count={len(diagnostics)}")
        
        # 更新所有等待该 URI 的请求
        with self._lock:
            if normalized_uri in self._waiters:
                waiters_list = self._waiters[normalized_uri]
                
                # 更新所有waiter的诊断数据
                for waiter in waiters_list:
                    waiter['diagnostics'] = diagnostics
                
                # 决定是否立即通知
                if has_diagnostics:
                    # 有诊断信息 → 取消Timer并立即通知所有waiter
                    self._notify_all_waiters(normalized_uri, cancel_timer=True)
                elif self._wait_for_stable:
                    # 空诊断且启用稳定性等待 → 如果没有Timer才创建
                    if normalized_uri not in self._timers:
                        timer = threading.Timer(self._stability_delay, self._notify_all_waiters, args=(normalized_uri,))
                        timer.start()
                        self._timers[normalized_uri] = timer
                else:
                    # 空诊断但不等待稳定 → 取消Timer并立即通知所有waiter
                    self._notify_all_waiters(normalized_uri, cancel_timer=True)
    
    def _register_waiter(self, uri: str) -> tuple[str, dict]:
        """
        注册等待者（内部方法）
        
        支持多个线程同时等待同一文件的诊断信息
        
        :param uri: 文件 URI
        :return: (waiter_id, waiter) 元组，waiter_id 用于后续注销
        """
        waiter_id = str(uuid.uuid4())
        waiter = {
            'id': waiter_id,
            'event': threading.Event(),
            'diagnostics': None,
        }
        
        with self._lock:
            if uri not in self._waiters:
                self._waiters[uri] = []
            self._waiters[uri].append(waiter)
        
        return waiter_id, waiter
    
    def _notify_all_waiters(self, uri: str, cancel_timer: bool = False) -> None:
        """
        通知所有等待者（Timer回调或立即通知）
        
        :param uri: 文件 URI
        :param cancel_timer: 是否需要先取消正在运行的Timer
        """
        with self._lock:
            # 如果需要，先取消Timer
            if cancel_timer and uri in self._timers:
                self._timers[uri].cancel()
            
            # 通知所有waiter
            if uri in self._waiters:
                for w in self._waiters[uri]:
                    w['event'].set()
            
            # 清理Timer
            if uri in self._timers:
                del self._timers[uri]
    
    def _unregister_waiter(self, uri: str, waiter_id: str) -> None:
        """
        取消注册等待者（内部方法）
        
        线程安全：整个操作在锁保护下完成
        
        :param uri: 文件 URI
        :param waiter_id: 等待者 ID
        """
        with self._lock:
            if uri not in self._waiters:
                # URI 不存在，可能已经被其他线程清理了
                return
            
            # 获取当前等待者列表
            waiters_list = self._waiters[uri]
            
            # 移除指定 ID 的 waiter
            new_waiters_list = [w for w in waiters_list if w['id'] != waiter_id]
            
            # 如果列表为空，删除键并清理Timer；否则更新列表
            if not new_waiters_list:
                del self._waiters[uri]
                # 最后一个waiter移除，取消并清理Timer
                if uri in self._timers:
                    self._timers[uri].cancel()
                    del self._timers[uri]
            else:
                self._waiters[uri] = new_waiters_list
    
    @staticmethod
    def _default_filter_diagnostics(diagnostics: list[dict], uri: str) -> list[ls_types.Diagnostic]:
        """
        默认的诊断信息过滤和转换（内部方法）
        
        将原始诊断信息转换为标准的 ls_types.Diagnostic 格式
        子类可以覆盖此方法以实现自定义过滤逻辑
        """
        ret: list[ls_types.Diagnostic] = []
        
        for item in diagnostics:
            new_item: ls_types.Diagnostic = {
                "uri": uri,
                "severity": item.get("severity", 1),
                "message": item.get("message", ""),
                "range": item["range"],
                "code": item.get("code"),
            }
            ret.append(new_item)
        
        return ret
    
    @staticmethod
    def _create_info_diagnostic(uri: str, message: str) -> list[ls_types.Diagnostic]:
        """创建信息级别的诊断消息（内部方法）"""
        return [{
            "uri": uri,
            "severity": 3,  # Information
            "message": message
        }]
    
    @staticmethod
    def _create_warning_diagnostic(uri: str, message: str) -> list[ls_types.Diagnostic]:
        """创建警告级别的诊断消息（内部方法）"""
        return [{
            "uri": uri,
            "severity": 2,  # Warning
            "message": message
        }]


class JavaPublishDiagnosticsHandler(PublishDiagnosticsHandler):
    """
    Java 语言服务器专用的 publishDiagnostics 处理器
    
    继承自 PublishDiagnosticsHandler，添加了 Java 特定的诊断过滤逻辑
    Java LSP 的诊断通知是稳定的，不需要等待机制
    """
    
    def __init__(self, language_server: 'SolidLanguageServer'):
        """
        初始化 Java 诊断处理器
        
        Java LSP 不需要稳定性等待，因为它的诊断通知是一次性完整的
        """
        super().__init__(language_server, wait_for_stable=False)
    
    @staticmethod
    def _default_filter_diagnostics(diagnostics: list[dict], uri: str) -> list[ls_types.Diagnostic]:
        """
        Java 特定的诊断信息过滤
        
        过滤掉 "Type safety:" 警告（Java 泛型的 unchecked 转换警告）
        """
        ret: list[ls_types.Diagnostic] = []
        
        for item in diagnostics:
            message = item.get("message", "")
            # 过滤类型安全警告（Java 泛型的 unchecked 转换警告）
            if message.startswith("Type safety:"):
                continue
            
            new_item: ls_types.Diagnostic = {
                "uri": uri,
                "severity": item.get("severity", 1),
                "message": message,
                "range": item["range"],
                "code": item.get("code"),
            }
            ret.append(new_item)
        
        return ret
