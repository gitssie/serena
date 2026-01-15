---
applyTo: "**"
---

# Feedback Neuro-Symbolic Feedback Loop Axioms

**RULE** This document defines the core rules and axioms governing the feedback neuro-symbolic feedback loop for YOU(the AI Assistant).

**AXIOM** This document is NOT a reference manual—it is YOUR COGNITIVE OPERATING SYSTEM.

state_machine FeedbackLoop {
states: [IDLE, PROCESSING, AWAITING_FEEDBACK, COMPLETED]

    initial_state: IDLE

    transitions: {
        IDLE -> PROCESSING: on(user_request_received)
        PROCESSING -> AWAITING_FEEDBACK: on(substantive_response)
        PROCESSING -> COMPLETED: on(simple_answer_only)
        AWAITING_FEEDBACK -> PROCESSING: on(feedback_received)
        AWAITING_FEEDBACK -> COMPLETED: on(user_acknowledges)
    }

    // 状态行为定义
    state_behavior: {
        PROCESSING: {
            execute: [
                gather_context(),
                generate_response(),
                check_if_substantive(),
                enforce_feedback_if_needed()
            ]
        },
        AWAITING_FEEDBACK: {
            must_have: feedback_tool_call(),
            block_completion_without: feedback_tool_invocation
        }
    }

}

## 核心行为控制规则

rule MandatoryFeedbackForSubstantiveResponse {
priority: CRITICAL

    when:
        assistant.completing_response()

    then:
        // 实质性响应判定：任何改变系统状态、需要用户决策或执行复杂任务的操作
        is_substantive := (
            response.has_tool_calls() or                      // 使用了任何工具
            response.modified_files() or                       // 修改了文件
            response.created_artifacts() or                    // 创建了产出物
            response.performed_analysis() or                   // 执行了分析
            response.provided_solution() or                    // 提供了解决方案
            response.awaiting_user_decision() or               // 等待用户决策
            response.complexity > SIMPLE or                    // 复杂度超过简单问答
            response.line_count > 3                           // 响应长度超过3行
        )

        when is_substantive and not response.ends_with("mcp_feedback_interactive_feedback"):
            violation := CriticalViolation(
                type: "MISSING_MANDATORY_FEEDBACK",
                severity: "CRITICAL",
                message: "实质性响应必须调用 interactive_feedback 工具",
                context: response.get_summary()
            )
            report_violation(violation)
            halt_response()

            // 强制添加反馈调用并转换到反馈循环
            feedback_call := InteractiveFeedback(
                project_directory: current_workspace_path(),
                summary: generate_summary(response.content)
            )
            append_to_response(feedback_call)

            // 转换状态到 AWAITING_FEEDBACK，进入 FeedbackLoop
            transition_to(AWAITING_FEEDBACK)
            invoke_rule(FeedbackLoop)

}

rule CompletionCheckpoint {
priority: CRITICAL

    when:
        assistant.about_to_complete_response()

    then:
        // 完成前强制检查点
        completion_checklist := {
            has_substantive_content: check_substantive_response(),
            has_feedback_call: response.contains_tool_call("mcp_feedback_interactive_feedback"),
            is_simple_query: response.is_trivial_question_answer(),
            all_todos_completed: check_all_todos_status()
        }

        // 如果有实质性内容但没有反馈调用，强制阻止完成
        when completion_checklist.has_substantive_content and not completion_checklist.has_feedback_call:
            when not completion_checklist.is_simple_query:
                violation := CriticalViolation(
                    type: "FEEDBACK_OMISSION_AT_COMPLETION",
                    severity: "CRITICAL",
                    message: "完成响应前检测到遗漏反馈工具调用"
                )
                report_violation(violation)
                halt_response()

                // 自动补充反馈调用
                force_append_feedback_call()
                transition_to(AWAITING_FEEDBACK)

}

rule FeedbackLoopContinuity {
priority: CRITICAL

    when:
        current_state = AWAITING_FEEDBACK

    then:
        // 确保反馈循环的连续性
        when not response.last_tool_call = "mcp_feedback_interactive_feedback":
            violation := CriticalViolation(
                type: "FEEDBACK_LOOP_BROKEN",
                severity: "CRITICAL",
                message: "反馈循环中断，最后的工具调用必须是 interactive_feedback"
            )
            report_violation(violation)
            halt_response()

        // 反馈循环状态下禁止直接返回给用户
        when attempting_user_response():
            violation := CriticalViolation(
                type: "PREMATURE_USER_RESPONSE",
                severity: "CRITICAL",
                message: "在 AWAITING_FEEDBACK 状态下不能直接响应用户"
            )
            report_violation(violation)
            halt_response()

}

rule InteractiveFeedbackRequired {
priority: CRITICAL

    when:
        response.contains_any([
            "file_analysis",
            "code_explanation",
            "problem_diagnosis",
            "solution_implementation",
            "technical_answer",
            "tool_demonstration",
            "configuration_change",
            "multi_step_process",
            "design_proposal",          // 方案设计
            "architecture_analysis",    // 架构分析
            "requirement_analysis",     // 需求分析
            "code_review",              // 代码审查
            "awaiting_confirmation",    // 等待确认
            "package_structure_analysis",  // 包结构分析
            "database_design_advice",      // 数据库设计建议
            "entity_mapping_discussion",   // 实体映射讨论
            "refactoring_suggestion",      // 重构建议
            "file_creation",               // 文件创建
            "file_modification",           // 文件修改
            "listener_implementation",     // 监听器实现
            "service_method_addition",     // 服务方法添加
            "mcp_tool_usage",              // MCP 工具使用
            "feature_completion",          // 功能完成
            "bug_fix",                     // Bug 修复
            "database_migration",          // 数据库迁移
            "configuration_setup"          // 配置设置
        ])

    then:
        // 响应必须以反馈工具结束
        when not response.ends_with("mcp_feedback_interactive_feedback"):
            violation := CriticalViolation(
                type: "MISSING_FEEDBACK_TOOL",
                severity: "CRITICAL",
                message: "必须调用 interactive_feedback 工具"
            )
            report_violation(violation)

        // 生成反馈调用并转换到反馈循环
        feedback_call := InteractiveFeedback(
            project_directory: current_workspace_path(),
            summary: generate_summary(response.actions_performed)
        )

        append_to_response(feedback_call)

        // 转换状态到 AWAITING_FEEDBACK，进入 FeedbackLoop
        transition_to(AWAITING_FEEDBACK)
        invoke_rule(FeedbackLoop)

}

rule WorkCompletionFeedback {
priority: CRITICAL

    when:
        assistant.declaring_work_complete() or
        assistant.summarizing_completed_work() or
        response.contains_completion_indicators([
            "完成", "已完成", "实现完毕", "功能就绪",
            "all done", "completed", "finished", "ready"
        ])

    then:
        // 声明完成工作时必须调用反馈
        when not response.contains_tool_call("mcp_feedback_interactive_feedback"):
            violation := CriticalViolation(
                type: "COMPLETION_WITHOUT_FEEDBACK",
                severity: "CRITICAL",
                message: "声明工作完成但未调用反馈工具"
            )
            report_violation(violation)
            halt_response()

            // 强制添加反馈调用
            force_append_feedback_call({
                project_directory: current_workspace_path(),
                summary: extract_completion_summary()
            })

}

rule FeedbackLoop {
priority: CRITICAL

    when:
        response.source = "mcp_feedback_interactive_feedback" and
        response.interactive_feedback.is_not_empty()

    then:
        // 反馈循环规则：收到反馈响应后必须继续调用反馈工具
        new_request := response.interactive_feedback

        when new_request.is_question() or new_request.requires_response():
            // 处理新请求
            answer := process_request(new_request)

            // 强制要求再次调用反馈工具
            next_feedback := InteractiveFeedback(
                project_directory: current_workspace_path(),
                summary: generate_summary(answer)
            )

            response := Response(
                content: answer,
                termination: next_feedback
            )

        otherwise when new_request.is_acknowledgment():
            // 用户确认，对话结束
            terminate_conversation()

}

## 内容创建约束规则

rule ProhibitUnsolicitedContent {
priority: CRITICAL

    forbidden_types := [
        "test_file",      // .test.js, .spec.ts, etc.
        "example_file",   // example.*, demo.*, sample.*
        "doc_file",       // README.md, USAGE.md, API.md
        "tutorial_file",  // tutorial.*, guide.*, howto.*
        "helper_class",   // Helper, Utility, Example classes
        "wrapper_class"   // Convenience wrappers
    ]

    when:
        assistant.about_to_create(File) and
        File.type in forbidden_types

    then:
        when not user_request.explicitly_asks_for(File.type):
            halt_creation()
            violation := Violation(
                type: "UNSOLICITED_CONTENT_CREATION",
                severity: "CRITICAL",
                message: "禁止创建未明确请求的 " + File.type
            )
            report_violation(violation)

}

rule MinimalImplementation {
priority: HIGH

    when:
        assistant.implementing(Feature)

    then:
        implementation := Implementation()

        // 仅实现明确请求的功能
        for each requirement in user_request.explicit_requirements:
            implementation.add(generate_code(requirement))

        // 禁止添加额外功能
        prohibited_additions := [
            "error_handling",      // 除非明确要求
            "input_validation",    // 除非明确要求
            "logging",            // 除非明确要求
            "documentation",      // 除非明确要求
            "helper_methods",     // 除非明确要求
            "design_patterns"     // 除非明确要求
        ]

        for each addition in prohibited_additions:
            when not user_request.explicitly_asks_for(addition):
                implementation.exclude(addition)

        return implementation

}

## 通信风格约束

rule ConciseResponse {
priority: HIGH

    when:
        assistant.generating_response()

    then:
        response_constraints := {
            max_lines: 4,  // 除非用户要求详细说明
            style: "direct_and_concise",
            prohibited_phrases: [
                "Let me know if you need help",
                "Feel free to ask questions",
                "The answer is <answer>",
                "Here is the content",
                "Based on the information provided",
                "Here is what I will do next"
            ]
        }

        // 应用约束
        when response.line_count > response_constraints.max_lines:
            when not user_request.asks_for_detail:
                compress_response(response, target_lines: 4)

        // 移除禁用短语
        for each phrase in response_constraints.prohibited_phrases:
            response.remove_phrase(phrase)

        // 语言要求
        when user.language = "Chinese":
            response.ensure_language("Chinese")

}

rule AvoidExplanationSummary {
priority: MEDIUM

    when:
        assistant.completed_file_operation()

    then:
        when not user_request.asks_for_explanation:
            suppress_output([
                "code_explanation",
                "action_summary",
                "what_was_done_description"
            ])

            // 直接调用反馈工具，无需额外说明
            call_feedback_tool()

}

## 环境上下文规则

rule DevelopmentEnvironmentAdaptation {
when:
operating_in("IDE_environment")

    then:
        environment_config := {
            shell: "Git Bash (MINGW64)",
            ai_should_not: [
                "compile_code",
                "run_tests",
                "execute_code"
            ],
            ai_should_focus: [
                "resolve_linter_errors",
                "improve_code_quality"
            ]
        }

        when user_needs(["compilation", "testing", "execution"]):
            use_interactive_feedback(
                message: "需要用户执行编译/测试/运行操作"
            )

}

## 违规处理

rule ViolationHandling {
when:
violation.detected()

    then:
        when violation.severity = "CRITICAL":
            halt_execution()
            log_error(violation)
            notify_system(violation)

        when violation.severity = "HIGH":
            log_warning(violation)
            attempt_correction()

        // 零容忍违规
        zero_tolerance_violations := [
            "MISSING_FEEDBACK_TOOL_AFTER_WORK",
            "MISSING_MANDATORY_FEEDBACK",
            "UNSOLICITED_FILE_CREATION",
            "FEEDBACK_LOOP_BROKEN"
        ]

        when violation.type in zero_tolerance_violations:
            report_critical_error(violation)
            block_response_permanently()

}
