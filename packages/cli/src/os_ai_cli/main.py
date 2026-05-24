import sys
import argparse
import os

from os_ai_core.adapters.llm.legacy_gateway import LegacyLLMGateway
from os_ai_core.adapters.tools.computer_specs import build_computer_tool_descriptor
from os_ai_core.application.services.prompt_builder import PromptBuilder
from os_ai_core.application.use_cases.run_agent import RunAgentCommand, RunAgentUseCase
from os_ai_core.di import create_container
from os_ai_core.utils.logger import setup_logging

from os_ai_core.orchestrator import Orchestrator
from os_ai_cli.events import CliEventSink

import pyautogui


def main() -> int:
    parser = argparse.ArgumentParser(description="Universal Computer Use agent (CLI)")
    parser.add_argument("--task", type=str, required=False, help="Задача (на естественном языке)")
    parser.add_argument("--debug", action="store_true", help="Включить DEBUG логи")
    parser.add_argument("--provider", type=str, required=False, help="Провайдер LLM: anthropic|openai")
    args = parser.parse_args()

    logger = setup_logging(debug=args.debug)
    logger.info(f"Screen size detected: {pyautogui.size()[0]}x{pyautogui.size()[1]}; pause={pyautogui.PAUSE}, failsafe={pyautogui.FAILSAFE}")

    if args.task:
        task_text = args.task
    else:
        logger.info("Awaiting task input from stdin...")
        print("Введите задачу:")
        task_text = sys.stdin.readline().strip()

    provider = args.provider  # None if not specified -> di.py uses LLM_PROVIDER from env/config
    inj = create_container(provider)
    from os_ai_llm.interfaces import LLMClient
    from os_ai_core.adapters.tools.composite_tool_gateway import CompositeToolGateway
    from os_ai_core.tools.registry import ToolRegistry
    client = inj.get(LLMClient)
    tools = inj.get(ToolRegistry)
    tool_gateway = inj.get(CompositeToolGateway)
    orch = Orchestrator(client, tools, tool_gateway=tool_gateway)

    # Resolve actual provider for tool type lookup
    actual_provider = provider or client.get_provider_name()
    tool_descs = [build_computer_tool_descriptor(actual_provider)]
    system_prompt = PromptBuilder().build_desktop_operator_prompt()
    total_in = 0
    total_out = 0

    try:
        if _application_runner_enabled():
            run_result = RunAgentUseCase(
                llm=LegacyLLMGateway(client),
                tools=tool_gateway,
                events=CliEventSink(),
            ).execute(
                RunAgentCommand(
                    job_id="cli-run",
                    task=task_text,
                    tool_descriptors=tool_descs,
                    system_prompt=system_prompt,
                    max_iterations=30,
                )
            )
            msgs = run_result.messages
            total_in = run_result.input_tokens
            total_out = run_result.output_tokens
        else:
            msgs = orch.run(task_text, tool_descs, system_prompt, max_iterations=30)
            total_in = getattr(orch, 'total_input_tokens', 0)
            total_out = getattr(orch, 'total_output_tokens', 0)
    except KeyboardInterrupt:
        try:
            from os_ai_core.utils.costs import estimate_cost
            model_name = client.get_model_name()
            in_cost, out_cost, total_cost, _tier = estimate_cost(model_name, int(total_in), int(total_out))
            print(f"\nInterrupted by user (Ctrl+C)\n📈 Usage total in={total_in} out={total_out} cost=${total_cost:.6f} (input=${in_cost:.6f}, output=${out_cost:.6f})")
        except Exception:
            print("\nInterrupted by user (Ctrl+C)")
        return 130

    final_texts = []
    for m in msgs:
        if getattr(m, "role", None) == "assistant":
            for p in (getattr(m, "content", []) or []):
                try:
                    if getattr(p, "type", None) == "text":
                        final_texts.append(str(getattr(p, "text", "")))
                except Exception:
                    pass
    if final_texts:
        print("\n".join(final_texts).strip())

    try:
        from os_ai_core.utils.costs import estimate_cost
        model_name = client.get_model_name()
        in_cost, out_cost, total_cost, _tier = estimate_cost(model_name, int(total_in), int(total_out))
        print(f"📈 Usage total in={total_in} out={total_out} cost=${total_cost:.6f} (input=${in_cost:.6f}, output=${out_cost:.6f})")
    except Exception:
        pass

    return 0


def _application_runner_enabled() -> bool:
    return os.environ.get("OS_AI_USE_APPLICATION_RUNNER", "1").lower() not in {"0", "false", "no", "off"}


if __name__ == "__main__":
    raise SystemExit(main())
