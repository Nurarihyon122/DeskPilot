"""
planner.py - DSPy-based Text-Only Planner
==========================================
Decides the next action using ONLY text state (no screenshots).
This is the DECIDE phase of the state machine.

DSPy handles:
- Deciding action type (keyboard vs wait vs done)
- Determining if vision escalation is needed
- Mapping intent to frozen Action schema
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Literal, List
import dspy


# ─────────────────────────────────────────────────────────────
# DATA STRUCTURES
# ─────────────────────────────────────────────────────────────

@dataclass
class TextState:
    """Non-visual state collected during OBSERVE phase."""
    active_app: str = ""
    window_title: str = ""
    focused_element: str = ""  # role + label if available
    current_url: Optional[str] = None  # Browser URL if active
    is_browser: bool = False  # True if Chrome is active window
    interactive_elements: str = ""  # Indexed list of clickable elements
    home_dir: str = "/root"  # Actual home directory in this environment
    desktop_path: str = "/root/Desktop"  # Actual Desktop path in this environment
    

@dataclass
class PlannerInput:
    """Everything the planner needs to decide next action."""
    goal: str
    step: int
    history: List[str] = None # List of [Action: Result] strings
    text_state: TextState = None
    
    def __post_init__(self):
        if self.history is None:
            self.history = []
        if self.text_state is None:
            self.text_state = TextState()


@dataclass  
class PlannerOutput:
    """Structured decision from the planner."""
    action_type: Literal[
        "PRESS_KEY", "TYPE", "WAIT", "DONE", "FAIL", "CLICK", "SCROLL",
        "BROWSER_NAVIGATE", "BROWSER_CLICK", "BROWSER_TYPE"
    ]
    action_param: str = ""  # key name, text to type, or seconds
    expected_window_title: str = "" # Anchor for local verification
    success_indicators: str = "" # Comma-separated success markers
    sub_goals: str = "" # Comma-separated list of steps to achieve the final goal
    reason: str = ""
    needs_vision: bool = False
    confidence: float = 0.8


# ─────────────────────────────────────────────────────────────
# DSPy SIGNATURE (defines input/output contract)
# ─────────────────────────────────────────────────────────────

class PlanNextAction(dspy.Signature):
    """
    Plan the next action sequence to achieve the goal using accessibility-first principles.

    STANDARD SKILLS:
    - Open App: PRESS_KEY("ESCAPE"); WAIT(0.5); PRESS_KEY("Alt+F2"); WAIT(1.5); TYPE("<bin_name>"); PRESS_KEY("ENTER")
      → bin_name MUST come from `app_knowledge`. Never guess.
    - Path Nav (Thunar): PRESS_KEY("Ctrl+L"); WAIT(1); TYPE("<absolute_path>"); PRESS_KEY("ENTER"); WAIT(1)
      → ALWAYS default to `/app` (the DeskPilot project folder) for all file operations unless `/root` is explicitly requested.
    - Save/Open Dialog: PRESS_KEY("Ctrl+S"); WAIT(1); TYPE("<filename>"); PRESS_KEY("ENTER")
      → Always WAIT(1) after Ctrl+S/Ctrl+O before typing.

    BROWSER ACTIONS (when is_browser=True):
    Use ONLY index-based actions from interactive_elements. Never use CSS selectors or TYPE/PRESS_KEY for web.
    - BROWSER_NAVIGATE(url) | BROWSER_CLICK(index) | BROWSER_TYPE(index, text)

    RULES:
    1. Sequences: Max 5-8 actions. Complete a sub-goal, verify, then plan next.
    2. Autocomplete/Dropdown: BROWSER_TYPE → STOP → observe → BROWSER_CLICK dropdown option.
       Never type AND click in same sequence. Indices change after typing.
    3. Popups/Modals: Dismiss FIRST (PRESS_KEY("ESCAPE") or BROWSER_CLICK close). Never interact behind popups.
    4. Success indicators: EMPTY for intermediate steps. Only set on the FINAL action completing the ENTIRE goal.
    5. Anti-loop: Same action failing 2× = stuck. Recovery: BROWSER_NAVIGATE to target, or escalate needs_vision=True.
    6. Goal decomposition: Break into 3–5 verifiable sub_goals (e.g. 'App opened', 'Page loaded').
    7. Multi-part goals ("go to X and search Y"): Step 1 navigates (success_indicators: ""), Step 2 searches.
       "search [site]" = BROWSER_NAVIGATE([site].com), NOT a Google search for the site name.
    8. Web behavior: Search results are intermediate. Navigate INTO a specific page before DONE.
    9. NEVER CHAIN BROWSER ACTIONS: You MUST output ONLY ONE browser action per sequence. Do NOT use semicolons to chain `BROWSER_TYPE`, `BROWSER_CLICK`, or `BROWSER_NAVIGATE` together. For example, if you want to type and click search, DO NOT chain them. Output ONLY `BROWSER_TYPE(...)` and stop. Let the engine observe, then output `BROWSER_CLICK(...)` on the next step.
    10. Window titles: Use generic titles (e.g. "Google Chrome"), not dynamic page titles.
    11. Launcher stuck ("app"/"application finder" in title): PRESS_KEY("ESCAPE"); WAIT(1). Stop — check if app launched behind it.
    12. Thunar navigation: Always use Ctrl+L path navigation. Never click folders directly.
    13. DONE: Only when the entire goal is complete. Use DONE("Goal reached") to end the task.
    14. You MUST provide at least one valid action in `action_sequence`. Never leave it empty. If finished, output DONE("Finished").
    """
    
    # Inputs
    goal: str = dspy.InputField(desc="The task goal to achieve")
    app_knowledge: str = dspy.InputField(desc="Knowledge base of app names and titles")
    history: str = dspy.InputField(desc="String representation of previous actions and their outcomes")
    step: int = dspy.InputField(desc="Current step number (1-indexed)")
    window_title: str = dspy.InputField(desc="Current window title (via xdotool)")
    active_app: str = dspy.InputField(desc="Currently focused application")
    current_url: str = dspy.InputField(desc="Current browser URL (if in Chrome, otherwise empty)")
    is_browser: bool = dspy.InputField(desc="True if currently in Chrome browser")
    focused_element: str = dspy.InputField(desc="Currently focused input element (if any)")
    interactive_elements: str = dspy.InputField(desc="Indexed list of clickable elements: [1] <A> 'Link', [2] <BUTTON> 'Submit' (browser only)")
    home_dir: str = dspy.InputField(desc="Actual home directory path (e.g. /root). Use this for ~ expansion.")
    desktop_path: str = dspy.InputField(desc="Actual Desktop directory path (e.g. /root/Desktop). ALWAYS use this when the goal refers to the Desktop.")
    
    # Outputs
    action_sequence: str = dspy.OutputField(desc="Semicolon-separated actions. BROWSER_NAVIGATE for websites. MUST NOT BE EMPTY.")
    expected_window_title: str = dspy.OutputField(desc="**SOFT ANCHOR**: Generic title like 'Google Chrome', never specific page names.")
    success_indicators: str = dspy.OutputField(desc="**CRITICAL**: Comma-separated strings visible ONLY when ENTIRE goal complete. EMPTY for intermediate steps.")
    sub_goals: str = dspy.OutputField(desc="Comma-separated sub-tasks (e.g., 'Navigate to site, Search for query, View results')")
    reason: str = dspy.OutputField(desc="Reasoning for this sequence")
    needs_vision: bool = dspy.OutputField(desc="Set to True ONLY if text signals are insufficient")


# ─────────────────────────────────────────────────────────────
# DSPy MODULE (the actual planner logic)
# ─────────────────────────────────────────────────────────────

class ActionPlanner(dspy.Module):
    """
    Multi-step text-only planner.
    """
    
    def __init__(self):
        super().__init__()
        self.planner = dspy.ChainOfThought(PlanNextAction)
        self.knowledge = self._load_knowledge()

    def _load_knowledge(self) -> str:
        try:
            import yaml
            from pathlib import Path
            # Resolve relative to repo root (3 levels up from this file: agent/ -> cua_backend/ -> src/ -> root)
            root = Path(__file__).resolve().parent.parent.parent.parent
            path = root / "configs" / "xfce_apps.yaml"
            if path.exists():
                with open(path, "r") as f:
                    return f.read() # Return raw YAML string
            return "No app knowledge available."
        except:
            return "No app knowledge available."
    
    def forward(self, inp: PlannerInput) -> PlannerOutput:
        """Run the planner and return a sequence-aware output."""
        # Convert history list to string for dspy
        history_str = "\n".join(inp.history) if inp.history else "none"
        
        result = self.planner(
            goal=inp.goal,
            app_knowledge=self.knowledge,
            history=history_str,
            step=inp.step,
            window_title=inp.text_state.window_title or "unknown",
            active_app=inp.text_state.active_app or "unknown",
            current_url=inp.text_state.current_url or "",
            is_browser=inp.text_state.is_browser,
            focused_element=inp.text_state.focused_element or "",
            interactive_elements=inp.text_state.interactive_elements or "",
            home_dir=inp.text_state.home_dir,
            desktop_path=inp.text_state.desktop_path,
        )
        
        return PlannerOutput(
            action_type="SEQUENCE",
            action_param=result.action_sequence,
            expected_window_title=result.expected_window_title,
            success_indicators=getattr(result, 'success_indicators', ""),
            sub_goals=getattr(result, 'sub_goals', ""),
            reason=result.reason,
            needs_vision=getattr(result, 'needs_vision', False),
        )


# ─────────────────────────────────────────────────────────────
# PLANNER WRAPPER (easy-to-use interface)
# ─────────────────────────────────────────────────────────────

class Planner:
    """
    High-level planner interface for the agent.
    
    Usage:
        planner = Planner()
        planner.configure("gemini/gemini-2.0-flash")
        
        output = planner.decide(PlannerInput(
            goal="Open Firefox and search for cats",
            step=1,
            text_state=TextState(window_title="Desktop", active_app="xfce4-panel")
        ))
    """
    
    def __init__(self):
        self._module = ActionPlanner()
        self._configured = False
    
    def configure(self, model: str = "gemini/gemini-2.5-flash"):
        """Configure DSPy with the specified model."""
        import os
        
        # If openrouter, we need to ensure LiteLLM has the key
        if model.startswith("openrouter/"):
            os.environ["OPENROUTER_API_KEY"] = os.getenv("OPENROUTER_API_KEY", "")
            
        lm = dspy.LM(model)
        dspy.configure(lm=lm)
        self._configured = True
    
    def decide(self, inp: PlannerInput) -> PlannerOutput:
        """
        Decide the next action based on text state.
        
        Args:
            inp: PlannerInput with goal, step, and text state
            
        Returns:
            PlannerOutput with action decision
        """
        if not self._configured:
            raise RuntimeError("Planner not configured. Call configure() first.")
        
        return self._module(inp)


# ─────────────────────────────────────────────────────────────
# ACTION MAPPING (convert planner output to frozen Action)
# ─────────────────────────────────────────────────────────────

import re


def _smart_split(s: str) -> list:
    """
    Split a semicolon-separated action sequence, but only on semicolons
    that are at the top level — NOT inside parentheses or quote strings.

    This prevents shell commands like
        TYPE(for file in *; do echo Hello > $file; done)
    from being torn apart at every inner semicolon.
    """
    parts: list[str] = []
    depth = 0
    in_single = False
    in_double = False
    current: list[str] = []

    for ch in s:
        if ch == "'" and not in_double:
            in_single = not in_single
            current.append(ch)
        elif ch == '"' and not in_single:
            in_double = not in_double
            current.append(ch)
        elif ch == '(' and not in_single and not in_double:
            depth += 1
            current.append(ch)
        elif ch == ')' and not in_single and not in_double:
            depth -= 1
            current.append(ch)
        elif ch == ';' and depth == 0 and not in_single and not in_double:
            parts.append(''.join(current))
            current = []
        else:
            current.append(ch)

    if current:
        parts.append(''.join(current))

    return parts


def _parse_repr_action(class_name: str, params_str: str, reason: str):
    """
    Parse the Python-repr format that DSPy sometimes emits, e.g.
        TypeAction(type='TYPE', reason='...', text='for file in *; do echo Hello > $file; done')
        PressKeyAction(type='PRESS_KEY', reason='...', key='ENTER')

    Returns an Action object or None when the class name is unrecognised.
    """
    from ..schemas.actions import (
        PressKeyAction, TypeAction, WaitAction, DoneAction, FailAction
    )

    def _extract(field: str, s: str) -> str:
        """Return the value for 'field=...' handling both quote styles."""
        # Try field='value' (single quotes)
        m = re.search(rf"{re.escape(field)}='((?:[^'\\]|\\.)*)'", s)
        if m:
            return m.group(1)
        # Try field="value" (double quotes)
        m = re.search(rf'{re.escape(field)}="((?:[^"\\]|\\.)*)"', s)
        if m:
            return m.group(1)
        return ""

    cn = class_name.lower()

    if cn == "typeaction":
        text = _extract("text", params_str)
        if text:
            return TypeAction(text=text, reason=reason)
    elif cn == "presskeyaction":
        key = _extract("key", params_str)
        if key:
            return PressKeyAction(key=key, reason=reason)
    elif cn == "waitaction":
        seconds_str = _extract("seconds", params_str)
        try:
            return WaitAction(seconds=float(seconds_str), reason=reason)
        except (ValueError, TypeError):
            return WaitAction(seconds=1.0, reason=reason)
    elif cn == "doneaction":
        final = _extract("final_answer", params_str) or "Goal reached"
        return DoneAction(final_answer=final, reason=reason)
    elif cn == "failaction":
        error = _extract("error", params_str) or "Task failed"
        return FailAction(error=error, reason=reason)
    elif cn == "scrollaction":
        from ..schemas.actions import ScrollAction
        amount_str = _extract("amount", params_str)
        try:
            return ScrollAction(amount=int(amount_str), reason=reason)
        except (ValueError, TypeError):
            return ScrollAction(amount=-10, reason=reason)

    return None


def parse_actions(output: PlannerOutput) -> list:
    """
    Parses a sequence string like 'PRESS_KEY(Alt+F2); TYPE(firefox)'
    into a list of Action objects.

    Also handles the Python-repr format that DSPy sometimes emits:
        TypeAction(type='TYPE', text='...'); PressKeyAction(key='ENTER')
    """
    from ..schemas.actions import (
        PressKeyAction, TypeAction, WaitAction, DoneAction, FailAction, ScrollAction,
        BrowserNavigateAction, BrowserClickAction, BrowserTypeAction, Action
    )

    actions = []
    # Use smart split so inner semicolons (e.g. shell for-loops) are NOT treated
    # as action separators.
    parts = [p.strip() for p in _smart_split(output.action_param) if p.strip()]
    
    def _strip_outer_quotes(s: str) -> str:
        """Strip matching outer quote pair only — never strips interior quotes."""
        s = s.strip()
        if len(s) >= 2 and (
            (s[0] == '"' and s[-1] == '"') or
            (s[0] == "'" and s[-1] == "'")
        ):
            return s[1:-1]
        return s

    for part in parts:
        # ── Standalone tokens ────────────────────────────────────
        if part.upper() == "DONE":
            actions.append(DoneAction(final_answer="Goal reached", reason=output.reason))
            continue
        if part.upper() == "FAIL":
            actions.append(FailAction(error="Task failed", reason=output.reason))
            continue

        # ── Python repr format (TypeAction(...), PressKeyAction(...)…) ──
        # DSPy sometimes emits this instead of the compact function-call format.
        repr_match = re.match(r"(\w+Action)\((.*)\)$", part, re.DOTALL)
        if repr_match:
            action = _parse_repr_action(repr_match.group(1), repr_match.group(2), output.reason)
            if action:
                actions.append(action)
                continue

        # ── Standard compact format: TYPE(...), PRESS_KEY(...) etc. ──
        # re.DOTALL so multi-line text inside TYPE() is captured correctly.
        match = re.match(r"(\w+)\((.*)\)$", part, re.DOTALL)
        if not match:
            continue

        a_type = match.group(1).upper()
        raw_param = match.group(2)

        # ── Browser actions (two-value params) ─────────────────
        if a_type.startswith("BROWSER_"):
            cleaned_params = []
            for p in raw_param.split(",", 1):
                p = p.strip().strip("'\"")
                if '=' in p:
                    p = p.split('=', 1)[1].strip("'\"")
                cleaned_params.append(p)

            if a_type == "BROWSER_NAVIGATE":
                url = cleaned_params[0] if cleaned_params else ""
                actions.append(BrowserNavigateAction(url=url, reason=output.reason))
            elif a_type == "BROWSER_CLICK":
                try:
                    index = int(cleaned_params[0]) if cleaned_params else 0
                    actions.append(BrowserClickAction(element_index=index, reason=output.reason))
                except ValueError:
                    pass
            elif a_type == "BROWSER_TYPE":
                try:
                    index = int(cleaned_params[0]) if len(cleaned_params) > 0 else 0
                    text = cleaned_params[1] if len(cleaned_params) > 1 else ""
                    actions.append(BrowserTypeAction(element_index=index, text=text, reason=output.reason))
                except ValueError:
                    pass
            continue

        # ── Regular desktop actions ─────────────────────────────
        if a_type == "PRESS_KEY":
            # Key names never contain meaningful outer quotes
            a_param = raw_param.strip().strip("'\"")
            actions.append(PressKeyAction(key=a_param, reason=output.reason))

        elif a_type == "TYPE":
            # Use safe outer-quote stripping so shell commands like
            #   TYPE(for file in *; do echo Hello > "$file"; done)
            # are preserved intact.
            a_param = _strip_outer_quotes(raw_param)
            actions.append(TypeAction(text=a_param, reason=output.reason))

        elif a_type == "WAIT":
            a_param = raw_param.strip().strip("'\"")
            try:
                sec = float(a_param) if a_param else 1.0
            except (ValueError, TypeError):
                sec = 1.0
            actions.append(WaitAction(seconds=sec, reason=output.reason))

        elif a_type == "DONE":
            a_param = _strip_outer_quotes(raw_param)
            actions.append(DoneAction(final_answer=a_param or "Goal reached", reason=output.reason))

        elif a_type == "SCROLL":
            a_param = raw_param.strip().strip("'\"")
            try:
                val = (
                    -10 if a_param.lower() == "down"
                    else 10 if a_param.lower() == "up"
                    else int(a_param)
                )
            except (ValueError, TypeError):
                val = -10
            actions.append(ScrollAction(amount=val, reason=output.reason))

        elif a_type == "FAIL":
            a_param = _strip_outer_quotes(raw_param)
            actions.append(FailAction(error=a_param or "Task failed", reason=output.reason))

    return actions or [FailAction(
        error=f"Failed to parse sequence: {output.action_param}",
        reason=output.reason,
    )]
