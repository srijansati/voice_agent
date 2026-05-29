"""
WiFi Router Troubleshooting Agent

Workflow structure:
    WifiTroubleshootingAgent (entry)
        ├── NoConnectionAgent        → can't connect at all
        │       ├── RouterPowerCheckAgent   → router has no power
        │       ├── NetworkVisibilityAgent  → WiFi name not visible
        │       └── AuthFailureAgent        → password / auth issues
        ├── SlowSpeedAgent           → slow internet
        │       ├── DeviceOverloadAgent     → too many devices
        │       └── SignalWeakAgent         → weak signal / distance
        ├── IntermittentDropsAgent   → connection keeps dropping
        └── RouterAdminAgent         → can't reach router settings page

LangChain plugin note:
    This agent is designed to use the LangChain plugin for LiveKit as the LLM backend.
    Install it alongside a LangChain provider:

        pip install livekit-agents[langchain] langchain-openai

    Then, in app.py, replace the LLM line in AgentSession with:

        from livekit.plugins import langchain as lk_langchain
        from langchain_openai import ChatOpenAI

        llm=lk_langchain.LLM(chat_model=ChatOpenAI(model="gpt-4o-mini")),

    The rest of the code below works unchanged — the LangChain LLM is a drop-in
    replacement for the OpenAI LLM already configured in AgentSession.
"""

from __future__ import annotations

from livekit.agents import Agent, RunContext, function_tool
from livekit.agents.llm.tool_context import ToolFlag


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

class WifiTroubleshootingAgent(Agent):
    """
    First agent in the workflow. Greets the user, asks what kind of WiFi
    problem they have, and hands off to the correct specialist agent.
    """

    def __init__(self) -> None:
        super().__init__(
            instructions="""
            You are a friendly WiFi router support assistant.

            Your only job in this step is to understand which category of problem
            the user is facing. Do NOT attempt to solve the problem yet.

            The four categories are:
              1. Cannot connect to WiFi at all
              2. Internet is slow or sluggish
              3. Connection keeps dropping / disconnecting
              4. Cannot access the router admin page (192.168.x.x)

            Once the user describes their issue, call the single tool that best
            matches the category. Ask one clarifying question if the category is
            genuinely unclear, then call the tool.
            """,
        )

    async def on_enter(self) -> None:
        self.session.generate_reply(
            instructions=(
                "Greet the user warmly and ask them to describe their WiFi problem. "
                "Mention the four common categories: cannot connect, slow speed, "
                "connection keeps dropping, and cannot reach router admin page."
            )
        )

    @function_tool(flags=ToolFlag.IGNORE_ON_ENTER)
    async def route_no_connection(self, ctx: RunContext) -> Agent:
        """
        Call this when the user says they cannot connect to the WiFi network at all,
        or that their device won't join the network.
        """
        return NoConnectionAgent()

    @function_tool(flags=ToolFlag.IGNORE_ON_ENTER)
    async def route_slow_speed(self, ctx: RunContext) -> Agent:
        """
        Call this when the user says their internet is slow, pages load slowly,
        or streaming keeps buffering.
        """
        return SlowSpeedAgent()

    @function_tool(flags=ToolFlag.IGNORE_ON_ENTER)
    async def route_intermittent_drops(self, ctx: RunContext) -> Agent:
        """
        Call this when the user says their WiFi keeps disconnecting, dropping,
        or cutting out periodically.
        """
        return IntermittentDropsAgent()

    @function_tool(flags=ToolFlag.IGNORE_ON_ENTER)
    async def route_router_admin(self, ctx: RunContext) -> Agent:
        """
        Call this when the user cannot access their router's admin/settings page
        (e.g. 192.168.1.1 or 192.168.0.1 does not load).
        """
        return RouterAdminAgent()


# ---------------------------------------------------------------------------
# Branch 1 — Cannot connect at all
# ---------------------------------------------------------------------------

class NoConnectionAgent(Agent):
    """
    Diagnoses a total inability to connect to WiFi.
    First checks whether the router has power, then branches further.
    """

    def __init__(self) -> None:
        super().__init__(
            instructions="""
            You are diagnosing why the user cannot connect to WiFi at all.

            Work through these questions one at a time — never ask more than one
            question at once:

              Step 1: "Is your router powered on? Can you see any lights on it?"

            Based on the answer call the correct tool:
              - No lights / no power → call router_has_no_power
              - Lights are on       → ask: "Can you see your WiFi network name
                                       in the list of available networks on your device?"
                  * No, not visible → call network_not_visible
                  * Yes, visible    → call network_visible_but_cant_connect
            """,
        )

    async def on_enter(self) -> None:
        self.session.generate_reply(
            instructions=(
                "Tell the user you will help them fix the connection issue. "
                "Start with the first diagnostic question: ask whether their router "
                "is powered on and whether they can see any lights on it."
            )
        )

    @function_tool(flags=ToolFlag.IGNORE_ON_ENTER)
    async def router_has_no_power(self, ctx: RunContext) -> Agent:
        """Call this when the router shows no lights / appears to have no power."""
        return RouterPowerCheckAgent()

    @function_tool(flags=ToolFlag.IGNORE_ON_ENTER)
    async def network_not_visible(self, ctx: RunContext) -> Agent:
        """
        Call this when the router is on but the user cannot see the WiFi network
        name in their device's network list.
        """
        return NetworkVisibilityAgent()

    @function_tool(flags=ToolFlag.IGNORE_ON_ENTER)
    async def network_visible_but_cant_connect(self, ctx: RunContext) -> Agent:
        """
        Call this when the WiFi name is visible but the device fails to join
        (wrong password, authentication error, etc.).
        """
        return AuthFailureAgent()


class RouterPowerCheckAgent(Agent):
    """Guides the user through power / cable checks for an unresponsive router."""

    def __init__(self) -> None:
        super().__init__(
            instructions="""
            You are helping the user with a router that appears to have no power.

            Walk them through these steps one at a time and confirm each step before
            moving on:

              1. "Check that the power cable is firmly plugged into both the router
                 and the wall outlet."
              2. "Try a different wall outlet or power strip to rule out a socket issue."
              3. "Unplug the power cable from the router, wait 10 seconds, then plug
                 it back in."
              4. "Wait 60 seconds for the router to fully boot and watch for any lights."

            After each step ask: "Do you see any lights on the router now?"

            If lights appear → summarise the fix and close the conversation warmly.
            If no lights after all steps → advise the user that the router hardware
            may be faulty and they should contact their ISP or replace the router.
            """,
        )

    async def on_enter(self) -> None:
        self.session.generate_reply(
            instructions=(
                "Acknowledge that the router seems to have no power. "
                "Begin with step 1: ask the user to check that the power cable is "
                "firmly plugged into both the router and the wall."
            )
        )


class NetworkVisibilityAgent(Agent):
    """
    Helps when the router is on but the WiFi SSID is not visible on the user's device.
    """

    def __init__(self) -> None:
        super().__init__(
            instructions="""
            The router is powered on but the user cannot see the WiFi network name.

            Guide them through these steps in order, one at a time:

              1. "Try toggling WiFi off and back on on your device — sometimes it
                 needs a refresh to pick up nearby networks."
              2. "Check whether the router's WiFi indicator light is on. If it is
                 blinking amber or off, the wireless radio may have been disabled."
              3. "Log into the router admin page from a wired connection (ethernet)
                 and verify that WiFi is enabled."
              4. "If WiFi cannot be enabled via admin, try a factory reset: hold the
                 reset button on the router for 10 seconds until all lights flash,
                 then wait 2 minutes for it to reboot."

            After each step ask if the network is now visible.
            If visible → congratulate and close warmly.
            If not visible after a factory reset → recommend contacting the ISP.
            """,
        )

    async def on_enter(self) -> None:
        self.session.generate_reply(
            instructions=(
                "Acknowledge that the router is on but the network is not visible. "
                "Start with step 1: ask the user to toggle WiFi off and on on their device."
            )
        )


class AuthFailureAgent(Agent):
    """Helps when the WiFi name is visible but the device cannot authenticate."""

    def __init__(self) -> None:
        super().__init__(
            instructions="""
            The user can see the WiFi network but cannot connect to it. This is
            usually a password or authentication issue.

            Work through these steps one at a time:

              1. "Make sure you are typing the password correctly — WiFi passwords
                 are case-sensitive. Try showing the password as you type it."
              2. "On your device, 'forget' the network and then reconnect from scratch,
                 re-entering the password carefully."
              3. "Restart your device and try connecting again."
              4. "Restart the router: unplug it for 30 seconds then plug it back in.
                 Wait 60 seconds and try connecting."
              5. "If nothing works, log into the router admin page from another device
                 and verify the WiFi password, or reset it to something simple to test."

            After each step ask if the device is now connected.
            If connected → congratulate and close warmly.
            """,
        )

    async def on_enter(self) -> None:
        self.session.generate_reply(
            instructions=(
                "Acknowledge that the network is visible but the device can't join it. "
                "Start with step 1: ask the user to double-check the password, "
                "reminding them it is case-sensitive."
            )
        )


# ---------------------------------------------------------------------------
# Branch 2 — Slow speed
# ---------------------------------------------------------------------------

class SlowSpeedAgent(Agent):
    """
    Diagnoses slow WiFi by first checking the number of connected devices
    and then the user's distance from the router.
    """

    def __init__(self) -> None:
        super().__init__(
            instructions="""
            You are diagnosing a slow WiFi issue.

            Ask these questions one at a time:

              Step 1: "Roughly how many devices are currently connected to your WiFi?
                       (phones, tablets, laptops, smart TVs, smart home devices, etc.)"

                  * Many devices (5+) → call too_many_devices
                  * Few devices       → continue to step 2

              Step 2: "How far are you from the router right now, and are there walls
                       or floors between you and it?"

                  * Far away or obstructed → call weak_signal
                  * Close by             → ask: "Have you restarted the router recently?"
                      - No  → advise a restart; if that doesn't help → call isp_issue
                      - Yes → call isp_issue
            """,
        )

    async def on_enter(self) -> None:
        self.session.generate_reply(
            instructions=(
                "Acknowledge the slow-speed complaint. Begin with step 1: ask how many "
                "devices are currently connected to their WiFi network."
            )
        )

    @function_tool(flags=ToolFlag.IGNORE_ON_ENTER)
    async def too_many_devices(self, ctx: RunContext) -> Agent:
        """Call this when many devices are connected and likely saturating bandwidth."""
        return DeviceOverloadAgent()

    @function_tool(flags=ToolFlag.IGNORE_ON_ENTER)
    async def weak_signal(self, ctx: RunContext) -> Agent:
        """Call this when the user is far from the router or separated by obstacles."""
        return SignalWeakAgent()

    @function_tool(flags=ToolFlag.IGNORE_ON_ENTER)
    async def isp_issue(self, ctx: RunContext) -> Agent:
        """
        Call this when the device count is low, the user is close to the router,
        and restarting has not helped — suggesting an ISP-side problem.
        """
        return ISPIssueAgent()


class DeviceOverloadAgent(Agent):
    """Guides the user when too many devices are consuming bandwidth."""

    def __init__(self) -> None:
        super().__init__(
            instructions="""
            The user has many devices connected, which is likely causing congestion.

            Suggest these steps in order:

              1. "Temporarily disconnect devices you are not actively using — especially
                 smart TVs, game consoles, or anything doing background updates."
              2. "Check if any device is running a large download or update in the
                 background and pause it."
              3. "Log into your router admin page and look for a 'QoS' (Quality of
                 Service) or 'Bandwidth Control' setting. You can prioritise traffic
                 for your most important device there."
              4. "Consider upgrading to a higher-speed internet plan if your household
                 regularly uses many devices simultaneously."

            After each step ask: "Has the speed improved?"
            """,
        )

    async def on_enter(self) -> None:
        self.session.generate_reply(
            instructions=(
                "Acknowledge that having many devices connected can slow things down. "
                "Suggest step 1: disconnect devices that are not currently in use."
            )
        )


class SignalWeakAgent(Agent):
    """Helps when the user is too far from the router or has physical obstructions."""

    def __init__(self) -> None:
        super().__init__(
            instructions="""
            The user is experiencing a weak signal due to distance or obstructions.

            Walk through these options:

              1. "Move closer to the router and check whether the speed improves."
              2. "Ensure the router is placed in a central, open location — not inside
                 a cabinet or behind a TV."
              3. "Switch to the 5 GHz band on your device if your router is dual-band —
                 it is faster at short range. Use 2.4 GHz for longer range."
              4. "Consider a WiFi extender or mesh node to cover areas the router
                 cannot reach well."

            After each suggestion ask: "Has the speed or signal strength improved?"
            """,
        )

    async def on_enter(self) -> None:
        self.session.generate_reply(
            instructions=(
                "Acknowledge the distance or obstruction issue. Begin with step 1: "
                "ask the user to move closer to the router to see if speed improves."
            )
        )


class ISPIssueAgent(Agent):
    """Handles cases where the problem is likely on the ISP side."""

    def __init__(self) -> None:
        super().__init__(
            instructions="""
            The user's slow speed is not caused by device count or signal — it is
            likely an ISP-side issue or a line problem.

            Guide them through:

              1. "Run a speed test at fast.com or speedtest.net and tell me the
                 download speed you get."
              2. "Compare it to the speed advertised in your internet plan."
              3. "If significantly below the plan speed, restart the router one more
                 time: unplug for 60 seconds, then replug."
              4. "If still slow after restart, contact your ISP and report the results
                 of the speed test — they may have a line fault in your area."

            Offer to help the user note down the speed test results to report to
            their ISP.
            """,
        )

    async def on_enter(self) -> None:
        self.session.generate_reply(
            instructions=(
                "Explain that the issue seems to be outside the home network and likely "
                "involves the ISP. Ask the user to run a speed test first."
            )
        )


# ---------------------------------------------------------------------------
# Branch 3 — Intermittent drops
# ---------------------------------------------------------------------------

class IntermittentDropsAgent(Agent):
    """
    Diagnoses a connection that keeps dropping.
    Checks interference, channel congestion, and hardware heat.
    """

    def __init__(self) -> None:
        super().__init__(
            instructions="""
            You are diagnosing a WiFi connection that keeps dropping or disconnecting.

            Ask these questions one at a time:

              Step 1: "How often does it drop — every few minutes, every hour, or
                       randomly throughout the day?"

              Step 2: "Are there any appliances near the router such as a microwave,
                       cordless phone, or baby monitor?"

                  * Yes → explain interference and suggest moving the router or
                    switching to 5 GHz, then continue to step 3.

              Step 3: "Is your router in a well-ventilated spot, or is it enclosed /
                       warm to the touch?"

                  * Hot → call router_overheating

              Step 4: "Have you tried changing the WiFi channel in the router settings?
                       Neighbouring routers on the same channel can cause drops."

              If none of the above apply, advise the user to:
              - Update the router firmware.
              - Perform a factory reset as a last resort.
              - Contact their ISP if drops persist.
            """,
        )

    async def on_enter(self) -> None:
        self.session.generate_reply(
            instructions=(
                "Acknowledge the intermittent drop issue and explain you will work "
                "through possible causes. Begin with step 1: ask how often the drops occur."
            )
        )

    @function_tool(flags=ToolFlag.IGNORE_ON_ENTER)
    async def router_overheating(self, ctx: RunContext) -> Agent:
        """Call this when the user confirms the router is hot or in an enclosed space."""
        return RouterOverheatAgent()


class RouterOverheatAgent(Agent):
    """Guides the user when the router is overheating and causing drops."""

    def __init__(self) -> None:
        super().__init__(
            instructions="""
            The router may be overheating, causing it to disconnect to protect itself.

            Walk the user through:

              1. "Turn off the router for 10 minutes to let it cool down, then turn
                 it back on."
              2. "Move the router to an open shelf or desk where air can circulate
                 freely around it."
              3. "Make sure the router's vents are not blocked by books, cables, or
                 other devices."
              4. "If the router feels hot even in an open spot, the internal fan or
                 cooling may be failing — contact your ISP or consider replacing it."

            After each step ask: "Has the dropping stopped?"
            """,
        )

    async def on_enter(self) -> None:
        self.session.generate_reply(
            instructions=(
                "Explain that an overheating router can cause random disconnections. "
                "Start with step 1: ask the user to turn the router off for 10 minutes "
                "to let it cool."
            )
        )


# ---------------------------------------------------------------------------
# Branch 4 — Cannot access router admin page
# ---------------------------------------------------------------------------

class RouterAdminAgent(Agent):
    """
    Helps users who cannot reach their router's admin/settings page.
    """

    def __init__(self) -> None:
        super().__init__(
            instructions="""
            You are helping the user access their router's admin page.

            Ask and work through these steps:

              Step 1: "Which address are you trying? The most common defaults are
                       192.168.1.1 and 192.168.0.1. Have you tried both?"

                  * Has not tried both → guide them to try both in a browser.

              Step 2: "Are you connected to the router via WiFi or an ethernet cable?
                       Try a wired connection — it is more reliable for admin access."

              Step 3: "Make sure you are typing the address into the browser's address
                       bar, not the search box."

              Step 4: "Check the sticker on the back or bottom of your router — it
                       usually lists the admin URL, default username, and password."

              Step 5: If still failing:
                  "Try restarting the router and attempt the admin page again after
                   it has fully booted (about 60 seconds)."

              Step 6: If credentials are wrong:
                  "The default credentials are usually admin/admin or admin/password.
                   If you have changed them and forgotten, a factory reset will restore
                   the defaults — hold the reset button for 10 seconds."

            After each step ask: "Were you able to reach the admin page?"
            """,
        )

    async def on_enter(self) -> None:
        self.session.generate_reply(
            instructions=(
                "Acknowledge the router admin access problem. Begin with step 1: "
                "ask which IP address the user has been trying and whether they have "
                "tried both 192.168.1.1 and 192.168.0.1."
            )
        )
