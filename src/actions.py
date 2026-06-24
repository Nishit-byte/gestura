import pyautogui

pyautogui.FAILSAFE = False   # disable corner-escape so hand can reach edges
pyautogui.PAUSE = 0.005

# All possible actions the user can assign to any gesture
AVAILABLE_ACTIONS = [
    "nothing",
    "left_click",
    "right_click",
    "double_click",
    "scroll_up",
    "scroll_down",
    "volume_up",
    "volume_down",
    "mute",
    "screenshot",
    "move_mouse",       # special: uses fingertip position
]

ACTION_LABELS = {
    "nothing":       "Do Nothing",
    "left_click":    "Left Click",
    "right_click":   "Right Click",
    "double_click":  "Double Click",
    "scroll_up":     "Scroll Up",
    "scroll_down":   "Scroll Down",
    "volume_up":     "Volume Up",
    "volume_down":   "Volume Down",
    "mute":          "Mute / Unmute",
    "screenshot":    "Screenshot",
    "move_mouse":    "Move Mouse",
}


def perform_action(action_name: str):
    """Execute the OS action corresponding to action_name."""
    if action_name == "nothing":
        pass

    elif action_name == "left_click":
        pyautogui.click()

    elif action_name == "right_click":
        pyautogui.rightClick()

    elif action_name == "double_click":
        pyautogui.doubleClick()

    elif action_name == "scroll_up":
        pyautogui.scroll(5)

    elif action_name == "scroll_down":
        pyautogui.scroll(-5)

    elif action_name == "volume_up":
        pyautogui.hotkey("volumeup")

    elif action_name == "volume_down":
        pyautogui.hotkey("volumedown")

    elif action_name == "mute":
        pyautogui.hotkey("volumemute")

    elif action_name == "screenshot":
        pyautogui.hotkey("ctrl", "printscreen")