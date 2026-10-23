
import re
import spacy

nlp = spacy.blank("en")

APPLIANCE_ALIASES = {
    "ac": ["ac", "air conditioner", "air conditioning"],
    "fan": ["fan", "ceiling fan", "bldc fan"],
    "refrigerator": ["fridge", "refrigerator"],
}

AC_MODES = {
    "off": ["off", "switch off", "turn off"],
    "eco": ["eco", "eco mode", "energy saving mode"],
    "dry": ["dry", "dry mode", "dehumidify"],
    "max_cool": [
        "max cool", "maximum cooling",
        "turbo cool", "max cooling"
    ],
}

POWER_ON_PHRASES = [
    "turn on", "switch on", "start",
    "activate", "keep running"
]

POWER_OFF_PHRASES = [
    "turn off", "switch off", "shut down",
    "stop", "deactivate"
]

ENERGY_SAVING_PHRASES = [
    "save energy", "save electricity",
    "reduce power consumption", "energy efficient",
    "reduce energy usage"
]

AWAY_PHRASES = [
    "while i'm away", "while i am away",
    "when i'm away", "when i am away",
    "when nobody is home", "when no one is home"
]


def contains_phrase(text, phrases):
    """Check if any of the given phrases appear in the text.
    
    This version uses a simpler approach: it checks if all words in a phrase
    appear in the text (in any order), making it suitable for natural language
    commands where intervening words are common.
    """
    text_lower = text.lower()
    for phrase in phrases:
        phrase_words = phrase.lower().split()
        # Check if all words in the phrase appear in the text
        if all(word in text_lower for word in phrase_words):
            return True
    return False


def find_appliance(text):
    aliases = sorted(
        [(alias, appliance)
         for appliance, names in APPLIANCE_ALIASES.items()
         for alias in names],
        key=lambda item: len(item[0]),
        reverse=True
    )
    for alias, appliance in aliases:
        if re.search(r"\b" + re.escape(alias) + r"\b", text):
            return appliance
    return None


def find_ac_mode(text):
    for mode, phrases in AC_MODES.items():
        for phrase in sorted(phrases, key=len, reverse=True):
            if re.search(r"\b" + re.escape(phrase) + r"\b", text):
                return mode
    return None


def parse_command(command):
    if not isinstance(command, str) or not command.strip():
        return {
            "command": command, "intent": "unknown",
            "appliance": None, "action": None, "value": None,
            "confidence": 0.0, "needs_clarification": True,
            "message": "Please enter a command."
        }

    text = re.sub(r"\\s+", " ", command.lower().strip())
    tokens = [token.text for token in nlp(text)]

    if contains_phrase(text, ENERGY_SAVING_PHRASES):
        away = contains_phrase(text, AWAY_PHRASES)
        return {
            "command": command, "intent": "energy_saving",
            "appliance": "whole_home", "action": "optimize_energy",
            "value": {
                "energy_saving": True,
                "occupancy_context": "away" if away else "unspecified"
            },
            "confidence": 0.95 if away else 0.85,
            "needs_clarification": False,
            "message": "Energy-saving preference identified.",
            "tokens": tokens
        }

    appliance = find_appliance(text)
    if appliance is None:
        return {
            "command": command, "intent": "unknown",
            "appliance": None, "action": None, "value": None,
            "confidence": 0.0, "needs_clarification": True,
            "message": "Which appliance do you mean?",
            "tokens": tokens
        }

    turn_on = contains_phrase(text, POWER_ON_PHRASES)
    turn_off = contains_phrase(text, POWER_OFF_PHRASES)

    if turn_on and turn_off:
        return {
            "command": command, "intent": "ambiguous",
            "appliance": appliance, "action": None, "value": None,
            "confidence": 0.0, "needs_clarification": True,
            "message": "The command contains conflicting power actions.",
            "tokens": tokens
        }

    if appliance == "ac":
        mode = find_ac_mode(text)
        if mode is not None:
            return {
                "command": command, "intent": "set_mode",
                "appliance": "ac", "action": "set_mode",
                "value": mode, "confidence": 0.98,
                "needs_clarification": False,
                "message": f"AC mode identified: {mode}.",
                "tokens": tokens
            }

    if turn_on or turn_off:
        value = "on" if turn_on else "off"
        return {
            "command": command, "intent": "power_control",
            "appliance": appliance, "action": "set_power",
            "value": value, "confidence": 0.95,
            "needs_clarification": False,
            "message": f"{appliance} power command identified: {value}.",
            "tokens": tokens
        }

    return {
        "command": command, "intent": "unknown",
        "appliance": appliance, "action": None, "value": None,
        "confidence": 0.0, "needs_clarification": True,
        "message": "Please specify the action or supported operating mode.",
        "tokens": tokens
    }
